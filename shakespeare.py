import argparse
import json
import os
import pdb
import pickle
import random
import re
import string
from collections import defaultdict
from glob import glob
from copy import deepcopy
import numpy as np
import torch; #torch.backends.cudnn.deterministic = False; 
torch.backends.cudnn.benchmark = True
from torch import nn
from torch import optim
from torch.nn.utils import rnn
from hyper import wrapped_fedex
from hyper import Server


BATCH = 100
CHARMAP = defaultdict(lambda: 1)
CHARMAP.update({char: i+2 for i, char in enumerate(string.printable)})
VOCAB = len(set(CHARMAP.values())) + 1
DATA = './data/shakespeare/'
SERVER = lambda: {
                  'lr': 10.0 ** np.random.uniform(low=-1.0, high=1.0),
                  'momentum': np.random.choice([0.0, 0.9]),
                  'step': 1,
                  'gamma': 1.0 - 10.0 ** np.random.uniform(low=-4.0, high=-2.0),
                  }
CLIENT = lambda: {
                  'lr': 10.0 ** np.random.uniform(low=-4.0, high=0.0),
                  'momentum': np.random.uniform(low=0.0, high=1.0),
                  'weight_decay': 10.0 ** np.random.uniform(low=-5.0, high=-1.0),
                  'epochs': 1, 
                  'batch': 2 ** np.random.choice(np.arange(3, 8)),
                  'mu': 10.0 ** np.random.uniform(low=-5.0, high=0.0),
                  'dropout': np.random.uniform(low=0.0, high=0.5),
                  }


def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('logdir')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--debug', default=0, type=int,
                        help='run in DEBUG mode if >0; sets number of clients and batches')

    # wrapper algorithm settings
    parser.add_argument('--rounds', default=800, type=int, 
                        help='maximum number of communication rounds')
    parser.add_argument('--total', default=4000, type=int, 
                        help='total number of communication rounds')
    parser.add_argument('--rate', default=3, type=int, help='elimination rate')
    parser.add_argument('--elim', default=0, type=int, help='number of elimination rounds')
    parser.add_argument('--eval', default=1, type=int, help='number of evaluation rounds')
    parser.add_argument('--discount', default=0.0, type=float,
                        help='discount factor for computing the validation score of an arm')

    # FedEx settings
    parser.add_argument('--batch', default=10, type=int, help='number of tasks per round')
    parser.add_argument('--configs', default=1, type=int,
                        help='''number of configs to optimize over with FedEx (use 1 for FedAvg):
                                - <-1: sample a random number between 1 and abs(args.configs)
                                - =-1: sample the number of arms given by the wrapper
                                - =0: sample a random number between 1 and the number of arms
                                - >0: sample the provided number, ignoring the number of arms''')
    parser.add_argument('--lr_only', action='store_true', help='tune only learning rate as a hyperparameter')
    parser.add_argument('--eps', default=0.1, type=float, help='multiplicative perturbation to client config, eps=0 is fedavg')
    parser.add_argument('--uniform', action='store_true',
                        help='run FedEx over a product set of single-parameter uniform grids')
    parser.add_argument('--random', action='store_true',
                        help='run FedEx over a product set of single-parameter random grids')
    parser.add_argument('--eta0', default=0.0, type=float, 
                        help='FedEx initial step size; if 0.0 uses FedEx default')
    parser.add_argument('--sched', default='aggressive', type=str, help='FedEx step size sched')
    parser.add_argument('--cutoff', default=0.0, type=float, 
                        help='stop updating FedEx config distribution if entropy below this cutoff')
    parser.add_argument('--baseline', default=-1.0, type=float,
                        help='''how FedEx computes the baseline:
                                - >=-1.0,<0.0: sample discount factor from [0.0, abs(args.baseline))
                                - =0.0: use the most recent value
                                - >0.0,<1.0: use geometrically discounted mean with this factor
                                - =1.0: use the mean of all values''')
    parser.add_argument('--diff', action='store_true', 
                        help='use difference between refine and global as FedEx objective')
    parser.add_argument('--stop', action='store_true',
                        help='stop updating FedEx config distribution after last elimination')

    # evaluation settings
    parser.add_argument('--mle', action='store_true', help='use MLE config at test time')
    parser.add_argument('--loss', action='store_true', help='use loss instead of error')
    parser.add_argument('--eval_global', action='store_true', help='use global error as elimination metric instead of refine')

    # data settings
    parser.add_argument('--length', default=80, type=int, help='sequence length')
    parser.add_argument('--iid', action='store_true', help='client data is iid')
    parser.add_argument('--val', default=0.1, type=float, help='val proportion')
    parser.add_argument('--test', default=0.1, type=float, help='test proportion')

    # model settings
    parser.add_argument('--hidden', default=256, type=int, help='number of hidden units in LSTM')
    parser.add_argument('--layers', default=2, type=int, help='number of layers in LSTM')

    return parser.parse_args()


def line2data(text, length=80):

    text = re.sub('  *', ' ', text.replace('\n', ' '))
    return [[text[i:i+length], text[i+length]] for i in range(len(text)-length)]


def file2tensor(fname, length=80):

    with open(fname, 'r') as f:
        data = line2data(f.read(), length=length)
    X, Y = torch.zeros(len(data), length).long(), torch.empty(len(data)).long()
    for i, (x, y) in enumerate(data):
        for j, char in enumerate(x):
            X[i,j] = CHARMAP[char]
        Y[i] = CHARMAP[y]
    return X, Y


def get_loader(task, val=0.1, test=0.1, iid=False):
    m = len(torch.load(task)['Y'])
    if m < 1.0 / min(1.0-val-test, val, test):
        return None
    partitions = {'train': (0, int((1.0-val-test)*m)), 
                  'val': (int((1.0-val-test)*m), int((1.0-test)*m)), 
                  'test': (int((1.0-test)*m), m)}
    if iid:
        randperm = torch.randperm(m)
    dump = torch.load(task)
    X, Y = (dump['X'][randperm], dump['Y'][randperm]) if iid else (dump['X'], dump['Y'])
    data = {key: (X[a:b].pin_memory(), Y[a:b].pin_memory()) for key, (a, b) in partitions.items()}
    def loader(*args):
        output = []
        for arg in args:
            Xarg, Yarg = data[arg]
            output.append(Xarg.cuda(non_blocking=True))
            output.append(Yarg.cuda(non_blocking=True))
        return output
    return loader


class CharLSTM(nn.Module):

    def __init__(self, input_size=8, hidden_size=256, **kwargs):

        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=VOCAB, 
                                      embedding_dim=input_size, padding_idx=0)
        self.lstm= nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                           batch_first=True, bidirectional=False, **kwargs)
        self.linear = nn.Linear(hidden_size, VOCAB)

    def forward(self, X, lengths):

        X = self.embedding(X)
        X = rnn.pack_padded_sequence(X, lengths, batch_first=True)
        self.lstm.flatten_parameters()
        X, _ = self.lstm(X)
        X, _ = rnn.pad_packed_sequence(X, batch_first=True)
        return self.linear(X[:,-1])


def get_lengths(X):

    lengths = X.shape[1] - (X == 0).sum(1)
    return lengths.sort(0, descending=True)


def get_prox(model, criterion=nn.CrossEntropyLoss(), mu=0.0):

    if not mu:
        return criterion

    mu *= 0.5
    model0 = [param.data.clone() for param in model.parameters()]

    def objective(*args, **kwargs):

        prox = sum((param-param0).pow(2).sum() 
                   for param, param0 in zip(model.parameters(), model0))
        return criterion(*args, **kwargs) + mu * prox

    return objective


def train(model, X, Y, batch=32, dropout=0.0, epochs=1, mu=0.0, **kwargs):

    optimizer = optim.SGD(model.parameters(), **kwargs)
    criterion = get_prox(model, mu=mu)
    model.lstm.dropout = dropout
    model.train()
    m = len(Y)
    for e in range(epochs):
        randperm = torch.randperm(m)
        X, Y = X[randperm], Y[randperm]
        for i in range(0, m, batch):
            Xbatch, Ybatch =X[i:i+batch], Y[i:i+batch]
            lengths, sortperm = get_lengths(Xbatch)
            pred = model(Xbatch[sortperm], lengths.cpu())
            loss = criterion(pred, Ybatch[sortperm])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def test_batch(model, X, Y):

    lengths, sortperm = get_lengths(X)
    pred = model(X[sortperm], lengths.cpu())
    Yperm = Y[sortperm]
    return (Yperm != pred.argmax(1)).sum().float(), nn.CrossEntropyLoss(reduction='sum')(pred, Yperm).float()


def test(model, X, Y, batch=BATCH):

    model.eval()
    with torch.no_grad():
        errors, losses = zip(*(test_batch(model, X[i:i+batch], Y[i:i+batch]) 
                               for i in range(0, len(Y), batch)))
        return float(sum(errors)) / len(Y), float(sum(losses)) / len(Y)


def main():

    args = parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = []
    for i, fname in enumerate(glob(os.path.join(DATA, '*.txt'))):
        task = os.path.join(DATA, fname.split('/')[-1][:-4] + '-' + str(args.length) + 'char.pt')
        if not os.path.isfile(task):
            print('\rCaching task', i+1, end='')
            dump = {}
            dump['X'], dump['Y'] = file2tensor(fname, length=args.length)
            torch.save(dump, task)
        loader = get_loader(task, val=args.val, test=args.test, iid=args.iid)
        if not loader is None:
            tasks.append(loader)
    if args.debug:
        tasks = tasks[:args.debug]
        print('DEBUG MODE')
    print('\rCompleted cache-check')

    def local_train(model, X, Y, **kwargs):

        if args.debug:
            return train(model, X[:args.debug*args.batch], Y[:args.debug*args.batch], **kwargs)
        return train(model, X, Y, **kwargs)

    def local_test(model, X, Y, **kwargs):

        return test(model, X, Y, **kwargs)[args.loss]

    def get_server():

        model = CharLSTM(hidden_size=args.hidden, num_layers=args.layers)
        return Server(model, tasks, local_train, local_test, batch=args.batch, **SERVER())


    def get_client(n_clients=1):
        '''performs local tuning for each hyperparameter'''
        if args.lr_only:
            return [SIMPLE_CLIENT()]

        initial_client = CLIENT()
        client_arr = [initial_client]
        eps = args.eps

        for i in range(n_clients-1):
            other_client = deepcopy(initial_client)
            
            log_lr = np.log10(other_client['lr'])
            other_client['lr'] = 10 ** np.clip(log_lr + np.random.uniform(-eps * 4, eps * 4), -4.0, 0.0)
            
            other_client['momentum'] = np.clip(initial_client['momentum'] + np.random.uniform(-eps, eps), 0, 1.0)
            
            log_wd = np.log10(other_client['weight_decay'])
            other_client['weight_decay'] = 10 ** np.clip(log_wd + np.random.uniform(-eps*4, eps*4),-5.0, -1.0)

            log_batch = int(np.log2(other_client['batch']))
            other_client['batch'] = 2 ** np.clip(np.random.choice([log_batch-1, log_batch, log_batch+1]), 3, 7)
            
            log_mu = np.log10(other_client['mu'])
            other_client['mu'] = 10 ** np.clip(log_mu + np.random.uniform(-eps*5, eps*5), -5.0 , 0.0)
            
            other_client['dropout'] = np.clip(initial_client['dropout'] + np.random.uniform(-eps*0.5, eps*0.5),0, 0.5)

            client_arr.append(other_client)

        return [UNIFORM()] if args.uniform else [RANDOM()] if args.random else client_arr


    print('Tuning',
          'FedAvg' if args.configs == 1 and not (args.uniform or args.random) else 'FedEx',
          'on Shakespeare')
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    wrapped_fedex(
                  get_server, 
                  get_client, 
                  num_configs=args.configs,
                  prod=args.uniform or args.random,
                  stepsize_init=args.eta0 if args.eta0 else 'auto',
                  stepsize_sched=args.sched,
                  cutoff=args.cutoff, 
                  baseline_discount=args.baseline,
                  diff=args.diff, 
                  mle=args.mle, 
                  logdir=args.logdir, 
                  val_discount=args.discount,
                  last_stop=args.stop,
                  max_resources=args.rounds, 
                  total_resources=args.total,
                  elim_rate=args.rate, 
                  num_elim=args.elim,
                  num_eval=args.eval,
                  eval_global=args.eval_global
                  )


if __name__ == '__main__':

    main()
