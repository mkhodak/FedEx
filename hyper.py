
import argparse
import os
import pdb
import pickle
import random
from copy import deepcopy
from glob import glob
from heapq import nsmallest
from itertools import product
from math import ceil
from math import log
from operator import itemgetter
import numpy as np
import torch
from numpy.linalg import norm
from scipy.special import logsumexp
from tensorboardX import SummaryWriter
from torch import optim


def discounted_mean(trace, factor=1.0):

    weight = factor ** np.flip(np.arange(len(trace)), axis=0)

    return np.inner(trace, weight) / weight.sum()


class FedEx:
    '''runs hyperparameter optimization given a federated learning server'''

    def entropy(self):

        entropy = 0.0
        for probs in product(*(theta[theta>0.0] for theta in self._theta)):
            prob = np.prod(probs)
            entropy -= prob * np.log(prob)
        return entropy

    def mle(self):
    
        return np.prod([theta.max() for theta in self._theta])

    def __init__(
                 self, 
                 server, 
                 configs, 
                 eta0='auto', 
                 sched='auto', 
                 cutoff=0.0, 
                 baseline=0.0, 
                 diff=False,
                 ):
        '''
        Args:
            server: Object that implements two methods, 'communication_round' and 'full_evaluation'
                    taking as input a single argument, 'get_config', itself a function that takes 
                    no inputs and outputs an element of the provided list 'configs'. 
                    - 'communication_round' samples a batch of clients, assigns a config to each 
                    using 'get_config', and runs local training using that config. It then 
                    aggregates the local models to to take a training step and returns three lists 
                    or arrays: a list of each client's validation error before local training, a 
                    list of each client's validation error after local training, and a list of each 
                    client's weight (e.g. size of its validation set). 
                    - 'full_evaluation' assigns a config to each client using 'get_config' and runs
                    local training using that config. It then returns three lists or arrays: a list
                    of each client's test error before local training, a list of each client's test
                    error after local training, and a list of each client's weight (e.g. size of 
                    its test set).
            configs: list of configs used for local training and testing by 'server' 
                     OR dict of (string, list) pairs denoting a grid of configs
            eta0: base exponentiated gradient step size; if 'auto' uses sqrt(2*log(len(configs)))
            sched: learning rate schedule for exponentiated gradient:
                    - 'adaptive': uses eta0 / sqrt(sum of squared gradient l-infinity norms)
                    - 'aggressive': uses eta0 / gradient l-infinity norm
                    - 'auto': uses eta0 / sqrt(t) for t the number of rounds
                    - 'constant': uses eta0
                    - 'scale': uses sched * sqrt(2 * log(len(configs)))
            cutoff: entropy level below which to stop updating the config probability and use MLE
            baseline: discount factor when computing baseline; 0.0 is most recent, 1.0 is mean
            diff: if True uses performance difference; otherwise uses absolute performance
        '''

        self._server = server
        self._configs = configs
        self._grid = [] if type(configs) == list else sorted(configs.keys())

        sizes = [len(configs[param]) for param in self._grid] if self._grid else [len(configs)]
        self._eta0 = [np.sqrt(2.0 * np.log(size)) if eta0 == 'auto' else eta0 for size in sizes]
        self._sched = sched
        self._cutoff = cutoff
        self._baseline = baseline
        self._diff = diff
        self._z = [np.full(size, -np.log(size)) for size in sizes]
        self._theta = [np.exp(z) for z in self._z]

        self._store = [0.0 for _ in sizes]
        self._stopped = False
        self._trace = {'global': [], 'refine': [], 'entropy': [self.entropy()], 'mle': [self.mle()]}

    def stop(self):

        self._stopped = True

    def sample(self, mle=False, _index=[]):
        '''samples from configs using current probability vector'''

        if mle or self._stopped:
            if self._grid:
                return {param: self._configs[param][theta.argmax()] 
                        for theta, param in zip(self._theta, self._grid)}
            return self._configs[self._theta[0].argmax()]
        _index.append([np.random.choice(len(theta), p=theta) for theta in self._theta])

        if self._grid:
            return {param: self._configs[param][i] for i, param in zip(_index[-1], self._grid)}
        return self._configs[_index[-1][0]]

    def settings(self):
        '''returns FedEx input settings'''

        output = {'configs': deepcopy(self._configs)}
        output['eta0'], output['sched'] = self._eta0, self._sched
        output['cutoff'], output['baseline'] = self._cutoff, self._baseline 
        if self._trace['refine']:
            output['theta'] = self.theta()
        return output

    def step(self):
        '''takes exponentiated gradient step (calls 'communication_round' once)'''

        index = []
        before, after, weight = self._server.communication_round(lambda: self.sample(_index=index))        
        before, after = np.array(before), np.array(after)
        weight = np.array(weight, dtype=np.float64) / sum(weight)

        if self._trace['refine']:
            trace = self.trace('refine')
            if self._diff:
                trace -= self.trace('global')
            baseline = discounted_mean(trace, self._baseline)
        else:
            baseline = 0.0
        self._trace['global'].append(np.inner(before, weight))
        self._trace['refine'].append(np.inner(after, weight))
        if not index:
            self._trace['entropy'].append(0.0)
            self._trace['mle'].append(1.0)
            return

        for i, (z, theta) in enumerate(zip(self._z, self._theta)):
            grad = np.zeros(len(z))
            for idx, s, w in zip(index, after-before if self._diff else after, weight):
                grad[idx[i]] += w * (s - baseline) / theta[idx[i]]
            if self._sched == 'adaptive':
                self._store[i] += norm(grad, float('inf')) ** 2
                denom = np.sqrt(self._store[i])
            elif self._sched == 'aggressive':
                denom = 1.0 if np.all(grad == 0.0) else norm(grad, float('inf'))
            elif self._sched == 'auto':
                self._store[i] += 1.0
                denom = np.sqrt(self._store[i])
            elif self._sched == 'constant':
                denom = 1.0
            elif self._sched == 'scale':
                denom = 1.0 / np.sqrt(2.0 * np.log(len(grad))) if len(grad) > 1 else float('inf')
            else:
                raise NotImplementedError
            eta = self._eta0[i] / denom
            z -= eta * grad
            z -= logsumexp(z)
            self._theta[i] = np.exp(z)

        self._trace['entropy'].append(self.entropy())
        self._trace['mle'].append(self.mle())
        if self._trace['entropy'][-1] < self._cutoff:
            self.stop()

    def test(self, mle=False):
        '''evaluates found config (calls 'full_evaluation' once)
        Args:
            mle: use MLE config instead of sampling
        Returns:
            output of 'full_evaluation'
        '''

        before, after, weight = self._server.full_evaluation(lambda: self.sample(mle=mle))
        return {'global': np.inner(before, weight) / weight.sum(),
                'refine': np.inner(after, weight) / weight.sum()}

    def theta(self):
        '''returns copy of config probability vector'''

        return deepcopy(self._theta)

    def trace(self, key):
        '''returns trace of one of three tracked quantities
        Args:
            key: 'entropy', 'global', or 'refine'
        Returns:
            numpy vector with length equal to number of calls to 'step'
        '''

        return np.array(self._trace[key])


def frac(p, q):

    return str(p) + '/' + str(q)


class Server:
    '''object for federated training implementing methods required by FedEx'''

    def _set_test_state(self):

        state = (np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())
        if self._state is None:
            self._state = state
        else:
            np.random.set_state(self._state[0])
            torch.set_rng_state(self._state[1])
            torch.cuda.set_rng_state(self._state[2])
        return state

    def _reset_state(self, state):

        np.random.set_state(state[0])
        torch.set_rng_state(state[1])
        torch.cuda.set_rng_state(state[2])

    def __init__(
                 self, 
                 model, 
                 clients, 
                 train, 
                 test, 
                 lr=1.0, 
                 momentum=0.0, 
                 step=1, 
                 gamma=1.0, 
                 batch=10,
                 state=None,
                 ):
        '''
        Args:
            model: PyTorch model
            clients: list of clients, each a function that takes one or more strings 'train',
                     'val', 'test' and returns, as one tuple, input and output tensors for each
            train: method that takes as argument a PyTorch model, an input tensor, an output
                   tensor, and optional kwargs and returns the same PyTorch model
            test: method that takes as argument a PyTorch model, an input tensor, and an output
                  tensor and returns the model's error
            lr: server learning rate
            momentum: server momentum
            step: server learning rate decay interval
            gamma: server learning rate decay factor
            batch: number of clients to sample per communication round
            state: np.random, torch, torch.cuda random state tuple; if None uses current states
        '''

        self._model = model
        self._clients = clients
        self._train = train
        self._test = test
        self._opt = optim.SGD(self._model.parameters(), lr=lr, momentum=momentum)
        self._sched = optim.lr_scheduler.StepLR(self._opt, step, gamma=gamma)
        self._batch = batch
        self._state = state
        self._reset_state(self._set_test_state())

    def communication_round(self, get_config):
        '''runs one step of local training and model aggregation
        Args:
            get_config: returns kwargs for 'train' as a dict
        Returns:
            np.array objects for global val error, local val error, and val size of each client
        '''

        self._model.cuda()
        before, after, weight = [np.zeros(self._batch) for _ in range(3)]
        total = 0.0

        for i in range(self._batch):
            Xtrain, Ytrain, Xval, Yval = random.choice(self._clients)('train', 'val')
            before[i] = self._test(self._model, Xval, Yval)
            model = self._train(deepcopy(self._model), Xtrain, Ytrain, **get_config())
            after[i] = self._test(model, Xval, Yval)
            weight[i] = len(Yval)
            total += len(Ytrain)
            if i:
                for agg, param in zip(aggregate.parameters(), model.parameters()):
                    agg.data += len(Ytrain) * param.data
            else:
                for param in model.parameters():
                    param.data *= len(Ytrain)
                aggregate = model

        self._opt.zero_grad()
        for agg, param in zip(aggregate.parameters(), self._model.parameters()):
            param.grad = param.data - agg / total
        self._opt.step()
        self._opt.zero_grad()
        self._sched.step()
        self._model.cpu()
        return before, after, weight

    def full_evaluation(self, get_config):
        '''evaluates personalization on each client
        Args:
            get_config: returns kwargs for 'train' as a dict
        Returns:
            np.array objects for global test error, local test error, and test size of each client
        '''

        state = self._set_test_state()
        self._model.cuda()
        before, after, weight = [np.zeros(len(self._clients)) for _ in range(3)]
        for i, client in enumerate(self._clients):
            Xtrain, Ytrain, Xtest, Ytest = client('train', 'test')
            before[i] = self._test(self._model, Xtest, Ytest)
            after[i] = self._test(self._train(deepcopy(self._model), 
                                              Xtrain, Ytrain, **get_config()), 
                                  Xtest, Ytest)
            weight[i] = len(Ytest)
            print('\r\tEvaluated client', frac(i+1, len(self._clients)),
                  '    global error:', round(np.inner(before, weight) / weight.sum(), 4),
                  '    refine error:', round(np.inner(after, weight) / weight.sum(), 4), 
                  end=32*' ')
        self._model.cpu()
        self._reset_state(state)
        return before, after, weight


def random_search(max_resources=500, total_resources=2000):
    '''returns a random search rate and schedule for use by 'successive_elimination'
    Args:
        max_resources: most resources (steps) assigned to single arm
        total_resources: overall resource limit
    Returns:
        elimination rate as an int, elimination schedule as a list
    '''

    assert max_resources > 0, "max_resources must be positive"
    return int(total_resources / max_resources), [max_resources]


def get_schedule(
                 max_resources=500, 
                 total_resources=2000, 
                 elim_rate=3, 
                 num_elim=0, 
                 num_eval=1,
                 ):
    '''returns rate and schedule for use by 'successive_elimination'
    Args:
        max_resources: most resources (steps) assigned to single arm
        total_resources: overall resource limit
        elim_rate: multiplicative elimination rate
        num_elim: number of elimination rounds; if 0 runs random search
        num_eval: number of evaluation rounds
    Returns:
        elimination rate as an int, elimination schedule as a list, evaluation schedule as a list
    '''

    assert max_resources <= total_resources, "max_resources cannot be greater than total_resources"
    assert elim_rate > 1, "elim_rate must be greater than 1"
    assert num_eval <= total_resources, "num_eval cannot be greater than total_resources"

    if num_elim:
        diff = total_resources - max_resources
        geos = (elim_rate**(num_elim+1) - 1) / (elim_rate-1)
        u = int(diff / (geos-num_elim-1))
        resources = 0
        v = lambda i: 1 + ceil((diff+(num_elim-geos+elim_rate**i)*u) / (elim_rate**i-1))
        for opt in product(*(range(u, v(i)) for i in reversed(range(1, num_elim+1)))):
            used = max_resources + sum((elim_rate**i-1)*r 
                                       for i, r in zip(reversed(range(1, num_elim+1)), opt))
            if resources <= used <= total_resources:
                best, resources = opt, used
        assert not 0 in best, "invalid: use more resources or fewer eliminations, or increase rate"
        elim_sched = list(np.cumsum(best)) + [max_resources]
    else:
        elim_rate, elim_sched = random_search(max_resources=max_resources, 
                                              total_resources=total_resources)

    eval_sched = [int(step) for step in np.linspace(0, total_resources, num_eval+1)[1:]]
    return elim_rate, elim_sched, eval_sched



def successive_elimination(
                           sampler, 
                           eval_traces, 
                           logdir=None, 
                           val_discount=0.0, 
                           elim_rate=3, 
                           elim_sched=[1], 
                           eval_sched=[-1], 
                           traces=[], 
                           last_round=None,
                           eval_global=False,
                           **kwargs,
                           ):
    '''runs successive elimination according to provided schedule
    Args:
        sampler: function of n returning an iterable of n objects with methods 'step' and 'trace'
        eval_traces: list of strings of traces measuring performance; element 0 used for elimination
        logdir: directory to store tensorboard logs; if None does not log anything
        val_discount: discount factor when computing score for a trace; 0.0 is most recent, 1.0 is mean
        elim_rate: multiplicative elimination rate
        elim_sched: list of steps at which to run an elimination
        eval_sched: list of steps at which to call 'test' method of the best config
        traces: list of strings of traces to collect
        last_round: str name of function that last config executes before final round
        kwargs: passed to 'test' method of each config
    Returns:
        best config;
        also dumps tensorboard logs and results.pkl to folder 'logdir', if specified
    '''
    assert len(elim_sched) > 0, "'elim_sched' must be a list of positive length"
    assert type(elim_rate) == int, "'elim_rate' must be an int"
    logger = False if logdir is None else SummaryWriter(logdir)
    traces, eval_sched = deepcopy(traces), deepcopy(eval_sched)
    for trace in eval_traces:
        if not trace in traces:
            traces.append(trace)
    #each config is (index, config_settings())
    configs = list(enumerate(sampler(elim_rate ** max(1, len(elim_sched)-1))))
    output = {index: {'settings': config.settings()} for index, config in configs}
    for trace in eval_traces:
        output[trace+' val'] = []
        output[trace+' test'] = []
    output['eval step'] = []
    changed = {index: True for index, _ in configs}

    #evaluate fedex object with best score across all time
    best_score = 100.0
    best_config = None 
    best_config_idx = None

    start, last_start, used = 0, 0, 0
    for i, stop in enumerate(elim_sched):
        if len(configs) == 1 and not last_round is None:
            getattr(configs[0][1], last_round)()
        scores = []
        for j, (index, config) in enumerate(configs):
            scores.append(float('inf'))
            for k in range(start, stop):
                changed[index] = True
                print('\r\tRound', frac(i+1, len(elim_sched)), 
                      'config', frac(j+1, len(configs)), 
                      'step', frac(k+1, stop), end=4*' ')
                config.step()
                for trace in traces:
                    output[index][trace] = config.trace(trace)
                    print(trace+':', round(output[index][trace][-1], 4), end=4*' ')
                    if logger:
                        logger.add_scalars(trace, {str(index): output[index][trace][-1]}, k+1)
                #use refine error, if eval_global use global error
                if eval_global:
                    scores[-1] = discounted_mean(output[index][eval_traces[1]][start:],val_discount)
                else:
                    scores[-1] = discounted_mean(output[index][eval_traces[0]][start:], val_discount)
                used += 1
                current_best, score = min(enumerate(scores), key=itemgetter(1))

                #check if val score beats best score so far
                if score < best_score: 
                    best_config_idx = configs[current_best][0]
                    best_config = deepcopy(configs[current_best][1])
                    best_score = score 

                print('best:', round(best_score, 4), end=8*' ')

                for trace in eval_traces:
                    if len(output[best_config_idx][trace][start:])==0:
                        val = discounted_mean(output[best_config_idx][trace][last_start:], val_discount)
                    else:
                        val = discounted_mean(output[best_config_idx][trace][start:], val_discount)
                    output[trace+' val'].append(val)
                    if logger:
                        logger.add_scalar(trace+' val', val, used)
                if used in eval_sched:
                    if changed[best_config_idx]:
                        results = best_config.test(**kwargs)
                        changed[best_config_idx] = False
                    print('\r\tStep', used, 'test error', end='')
                    for trace in reversed(eval_traces):
                        output[trace+' test'].append(results[trace])
                        if logger:
                            logger.add_scalar(trace+' test', results[trace], used)
                        print('    '+trace, round(results[trace], 4), end='')
                    print(64*' ')
                    output['eval step'].append(eval_sched.pop(eval_sched.index(used)))
        if len(configs) == 1:
            break

        #select top n configs, index of each config is preserved
        _, configs = zip(*nsmallest(int(len(configs) / elim_rate), 
                                    zip(scores, configs), 
                                    key=itemgetter(0)))
        last_start = start 
        start = stop

    #best, config = configs[0]
    best, config = best_config_idx, best_config 
    output['best'], output[best]['settings'] = best, config.settings()
    if eval_sched:
        if changed[best]:
            results = config.test(**kwargs)
        print('\r\tStep', used, 'test error', end='')
        for trace in reversed(eval_traces):
            output[trace+' test'].append(results[trace])
            if logger:
                logger.add_scalar(trace+' test', results[trace], used)
            print('    '+trace, round(results[trace], 4), end='')
        print(64*' ')
        output['eval step'].append(used)

    if logger:
        with open(os.path.join(logdir, 'results.pkl'), 'wb') as f:
            pickle.dump(output, f)
        try:
            logger.flush()
        except AttributeError:
            pass
    return config



def wrapped_fedex(
                  get_server,
                  get_client,
                  num_configs=1,
                  prod=False,
                  stepsize_init='auto', 
                  stepsize_sched='aggressive', 
                  cutoff=1E-4, 
                  baseline_discount=-1.0, 
                  diff=False,
                  mle=False, 
                  logdir=None,
                  val_discount=0.0, 
                  last_stop=False,
                  eval_global=False,
                  **kwargs,
                  ):
    '''evaluates FedEx wrapped with successive elimination algorithm;
       uses FedAvg when num_configs = 1 and prod = False
    Args:
        get_server: function that takes no input and returns an object that can be passed as the 
                    first argument to FedEx.__init__, e.g. a Server object
        get_client: function that takes no input and returns a dict of local training configs, a
                    list of which is passed as the second argument to 'FedEx.__init__'; can also
                    return a dict of (string, list) pairs to be passed directly to 'FedEx.__init__'
        num_configs: determines number of configs in the list passed to 'FedEx.__init__':
                     - >0: use this value directly
                     - =0: value drawn at random from Unif[1, number of arms given by the wrapper]
                     - =-1: use the number of arms given by the wrapper
                     - else: value drawn at random from Unif{1, ..., abs(num_configs)}
        prod: run FedEx over a product set of single-parameter grids; must be 'True' in the case
                  when 'get_client' returns an object to be passed directly to 'FedEx.__init__'
        stepsize_init: passed to 'eta0' kwarg of 'FedEx.__init__'
        stepsize_sched: passed to 'sched' kwarg of 'FedEx.__init__'
        baseline_discount: determines 'baseline' kwarg of 'FedEx.__init__':
                           - >0.0: use this value directly
                           - else: value drawn at random from Unif[0.0, abs(baseline_discount)]
        diff: passed to 'diff' kwarg of 'FedEx.__init__'
        mle: passed to 'mle' kwarg of 'FedEx.test' via the kwargs of 'successive_elimination'
        logdir: passed to 'logdir' kwarg of 'successive_elimination'
        val_discount: passed to 'val_discount' kwarg of 'successive_elimination'
        last_stop: if True sets 'last_round' kwarg of 'successive_elimination' to 'stop'
        kwargs: passed to 'get_schedule'
    Returns:
        FedEx object
    '''

    elim_rate, elim_sched, eval_sched = get_schedule(**kwargs)
    print('Wrapping with', 'random search' if len(elim_sched) == 1 else 'successive elimination')

    if num_configs < -1:
        samples = lambda n: random.randint(1, -num_configs)
    elif num_configs == -1:
        samples = lambda n: n
    elif num_configs == 0:
        samples = lambda n: random.randint(1, n)
    else:
        samples = lambda n: num_configs

    if baseline_discount < 0.0:
        baseline = lambda: random.uniform(0.0, -baseline_discount)
    else:
        baseline = lambda: baseline_discount

    def sampler(n):

        for _ in range(n):
            yield FedEx(
                        get_server(), 
                        get_client() if prod else get_client(samples(n)),
                        eta0=stepsize_init, 
                        sched=stepsize_sched, 
                        cutoff=cutoff, 
                        baseline=baseline(),
                        diff=diff,
                        )

    return successive_elimination(
                                  sampler, 
                                  ['refine', 'global'], 
                                  logdir=logdir, 
                                  val_discount=val_discount,
                                  elim_rate=elim_rate, 
                                  elim_sched=elim_sched, 
                                  eval_sched=eval_sched,
                                  traces=['entropy', 'mle', 'global', 'refine'], 
                                  last_round='stop' if last_stop else None,
                                  mle=mle,
                                  eval_global=eval_global,
                                  )


def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='*', help='parent directory of input logdirs')
    parser.add_argument('--output', default='.', help='output directory for tensorboard log')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse()
    results = {}
    for fname in glob(os.path.join(args.input, '*/results.pkl')):
        with open(fname, 'rb') as f:
            key = '/'.join(fname.split('/')[:-1])
            results[key] = pickle.load(f)
    
    logger = SummaryWriter(args.output)
    for mode in ['global', 'refine']:
        for partition in ['val', 'test']:
            trace = mode + ' ' + partition
            for j, scores in enumerate(zip(*(val[trace] for val in results.values()))):
                step = j+1 if partition == 'val' else results[key]['eval step'][j]
                logger.add_scalar('avg '+trace+' error', np.mean(scores), step)
                logger.add_scalar('std '+trace+' error', np.std(scores), step)
                logger.add_histogram(trace+' error', np.array(scores), step)
            if partition == 'test':
                print('Average final '+trace+' error:', np.mean(scores))
                print('Standard deviation', np.std(scores))
    try:
        logger.flush()
    except AttributeError:
        pass
