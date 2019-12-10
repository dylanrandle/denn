import torch
import torch.nn as nn
import argparse
import numpy as np

from denn.algos import train_L2, train_GAN
from denn.models import MLP
from denn.problems import Exponential, SimpleOscillator, NonlinearOscillator
from denn.config import *
from denn.utils import handle_overwrite

def L2_experiment(problem, seed=0, model_kwargs={}, train_kwargs={}):
    torch.manual_seed(seed)
    model = MLP(**model_kwargs)
    res = train_L2(model, problem, **train_kwargs)
    return res

def gan_experiment(problem, seed=0, gen_kwargs={}, disc_kwargs={}, train_kwargs={}):
    torch.manual_seed(seed)
    gen = MLP(**gen_kwargs)
    disc = MLP(**disc_kwargs)
    res = train_GAN(gen, disc, problem, **train_kwargs)
    return res

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--gan', action='store_true', default=False,
        help='whether to use GAN-based training, default False (use L2-based)')
    args.add_argument('--exp', action='store_true', default=False,
        help='whether to fit Exponential problem')
    args.add_argument('--sho', action='store_true', default=False,
        help='whether to fit SimpleOscillator problem')
    args.add_argument('--nlo', action='store_true', default=False,
        help='whether to fit NonlinearOscillator problem')
    args.add_argument('--niters', type=int, default=100,
        help='how many iterations of the algorithm to run (default: 100)')
    args.add_argument('--seed', type=int, default=0,
        help='value of random seed set for reproducibility (default: 0)')
    args.add_argument('--semisup', action='store_true', default=False,
        help='if training is semisupervised, default False (use unsupervised)')
    args.add_argument('--sup', action='store_true', default=False,
        help='if training is supervised, default False (use unsupervised)')
    args.add_argument('--obs_every', type=int, default=1,
        help='setting the observer frequency, default 1 (no missing)')
    args.add_argument('--fname', type=str, default=None,
        help='file name to save figure (default None: constructed from args)')
    args = args.parse_args()

    if args.exp:
        print('Solving Exponential problem')
        problem = Exponential(**problem_kwargs)
        problem_key = 'EXP'
    elif args.sho:
        print('Solving SimpleOscillator problem')
        problem = SimpleOscillator(**problem_kwargs)
        problem_key = 'SHO'
    elif args.nlo:
        print('Solving NonlinearOscillator problem')
        problem = NonlinearOscillator(**problem_kwargs)
        problem_key = 'NLO'
    else:
        print('Did not receive a problem flag (e.g. --simple or --nonlinear) set to true.')
        exit(0)

    if args.semisup:
        method = 'semisupervised'
    elif args.sup:
        method = 'supervised'
    else:
        method = 'unsupervised'

    print('Method: {}'.format(method))

    if args.gan:
        print(f'Running GAN training for {args.niters} steps')
        gan_kwargs['method'] = method
        gan_kwargs['fname'] = args.fname if args.fname else f'GAN_{method}_{problem_key}.png'
        gan_kwargs['niters'] = args.niters
        gan_kwargs['obs_every'] = args.obs_every
        gan_experiment(problem, seed=args.seed, gen_kwargs=gen_kwargs, disc_kwargs=disc_kwargs, train_kwargs=gan_kwargs)
    else:
        print(f'Running L2 training for {args.niters} steps')
        L2_kwargs['method'] = method
        L2_kwargs['fname'] = args.fname if args.fname else f'L2_{method}_{problem_key}.png'
        L2_kwargs['niters'] = args.niters
        L2_kwargs['obs_every'] = args.obs_every
        L2_experiment(problem, seed=args.seed, model_kwargs=L2_mlp_kwargs, train_kwargs=L2_kwargs)
