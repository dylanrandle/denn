import torch
import torch.nn as nn
import argparse
import numpy as np

from denn.algos import train_L2, train_GAN
from denn.models import MLP
from denn.problems import get_problem
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
    args.add_argument('--pkey', type=str, default='EXP',
        help='problem to run (exp=Exponential, sho=SimpleOscillator, nlo=NonlinearOscillator)')
    args.add_argument('--seed', type=int, default=0,
        help='value of random seed set for reproducibility (default: 0)')
    args.add_argument('--fname', type=str, default=None,
        help='file name to save figure (default None: use function default)')
    args = args.parse_args()

    problem = get_problem(args.pkey)

    if args.gan:
        print('Running GAN training...')
        if args.fname:
            gan_kwargs['fname'] = args.fname
        gan_experiment(problem, seed=args.seed, gen_kwargs=gen_kwargs, disc_kwargs=disc_kwargs, train_kwargs=gan_kwargs)
    else:
        print('Running L2 training...')
        if args.fname:
            L2_kwargs['fname'] = args.fname
        L2_experiment(problem, seed=args.seed, model_kwargs=L2_mlp_kwargs, train_kwargs=L2_kwargs)
