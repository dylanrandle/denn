from algos import train_MSE, train_GAN
from models import MLP
from problems import Exponential, SimpleOscillator, NonlinearOscillator
import torch
import torch.nn as nn
import argparse
import numpy as np

def mse_experiment(problem, seed=0, model_kwargs={}, train_kwargs={}):
    torch.manual_seed(seed)
    model = MLP(**model_kwargs)
    res = train_MSE(model, problem, **train_kwargs)
    return res

def gan_experiment(problem, seed=0, gen_kwargs={}, disc_kwargs={}, train_kwargs={}):
    torch.manual_seed(seed)
    gen = MLP(**gen_kwargs)
    disc = MLP(**disc_kwargs)
    res = train_GAN(gen, disc, problem, **train_kwargs)
    return res

gen_kwargs = dict(
    in_dim=1,
    out_dim=1,
    n_hidden_units=32,
    n_hidden_layers=4,
    activation=nn.Tanh(),
    residual=True,
    regress=True,
)

disc_kwargs = dict(
    in_dim=2,
    out_dim=1,
    n_hidden_units=32,
    n_hidden_layers=4,
    activation=nn.Tanh(),
    residual=True,
    regress=True, # true for WGAN
)

gan_kwargs = dict(
    niters=100,
    g_lr=1e-3,
    g_betas=(0.0, 0.9),
    d_lr=1e-3,
    d_betas=(0.0, 0.9),
    lr_schedule=True,
    obs_every=1,
    d1=1.,
    d2=1.,
    G_iters=1,
    D_iters=1,
    wgan=True,
    gp=0.1,
    plot=True,
    save=True,
)

mse_mlp_kwargs = dict(
    in_dim=1,
    out_dim=1,
    n_hidden_units=32,
    n_hidden_layers=4,
    activation=nn.Tanh(),
    residual=True,
    regress=True,
)

mse_kwargs = dict(
    niters=100,
    lr=1e-3,
    betas=(0, 0.9),
    lr_schedule=True,
    obs_every=1,
    d1=1,
    d2=1,
    plot=True,
    save=True,
)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--mse', action='store_true', default=False,
        help='whether to use MSE-based training (Lagaris method): default is False (use GAN method)')
    args.add_argument('--exp', action='store_true', default=False,
        help='whether to fit Exponential problem')
    args.add_argument('--sho', action='store_true', default=False,
        help='whether to fit SimpleOscillator problem')
    args.add_argument('--nlo', action='store_true', default=False,
        help='whether to fit NonlinearOscillator problem')
    args.add_argument('--niters', type=int, default=100,
        help='how many iterations of the algorithm to run (default: 100)')
    args = args.parse_args()

    if args.exp:
        print('Solving Exponential problem')
        problem = Exponential()
        problem_key = 'EXP'
    elif args.sho:
        print('Solving SimpleOscillator problem')
        problem = SimpleOscillator()
        problem_key = 'SHO'
    elif args.nlo:
        print('Solving NonlinearOscillator problem')
        problem = NonlinearOscillator()
        problem_key = 'NLO'
    else:
        print('Did not receive a problem flag (e.g. --simple or --nonlinear) set to true.')
        exit(0)

    if args.mse:
        for method in ['unsupervised', 'semisupervised', 'supervised']:
            print('Method: {}'.format(method))
            print('Running MSE...')
            mse_kwargs['method'] = method
            mse_kwargs['fname'] = f'train_MSE_{method}_{problem_key}.png'
            mse_kwargs['niters'] = args.niters
            mse_experiment(problem, seed=0, model_kwargs=mse_mlp_kwargs, train_kwargs=mse_kwargs)
    else:
        method = 'unsupervised'
        print('Method: {}'.format(method))
        print('Running GAN...')
        gan_kwargs['method'] = method
        gan_kwargs['fname'] = f'train_GAN_{method}_{problem_key}.png'
        gan_kwargs['niters'] = args.niters
        gan_experiment(problem, seed=0, gen_kwargs=gen_kwargs, disc_kwargs=disc_kwargs, train_kwargs=gan_kwargs)
