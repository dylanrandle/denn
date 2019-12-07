from algos import train_MSE
from models import MLP
from problems import Exponential, SimpleOscillator, NonlinearOscillator
import torch
import torch.nn as nn
import argparse

def mse_experiment(problem, seed=0, model_kwargs={}, train_kwargs={}):
    torch.manual_seed(seed)
    model = MLP(**model_kwargs)
    res = train_MSE(model, problem, **train_kwargs)
    return res

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--exp', action='store_true', default=False)
    args.add_argument('--simple', action='store_true', default=False)
    args.add_argument('--nonlinear', action='store_true', default=False)
    args = args.parse_args()

    if args.exp:
        problem = Exponential()
        problem_key = 'EXP'
    elif args.simple:
        problem = SimpleOscillator()
        problem_key = 'SHO'
    elif args.nonlinear:
        problem = NonlinearOscillator()
        problem_key = 'NLO'
    else:
        print('Did not receive a problem flag (e.g. --simple or --nonlinear) set to true.')
        exit(0)

    for method in ['unsupervised', 'semisupervised', 'supervised']:

        print('Method: {}'.format(method))

        model_kwargs = dict(
            in_dim=1,
            out_dim=1,
            n_hidden_units=32,
            n_hidden_layers=4,
            activation=nn.Tanh(),
            residual=True,
            regress=True,
        )

        train_kwargs = dict(
            method=method,
            niters=1000,
            lr=0.001,
            betas=(0, 0.9),
            lr_schedule=True,
            obs_every=1,
            d1=1,
            d2=1,
            plot=True,
            save=True,
            fname=f'train_{method}_{problem_key}_plot.png'
        )

        mse_experiment(problem, seed=0, model_kwargs=model_kwargs, train_kwargs=train_kwargs)
