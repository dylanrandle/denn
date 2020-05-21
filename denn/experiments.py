import torch
import torch.nn as nn
import argparse
import numpy as np

from denn.algos import train_L2, train_GAN
from denn.models import MLP
from denn.config.config import get_config
from denn.utils import handle_overwrite
import denn.problems as pb

def get_problem(pkey, params):
    """ helper to parse problem key and return appropriate problem
    """
    pkey = pkey.lower().strip()
    if pkey == 'exp':
        return pb.Exponential(**params['problem'])
    elif pkey == 'sho':
        return pb.SimpleOscillator(**params['problem'])
    elif pkey == 'nlo':
        return pb.NonlinearOscillator(**params['problem'])
    elif pkey == 'pos':
        return pb.PoissonEquation(**params['problem'])
    elif pkey == 'rans':
        return pb.ReynoldsAveragedNavierStokes(**params['problem'])
    elif pkey == 'sir':
        return pb.SIRModel(**params['problem'])
    elif pkey == 'coo':
        return pb.CoupledOscillator(**params['problem'])
    else:
        raise RuntimeError(f'Did not understand problem key (pkey): {pkey}')

def L2_experiment(pkey, params):
    torch.manual_seed(0)
    np.random.seed(0)
    model = MLP(**params['generator'])

    np.random.seed(params['training']['seed'])
    torch.manual_seed(params['training']['seed'])
    problem = get_problem(pkey, params)
    res = train_L2(model, problem, **params['training'], config=params)
    return res

def gan_experiment(pkey, params):
    torch.manual_seed(0)
    np.random.seed(0)
    gen = MLP(**params['generator'])
    disc = MLP(**params['discriminator'])

    torch.manual_seed(params['training']['seed'])
    np.random.seed(params['training']['seed'])
    problem = get_problem(pkey, params)
    res = train_GAN(gen, disc, problem, **params['training'], config=params)
    return res

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--gan', action='store_true', default=False,
        help='whether to use GAN-based training, default False (use L2-based)')
    args.add_argument('--pkey', type=str, default='EXP',
        help='problem to run (exp=Exponential, sho=SimpleOscillator, nlo=NonlinearOscillator)')
    args = args.parse_args()

    params = get_config(args.pkey)

    if args.gan:
        print(f'Running GAN training for {args.pkey} problem...')
        gan_experiment(args.pkey, params)
    else:
        print(f'Running classical training for {args.pkey} problem...')
        L2_experiment(args.pkey, params)
