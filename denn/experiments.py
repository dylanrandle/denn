import torch
import torch.nn as nn
import argparse
import numpy as np

from denn.algos import train_L2, train_L2_2D, train_GAN, train_GAN_2D
from denn.models import MLP
from denn.pretrained import get_pretrained_fcnn
from denn.config.config import get_config
import denn.ode_problems as ode
import denn.pde_problems as pde

def get_problem(pkey, params):
    """ helper to parse problem key and return appropriate problem
    """
    pkey = pkey.lower().strip()
    if pkey == 'exp':
        return ode.Exponential(**params['problem'])
    elif pkey == 'sho':
        return ode.SimpleOscillator(**params['problem'])
    elif pkey == 'nlo':
        return ode.NonlinearOscillator(**params['problem'])
    elif pkey == 'pos':
        return pde.PoissonEquation(**params['problem'])
    elif pkey == 'rans':
        return ode.ReynoldsAveragedNavierStokes(**params['problem'])
    elif pkey == 'sir':
        return ode.SIRModel(**params['problem'])
    elif pkey == 'coo':
        return ode.CoupledOscillator(**params['problem'])
    elif pkey == 'eins':
        return ode.EinsteinEquations(**params['problem'])
    elif pkey == 'wav':
        return pde.WaveEquation(**params['problem'])
    elif pkey == 'bur':
        return pde.BurgersEquation(**params['problem'])
    elif pkey == 'burv':
        return pde.BurgersViscous(**params['problem'])
    elif pkey == 'hea':
        return pde.HeatEquation(**params['problem'])
    elif pkey == 'aca':
        return pde.AllenCahn(**params['problem'])
    elif pkey == 'kur':
        return pde.KuramotoSivashinsky(**params['problem'])
    else:
        raise RuntimeError(f'Did not understand problem key (pkey): {pkey}')

def L2_experiment(pkey, params):
    # model init seed
    torch.manual_seed(0)
    np.random.seed(0)

    # model
    model = MLP(**params['generator'])

    # experiment seed
    np.random.seed(params['training']['seed'])
    torch.manual_seed(params['training']['seed'])

    # run
    problem = get_problem(pkey, params)
    if pkey.lower().strip() in ["pos", "wav", "bur", "burv", "hea", "aca", "kur"]:
        res = train_L2_2D(model, problem, **params['training'], config=params)
    else:
        res = train_L2(model, problem, **params['training'], config=params)

    return res

def gan_experiment(pkey, params):
    # model init seed
    torch.manual_seed(0)
    np.random.seed(0)

    # models
    if params['generator']['pretrained']:
        gen = get_pretrained_fcnn(pkey)
    else:
        gen = MLP(**params['generator'])
    disc = MLP(**params['discriminator'])

    # experiment seed
    torch.manual_seed(params['training']['seed'])
    np.random.seed(params['training']['seed'])

    # run
    problem = get_problem(pkey, params)
    if pkey.lower().strip() in ["pos", "wav", "bur", "burv", "hea", "aca", "kur"]:
        res = train_GAN_2D(gen, disc, problem, **params['training'], config=params)
    else:
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
