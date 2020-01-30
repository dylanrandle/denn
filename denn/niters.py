import pandas as pd
import multiprocessing as mp
import numpy as np
import argparse

from denn.experiments import gan_experiment, L2_experiment, get_problem
import denn.config as cfg
from denn.utils import handle_overwrite, dict_product

def gan_exp_with_seed(i, q, gen_kwargs, disc_kwargs, gan_kwargs):
    res = gan_experiment(
        problem = _PROBLEM,
        seed = i,
        gen_kwargs = gen_kwargs,
        disc_kwargs = disc_kwargs,
        train_kwargs = gan_kwargs,
    )
    q.put(res['final_mse'])

def gan_exp_with_hypers(hypers):
    """
    take hypers dict and map values onto appropriate kwargs dict
    to run gan_experiment
    """
    disc_kwargs = cfg.disc_kwargs
    gan_kwargs = cfg.gan_kwargs
    gen_kwargs = cfg.gen_kwargs

    gan_kwargs['plot'] = False # ensure plotting is off

    for k, v in hypers.items():
        if k.startswith('gan_'):
            gan_kwargs[k.replace('gan_', '')] = v
        elif k.startswith('gen_'):
            gen_kwargs[k.replace('gen_', '')] = v
        elif k.startswith('disc_'):
            disc_kwargs[k.replace('disc_', '')] = v

    q = mp.Queue()

    procs = []
    for i in range(_N_REPS):
        p = mp.Process(target=gan_exp_with_seed, args=(i, q, gen_kwargs, disc_kwargs, gan_kwargs))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()   

    reps = []
    while not q.empty():
        reps.append(q.get())

    res = {'mse': reps, 'hypers': hypers}
    print(f'Result: {res}')
    return res

def L2_exp_with_seed(i, q, model_kwargs, train_kwargs):
    res = L2_experiment(
        problem = _PROBLEM,
        seed = i,
        model_kwargs = model_kwargs,
        train_kwargs = train_kwargs,
    )
    q.put(res['final_mse'])

def L2_exp_with_hypers(hypers):
    """
    take hypers dict and map values onto appropriate kwargs dict
    to run gan_experiment
    """
    model_kwargs = cfg.L2_mlp_kwargs
    train_kwargs = cfg.L2_kwargs

    train_kwargs['plot'] = False # ensure plotting is off

    for k, v in hypers.items():
        if k.startswith('model_'):
            model_kwargs[k.replace('model_', '')] = v
        elif k.startswith('train_'):
            train_kwargs[k.replace('train_', '')] = v

    q = mp.Queue()

    procs = []
    for i in range(_N_REPS):
        p = mp.Process(target=L2_exp_with_seed, args=(i, q, model_kwargs, train_kwargs))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    reps = []
    while not q.empty():
        reps.append(q.get())

    res = {'mse': reps, 'hypers': hypers}
    print(f'Result: {res}')
    return res

def get_niters(pkey, gan):
    name = 'gan' if gan else 'L2'
    name = name + '_' + pkey
    return eval('cfg.' + name + '_niters')

if __name__== "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--gan', action='store_true', default=False,
        help='whether to use GAN-based training, default False (use L2-based)')
    args.add_argument('--fname', type=str, default='SHO_hypertune.csv',
        help='path to save results of replications')
    args.add_argument('--nreps', type=int, default=20,
        help='number of replications to run')
    args.add_argument('--pkey', type=str, default='sho',
        help='problem key as a string')
    args = args.parse_args()

    _N_REPS = args.nreps
    _PROBLEM = get_problem(args.pkey)

    handle_overwrite(args.fname)

    hyper_space = get_niters(args.pkey, args.gan)
    hyper_space = dict_product(hyper_space)

    if args.gan:
        results = map(gan_exp_with_hypers, hyper_space)
    else:
        results = map(L2_exp_with_hypers, hyper_space)

    pd.DataFrame().from_records(results).to_csv(args.fname)
    print(f'Saved results to {args.fname}')
