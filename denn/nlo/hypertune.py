import pandas as pd
import multiprocessing as mp
import numpy as np
import argparse

from denn.problems import NonlinearOscillator
from denn.experiments import gan_experiment
import denn.config as cfg
from denn.utils import handle_overwrite, dict_product

def gan_exp_with_hypers(hypers):
    """
    take hypers dict and map values onto appropriate kwargs dict
    to run gan_experiment
    """
    disc_kwargs = cfg.disc_kwargs
    gan_kwargs = cfg.gan_kwargs
    gen_kwargs = cfg.gen_kwargs

    for k, v in hypers.items():
        if k.startswith('gan_'):
            gan_kwargs[k.replace('gan_', '')] = v
        elif k.startswith('gen_'):
            gen_kwargs[k.replace('gen_', '')] = v
        elif k.startswith('disc_'):
            disc_kwargs[k.replace('disc_', '')] = v

    exp_res = gan_experiment(
        problem = NonlinearOscillator(**cfg.problem_kwargs),
        seed = 0,
        gen_kwargs = gen_kwargs,
        disc_kwargs = disc_kwargs,
        train_kwargs = gan_kwargs,
    )
    res = {'mse': exp_res['final_mse'], 'hypers': hypers}
    print(f'Result: {res}')
    return res


if __name__== "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--ncpu', type=int, default=4,
        help='number of cpus to use (default: 4)')
    args.add_argument('--fname', type=str, default='NLO_hypertune.csv',
        help='path to save results of replications')
    args = args.parse_args()

    handle_overwrite(args.fname)

    # define search space
    hyper_space = dict(
        # disc_kwargs
        disc_n_hidden_units = [64, 32, 16],
        disc_n_hidden_layers = [2, 3, 4],
        # gen_kwargs
        gen_n_hidden_units = [32, 16],
        gen_n_hidden_layers = [3],
    )

    hyper_space = dict_product(hyper_space)

    pool = mp.Pool(args.ncpu)
    results = pool.map(gan_exp_with_hypers, hyper_space)

    pd.DataFrame().from_records(results).to_csv(args.fname)
    print(f'Saved results to {args.fname}')
