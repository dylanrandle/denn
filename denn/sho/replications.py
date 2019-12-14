import pandas as pd
import multiprocessing as mp
import argparse

from denn.problems import get_problem
from denn.experiments import gan_experiment, L2_experiment
import denn.config as cfg
from denn.utils import handle_overwrite


if __name__== "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--gan', action='store_true', default=False,
        help='whether to use GAN-based training, default False (use L2-based)')
    args.add_argument('--pkey', type=str, default='EXP',
        help='problem to run (exp=Exponential, sho=SimpleOscillator, nlo=NonlinearOscillator)')
    args.add_argument('--ncpu', type=int, default=4,
        help='number of cpus to use (default: 4)')
    args.add_argument('--nreps', type=int, default=10,
        help='number of replications to perform (default: 10)')
    args.add_argument('--fname', type=str, default='EXP_replications.csv',
        help='path to save results of replications')
    args = args.parse_args()

    handle_overwrite(args.fname)

    seeds = list(range(args.nreps))
    pool = mp.Pool(args.ncpu)

    def gan_exp_with_seed(s):
        """ wrap gan_experiment for pool.map """
        return gan_experiment(
            get_problem(args.pkey),
            seed=s,
            gen_kwargs=cfg.gen_kwargs,
            disc_kwargs=cfg.disc_kwargs,
            train_kwargs=cfg.gan_kwargs
        )

    def L2_exp_with_seed(s):
        """ wrap L2_experiment for pool.map """
        return L2_experiment(
            get_problem(args.pkey),
            seed=s,
            model_kwargs=cfg.L2_mlp_kwargs,
            train_kwargs=cfg.L2_kwargs
        )

    if args.gan:
        results = pool.map(gan_exp_with_seed, seeds)
    else:
        results = pool.map(L2_exp_with_seed, seeds)
    pd.DataFrame().from_records(results).to_csv(args.fname)
    print(f'Saved results to {args.fname}')
