import argparse
import numpy as np

from denn.config.config import get_config
from denn.experiments import gan_experiment, L2_experiment
from denn.utils import handle_overwrite

import multiprocessing as mp

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--gan', action='store_true', default=False,
        help='whether to use GAN-based training, default False (use L2-based)')
    args.add_argument('--pkey', type=str, default='EXP',
        help='problem to run (exp=Exponential, sho=SimpleOscillator, nlo=NonlinearOscillator)')
    args.add_argument('--nreps', type=int, default=10,
        help='number of random seeds to try')
    args.add_argument('--fname', type=str, default='rand_reps',
        help='file to save numpy results of MSEs')
    args = args.parse_args()

    handle_overwrite(args.fname)
    handle_overwrite(args.fname+'.npy')

    params = get_config(args.pkey)

    # turn off plotting / saving / logging
    params['training']['log'] = False
    params['training']['plot'] = False
    params['training']['save'] = False

    np.random.seed(42)
    seeds = np.random.randint(int(1e16), size=args.nreps)

    results = []
    for s in seeds:
        print(f'Seed = {s}')
        params['training']['seed'] = s

        if args.gan:
            print(f'Running GAN training for {args.pkey} problem...')
            res = gan_experiment(args.pkey, params)
        else:
            print(f'Running L2 training for {args.pkey} problem...')
            res = L2_experiment(args.pkey, params)

        results.append(res['mses']['val'])

    results = np.vstack(results)
    np.save(args.fname, results)