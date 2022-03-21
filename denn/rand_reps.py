import argparse
import numpy as np
import pandas as pd
import os

from denn.config.config import get_config
from denn.experiments import gan_experiment, L2_experiment
from denn.utils import handle_overwrite

import multiprocessing as mp

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--pkey', type=str, default='EXP',
        help='problem to run (exp=Exponential, sho=SimpleOscillator, nlo=NonlinearOscillator)')
    args.add_argument('--loss', type=str, default='GAN',
        help='loss function to use, default GAN (otherwise classical will be used))')
    args.add_argument('--nreps', type=int, default=10,
        help='number of random seeds to try')
    args.add_argument('--sensitivity', action='store_true', default=False, 
        help='whether to run a sensitivity analysis with 500 random repetitions, default False')
    args.add_argument('--fname', type=str, default='rand_reps',
        help='file to save numpy results of MSEs')
    args = args.parse_args()

    handle_overwrite(args.fname)
    handle_overwrite(args.fname+'.npy')

    params = get_config(args.pkey)

    this_dir = os.path.dirname(os.path.abspath(__file__)) 
    dirname = os.path.join(this_dir, '../experiments/reps', args.pkey)

    # turn off plotting / logging
    #params['training']['log'] = False
    params['training']['plot'] = False
    # turn off saving
    params['training']['save'] = False
    params['training']['save_for_animation'] = False

    # add the loss function to params
    params['training']['loss_fn'] = args.loss

    # initialize lists
    val_mse = []
    lhs_vals = []

    if args.sensitivity:

        # seed the seeds and learning rates sampled
        np.random.seed(42)
        lr_bound = (0.01, 0.1)

        # initialize lists
        col_names = []
        results = []

        # randomly sample seeds and learning rates
        seeds = np.random.choice(args.nreps, size=500)
        gen_lrs = np.random.uniform(*lr_bound, size=500)
        disc_lrs = np.random.uniform(*lr_bound, size=500)

        for i in range(500):

            s = seeds[i]
            g_lr = gen_lrs[i]
            d_lr = disc_lrs[i]

            params['training']['seed'] = s
            params['training']['g_lr'] = g_lr
            params['training']['d_lr'] = d_lr

            if args.loss == 'GAN':
                print(f'Running GAN training for {args.pkey} problem...')
                res = gan_experiment(args.pkey, params)
            else:
                print(f'Running classical training for {args.pkey} problem...')
                res = L2_experiment(args.pkey, params)

            col_names.append(f"{s}_{g_lr:.5f}_{d_lr:.5f}")
            val_mse.append(res['mses']['val'])
            lhs_vals.append(res['losses']['LHS'])
            final_val_mse = res['mses']['val'][-1]
            run = [s, g_lr, d_lr, final_val_mse]
            results.append(run)
            
            print(f"Run {i}: seed={s}, g_lr={g_lr:.4f}, d_lr={d_lr:.4f}. Final Val MSE={final_val_mse:.4e}")

        val_mse = np.vstack(val_mse)
        lhs_vals = np.vstack(lhs_vals)
        val_mse_df = pd.DataFrame(val_mse.T, columns=col_names)
        lhs_vals_df = pd.DataFrame(lhs_vals.T, columns=col_names)
        results_df = pd.DataFrame(results, columns=['seed', 'g_lr', 'd_lr', 'mean_squared_error'])
        val_mse_df = val_mse_df.to_csv(f"{args.fname}_mse.csv", index=False)
        lhs_vals_df = lhs_vals_df.to_csv(f"{args.fname}_lhs.csv", index=False)
        results_df.to_csv(f"{args.fname}.csv", index=False)
    
    else:
        seeds = list(range(args.nreps))
        print("Using seeds: ", seeds)

        for s in seeds:
            print(f'Seed = {s}')
            params['training']['seed'] = s

            if args.loss == 'GAN':
                print(f'Running GAN training for {args.pkey} problem...')
                res = gan_experiment(args.pkey, params)
            else:
                print(f'Running classical training for {args.pkey} problem...')
                res = L2_experiment(args.pkey, params)

            val_mse.append(res['mses']['val'])
            # lhs_vals.append(res['losses']['LHS'])

        if not os.path.exists(dirname):
            os.mkdir(dirname)
        np.save(os.path.join(dirname, args.fname), val_mse)
        # np.save(os.path.join(dirname, args.fname+'_lhs'), lhs_vals)