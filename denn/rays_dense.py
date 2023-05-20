import argparse
import numpy as np
import pandas as pd
import os
import time

from denn.config.config import get_config
from denn.experiments import gan_experiment

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--pretrained', action='store_true', default=False, 
        help='whether to run a sensitivity analysis with 500 random repetitions, default False')
    args = args.parse_args()

    pkey = "rays"
    params = get_config(pkey)

    this_dir = os.path.dirname(os.path.abspath(__file__)) 
    dirname = os.path.join(this_dir, '../experiments/reps/rays/dense')

    # turn off plotting / logging
    params['training']['log'] = False
    params['training']['plot'] = True
    # turn off saving
    params['training']['save'] = True
    params['training']['save_G'] = False
    params['training']['save_for_animation'] = False

    # turn noise on
    params['training']['noise'] = True

    # whether to use pretrained base generator (transfer learning)
    if args.pretrained:
        params['training']['multihead'] = True
        params['generator']['n_heads'] = 11
        params['generator']['pretrained'] = True
        params['discriminator']['n_heads'] = 1
    else:
        params['training']['multihead'] = False
        params['generator']['n_heads'] = 1
        params['generator']['pretrained'] = False
        params['discriminator']['n_heads'] = 1

    # initialize results dict
    results_dict = {}

    # loop through initial conditions
    # y0s = np.linspace(0, 1, 100)
    y0s = [0.3]
    for y0 in y0s:
        print(f"Running RAYS problem for y0={y0}.")
        params['problem']['y0'] = [y0]
        # start_time = time.time()
        res = gan_experiment("rays", params)
        # end_time = time.time() - start_time
        # print(f"Run completed in {end_time} seconds.")
        results_dict[y0] = res["preds"]

    # if not os.path.exists(dirname):
    #     os.mkdir(dirname)
    # np.save(os.path.join(dirname, "rays_dense"), results_dict)
