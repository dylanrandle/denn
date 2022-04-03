import numpy as np
import pandas as pd
import os

from denn.config.config import get_config
from denn.experiments import gan_experiment

if __name__ == '__main__':
    pkey = "rays"
    params = get_config(pkey)

    this_dir = os.path.dirname(os.path.abspath(__file__)) 
    dirname = os.path.join(this_dir, '../experiments/reps/rays/dense')

    # turn off plotting / logging
    params['training']['log'] = False
    params['training']['plot'] = False
    # turn off saving
    params['training']['save'] = False
    params['training']['save_G'] = False
    params['training']['save_for_animation'] = False

    # turn noise on
    params['training']['noise'] = True

    # set to single head configuration
    params['training']['multihead'] = False
    params['generator']['n_heads'] = 1
    params['generator']['pretrained'] = False
    params['discriminator']['n_heads'] = 1

    # initialize results dict
    results_dict = {}

    # loop through initial conditions
    y0s = np.linspace(0, 1, 100)[50:]
    for y0 in y0s:
        print(f"Running RAYS problem for y0={y0}.")
        params['problem']['y0'] = [y0]
        res = gan_experiment("rays", params)
        results_dict[y0] = res["preds"]

    if not os.path.exists(dirname):
        os.mkdir(dirname)
    np.save(os.path.join(dirname, "rays_dense2"), results_dict)
