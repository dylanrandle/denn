from copy import deepcopy
import argparse
import numpy as np

import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler, MedianStoppingRule

from denn.experiments import gan_experiment
from denn.config.config import get_config

N_REPS = 5
LAST_PCT = 0.05

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--pkey', type=str, default='EXP',
        help='problem to run (exp=Exponential, sho=SimpleOscillator, nlo=NonlinearOscillator)')
    args = args.parse_args()

    params = get_config(args.pkey)

    # turn off plotting / saving / logging
    params['training']['log'] = False
    params['training']['plot'] = False
    params['training']['save'] = False

    def gan_tuning(config):
        res = gan_experiment(args.pkey, config)

    search_space = deepcopy(params)

    min_lr_power = 8

    search_space['training']['g_lr'] = tune.sample_from(lambda spec: 10**(-min_lr_power * np.random.rand()))
    search_space['training']['d_lr'] = tune.sample_from(lambda spec: 10**(-min_lr_power * np.random.rand()))
    # search_space['training']['niters'] = tune.sample_from(lambda spec: np.random.choice([750, 1000, 1500]))
    search_space['problem']['n'] = tune.sample_from(lambda spec: int(np.random.choice([100, 200])))
    search_space['generator']['n_hidden_units'] = tune.sample_from(lambda spec: np.random.choice([20, 30, 40]))
    search_space['generator']['n_hidden_layers'] = tune.sample_from(lambda spec: np.random.choice([2, 3, 4]))
    search_space['discriminator']['n_hidden_units'] = tune.sample_from(lambda spec: np.random.choice([20, 30, 40]))
    search_space['discriminator']['n_hidden_layers'] = tune.sample_from(lambda spec: np.random.choice([2, 3, 4]))

    # Uncomment this to enable distributed execution
    # `ray.init(address=...)`
    ray.init(num_cpus=8)

    scheduler = AsyncHyperBandScheduler(
        time_attr='training_iteration',
        metric='mean_squared_error',
        mode='min',
        # tune is tracked every 10 iters
        # ==> e.g. 25 x 10 = 250 real iters
        grace_period=20,
        max_t=100 # ==> e.g. 100 x 10 = 1000 real iters
    )

    analysis = tune.run(
        gan_tuning,
        name=str(f'gan_tuning_{args.pkey}'),
        config=search_space,
        scheduler=scheduler,
        num_samples=500
    )

    df = analysis.dataframe(metric="mean_squared_error", mode="min")
    print(df)
    print("Best config is:", analysis.get_best_config(metric="mean_squared_error", mode="min"))
    best_logdir = analysis.get_best_logdir(metric="mean_squared_error", mode="min")
    best_mse = df.loc[df.logdir==best_logdir]
    print("Best MSE is:", best_mse.mean_squared_error)
