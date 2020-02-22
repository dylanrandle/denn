from copy import deepcopy
import argparse
import numpy as np

from ray import tune
from ray.tune import track
from ray.tune.schedulers import ASHAScheduler

from denn.experiments import gan_experiment
from denn.config.config import get_config

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
        return gan_experiment(args.pkey, config)

    search_space = deepcopy(params)

    search_space['training']['g_lr'] = tune.sample_from(lambda spec: 10**(-10 * np.random.rand()))
    search_space['training']['d_lr'] = tune.sample_from(lambda spec: 10**(-10 * np.random.rand()))

    # Uncomment this to enable distributed execution
    # `ray.init(address=...)`

    analysis = tune.run(
        gan_tuning,
        config=search_space,
        num_samples=5)

    # df = analysis.dataframe(metric="mean_squared_error", mode="min")
    # print(df.head())
    print("Best trial is:", analysis.get_best_trial(metric="mean_squared_error", mode="min"))
    print("Best config is:", analysis.get_best_config(metric="mean_squared_error", mode="min"))
