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
        res = gan_experiment(args.pkey, config)
        train_mse = res['mses']['val'] # val is the fixed grid
        last_pct = train_mse[-int(0.05 * len(train_mse)):]
        track.log(mean_squared_error=np.mean(last_pct))
        return res

    search_space = deepcopy(params)

    search_space['training']['g_lr'] = tune.sample_from(lambda spec: 10**(-10 * np.random.rand()))
    search_space['training']['d_lr'] = tune.sample_from(lambda spec: 10**(-10 * np.random.rand()))
    search_space['training']['niters'] = tune.sample_from(lambda _: np.random.choice(range(1000,5001)))
    # search_space['training']['lr_schedule'] = tune.sample_from(lambda _: np.random.choice([True, False]))
    search_space['generator']['n_hidden_units'] = tune.sample_from(lambda _: np.random.choice(range(20,51)))
    search_space['generator']['n_hidden_layers'] = tune.sample_from(lambda _: np.random.choice(range(2,6)))
    search_space['discriminator']['n_hidden_units'] = tune.sample_from(lambda _: np.random.choice(range(20,51)))
    search_space['discriminator']['n_hidden_layers'] = tune.sample_from(lambda _: np.random.choice(range(2,6)))

    # Uncomment this to enable distributed execution
    # `ray.init(address=...)`

    analysis = tune.run(
        gan_tuning,
        name=str(f'gan_tuning_{args.pkey}'),
        config=search_space,
        num_samples=500
    )

    df = analysis.dataframe(metric="mean_squared_error", mode="min")
    print(df)
    print("Best config is:", analysis.get_best_config(metric="mean_squared_error", mode="min"))
    best_logdir = analysis.get_best_logdir(metric="mean_squared_error", mode="min")
    best_mse = df.loc[df.logdir==best_logdir]
    print("Best MSE is:", best_mse.mean_squared_error)
