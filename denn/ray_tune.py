from copy import deepcopy
import argparse
import numpy as np

import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler, MedianStoppingRule

from denn.experiments import gan_experiment, L2_experiment
from denn.config.config import get_config

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--pkey', type=str, default='EXP',
        help='problem to run (exp=Exponential, sho=SimpleOscillator, nlo=NonlinearOscillator)')
    args.add_argument('--classical', action='store_true', default=False,
        help='whether to use classical training, default False (use GAN))')
    args.add_argument('--ncpu', type=int, default=10)
    args.add_argument('--nsample', type=int, default=100)
    args = args.parse_args()

    params = get_config(args.pkey)
    _niters = params['training']['niters']

    # turn off plotting / saving / logging
    params['training']['log'] = False
    params['training']['plot'] = False
    params['training']['save'] = False

    def gan_tuning(config):
        res = gan_experiment(args.pkey, config)

    def classical_tuning(config):
        res = L2_experiment(args.pkey, config)

    search_space = deepcopy(params)

    # Bounds of Search
    lr_bound = (1e-6, 1e-2)
    gamma_bound = (0.99, 0.9999)
    beta_bound = (0, 0.999)
    n_nodes = [20, 30, 40]
    n_layers = [2, 3, 4]

    # LRs
    search_space['training']['g_lr'] = tune.sample_from(lambda s: np.random.uniform(*lr_bound))
    search_space['training']['d_lr'] = tune.sample_from(lambda s: np.random.uniform(*lr_bound))

    # Decay / Moment
    search_space['training']['gamma'] = tune.sample_from(lambda s: np.random.uniform(*gamma_bound))
    search_space['training']['g_betas'] = tune.sample_from(lambda s: np.random.uniform(*beta_bound, size=2))
    search_space['training']['d_betas'] = tune.sample_from(lambda s: np.random.uniform(*beta_bound, size=2))
    
    # Generator
    search_space['generator']['n_hidden_units'] = tune.sample_from(lambda s: int(np.random.choice(n_nodes)))
    search_space['generator']['n_hidden_layers'] = tune.sample_from(lambda s: int(np.random.choice(n_layers)))

    # Discriminator
    search_space['discriminator']['n_hidden_units'] = tune.sample_from(lambda s: int(np.random.choice(n_nodes)))
    search_space['discriminator']['n_hidden_layers'] = tune.sample_from(lambda s: int(np.random.choice(n_layers)))

    # for testing at different seeds
    # note: need to change experiments.py to init models below seed setting
    # nseeds = 10
    # search_space['training']['seed'] = tune.sample_from(lambda s: np.random.choice(nseeds))

    # Uncomment this to enable distributed execution
    # ray.init(address='auto', redis_password='5241590000000000')

    # Uncomment this to specify num cpus
    ray.init(num_cpus=args.ncpu)

    scheduler = AsyncHyperBandScheduler(
        time_attr='training_iteration',
        metric='mean_squared_error',
        mode='min',
        reduction_factor=4,
        brackets=1,
        max_t=int(_niters/100), # ==> e.g. 100 x 10 = 1000 real iters
        grace_period=int(_niters/1000), # tune is tracked every 10 iters
    )                    # ==> e.g. 25 x 10 = 250 real iters

    if args.classical:
        print('Using classical method')
        _fn = classical_tuning
        _jobname = f"classical_tuning_{args.pkey}"
    else:
        print('Using GAN method')
        _fn = gan_tuning
        _jobname = f"gan_tuning_{args.pkey}"

    analysis = tune.run(
        _fn,
        name=_jobname,
        config=search_space,
        scheduler=scheduler,
        num_samples=args.nsample,
    )

    df = analysis.dataframe(metric="mean_squared_error", mode="min")
    print("Sorted top results")
    print(df.sort_values(by="mean_squared_error").head(10))
    print("Best config is:", analysis.get_best_config(metric="mean_squared_error", mode="min"))
    best_logdir = analysis.get_best_logdir(metric="mean_squared_error", mode="min")
    best_mse = df.loc[df.logdir==best_logdir]
    print("Best MSE is: ", best_mse.mean_squared_error)
