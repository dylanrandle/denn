from denn.sho.gan_sho import train_GAN_SHO
import pandas as pd
import multiprocessing as mp
import itertools
import numpy as np

def collect_results(kwargs):
    """ k is tuple of dict from pool.Map"""
    print(kwargs)
    mses = []
    for i in range(3):
        result = train_GAN_SHO(seed=i, **kwargs)
        mses.append(result['final_mse'])
    res = {'kwargs': kwargs, 'mean_mse': np.mean(mses), 'std_mse': np.std(mses)}
    return res

def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

if __name__== "__main__":

    # Oct 30, done
    # hyper_space = dict(
    #     g_units=[16, 32, 64],
    #     g_layers=[2, 4, 8],
    #     d_units=[16, 32, 64],
    #     d_layers=[2, 4, 8],
    #     d2_units=[16, 32, 64],
    #     d2_layers=[2, 4, 8],
    #     d2 = [0.1, 1, 10],
    # )

    # Not executed
    # G_iters=[1, 2, 4],
    # D_iters=[1, 2, 4],
    # d1 = [0.1, 1, 10],

    # Nov 1, done
    # hyper_space = dict(
    #     d2 = [0.001, 0.01, 0.1],
    #     num_epochs = [10000, 20000, 50000],
    #     gp = [0.1, 1, 10],
    #     d_lr = [1e-4, 2e-4, 1e-3],
    #     g_lr = [1e-4, 2e-4, 1e-3],
    #     eq_lr = [1e-4, 2e-4, 1e-3]
    # )

    hyper_space = dict(
        d2 = [1000],
        num_epochs = [10000],
        d_lr = [1e-4, 2e-4, 1e-3],
        g_lr = [1e-4, 2e-4, 1e-3],
        g_units=[16, 32, 64],
        g_layers=[2, 4, 8],
        d2_units=[16, 32, 64],
        d2_layers=[2, 4, 8],
    )

    n_iters = np.product([len(v) for k, v in hyper_space.items()])
    hyper_space = dict_product(hyper_space)

    max_cpus = 256
    n_cpus = n_iters if n_iters < max_cpus else max_cpus
    p = mp.Pool(n_cpus)
    results = p.map(collect_results, hyper_space)

    resdf = pd.DataFrame().from_records(results)
    resdf.to_csv('GAN_hyper_results.csv')
