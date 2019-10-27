from denn.sho.gan_sho import train_GAN_SHO
import pandas as pd
import multiprocessing as mp
import itertools
import numpy as np

def collect_results(args):
    """ k is tuple of dict from pool.Map"""
    print(args)
    mses = []
    for i in range(5):
        result = train_GAN_SHO(num_epochs=10000, seed=i, **args)
        mses.append(result['final_mse'])
    res = {'kwargs': args, 'mean_mse': np.mean(mses), 'std_mse': np.std(mses)}
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

    hyper_space = dict(
        g_units=[16, 32, 64],
        g_layers=[2, 4, 8],
        d_units=[16, 32, 64],
        d_layers=[2, 4, 8],
        d2_units=[16, 32, 64],
        d2_layers=[2, 4, 8],
        G_iters=[1, 2, 4],
        D_iters=[1, 2, 4],
        d1 = [0.1, 1, 10],
        d2 = [0.1, 1, 10],
    )

    hyper_space = dict_product(hyper_space)

    n_cpus = 256
    p = mp.Pool(n_cpus)
    results = p.map(collect_results, hyper_space)

    resdf = pd.DataFrame().from_records(results)
    resdf.to_csv('GAN_hyper_results.csv')
