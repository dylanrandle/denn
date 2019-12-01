import pandas as pd
import multiprocessing as mp
import itertools
import numpy as np

from denn.sho.gan_sho import train_GAN_SHO_unsupervised
from denn.utils import Generator, Discriminator

def collect_results(kwargs):
    """ k is tuple of dict from pool.Map"""
    print(kwargs)
    num_epochs = kwargs['num_epochs']
    g_lr = kwargs['g_lr']
    d_lr = kwargs['d_lr']
    g_units = kwargs['g_units']
    d_units = kwargs['d_units']
    g_layers = kwargs['g_layers']
    d_layers = kwargs['d_layers']
    g_iters = kwargs['g_iters']
    d_iters = kwargs['d_iters']
    gp = kwargs['gp']

    mses = []
    for i in range(3):
        G = Generator(in_dim=1, out_dim=1,
                      n_hidden_units=g_units,
                      n_hidden_layers=g_layers,
                      output_tan=False,
                      residual=True)

        D = Discriminator(in_dim=2, out_dim=1,
                          n_hidden_units=d_units,
                          n_hidden_layers=d_layers,
                          unbounded=True,
                          residual=True)

        result = train_GAN_SHO_unsupervised(G, D, seed=i, g_lr=g_lr, d_lr=d_lr,
            G_iters=g_iters, D_iters=d_iters, gp=gp, num_epochs=num_epochs)

        mses.append(result['final_mse'])

    res = {'kwargs': kwargs, 'mean_mse': np.mean(mses), 'std_mse': np.std(mses)}
    print(f'Search result: {res}')
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

    # Nov 1, done
    # hyper_space = dict(
    #     d2 = [0.001, 0.01, 0.1],
    #     num_epochs = [10000, 20000, 50000],
    #     gp = [0.1, 1, 10],
    #     d_lr = [1e-4, 2e-4, 1e-3],
    #     g_lr = [1e-4, 2e-4, 1e-3],
    #     eq_lr = [1e-4, 2e-4, 1e-3]
    # )

    # hyper_space = dict(
    #     num_epochs = [10000],
    #     d_lr = [1e-4, 2e-4, 1e-3],
    #     g_lr = [1e-4, 2e-4, 1e-3],
    #     g_units=[16, 32, 64],
    #     g_layers=[2, 4, 8],
    #     d_units=[16, 32, 64],
    #     d_layers=[2, 4, 8],
    # )

    # For new, working, unsupervised GAN method
    hyper_space = dict(
        num_epochs = [10000],
        d_lr = [2e-4, 1e-3],
        g_lr = [2e-4, 1e-3],
        g_units=[64, 32],
        g_layers=[6, 4],
        d_units=[64, 32],
        d_layers=[6, 4],
        gp=[0.01, 0.1],
        g_iters = [1],
        d_iters = [1],
    )

    # hyper_space = dict(
    #     num_epochs = [1000],
    #     d_lr = [2e-4],
    #     g_lr = [2e-4],
    #     g_units=[64],
    #     g_layers=[6],
    #     d_units=[64],
    #     d_layers=[6],
    #     gp=[0.01],
    #     g_iters = [1],
    #     d_iters = [1],
    # )

    n_iters = np.product([len(v) for k, v in hyper_space.items()])
    print(f'Searching {n_iters} possibilities...')
    hyper_space = dict_product(hyper_space)

    max_cpus = 1
    n_cpus = n_iters if n_iters < max_cpus else max_cpus
    p = mp.Pool(n_cpus)
    results = p.map(collect_results, hyper_space)

    resdf = pd.DataFrame().from_records(results)
    resdf.to_csv('GAN_hyper_results.csv')
