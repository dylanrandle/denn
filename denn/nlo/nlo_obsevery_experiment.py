import pandas as pd
import multiprocessing as mp
import numpy as np
from denn.utils import Generator, Discriminator
from denn.nlo.mse_nlo import train_MSE
from denn.nlo.gan_nlo import train_GAN_NLO

N_REPS = 5
N_ITERS = 10000
N_HIDDEN = 64
N_LAYERS = 8
N_PERIODS = 4
OBS_EVERY = [1, 2, 4, 8, 10, 20]

def run_gan(obs_every, queue, seed):
    D = Discriminator(in_dim=2, out_dim=1,
                      n_hidden_units=32,
                      n_hidden_layers=8,
                      unbounded=True,
                      residual=True)
    G = Generator(in_dim=1, out_dim=1,
                  n_hidden_units=N_HIDDEN,
                  n_hidden_layers=N_LAYERS,
                  output_tan=True,
                  residual=True)
    result = train_GAN_NLO(G, D, num_epochs=N_ITERS, observe_every=obs_every, seed=seed)
    queue.put(result['final_mse'])

def run_semisup(obs_every, queue, seed):
    resgnet = Generator(n_hidden_units=N_HIDDEN, n_hidden_layers=N_LAYERS, residual=True, output_tan=True)
    result = train_MSE(resgnet, method='semisupervised', niters=N_ITERS, observe_every=obs_every, seed=seed)
    queue.put(result['final_mse'])

def run_sup(obs_every, queue, seed):
    resgnet = Generator(n_hidden_units=N_HIDDEN, n_hidden_layers=N_LAYERS, residual=True, output_tan=True)
    result = train_MSE(resgnet, method='supervised', niters=N_ITERS, observe_every=obs_every, seed=seed)
    queue.put(result['final_mse'])

def collect_mse(obs_every, queue, method):
    if method == 'gan':
        target = run_gan
    elif method == 'semisupervised':
        target = run_semisup
    else: # method == 'supervised':
        target = run_sup

    ## PARALELLIZE OVER N_REPS
    q = mp.Queue()
    procs = []
    for seed in range(N_REPS):
        p = mp.Process(target=target, args=(obs_every, q, seed))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    mses = []
    while not q.empty():
        mses.append(q.get())

    res = {'obs_every': obs_every, 'mse_mean': np.mean(mses), 'mse_std': np.std(mses)}
    queue.put(res)

def map_processes(method):
    assert method in ['supervised', 'semisupervised', 'gan']

    ## PARALLELIZE OVER OBS_EVERY

    # queue used for shared memory
    q = mp.Queue()

    # spawn processes and store them in a list
    processes = []
    for o in OBS_EVERY:
        p = mp.Process(target=collect_mse, args=(o, q, method))
        p.start()
        processes.append(p)

    # join processes, will block until process exits
    for p in processes:
        p.join()

    # collect results from Queue, store in DataFarme
    mse_results = []
    while not q.empty():
        mse_results.append(q.get())
    resdf = pd.DataFrame().from_records(mse_results)

    # write to CSV
    if method=='semisupervised':
        resdf.to_csv('LAG_obsevery_mse_results_NLO.csv')
    elif method=='supervised':
        resdf.to_csv('SUP_obsevery_mse_results_NLO.csv')
    else: # method == 'gan':
        resdf.to_csv('GAN_obsevery_mse_results_NLO.csv')

if __name__== "__main__":
    map_processes('gan')
    map_processes('semisupervised')
    map_processes('supervised')
