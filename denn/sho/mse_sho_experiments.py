from denn.sho.mse_sho import train_MSE
from denn.utils import Generator
import pandas as pd
import multiprocessing as mp

def collect_mse(obs_every, queue):
    mses = []
    for i in range(10):
        resnet = Generator(n_hidden_units=30, n_hidden_layers=7, residual=True)
        result = train_MSE(resnet, niters=100000, observe_every=obs_every, seed=i, make_plot=False) # random seed
        final_mse = result['final_mse']
        mses.append(final_mse)
    res = {'obs_every': obs_every, 'mses': mses}
    queue.put(res)

if __name__== "__main__":
    # queue used for shared memory
    q = mp.Queue()
    # spawn processes and store them in a list
    processes = []
    obs_every = [1, 2, 4, 8, 16, 25, 32, 50]
    for o in obs_every:
        p = mp.Process(target=collect_mse, args=(o,q))
        p.start()
        processes.append(p)
    # join will block until process exits
    for p in processes:
        p.join()
    # the results from each process are stored in the Queue
    while not q.empty():
        mse_results.append(q.get())
    # write to CSV
    resdf = pd.DataFrame().from_records(mse_results)
    resdf.to_csv('LAG_obsevery_mse_results.csv')
