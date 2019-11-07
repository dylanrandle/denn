from denn.sho.gan_sho import train_GAN_SHO
import pandas as pd
import multiprocessing as mp
import random

def collect_mse(obs_every, queue):
    mses = []
    for i in range(10):
        result = train_GAN_SHO(num_epochs=100000, observe_every=obs_every, seed=i)
        final_mse = result['final_mse']
        mses.append(final_mse)
    res = {'obs_every': obs_every, 'mses': mses}
    queue.put(res)

if __name__== "__main__":
    # queue used for shared memory
    q = mp.Queue()
    # spawn processes and store them in a list
    processes = []
    obs_every = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    for o in obs_every:
        p = mp.Process(target=collect_mse, args=(o,q))
        p.start()
        processes.append(p)
    # join will block until process exits
    for p in processes:
        p.join()
    # the results from each process are stored in the Queue
    mse_results = []
    while not q.empty():
        mse_results.append(q.get())
    # write to CSV
    resdf = pd.DataFrame().from_records(mse_results)
    resdf.to_csv('GAN_obsevery_mse_results.csv')
