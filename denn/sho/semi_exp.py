import denn.experiments as exp
import denn.config as cfg
import pandas as pd
import argparse
from denn.utils import handle_overwrite

sho = exp.get_problem('sho')

gan = cfg.gan_kwargs
gen = cfg.gen_kwargs
disc = cfg.disc_kwargs

l2_mlp = cfg.L2_mlp_kwargs
l2_train = cfg.L2_kwargs

d1_factor = 1e-3

def run_gan(obs_freqs, niters=100):
    mses = []
    gan['method'] = 'semisupervised'
    gan['plot'] = False # turn off plotting
    gan['niters'] = niters
    for o in obs_freqs:
        gan['obs_every'] = o
        gan['d1'] = d1_factor
        gan['d2'] = 1/o
        print('Running GAN at obs_every = {}'.format(o))
        res = exp.gan_experiment(sho, gen_kwargs=gen, disc_kwargs=disc, train_kwargs=gan)
        mses.append(res['final_mse'])
    return mses

def run_L2(obs_freqs, niters=100):
    mses = []
    l2_train['method'] = 'semisupervised'
    l2_train['plot'] = False
    l2_train['niters'] = niters
    for o in obs_freqs:
        l2_train['obs_every'] = o
        l2_train['d1'] = d1_factor
        l2_train['d2'] = 1/o
        print('Running L2 at obs_every = {}'.format(o))
        res = exp.L2_experiment(sho, model_kwargs=l2_mlp, train_kwargs=l2_train)
        mses.append(res['final_mse'])
    return mses

def run_sup(obs_freqs, niters=100):
    mses = []
    l2_train['method'] = 'supervised'
    l2_train['plot'] = False
    l2_train['niters'] = niters
    for o in obs_freqs:
        print('Running supervised (L2) at obs_every = {}'.format(o))
        l2_train['obs_every'] = o
        res = exp.L2_experiment(sho, model_kwargs=l2_mlp, train_kwargs=l2_train)
        mses.append(res['final_mse'])
    return mses

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--seed', type=int, default=0,
        help='value of random seed set for reproducibility (default: 0)')
    args.add_argument('--fname', type=str, default=None,
        help='file name to save figure (default None: use function default)')
    args = args.parse_args()

    niters = 5000
    obs_freqs = [1, 2, 5, 10, 20, 50, 100]

    gan_fname = 'gan_'+args.fname if args.fname else 'gan_mse.csv'
    l2_fname = 'l2_'+args.fname if args.fname else 'l2_mse.csv'
    sup_fname = 'sup_'+args.fname if args.fname else 'sup_mse.csv'

    handle_overwrite(gan_fname)

    gan_mse = run_gan(obs_freqs, niters=niters)
    pd.DataFrame(list(zip(gan_mse, obs_freqs)), columns=['mse', 'obs_freq']).to_csv(gan_fname)

    l2_mse = run_L2(obs_freqs, niters=niters)
    pd.DataFrame(list(zip(l2_mse, obs_freqs)), columns=['mse', 'obs_freq']).to_csv(l2_fname)

    sup_mse = run_sup(obs_freqs, niters=niters)
    pd.DataFrame(list(zip(sup_mse, obs_freqs)), columns=['mse', 'obs_freq']).to_csv(sup_fname)
