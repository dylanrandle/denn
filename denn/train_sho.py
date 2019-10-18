""" Driver script for hyperparameter search for Simple Harmonic Oscillator """
from denn.gan_sho import train_GAN_SHO
import numpy as np
import torch
from torch import nn
import sys
import os
import multiprocessing as mp

# set seed for reproducibility
torch.manual_seed(42)

# use this file to change default backend if parallel problems when saving figure
import matplotlib
print('Matplotlib rc file: {}'.format(matplotlib.matplotlib_fname()))

g_units = [40,80]
g_layers = [6,8]
# d_units= [20,40,80,120]
# d_layers = [1,2,4,6]
epochs = [10000, 16666]
n_pts = [100, 200]
gp_hyper = [1., 5., 10.]

settings = []
for gu in g_units:
    for gl in g_layers:
        #for du in d_units:
            #for dl in d_layers:
        for e in epochs:
            for n in n_pts:
                for gp in gp_hyper:
                     settings.append((gu, gl, e,n,gp))

# epochs=5000
# print('Training each setting for {} epochs'.format(epochs))

def run_at_params(gu, gl, e, n, gp):
    torch.manual_seed(42)
    args = dict(  # NETWORKS
                  activation=nn.Tanh(),
                  g_hidden_units=gu,
                  g_hidden_layers=gl,
                  d_hidden_units=20,
                  d_hidden_layers=2,

                  G_iters=1,
                  D_iters=5,

                  # FROM WGAN PAPER
                  d_lr=0.0001,
                  g_lr=0.0001,
                  d_betas=(0., 0.9),
                  g_betas=(0., 0.9),

                  # PROBLEM
                  t_low=0,
                  t_high=2*np.pi,
                  n=n,
                  x0=0.,
                  dx_dt0=.5,

                  # VIZ
                  logging=False,
                  realtime_plot=False,

                  # Hacks
                  real_data=False,
                  soft_labels=False,

                  # WGAN
                  wgan=True,
                  gradient_penalty=True,
                  gp_hyper=gp,

                  # SYSTEM
                  systemOfODE=True
                  )
    experiment_name = 'WGANSystem_{}keps_{}x{}gen_{}pts_{}gp.png'.format(e//1000, gu, gl, n, gp)
    G,D,G_loss,D_loss = train_GAN_SHO(e,**args,savefig=True,fname=experiment_name)

    t_np = np.linspace(args['t_low'], args['t_high'], args['n']).reshape(-1,1)
    t_torch = torch.linspace(args['t_low'], args['t_high'], args['n'], dtype=torch.float, requires_grad=True).reshape(-1,1)
    analytic_oscillator_np = lambda t: args['x0']*np.cos(t) + args['dx_dt0']*np.sin(t)
    true_sol = analytic_oscillator_np(t_np)
    pred_sol = G(t_torch).detach().numpy()
    mse = np.mean((pred_sol - true_sol) ** 2)
    print('\n========================')
    print('MSE = {}'.format(mse))
    print('Params = eps:{}, g_unit:{}, g_layer:{}, pts:{}, gp:{}'.format(e, gu, gl, n, gp))
    print('========================')

print("Total searches to try {}".format(len(settings)))
print('Starting training')

print('Using {} cpus'.format(mp.cpu_count()))
p = mp.Pool(mp.cpu_count())
p.starmap(run_at_params, settings)
print('Done')
