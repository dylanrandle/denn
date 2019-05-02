""" Driver script for hyperparameter search for Simple Harmonic Oscillator """
from gan import train_GAN_SHO, plot_losses_and_preds
import numpy as np
import torch
from torch import nn
import sys
import os
import multiprocessing as mp

# set seed for reproducibility
torch.manual_seed(42)

# use this file to change default backend if parallel problems
import matplotlib
print('Matplotlib rc file: {}'.format(matplotlib.matplotlib_fname()))

g_units = [20,30,40]
g_layers = [2,3,4]
d_units= [20,30,40]
d_layers = [2,3,4]

settings = []
for gu in g_units:
    for gl in g_layers:
        for du in d_units:
            for dl in d_layers:
                settings.append((gu,gl,du,dl))

epochs=10000
print('Training each setting for {} epochs'.format(epochs))

def run_at_params(gunit, glayer, dunit, dlayer):
    args = dict(g_hidden_units=gunit,
                g_hidden_layers=glayer,
                d_hidden_units=dunit,
                d_hidden_layers=dlayer,
                d_lr=0.001,
                g_lr=0.001,
                t_low=0,
                t_high=2*np.pi,
                logging=False,
                G_iters=9,
                D_iters=1,
                n=100,
                x0=0.,
                dx_dt0=.5,
                realtime_plot=False,
                activation=nn.Tanh(),
                wgan=False,
                soft_labels=False,
                real_data=False,
                gradient_penalty=False,
                gp_hyper=0.,
                systemOfODE=True)
    experiment_name = 'systemODE_{}keps_{}x{}gen_{}x{}disc'.format(epochs//1000, gunit, glayer, dunit, dlayer)
    train_GAN_SHO(epochs,**args,savefig=True,fname=experiment_name)

print("Total searches {}".format(len(settings)))
print('Starting training')

print('Using {} cpus'.format(mp.cpu_count()))
p = mp.Pool(mp.cpu_count())
p.starmap(run_at_params, settings)
print('Done')
