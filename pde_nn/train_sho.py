""" Driver script for hyperparameter search for Simple Harmonic Oscillator """
from gan import train_GAN_SHO, plot_losses_and_preds
import numpy as np
import torch
from torch import nn
import sys

try:
    experiment_name = str(sys.argv[1])
except:
    raise Exception('Provide an experiment name on the command line')

# set seed for reproducibility
torch.manual_seed(42)


args = dict(g_hidden_units=20,
            g_hidden_layers=2,
            d_hidden_units=20,
            d_hidden_layers=2,
            d_lr=0.001,
            g_lr=0.001,
            t_low=0,
            t_high=np.pi,
            logging=False,
            G_iters=1,
            D_iters=4,
            n=100,
            # clip=.1,
            # max_while=10,
            x0=0.,
            dx_dt0=.5,
            realtime_plot=False,
            activation=nn.Tanh(),
            wgan=False,
            soft_labels=False,          ## soft labels not work
            real_data=False,
            gradient_penalty=False,
            gp_hyper=0.)

epochs=100

print('Starting training')
G,D,G_loss,D_loss = train_GAN_SHO(epochs, **args, savefig=True, fname=experiment_name)
print('Done')
# loss_ax, pred_ax = plot_losses_and_preds(np.exp(G_loss),
#                                          np.exp(D_loss),
#                                          G,
#                                          t,
#                                          analytic_oscillator,
#                                          savefig=True,
#                                          fname=fname)
