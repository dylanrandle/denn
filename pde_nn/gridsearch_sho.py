""" Driver script for hyperparameter search for Simple Harmonic Oscillator """
from gan import train_GAN_SHO, plot_losses_and_preds
import numpy as np
import torch

# set seed for reproducibility
torch.manual_seed(42)

# analytic solution
tlo,thi=0,10
t = np.linspace(tlo,thi,100)
analytic_oscillator = lambda t: 1*np.cos(t) + -1*np.sin(t)

args = dict(g_hidden_units=40,
            g_hidden_layers=3,
            d_hidden_units=20,
            d_hidden_layers=2,
            d_lr=0.001,
            g_lr=0.001,
            t_low=tlo,
            t_high=thi,
            n=100,
            real_label=1,
            fake_label=-1,
            logging=True,
            G_iters=1,
            D_iters=1,
            m=1,
            k=1,
            clip=1,
            loss_diff=.1,
            max_while=20,
            grad_penalty=1,
            x0=1,
            dx_dt0=-1)
epochs=2000
fname='epochs'+str(epochs)+'_'+'_'.join([str(key)+str(val) for key,val in list(zip(args.keys(), args.values()))])+'.png'

G,D,G_loss,D_loss = train_GAN_SHO(epochs, **args)
loss_ax, pred_ax = plot_losses_and_preds(np.exp(G_loss),
                                         np.exp(D_loss),
                                         G,
                                         t,
                                         analytic_oscillator,
                                         savefig=True,
                                         fname=fname)
