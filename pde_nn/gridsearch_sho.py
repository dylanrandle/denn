""" Driver script for hyperparameter search for Simple Harmonic Oscillator """
from gan import train_GAN_SHO, plot_losses_and_preds
import numpy as np

t = np.linspace(0,10,100)
analytic_oscillator = lambda t: np.cos(t)

args = dict(g_hidden_units=20,
            g_hidden_layers=2,
            d_hidden_units=20,
            d_hidden_layers=3,
            d_lr=0.001,
            g_lr=0.001,
            t_low=0,
            t_high=10,
            n=100,
            real_label=1,
            fake_label=-1,
            logging=True,
            G_iters=1,
            D_iters=1,
            m=1,
            k=1,
            clip=.1,
            loss_diff=.1,
            max_while=50)
G,D,G_loss,D_loss = train_GAN_SHO(1000, **args)
loss_ax, pred_ax = plot_losses_and_preds(np.exp(G_loss),
                                         np.exp(D_loss),
                                         G,
                                         t,
                                         analytic_oscillator,
                                         savefig=True,
                                         fname='test')
