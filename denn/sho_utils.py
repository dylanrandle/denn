import matplotlib.pyplot as plt
import numpy as np
import torch
from denn.utils import diff
from IPython.display import clear_output

def produce_SHO_preds(G, t, x0=0, dx_dt0=0.5):
    """ produce sho preds that satisfy system, without u adjustment """
    x_raw = G(t)

    # adjust for initial conditions on x and dx_dt
    x_adj = x0 + (1 - torch.exp(-t)) * dx_dt0 + ((1 - torch.exp(-t))**2) * x_raw

    dx_dt = diff(x_adj, t)

    d2x_dt2 = diff(dx_dt, t)

    return x_adj, dx_dt, d2x_dt2

def produce_SHO_preds_system(G, t, x0=0, dx_dt0=0.5):
    """ produce preds that satisfy equation """
    x_pred = G(t)

    # x condition
    x_adj = x0 + (1 - torch.exp(-t)) * dx_dt0 + ((1 - torch.exp(-t))**2) * x_pred

    # dx_dt (directly from NN output, x_pred)
    dx_dt = diff(x_pred, t)

    # u condition guarantees that dx_dt = u (first equation in system)
    u_adj = torch.exp(-t) * dx_dt0 + 2 * (1 - torch.exp(-t)) * torch.exp(-t) * x_pred + (1 - torch.exp(-t)) * dx_dt

    # compute du_dt = d2x_dt2
    du_dt = diff(u_adj, t)

    return x_adj, u_adj, du_dt

def plot_SHO(g_loss, d_loss, t, analytic, G, pred_fn, clear=False, savefig=False, fname=None):
    """ helpful plotting function for Simple Harmonic Oscillator problem """
    # clear the cell
    if clear:
      clear_output(True)

    # get all preds/derivatives
    x_adj, dx_dt, d2x_dt2 = pred_fn(G, t)

    # convert to numpy
    preds = x_adj.cpu().detach().numpy()[:,0]
    dx_dt = dx_dt.cpu().detach().numpy()
    d2x_dt2 = d2x_dt2.cpu().detach().numpy()[:,0]
    t = t.cpu().detach().numpy()
    analytic = analytic.cpu().detach().numpy()[:,0]

    # make plots
    fig, ax = plt.subplots(1,3,figsize=(20,6))
    steps = len(g_loss)
    epochs = np.arange(steps)

    # Losses
    ax[0].plot(epochs, [g[0] for g in g_loss], label='g_loss1', color='orange')
    ax[0].plot(epochs, [g[1] for g in g_loss], label='g_loss2', color='green')

    ax[0].plot(epochs, [d[0] for d in d_loss], label='d_loss1', color='blue')
    ax[0].plot(epochs, [d[1] for d in d_loss], label='d_loss2', color='red')

    ax[0].legend()
    ax[0].set_title('Losses')

    # Prediction
    ax[1].plot(t, preds, label='pred', color='orange')
    ax[1].plot(t, analytic, '--', label='true', color='blue')
    ax[1].legend()
    ax[1].set_title('Prediction')

    # Derivatives
    ax[2].plot(t, preds, label='x', color='orange')
    ax[2].plot(t, -d2x_dt2, '--', label="-x''", color='green')
    ax[2].legend()
    ax[2].set_title('Derivatives')

    if not savefig:
        plt.show()
    else:
        plt.savefig(fname)
