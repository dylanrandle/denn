# Utilities used throughout
import numpy as np
import torch

def calc_renot(u_bar, delta, nu):
    """ calculates Re_not where Re stands for Reynolds number"""
    n = u_bar.shape[0]
    U_0 = u_bar[n//2][0]
    renot = U_0 * delta / nu
    return renot

def calc_renum(u_bar, ygrid, delta, nu):
    """ calculates Reynolds number """
    n = u_bar.shape[0]
    U_bar = (1/delta) * np.trapz(u_bar[:n//2], x=ygrid[:n//2], axis=0)[0] # integrate from wall to center
    renum = 2 * delta * U_bar / nu
    return renum

def calc_retau(delta, dp_dx, rho, nu):
    """calculates Re_tau (Re stands for Reynolds number)"""
    tau_w = -delta * dp_dx
    u_tau = np.sqrt(tau_w / rho)
    re_tau = u_tau * delta / nu
    return re_tau

def convert_dns(delta, hypers, dns):
    """re-dimensionalizes DNS data"""
    tau_w = -delta * hypers['dp_dx']
    u_tau = np.sqrt(tau_w / hypers['rho'])
    h_v = hypers['nu'] / u_tau
    half_y = dns[['y+,']]*h_v
    half_u = dns[['u_1,']]
    return half_u,  half_y

def plot_dns(handle, half_u, half_y, delta):
    """plots DNS data (reflected over the center axis)"""
    handle.plot(half_u, half_y-1, color='red', label='DNS')
    handle.plot(half_u, -half_y+2*delta-1, color='red')
    handle.legend()

def loss_vs_distance(ax, ymin, ymax, model, hypers, reynolds_stress):
    """ plots model loss vs. distance in the channel """
    y = torch.tensor(torch.linspace(ymin,ymax,1000).reshape(-1,1), requires_grad=True)
    u_bar = model.predict(y)
    axial_eqn = model.compute_diffeq(u_bar, y, reynolds_stress, hypers['nu'], hypers['rho'], hypers['dp_dx'])
    ax.plot(y.detach().numpy(), np.power(axial_eqn.detach().numpy(), 2), 'o', markersize=2, lw=0.5, label='square')
    ax.set_title('Loss as a function of distance on ({}, {})'.format(ymin, ymax))
    ax.set_ylabel('Loss (f^2 or |f|)')
    ax.set_xlabel('position (y)')
    ax.legend()

def make_plots(ax, losses,  model, hypers, retau, dns_u=None, dns_y=None):
    """ plot loss and prediction of model at retau """
    # losses
    ax[0].loglog(np.arange(len(losses)), losses, color='blue')
    ax[0].set_title('Log mean loss vs. log epoch at Retau={}'.format(retau))
    ax[0].set_xlabel('log( epoch )')
    ax[0].set_ylabel('log( mean loss )')
    # preds
    y_space = torch.linspace(hypers['ymin'], hypers['ymax'], 1000).reshape(-1,1)
    preds = model.predict(y_space).detach().numpy()
    ax[1].plot(preds, y_space.detach().numpy(), alpha=1, color='blue', label='NN')
    if dns_u is not None and dns_y is not None:
        ax[1].plot(dns_u, dns_y, alpha=1, color='red', label='DNS')
    ax[1].set_title('Predicted $<u>$ at Retau={}'.format(retau))
    ax[1].set_ylabel('y')
    ax[1].set_xlabel('$<u>$')
    ax[1].legend()
