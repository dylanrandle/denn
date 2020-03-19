import denn.rans.channel_flow as chan
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

# global plot params
plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=12)
plt.rc('legend', fontsize=12)

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

def convert_dns(hypers, dns):
    """re-dimensionalizes DNS data"""
    tau_w = -hypers['delta'] * hypers['dp_dx']
    u_tau = np.sqrt(tau_w / hypers['rho'])
    h_v = hypers['nu'] / u_tau
    half_y = dns[['y+,']]*h_v
    half_u = dns[['u_1,']]
    return half_u,  half_y

def plot_dns(handle, dns, hypers, color='red', label='DNS'):
    half_u, half_y = convert_dns(hypers, dns)
    """plots DNS data (reflected over the center axis)"""
    delta=hypers['delta']
    handle.plot(half_u, half_y-1, color=color, label=label)
    handle.plot(half_u, -half_y+2*delta-1, color=color)
    handle.legend()

def loss_vs_distance(ax, ymin, ymax, model, n):
    """ plots model loss vs. distance in the channel """
    y = torch.linspace(ymin,ymax,n, requires_grad=True).reshape(-1,1)
    u_bar = model.predict(y)
    axial_eqn = model.compute_diffeq(u_bar, y)
    ax.plot(y.detach().numpy(), np.abs(axial_eqn.detach().numpy()), 'o',
        alpha=0.5, markersize=2, color='crimson', lw=0.5, label='square')
    # ax.set_title('Loss as a function of distance on ({}, {})'.format(ymin, ymax))
    ax.set_ylabel('Loss')
    ax.set_xlabel('$y$')
    # ax.legend()

def make_plots(ax, train_loss, val_loss,  preds, hypers, retau, numerical):
    """ plot loss and prediction of model at retau for RANS equations """
    # losses
    ax[0].loglog(np.arange(train_loss.shape[0]), train_loss, color='crimson', label='train')
    ax[0].loglog(np.arange(val_loss.shape[0])*100, val_loss, color='blue', label='val', alpha=0.7)
    # ax[0].set_title('Log mean loss vs. log epoch at Retau={}'.format(retau))
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='lower left')
    # preds
    y_space = torch.linspace(hypers['ymin'], hypers['ymax'], hypers['n']).reshape(-1,1)
    y=y_space.detach().numpy()
    ax[1].plot(preds, y, alpha=1, color='crimson', label='NN')
    ax[1].plot(numerical, y, label='FD', color='blue')
    # ax[1].set_title('Predicted $<u>$ at Retau={}'.format(retau))
    ax[1].set_ylabel('$y$')
    ax[1].set_xlabel('$\\bar{u}$')
    ax[1].legend(loc='center left')

def expose_results(folder_timestamp, top_dir='experiments/', dns_file='data/LM_Channel_Retau180.txt', numerical_file='data/mixlen_numerical_u180.npy'):
    """ useful function for loading results """
    # load everything from disk
    preds = np.load(top_dir+'{}/preds.npy'.format(folder_timestamp))
    train_loss = np.load(top_dir+'{}/train_loss.npy'.format(folder_timestamp))
    val_loss = np.load(top_dir+'{}/val_loss.npy'.format(folder_timestamp))
    hypers = np.load(top_dir+'{}/hypers.npy'.format(folder_timestamp), allow_pickle=True)
    hypers = hypers.item()
    print('hypers: \n{}'.format(hypers))
    pdenn = chan.Chanflow(**hypers)
    pdenn.load_state_dict(torch.load(top_dir+'{}/model.pt'.format(folder_timestamp)))
    # dns = pd.read_csv(dns_file, delimiter=' ')
    # half_u, half_y = convert_dns(hypers, dns)
    numerical = np.load(numerical_file)
    retau=hypers['retau']

    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    # plot_dns(ax[1], dns, hypers)
    make_plots(ax, train_loss, val_loss, preds, hypers, retau, numerical)
    fig.tight_layout()
    # plt.savefig('rans_nn_vs_fd_perturbed.png')

    # y = np.linspace(hypers['ymin'], hypers['ymax'], hypers['n'])
    # fig, ax = plt.subplots(1, 1, figsize=(10,8))
    # ax.plot(preds, y, 'o', label='NN', color='blue', alpha=1, markersize=0.5, linewidth=1)
    # ax.plot(numerical, y, label='FD', color='black')
    # # ax.set_title('$<U>$ @ Re_tau = {}'.format(retau))
    # ax.set_xlabel('$<U>$')
    # ax.set_ylabel('$y$')
    # ax.legend()

    npts = 10000
    fig, ax = plt.subplots(2,2, figsize=(6,5))
    loss_vs_distance(ax[0,0],  -1, 1, pdenn, npts)
    loss_vs_distance(ax[0,1], -.5, .5, pdenn, npts)
    loss_vs_distance(ax[1,0], -1, -0.9, pdenn, npts)
    loss_vs_distance(ax[1,1], 0.9, 1, pdenn, npts)
    fig.tight_layout()

    plt.show()
