# Utilities used throughout
import numpy as np
import torch
import channel_flow as chan
import pandas as pd
import matplotlib.pyplot as plt

# def Swish(x):
#     """input: x - a pytorch tensor
#        output: out - swish activation as pytorch tensor"""
#     return x * x.sigmoid()
class Swish(torch.nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, input):
        return input * torch.sigmoid(self.beta * input)

    def extra_repr(self):
        return 'beta={}'.format(self.beta)

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
    y = torch.tensor(torch.linspace(ymin,ymax,n).reshape(-1,1), requires_grad=True)
    u_bar = model.predict(y)
    axial_eqn = model.compute_diffeq(u_bar, y)
    ax.plot(y.detach().numpy(), np.power(axial_eqn.detach().numpy(), 2), 'o', markersize=2, lw=0.5, label='square')
    ax.set_title('Loss as a function of distance on ({}, {})'.format(ymin, ymax))
    ax.set_ylabel('Loss (f^2 or |f|)')
    ax.set_xlabel('position (y)')
    ax.legend()

def make_plots(ax, train_loss, val_loss,  preds, hypers, retau, numerical):
    """ plot loss and prediction of model at retau """
    # losses
    ax[0].loglog(np.arange(train_loss.shape[0]), train_loss, color='blue', label='train')
    ax[0].loglog(np.arange(val_loss.shape[0])*100, val_loss, color='orange', label='val', alpha=0.7)
    ax[0].set_title('Log mean loss vs. log epoch at Retau={}'.format(retau))
    ax[0].set_xlabel('log( epoch )')
    ax[0].set_ylabel('log( mean loss )')
    ax[0].legend()
    # preds
    y_space = torch.linspace(hypers['ymin'], hypers['ymax'], hypers['n']).reshape(-1,1)
    y=y_space.detach().numpy()
    ax[1].plot(preds, y, alpha=1, color='blue', label='NN')
    ax[1].plot(numerical, y, label='FD', color='black')
    ax[1].set_title('Predicted $<u>$ at Retau={}'.format(retau))
    ax[1].set_ylabel('y')
    ax[1].set_xlabel('$<u>$')
    ax[1].legend()

def expose_results(folder_timestamp, top_dir='data/', dns_file='data/LM_Channel_Retau180.txt', numerical_file='data/mixlen_numerical_u180.npy'):

    # load everything from disk
    preds = np.load(top_dir+'{}/preds.npy'.format(folder_timestamp))
    train_loss = np.load(top_dir+'{}/train_loss.npy'.format(folder_timestamp))
    val_loss = np.load(top_dir+'{}/val_loss.npy'.format(folder_timestamp))
    hypers = np.load(top_dir+'{}/hypers.npy'.format(folder_timestamp))
    hypers = hypers.item()
    print('hypers: \n{}'.format(hypers))
    pdenn = chan.Chanflow(**hypers)
    pdenn.load_state_dict(torch.load(top_dir+'{}/model.pt'.format(folder_timestamp)))
    dns = pd.read_csv(dns_file, delimiter=' ')
    # half_u, half_y = convert_dns(hypers, dns)
    numerical = np.load(numerical_file)
    retau=hypers['retau']

    fig, ax = plt.subplots(1, 2, figsize=(20,10))
    plot_dns(ax[1], dns, hypers)
    make_plots(ax, train_loss, val_loss, preds, hypers, retau, numerical)

    y = np.linspace(hypers['ymin'], hypers['ymax'], hypers['n'])
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    ax.plot(preds, y, 'o', label='NN', color='blue', alpha=1, markersize=0.5, linewidth=1)
    ax.plot(numerical, y, label='FD', color='black')
    ax.set_title('$<U>$ @ Re_tau = {}'.format(retau))
    ax.set_xlabel('$<U>$')
    ax.set_ylabel('$y$')
    ax.legend()

    fig, ax = plt.subplots(2,2, figsize=(12,10))
    loss_vs_distance(ax[0,0],  -1, 1, pdenn, 1000)
    loss_vs_distance(ax[0,1], -.5, .5, pdenn, 1000)
    loss_vs_distance(ax[1,0], .8, 1, pdenn, 1000)
    loss_vs_distance(ax[1,1], .99, 1, pdenn, 1000)

    plt.show()
