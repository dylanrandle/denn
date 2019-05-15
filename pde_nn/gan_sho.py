"""
Implementation of GAN for Unsupervised deep learning of differential equations

Equation:
dx/dt = L * x

Analytic Solution:
x = exp(L * t)
"""
import torch
import torch.nn as nn
from torch import tensor, autograd
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from IPython.display import clear_output

class Generator(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, n_hidden_units=20, n_hidden_layers=2, activation=nn.Tanh(), x0=1,
                output_tan=True):
        super(Generator, self).__init__()

        # initial condition
        self.x0 = x0

        layers = [('lin1', nn.Linear(in_dim, n_hidden_units)), ('act1', activation)]
        for i in range(n_hidden_layers):
            layer_id = i+2
            layers.append(('lin{}'.format(layer_id), nn.Linear(n_hidden_units, n_hidden_units)))
            layers.append(('act{}'.format(layer_id), activation))
        layers.append(('linout', nn.Linear(n_hidden_units, out_dim)))
        if output_tan:
            layers.append(('actout', nn.Tanh()))

        layers = OrderedDict(layers)
        self.main = nn.Sequential(layers)

    def forward(self, x):
        output = self.main(x)
        return output

    def predict(self, t):
        x_pred = self(t)
        # use Marios adjustment for initial condition
        x_adj = self.x0 + (1 - torch.exp(-t)) * x_pred
        return x_adj

class Discriminator(nn.Module):
    def __init__(self, vec_dim=1, n_hidden_units=20, n_hidden_layers=2, activation=nn.Tanh(), unbounded=False):
        super(Discriminator, self).__init__()

        layers = [('lin1', nn.Linear(vec_dim, n_hidden_units)), ('act1', activation)]
        for i in range(n_hidden_layers):
            layer_id = i+2
            layers.append(('lin{}'.format(layer_id), nn.Linear(n_hidden_units, n_hidden_units)))
            layers.append(('act{}'.format(layer_id), activation))
        layers.append(('linout', nn.Linear(n_hidden_units, vec_dim)))
        if not unbounded:
            # unbounded used for WGAN (no sigmoid)
            layers.append(('actout', nn.Sigmoid()))

        layers = OrderedDict(layers)
        self.main = nn.Sequential(layers)

    def forward(self, x):
        output = self.main(x)
        return output

def plot_loss(G_loss, D_loss, ax):
    epochs=np.arange(len(G_loss))
    ax.plot(epochs, np.log(G_loss), label='G Loss')
    ax.plot(epochs, np.log(D_loss), label='D Loss')
    ax.set_title('Loss of D and G')
    ax.set_xlabel('epoch')
    ax.set_ylabel('log-loss')
    ax.legend()
    return ax

def plot_preds(G, t, analytic, ax):
    ax.plot(t, analytic(t), label='analytic')
    t_torch = tensor(t, dtype=torch.float, requires_grad=True).reshape(-1,1)
    pred = G.predict(t_torch)
    ax.plot(t, pred.detach().numpy().flatten(), '--', label='pred')
    ax.set_title('Pred and Analytic')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend()
    return ax

def plot_losses_and_preds(G_loss, D_loss, G, t, analytic, figsize=(15,5), savefig=False, fname=None):
    fig, ax = plt.subplots(1,2,figsize=figsize)
    ax1 = plot_loss(G_loss, D_loss, ax[0])
    ax2 = plot_preds(G, t, analytic, ax[1])
    if not savefig:
       plt.show()
    else:
       plt.savefig(fname)
    return ax1, ax2

def realtime_vis(g_loss, d_loss, t, preds, analytic_fn, dx_dt, d2x_dt2, savefig=False, fname=None):
    fig, ax = plt.subplots(1,3,figsize=(20,6))
    steps = len(g_loss)
    epochs = np.arange(steps)

    ax[0].plot(epochs, d_loss, label='d_loss')
    ax[0].plot(epochs, g_loss, label='g_loss')
    ax[0].legend()
    ax[0].set_title('Losses')

    ax[1].plot(t, analytic_fn(t), label='true')
    ax[1].plot(t, preds, '--', label='pred')
    ax[1].legend()
    ax[1].set_title('X Pred')

    ax[2].plot(t, dx_dt, label='dx_dt')
    ax[2].plot(t, d2x_dt2, label='d2x_dt2')
    ax[2].plot(t, -preds, '--', label='-x')
    ax[2].legend()
    ax[2].set_title('Derivatives')

    if not savefig:
       plt.show()
    else:
       plt.savefig(fname)

def realtime_vis_system(g_loss, d_loss, t, preds, analytic_fn, u, dx_dt, du_dt, savefig=False, fname=None):
    fig, ax = plt.subplots(1,3,figsize=(20,6))
    steps = len(g_loss)
    epochs = np.arange(steps)

    ax[0].plot(epochs, [dl[0] for dl in d_loss], label='d1_loss')
    ax[0].plot(epochs, [dl[1] for dl in d_loss], label='d2_loss')

    ax[0].plot(epochs, g_loss, label='g_loss')
    ax[0].legend()
    ax[0].set_title('Losses')

    ax[1].plot(t, analytic_fn(t), label='true')
    ax[1].plot(t, preds, '--', label='pred')
    ax[1].legend()
    ax[1].set_title('X Pred')

    ax[2].plot(t, dx_dt, label='dx_dt')
    ax[2].plot(t, du_dt, label='du_dt')
    ax[2].plot(t, -preds, '--', label='-x')
    ax[2].plot(t, u, '--', label='u')
    ax[2].legend()
    ax[2].set_title('Derivatives')

    if not savefig:
       plt.show()
    else:
       plt.savefig(fname)

def compute_first_derivative(x, t):
    dx_dt, = autograd.grad(x, t,
                           grad_outputs=x.data.new(x.shape).fill_(1),
                           create_graph=True)
    return dx_dt

def compute_second_derivative(dx_dt, t):
    d2x_dt2, = autograd.grad(dx_dt, t,
                             grad_outputs=dx_dt.data.new(dx_dt.shape).fill_(1),
                             create_graph=True)
    return d2x_dt2

def train_GAN_SHO(num_epochs,
          g_hidden_units=10,
          d_hidden_units=10,
          g_hidden_layers=2,
          d_hidden_layers=2,
          d_lr=0.001,
          g_lr=0.001,
          t_low=0,
          t_high=10,
          n=100,
          real_label=1,
          fake_label=0,
          logging=True,
          G_iters=1,
          D_iters=1,
          m=1.,
          k=1.,
          clip=.1,
          loss_diff=.1,
          max_while=20,
          gp_hyper=0.1,
          x0=0,
          dx_dt0=.5,
          activation=nn.Tanh(),
          realtime_plot=False,
          wgan=False,
          soft_labels=False,
          real_data=False,
          gradient_penalty=False,
          savefig=False,
          fname=None,
          systemOfODE=False):

    """
    function to perform training of generator and discriminator for num_epochs
    equation: simple harmonic oscillator (SHO)
    gan hacks:
        - wasserstein + clipping / wasserstein GP
        - label smoothing
        - while loop iters
    """
    if savefig and realtime_plot:
        raise Exception('savefig and realtime_plot both True. Assuming you dont want that.')

    if wgan:
        fake_label = -1

    out_dim=1
    if systemOfODE:
        out_dim=1 # symplectic: only 1 output

    # initialize nets
    G = Generator(in_dim=1, out_dim=out_dim,
                  n_hidden_units=g_hidden_units,
                  n_hidden_layers=g_hidden_layers,
                  activation=activation, # twice diff'able activation
                  x0=x0,
                  output_tan=True) # output range should be (-1,1) if True

    D = Discriminator(vec_dim=1,
                      n_hidden_units=d_hidden_units,
                      n_hidden_layers=d_hidden_layers,
                      activation=activation,
                      unbounded=wgan) # true for WGAN

    # if systemOfODE: # need second disc (and here we use second generator too)
    #     # G1 = G
    #     # G2 = Generator(in_dim=1, out_dim=out_dim,
    #     #               n_hidden_units=g_hidden_units,
    #     #               n_hidden_layers=g_hidden_layers,
    #     #               activation=activation,
    #     #               x0=x0,
    #     #               output_tan=True)
    #     D1 = D
    #     D2 = Discriminator(vec_dim=1,
    #                       n_hidden_units=d_hidden_units,
    #                       n_hidden_layers=d_hidden_layers,
    #                       activation=activation,
    #                       unbounded=wgan)

    # grid
    t_torch = torch.linspace(t_low, t_high, n, dtype=torch.float, requires_grad=True).reshape(-1,1)
    t_np = np.linspace(t_low, t_high, n).reshape(-1,1)

    delta_t = t_torch[1]-t_torch[0]
    def get_batch():
        """ perturb grid """
        return t_torch + delta_t * torch.randn_like(t_torch) / 3

    # labels
    real_label_vec = torch.full((n,), real_label).reshape(-1,1)
    fake_label_vec = torch.full((n,), fake_label).reshape(-1,1)

    # optimization
    if wgan:
        criterion = lambda y_true, y_pred: torch.mean(y_true * y_pred)
    else:
        criterion = nn.BCELoss()

    optiG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.9, 0.999))
    if not systemOfODE:
        optiD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.9, 0.999))
    else:
        optiD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.9, 0.999))
        # optiD1 = torch.optim.Adam(D1.parameters(), lr=d_lr, betas=(0.9, 0.999))
        # optiD2 = torch.optim.Adam(D2.parameters(), lr=d_lr, betas=(0.9, 0.999))

    # logging
    D_losses = []
    G_losses = []

    analytic_oscillator = lambda t: x0*torch.cos(t) + dx_dt0*torch.sin(t)
    analytic_oscillator_np = lambda t: x0*np.cos(t) + dx_dt0*np.sin(t)

    best_mse = 1e6

    def produce_SHO_preds(G, t):
        x_raw = G(t)
        # adjust for initial conditions on x and dx_dt
        x_adj = x0 + (1 - torch.exp(-t)) * dx_dt0 + ((1 - torch.exp(-t))**2) * x_raw

        dx_dt = compute_first_derivative(x_adj, t)

        d2x_dt2 = compute_second_derivative(dx_dt, t)

        return x_adj, dx_dt, d2x_dt2

    # def produce_SHO_preds_systemODE(G, t):
    #     pred_vec = G(t)
    #     x_pred, u_pred = pred_vec[:,0].reshape(-1,1), pred_vec[:,1].reshape(-1,1)
    #     # adjust for x condition
    #     x_adj = x0 + (1 - torch.exp(-t)) * dx_dt0 + ((1 - torch.exp(-t))**2) * x_pred
    #     # adjust for u condition
    #     u_adj = dx_dt0 + (1 - torch.exp(-t)) * u_pred
    #     # compute dx_dt = u
    #     dx_dt = compute_first_derivative(x_adj, t)
    #     # compute du_dt = d2x_dt2
    #     du_dt = compute_first_derivative(u_adj, t)
    #
    #     return x_adj, u_adj, dx_dt, du_dt

    def produce_SHO_preds_systemODE(G, t):
        x_pred = G(t)
        # x condition
        x_adj = x0 + (1 - torch.exp(-t)) * dx_dt0 + ((1 - torch.exp(-t))**2) * x_pred
        # dx_dt
        dx_dt = compute_first_derivative(x_pred, t)
        # u condition
        u_adj = torch.exp(-t) * dx_dt0 + 2 * (1 - torch.exp(-t)) * torch.exp(-t) * x_pred + (1 - torch.exp(-t)) * dx_dt
        # compute du_dt = d2x_dt2
        du_dt = compute_first_derivative(u_adj, t)

        return x_adj, u_adj, du_dt

    for epoch in range(num_epochs):

        ## =========
        ##  TRAIN G
        ## =========

        for p in D.parameters():
            p.requires_grad = False # turn off computation for D

        t = get_batch()
        if real_data:
            real = analytic_oscillator(t)

        for i in range(G_iters):
#         it_counter=0
#         while True:
#             it_counter+=1

            if systemOfODE:
                # x_adj, u_adj, dx_dt, du_dt = produce_SHO_preds_systemODE(G,t)
                x_adj, u_adj, d2x_dt2 = produce_SHO_preds_systemODE(G, t)
            else:
                x_adj, dx_dt, d2x_dt2 = produce_SHO_preds(G, t)

            if real_data:
                fake = x_adj
            else:
                if not systemOfODE:
                    real = x_adj
                    fake = -(m/k)*d2x_dt2
                else:
                    # # equation 1: x = -du/dt
                    # real1 = x_adj
                    # fake1 = -du_dt
                    # # equation 2: u = dx_dt
                    # real2 = u_adj
                    # fake2 = dx_dt
                    real = x_adj
                    fake = d2x_dt2

            # generator loss
            if not systemOfODE:
                g_loss = criterion(D(fake), real_label_vec)
                # g_loss = torch.mean(-D(fake))
            else:
                # g_loss1 = criterion(D1(fake1), real_label_vec)
                # g_loss2 = criterion(D2(fake2), real_label_vec)
                # g_loss = g_loss1 + g_loss2
                g_loss = criterion(D(fake), real_label_vec)

            optiG.zero_grad() # zero grad before backprop
            g_loss.backward(retain_graph=True)
            if wgan and not gradient_penalty:
                g_grad_norm = nn.utils.clip_grad_norm_(G.parameters(), clip)
            optiG.step()

#             if epoch < 1 or g_loss.item() < d_loss.item() or it_counter > max_while:
#                 break

        ## =========
        ##  TRAIN D
        ## =========

        for p in D.parameters():
            p.requires_grad = True # turn on computation for D

        for i in range(D_iters):

#         it_counter=0
#         while True:
#             it_counter+=1

            if soft_labels:
                real_label_vec_ = real_label_vec + (-.3 + .6 * torch.rand_like(real_label_vec))
                fake_label_vec_ = fake_label_vec + (-.3 + .6 * torch.rand_like(fake_label_vec))
            else:
                real_label_vec_ = real_label_vec
                fake_label_vec_ = fake_label_vec

            norm_penalty = torch.zeros(1)
            if wgan and gradient_penalty:
                total_norm = torch.zeros(1)

                if epoch > 0:
                    eps_mix = torch.rand(1)
                    x_mix = eps_mix * real + (1-eps_mix) * fake

                    mix_loss = torch.mean(D(x_mix))
                    # zero grad before computing the mix grad norm
                    optiD.zero_grad()
                    mix_loss.backward(retain_graph=True)

                    for p in D.parameters():
                        param = p.grad.data.norm(2).item()
                        total_norm += param ** 2
                    total_norm = total_norm ** (1. / 2)

                    norm_penalty = gp_hyper * torch.pow(total_norm - 1, 2)

            # discriminator loss
            if not systemOfODE:
                real_loss = criterion(D(real), real_label_vec_)
                fake_loss = criterion(D(fake), fake_label_vec_)
                d_loss = (real_loss + fake_loss)/2 + norm_penalty
                # d_loss = torch.mean(D(fake) - D(real) + norm_penalty)
                optiD.zero_grad()
                d_loss.backward(retain_graph=True)
                if wgan and not gradient_penalty:
                    d_grad_norm = nn.utils.clip_grad_norm_(D.parameters(), clip)
                optiD.step()

            else:
                # real_loss1 = criterion(D1(real1), real_label_vec_)
                # fake_loss1 = criterion(D1(fake1), fake_label_vec_)
                # real_loss2 = criterion(D2(real2), real_label_vec_)
                # fake_loss2 = criterion(D2(fake2), fake_label_vec_)
                #
                # d1_loss = (real_loss1 + fake_loss1)/2
                # d2_loss = (real_loss2 + fake_loss2)/2
                #
                # optiD1.zero_grad()
                # optiD2.zero_grad()
                #
                # d1_loss.backward(retain_graph=True)
                # d2_loss.backward(retain_graph=True)
                #
                # optiD1.step()
                # optiD2.step()

                real_loss = criterion(D(real), real_label_vec_)
                fake_loss = criterion(D(fake), fake_label_vec_)
                d_loss = (real_loss + fake_loss)/2 + norm_penalty
                optiD.zero_grad()
                d_loss.backward(retain_graph=True)
                optiD.step()

    #             if epoch < 1 or d_loss.item() < g_loss.item() or it_counter > max_while:
    #                 break

            ## ========
            ## Logging
            ## ========

        if logging:
            print('[%d/%d] D_Loss : %.4f Loss_G: %.4f' % (epoch, num_epochs, d_loss.item(), g_loss.item()))

        if not systemOfODE:
            D_losses.append(d_loss.item())
        else:
            # D_losses.append([d1_loss.item(), d2_loss.item()])
            D_losses.append(d_loss.item())

        G_losses.append(g_loss.item())
        # this_mse = np.mean((x_adj.detach().numpy() - analytic_oscillator_np(t_np))**2)
        # if this_mse < best_mse:
        #     best_mse = this_mse
        #     best_gen = deepcopy(G)
        #     best_disc = deepcopy(D)

        if (realtime_plot or epoch == num_epochs - 1):
            # either every time or on last epoch, show plots
            # if savefig is True, the figure will be saved (only on last epoch)
            clear_output(True)
            if not real_data:
                if not systemOfODE:
                    x_adj, dx_dt, d2x_dt2 = produce_SHO_preds(G, t_torch)
                    realtime_vis(G_losses, D_losses, t_np, x_adj.detach().numpy(), analytic_oscillator_np,
                                dx_dt.detach().numpy(), d2x_dt2.detach().numpy(), savefig=savefig, fname=fname)
                else:
                    # x_adj, u_adj, dx_dt, du_dt = produce_SHO_preds_systemODE(G, t_torch)
                    x_adj, u_adj, d2x_dt2 = produce_SHO_preds_systemODE(G, t)
                    # realtime_vis_system(G_losses, D_losses, t_np, x_adj.detach().numpy(), analytic_oscillator_np,
                                # u_adj.detach().numpy(), dx_dt.detach().numpy(), du_dt.detach().numpy(),
                                # savefig=savefig, fname=fname)
                    realtime_vis(G_losses, D_losses, t_np, x_adj.detach().numpy(), analytic_oscillator_np,
                                u_adj.detach().numpy(), d2x_dt2.detach().numpy(), savefig=savefig, fname=fname)
            else:
                loss_ax, pred_ax = plot_losses_and_preds(G_losses, D_losses, G, t_np, analytic_oscillator_np)
                plt.show()

    # if savefig:
    #     if not real_data:
    #         if not systemOfODE:
    #             x_adj, dx_dt, d2x_dt2 = produce_SHO_preds(G, t_torch)
    #             realtime_vis(G_losses, D_losses, t_np, x_adj.detach().numpy(), analytic_oscillator_np,
    #                         dx_dt.detach().numpy(), d2x_dt2.detach().numpy(), savefig=savefig, fname=fname)
    #         else:
    #             x_adj, u_adj, dx_dt, du_dt = produce_SHO_preds_systemODE(G, t_torch)
    #             realtime_vis(G_losses, D_losses, t_np, x_adj.detach().numpy(), analytic_oscillator_np,
    #                         dx_dt.detach().numpy(), du_dt.detach().numpy(), savefig=savefig, fname=fname)
    #
    #     else:
    #         loss_ax, pred_ax = plot_losses_and_preds(G_losses, D_losses, G, t_np, analytic_oscillator_np,
    #                                                 savefig=savefig, fname=fname)

    if not systemOfODE:
        return G, D, G_losses, D_losses
    else:
        return G, D1, D2, G_losses, D_losses

if __name__ == "__main__":
    raise NotImplementedError()
