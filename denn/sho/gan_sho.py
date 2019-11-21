"""
Implementation of GAN for Unsupervised deep learning of differential equations

Equation: Simple Harmonic Oscillator (SHO)
x'' + x = 0
"""
import torch
import torch.nn as nn
from torch import tensor, autograd
import numpy as np
from copy import deepcopy
from denn.utils import Generator, Discriminator, diff, LambdaLR, calc_gradient_penalty, exponential_weight_average
from denn.sho.sho_utils import produce_SHO_preds, produce_SHO_preds_system, plot_SHO

def train_GAN_SHO_unsupervised(G, D, num_epochs=10000, eq=False, eq_k=0, eq_lr=0.001,
    d_lr=0.001, g_lr=0.001, d_betas=(0.0, 0.9), g_betas=(0.0, 0.9), G_iters=1,
    D_iters=1, t_low=0, t_high=4*np.pi, x0=0, dx_dt0=.5, n=100, m=1., k=1.,
    real_label=1, fake_label=0, perturb=True, wgan=True, gp=0.1, d1=1., d2=1.,
    system_of_ODE=True, lr_schedule=True, decay_start_epoch=0,
    savefig=False, fname='GAN_SHO.png', check_every=1000, logging=False, realtime_plot=False,
    final_plot=False, seed=0):
    """
    function to perform training of generator and discriminator for num_epochs
    equation: simple harmonic oscillator (SHO)
    """
    torch.manual_seed(seed) # reproducibility
    cuda = torch.cuda.is_available()

    if wgan:
        fake_label = -1

    # grid
    t_torch = torch.linspace(t_low, t_high, n, dtype=torch.float, requires_grad=True).reshape(-1,1)
    zeros = torch.zeros_like(t_torch)
    # analytical solution
    omega = np.sqrt(k/m)
    analytic_oscillator = lambda t: x0 * torch.cos(omega * t) + (dx_dt0/omega) * torch.sin(omega * t)
    solution = analytic_oscillator(t_torch)

    # inter-point spacing
    delta_t = t_torch[1,0]-t_torch[0,0]
    # batch getter
    def get_batch(perturb=False):
        """ perturb grid """
        if perturb:
            return t_torch + delta_t * torch.randn_like(t_torch) / 3
        else:
            return t_torch

    # labels
    real_label_vec = torch.full((n,), real_label).reshape(-1,1)
    fake_label_vec = torch.full((n,), fake_label).reshape(-1,1)

    # optimization objective
    if wgan:
        criterion = lambda y_true, y_pred: torch.mean(y_true * y_pred)
    else:
        criterion = nn.BCELoss()

    mse_loss = nn.MSELoss()

    # optimizers
    optiG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)
    optiD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)

    # lr schedulers
    start_epoch = 0
    if lr_schedule:
      lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optiG, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_start_epoch).step)
      lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optiD, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_start_epoch).step)

    # logging
    D_losses = []
    G_losses = []

    # WGAN - GP penalty placeholder
    null_norm_penalty = torch.zeros(1)
    _pred_fn = produce_SHO_preds_system if system_of_ODE else produce_SHO_preds

    if cuda:
        G.cuda()
        D.cuda()
        t_torch = t_torch.cuda()
        null_norm_penalty = null_norm_penalty.cuda()
        analytic = analytic.cuda()
        delta_t = delta_t.cuda()
        real_label_vec = real_label_vec.cuda()
        fake_label_vec = fake_label_vec.cuda()

    for epoch in range(num_epochs):

        ## =========
        ##  TRAIN G
        ## =========

        for p in D.parameters():
            p.requires_grad = False # turn off computation for D

        for i in range(G_iters):
            # unsupervised: G try to fool D
            t = get_batch(perturb=perturb)
            x_adj, dx_dt, d2x_dt2 = _pred_fn(G, t, x0=x0, dx_dt0=dx_dt0)
            eq_out = d2x_dt2 + x_adj
            fake = torch.cat((eq_out, t), 1)
            real = torch.cat((zeros, t), 1)
            g_loss = criterion(D(fake), real_label_vec)

            optiG.zero_grad() # zero grad before backprop
            g_loss.backward(retain_graph=True)
            optiG.step()

        ## =========
        ##  TRAIN D
        ## =========

        for p in D.parameters():
            p.requires_grad = True # turn on computation for D

        for i in range(D_iters):

            if wgan:
                norm_penalty = calc_gradient_penalty(D, real, fake, gp, cuda=cuda)
            else:
                norm_penalty = null_norm_penalty

            # D1: discriminating between real and fake1
            real_loss = criterion(D(real), real_label_vec)
            fake_loss = criterion(D(fake), fake_label_vec)
            if eq:
                d_loss = real_loss + eq_k * fake_loss + norm_penalty
            else:
                d_loss = real_loss + fake_loss + norm_penalty

            # zero gradients
            optiD.zero_grad()
            d_loss.backward(retain_graph=True)
            optiD.step()

            if epoch > 0 and eq:
                gamma = g_loss.item() / d_loss.item()
                eq_k += eq_lr * (gamma * real_loss.item() - g_loss.item())

        # Update learning rates
        if lr_schedule:
          lr_scheduler_G.step()
          lr_scheduler_D.step()

        ## ========
        ## Logging
        ## ========

        D_losses.append(d_loss.item())
        G_losses.append(g_loss.item())

        if realtime_plot and (epoch % check_every) == 0:
            plot_SHO(G_losses, D_losses, t_torch, solution, G, _pred_fn, savefig=False, clear=True)

    if final_plot:
        plot_SHO(G_losses, D_losses, t_torch, solution, G, _pred_fn, savefig=savefig, fname=fname, clear=False)

    x_adj, dx_dt, d2x_dt2 = _pred_fn(G, t_torch, x0=x0, dx_dt0=dx_dt0)
    final_mse = mse_loss(x_adj, solution).item()
    print(f'Final MSE: {final_mse}')
    return {'final_mse': final_mse, 'model': G}

def train_GAN_SHO_semisupervised(G, D, D2, num_epochs=10000, eq=False, eq_k=0,
    eq_lr=0.001, d_lr=0.001, g_lr=0.001, d_betas=(0.0, 0.9), g_betas=(0.0, 0.9),
    G_iters=1, D_iters=1, t_low=0, t_high=4*np.pi, x0=0, dx_dt0=.5, n=100, m=1., k=1.,
    real_label=1, fake_label=0, perturb=True, observe_every=1, wgan=True, gp=0.1,
    d1=1., d2=1., system_of_ODE=True, lr_schedule=True, decay_start_epoch=0,
    savefig=False, fname='GAN_SHO.png', check_every=1000, logging=False,
    realtime_plot=False, final_plot=False, seed=0):
    """
    function to perform training of generator and discriminator for num_epochs
    equation: simple harmonic oscillator (SHO)
    """
    torch.manual_seed(seed) # reproducibility
    cuda = torch.cuda.is_available()

    # grid
    t_torch = torch.linspace(t_low, t_high, n, dtype=torch.float, requires_grad=True).reshape(-1,1)
    zeros = torch.zeros_like(t_torch)

    # analytical solution
    omega = np.sqrt(k/m)
    analytic_oscillator = lambda t: x0 * torch.cos(omega * t) + (dx_dt0/omega) * torch.sin(omega * t)
    solution = analytic_oscillator(t_torch)

    # this generates an index mask for our "observers"
    observers = torch.arange(0, n, observe_every)
    t_observers = t_torch[observers, :]
    y_observers = solution[observers, :]

    # conditional GAN
    real_observers = torch.cat((y_observers, t_observers), 1)

    delta_t = t_torch[1,0]-t_torch[0,0]
    def get_batch(perturb=False):
        """ perturb grid """
        if perturb:
            return t_torch + delta_t * torch.randn_like(t_torch) / 3
        else:
            return t_torch

    # labels
    if wgan:
        fake_label = -1
    real_label_vec = torch.full((n,), real_label).reshape(-1,1)
    fake_label_vec = torch.full((n,), fake_label).reshape(-1,1)

    # optimization objectives
    if wgan:
        criterion = lambda y_true, y_pred: torch.mean(y_true * y_pred)
    else:
        criterion = nn.BCELoss()
    mse_loss = nn.MSELoss()

    # optimizers
    optiG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)
    optiD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)
    optiD2 = torch.optim.Adam(D2.parameters(), lr=d_lr, betas=d_betas)

    # lr schedulers
    start_epoch = 0
    if lr_schedule:
      lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optiG, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_start_epoch).step)
      lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optiD, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_start_epoch).step)
      lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(optiD2, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_start_epoch).step)

    # logging
    D_losses = []
    G_losses = []
    D2_losses = []

    # WGAN - GP penalty placeholder
    null_norm_penalty = torch.zeros(1)
    _pred_fn = produce_SHO_preds_system if system_of_ODE else produce_SHO_preds

    if cuda:
        G.cuda()
        D.cuda()
        t_torch = t_torch.cuda()
        null_norm_penalty = null_norm_penalty.cuda()
        analytic = analytic.cuda()
        delta_t = delta_t.cuda()
        real_label_vec = real_label_vec.cuda()
        fake_label_vec = fake_label_vec.cuda()

    for epoch in range(num_epochs):

        ## =========
        ##  TRAIN G
        ## =========

        for p in D.parameters():
            p.requires_grad = False # turn off computation for D

        for p in D2.parameters():
            p.requires_grad = False # turn off computation for D2

        for i in range(G_iters):
            # unsupervised
            t = get_batch(perturb=perturb)
            real = torch.cat((zeros, t), 1)

            x_adj, dx_dt, d2x_dt2 = _pred_fn(G, t, x0=x0, dx_dt0=dx_dt0)
            eq_out = d2x_dt2 + x_adj
            fake = torch.cat((eq_out, t), 1)

            g_loss1 = criterion(D(fake), real_label_vec)

            # supervised
            x_adj_obs, _, _ = _pred_fn(G, t_observers, x0=x0, dx_dt0=dx_dt0)
            fake_observers = torch.cat((x_adj_obs, t_observers), 1)
            g_loss2 = criterion(D2(fake_observers), real_label_vec[observers, :])

            # combined
            g_loss = d1 * g_loss1 + d2 * g_loss2
            optiG.zero_grad() # zero grad before backprop
            g_loss.backward(retain_graph=True)
            optiG.step()

        ## =========
        ##  TRAIN D
        ## =========

        for p in D.parameters():
            p.requires_grad = True # turn on computation for D

        for i in range(D_iters):

            if wgan:
                norm_penalty = calc_gradient_penalty(D, real, fake, gp, cuda=cuda)
            else:
                norm_penalty = null_norm_penalty

            # D1: unsupervised
            real_loss = criterion(D(real), real_label_vec)
            fake_loss = criterion(D(fake), fake_label_vec)
            # if eq:
                # d_loss = real_loss + eq_k * fake_loss + norm_penalty
            # else:
            d_loss = real_loss + fake_loss + norm_penalty

            # zero gradients and step
            optiD.zero_grad()
            d_loss.backward(retain_graph=True)
            optiD.step()

            # if epoch > 0 and eq:
            #     gamma = g_loss.item() / d_loss.item()
            #     eq_k += eq_lr * (gamma * real_loss.item() - g_loss.item())

        ## =========
        ##  TRAIN D2
        ## =========

        for p in D2.parameters():
            p.requires_grad = True

        for i in range(D_iters):

            if wgan:
                norm_penalty = calc_gradient_penalty(D2, real_observers, fake_observers, gp, cuda=cuda)
            else:
                norm_penalty = null_norm_penalty

            # D1: unsupervised
            real_loss = criterion(D2(real_observers), real_label_vec[observers, :])
            fake_loss = criterion(D2(fake_observers), fake_label_vec[observers, :])
            # if eq:
            #     d_loss2 = real_loss + eq_k * fake_loss + norm_penalty
            # else:
            d_loss2 = real_loss + fake_loss + norm_penalty

            # zero gradients and step
            optiD2.zero_grad()
            d_loss2.backward(retain_graph=True)
            optiD2.step()

            # if epoch > 0 and eq:
            #     gamma = g_loss.item() / d_loss2.item()
            #     eq_k += eq_lr * (gamma * real_loss.item() - g_loss.item())

        # Update learning rates
        if lr_schedule:
            lr_scheduler_G.step()
            lr_scheduler_D.step()
            lr_scheduler_D2.step()

        D_losses.append(d_loss.item())
        G_losses.append(g_loss.item())
        D2_losses.append(d_loss2.item())

        if realtime_plot and (epoch % check_every) == 0:
            plot_SHO(G_losses, D_losses, t_torch, solution, G, _pred_fn, D2_losses=D2_losses, savefig=False, clear=True)

    if final_plot:
        plot_SHO(G_losses, D_losses, t_torch, solution, G, _pred_fn, D2_losses=D2_losses, savefig=savefig, fname=fname, clear=False)

    x_adj, dx_dt, d2x_dt2 = _pred_fn(G, t_torch, x0=x0, dx_dt0=dx_dt0)
    final_mse = mse_loss(x_adj, solution).item()
    print(f'Final MSE: {final_mse}')
    return {'final_mse': final_mse, 'model': G}

if __name__ == '__main__':
    G = Generator(in_dim=1, out_dim=1,
                  n_hidden_units=64,
                  n_hidden_layers=8,
                  activation=nn.Tanh(), # twice diff'able activation
                  output_tan=True,      # true output range should be (-1,1) if True
                  residual=True)

    D = Discriminator(in_dim=2, out_dim=1,
                      n_hidden_units=32,
                      n_hidden_layers=8,
                      activation=nn.Tanh(),
                      unbounded=True, # true for WGAN
                      residual=True)

    D2 = Discriminator(in_dim=2, out_dim=1,
                      n_hidden_units=32,
                      n_hidden_layers=8,
                      activation=nn.Tanh(),
                      unbounded=True, # true for WGAN
                      residual=True)

    res = train_GAN_SHO_unsupervised(G, D, d_lr=2e-4, final_plot=True, num_epochs=1000)
    res = train_GAN_SHO_semisupervised(G, D, D2, d_lr=2e-4, final_plot=True, num_epochs=1000, observe_every=10)
