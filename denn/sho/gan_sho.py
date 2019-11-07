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

def train_GAN_SHO(
    # Architecture
    num_epochs=100000,
    activation=nn.Tanh(),
    g_units=32,
    g_layers=4,
    d_units=32,
    d_layers=8,
    # D2 params
    # d2_units=16,
    # d2_layers=8,
    # start_d2=0,
    # equilibrium params
    eq=False,
    eq_k=0,
    eq_lr=0.001,
    # optimizer
    d_lr=0.001,
    g_lr=0.0002,
    d_betas=(0.0, 0.9),
    g_betas=(0.0, 0.9),
    G_iters=1,
    D_iters=1,
    # Problem
    t_low=0,
    t_high=4*np.pi,
    x0=0,
    dx_dt0=.5,
    n=100,
    m=1.,
    k=1.,
    real_label=1,
    fake_label=0,
    # Hacks
    perturb=True,
    residual=True,
    observe_every=1,
    wgan=True,
    gp=0.1,
    d1=1e-3,
    d2=1.,
    output_tan=True,
    system_of_ODE=True,
    conditional_GAN=True,
    lr_schedule=True,
    decay_start_epoch=0,
    weight_average=False,
    weight_average_start=50000,
    # Inspect
    savefig=False,
    fname='GAN_SHO.png',
    device=None,
    check_every=1000,
    logging=False,
    realtime_plot=False,
    final_plot=False,
    seed=0,
):

    """
    function to perform training of generator and discriminator for num_epochs
    equation: simple harmonic oscillator (SHO)
    """
    torch.manual_seed(seed) # reproducibility
    cuda = torch.cuda.is_available()

    if wgan:
        fake_label = -1

    # initialize nets
    G = Generator(in_dim=1, out_dim=1,
                  n_hidden_units=g_units,
                  n_hidden_layers=g_layers,
                  activation=activation, # twice diff'able activation
                  output_tan=output_tan, # true output range should be (-1,1) if True
                  residual=residual)
    if cuda:
      G.cuda()

    d_in_dim = 2 if conditional_GAN else 1
    D = Discriminator(in_dim=d_in_dim, out_dim=1,
                      n_hidden_units=d_units,
                      n_hidden_layers=d_layers,
                      activation=activation,
                      unbounded=wgan, # true for WGAN
                      residual=residual) # now using residual, before # no residual in D's

    # D2 = Discriminator(in_dim=d_in_dim, out_dim=1,
    #                   n_hidden_units=d2_units,
    #                   n_hidden_layers=d2_layers,
    #                   activation=activation,
    #                   unbounded=wgan, # true for WGAN
    #                   residual=residual) # now using residual, before # no residual in D's
    if cuda:
      D.cuda()
      # D2.cuda()

    # grid
    t_torch = torch.linspace(t_low, t_high, n, dtype=torch.float, requires_grad=True).reshape(-1,1)
    if cuda:
      t_torch = t_torch.cuda()
    t_np = t_torch.cpu().detach().numpy()

    # inter-point spacing
    delta_t = t_torch[1,0]-t_torch[0,0]
    if cuda:
      delta_t = delta_t.cuda()

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
    if cuda:
      real_label_vec = real_label_vec.cuda()
      fake_label_vec = fake_label_vec.cuda()

    # optimization objective
    if wgan:
        criterion = lambda y_true, y_pred: torch.mean(y_true * y_pred)
    else:
        criterion = nn.BCELoss()

    mse_loss = nn.MSELoss()

    # optimizers
    optiG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)
    optiD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)
    # optiD2 = torch.optim.Adam(D2.parameters(), lr=d_lr, betas=d_betas)

    # lr schedulers
    start_epoch = 0
    if lr_schedule:
      lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optiG, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_start_epoch).step)
      lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optiD, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_start_epoch).step)
      # lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(optiD2, lr_lambda=LambdaLR(num_epochs, start_epoch, decay_start_epoch).step)

    # logging
    D_losses = []
    G_losses = []

    # analytical solution
    omega = np.sqrt(k/m)
    analytic_oscillator = lambda t: x0 * torch.cos(omega * t) + (dx_dt0/omega) * torch.sin(omega * t)
    solution = analytic_oscillator(t_torch)

    # generate an index mask for our "observers"
    observers = torch.arange(0, n, observe_every)
    t_observers = t_torch[observers, :]
    analytic = analytic_oscillator(t_observers)
    if cuda:
        analytic.cuda()
    if conditional_GAN:
        analytic = torch.cat((analytic, t_observers), 1)

    # WGAN - GP penalty placeholder
    null_norm_penalty = torch.zeros(1)
    if cuda:
      null_norm_penalty = null_norm_penalty.cuda()

    # placeholder for average params
    # ema_params = [0.01, 0.1, 0.5, 0.8, 0.9, 0.99, .999, .9999]
    # average_params = [np.array(list(G.parameters())) for _ in range(len(ema_params))]

    _pred_fn = produce_SHO_preds_system if system_of_ODE else produce_SHO_preds

    for epoch in range(num_epochs):

        ## =========
        ##  TRAIN G
        ## =========

        for p in D.parameters():
            p.requires_grad = False # turn off computation for D

        # for p in D2.parameters():
        #     p.requires_grad = False

        for i in range(G_iters):

            t = get_batch(perturb=perturb)

            x_adj, dx_dt, d2x_dt2 = _pred_fn(G, t, x0=x0, dx_dt0=dx_dt0)
            g_loss2 = mse_loss(x_adj, -(m/k) * d2x_dt2)

            # FOR CONDITIONAL GAN:
            # input t with real/fake (to discriminator)
            if conditional_GAN:
              x_adj = torch.cat((x_adj, t), 1)
              d2x_dt2 = torch.cat((d2x_dt2, t), 1)

            real = analytic
            fake1 = x_adj
            fake2 = -(m/k) * d2x_dt2

            # # save previous params
            # prev_params = G.parameters()

            optiG.zero_grad() # zero grad before backprop

            # first loss is used in the typical GAN sense where "real" is actually the x pred from G
            # second loss is used in our GAN sense
            # the generator wants to fool D both with X and X''
            g_loss1 = criterion(D(fake1), real_label_vec)
            # g_loss2 = criterion(D2(fake2), real_label_vec)
            if epoch >= start_d2:
                g_loss = d1 * g_loss1 + d2 * g_loss2
            else:
                g_loss = d1 * g_loss1
            g_loss.backward(retain_graph=True)
            optiG.step()

            # store the exponential moving average of parameters in prev_params
            # see here: https://discuss.pytorch.org/t/set-model-weights-to-preset-tensor-with-torch/35051/2
            if weight_average:
                curr_params = G.parameters()
                # first step, just set estimate to current params
                if epoch == weight_average_start:
                    avg_params = []
                    for p in curr_params:
                        avg_params.append(p.data)
                    average_params = [deepcopy(avg_params) for _ in range(len(ema_params))]
                # future steps, compute exponential moving average
                elif epoch > weight_average_start:
                    # loop over ema params
                    for e in range(len(ema_params)):
                        for i, (a, c) in enumerate(zip(average_params[e], curr_params)):
                            new_ema = exponential_weight_average(a, c.data, beta=ema_params[e])
                            average_params[e][i].data = new_ema
                else:
                    pass

        ## =========
        ##  TRAIN D
        ## =========

        for p in D.parameters():
            p.requires_grad = True # turn on computation for D

        # for p in D2.parameters():
        #     p.requires_grad = True

        for i in range(D_iters):

            if wgan:
                norm_penalty = calc_gradient_penalty(D, real, fake1[observers, :], gp, cuda=cuda)
                # norm_penalty2 = calc_gradient_penalty(D2, fake1, fake2, gp, cuda=cuda)
            else:
                norm_penalty = null_norm_penalty
                # norm_penalty2 = null_norm_penalty

            # zero gradients
            optiD.zero_grad()
            # optiD2.zero_grad()

            # D1: discriminating between real and fake1
            real_loss = criterion(D(real), real_label_vec[observers, :])
            fake_loss = criterion(D(fake1), fake_label_vec)
            if eq:
                d_loss1 = real_loss + eq_k * fake_loss + norm_penalty
            else:
                d_loss1 = real_loss + fake_loss + norm_penalty
            d_loss1.backward(retain_graph=True)
            optiD.step()
            if epoch > 0 and eq:
                gamma = g_loss1.item() / d_loss1.item()
                eq_k += eq_lr * (gamma * real_loss.item() - g_loss1.item())

            # # D2: discriminating between pred and X''
            # real_loss = criterion(D2(fake1), real_label_vec)
            # fake_loss = criterion(D2(fake2), fake_label_vec)
            # if eq:
            #     d_loss2 = real_loss + eq_k * fake_loss + norm_penalty2
            # else:
            #     d_loss2 = real_loss + fake_loss + norm_penalty2
            # if epoch > start_d2:
            #     d_loss2.backward(retain_graph=True)
            #     optiD2.step()
            # if epoch > 0 and eq:
            #     gamma = g_loss2.item() / d_loss2.item()
            #     eq_k += eq_lr * (gamma * real_loss.item() - g_loss2.item())

        # Update learning rates
        if lr_schedule:
          lr_scheduler_G.step()
          lr_scheduler_D.step()
          # lr_scheduler_D2.step()

        ## ========
        ## Logging
        ## ========

        D_losses.append((d_loss1.item(), 0))
        G_losses.append((g_loss1.item(), g_loss2.item()))

        if realtime_plot and (epoch % check_every) == 0:
            plot_SHO(G_losses, D_losses, t_torch, solution, G, _pred_fn, savefig=False, clear=True)

    if weight_average:
        for e in range(len(ema_params)):
            print('Setting to EMA params for beta={}'.format(ema_params[e]))
            # set G's params to the average params
            for p, a in zip(G.parameters(), average_params[e]):
                p.data = a.data

    if final_plot:
        plot_SHO(G_losses, D_losses, t_torch, solution, G, _pred_fn, savefig=savefig, fname=fname, clear=False)

    x_adj, dx_dt, d2x_dt2 = _pred_fn(G, t_torch, x0=x0, dx_dt0=dx_dt0)
    analytic = analytic_oscillator(t_torch)
    final_mse = mse_loss(x_adj, analytic).item()
    return {'final_mse': final_mse, 'model': G}

if __name__ == '__main__':
    res = train_GAN_SHO(num_epochs=1000)
