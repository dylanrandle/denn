"""
Implementation of GAN for Unsupervised deep learning of differential equations

Equation: Simple Harmonic Oscillator (SHO)
d2x/dt2 =  -x

Analytic Solution:
x = sin(t)
"""
import torch
import torch.nn as nn
from torch import tensor, autograd
import numpy as np
from copy import deepcopy
from denn.utils import Generator, Discriminator, diff, LambdaLR, calc_gradient_penalty, exponential_weight_average
from denn.sho_utils import produce_SHO_preds, produce_SHO_preds_system, plot_SHO

def train_GAN_SHO(
    # Architecture
    num_epochs=100000,
    activation=nn.Tanh(),
    g_hidden_units=30,
    d_hidden_units=30,
    g_hidden_layers=5,
    d_hidden_layers=3,
    d_lr=0.001,
    g_lr=0.001,
    d_betas=(0.9, 0.999),
    g_betas=(0.9, 0.999),
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
    real_data=False,
    residual=False,
    semi_supervised=False,
    observe_every=1,
    wgan=True,
    gp_hyper=1.0,
    d2_hyper=1.0,
    outputTan=True,
    systemOfODE=True,
    conditionalGAN=True,
    lr_schedule=True,
    decay_start_epoch=5000,
    weight_average=False,
    weight_average_start=50000,
    # Inspect
    savefig=False,
    fname=None,
    device=None,
    check_every=100,
    logging=False,
    realtime_plot=True,
    seed=42,
):

    """
    function to perform training of generator and discriminator for num_epochs
    equation: simple harmonic oscillator (SHO)
    gan hacks:
        - wasserstein + clipping / wasserstein GP
        - label smoothing
        - while loop iters
    """
    torch.manual_seed(seed) # reproducibility

    cuda = torch.cuda.is_available()

    if savefig and realtime_plot:
        raise Exception('savefig and realtime_plot both True. Assuming you dont want that.')

    if wgan:
        fake_label = -1

    # initialize nets
    G = Generator(in_dim=1, out_dim=1,
                  n_hidden_units=g_hidden_units,
                  n_hidden_layers=g_hidden_layers,
                  activation=activation, # twice diff'able activation
                  output_tan=outputTan, # true output range should be (-1,1) if True
                  residual=residual)
    if cuda:
      G.cuda()

    d_in_dim = 2 if conditionalGAN else 1
    D = Discriminator(in_dim=d_in_dim, out_dim=1,
                      n_hidden_units=d_hidden_units,
                      n_hidden_layers=d_hidden_layers,
                      activation=activation,
                      unbounded=wgan, # true for WGAN
                      residual=False) # no residual in D's

    # make D2 an exact (deep) copy of D
    D2 = deepcopy(D)

    if cuda:
      D.cuda()
      D2.cuda()

    # grid
    t_torch = torch.linspace(t_low, t_high, n, dtype=torch.float, requires_grad=True).reshape(-1,1)
    if cuda:
      t_torch = t_torch.cuda()
    t_np = t_torch.cpu().detach().numpy() # np.linspace(t_low, t_high, n).reshape(-1,1)

    # inter-point spacing
    delta_t = t_torch[1]-t_torch[0]
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

    # analytical solution
    omega = np.sqrt(k/m)
    analytic_oscillator = lambda t: x0 * torch.cos(omega * t) + (dx_dt0/omega) * torch.sin(omega * t)
    analytic_oscillator_np = lambda t: x0 * np.cos(omega * t) + (dx_dt0/omega) * np.sin(omega * t)

    # if semi_supervised:
    # this generates an index mask for our "observers"
    observers = torch.arange(0, n, observe_every)

    # WGAN - GP penalty placeholder
    null_norm_penalty = torch.zeros(1)
    if cuda:
      null_norm_penalty = null_norm_penalty.cuda()

    # placeholder for average params
    ema_params = [0.01, 0.1, 0.5, 0.8, 0.9, 0.99, .999, .9999]
    # average_params = [np.array(list(G.parameters())) for _ in range(len(ema_params))]

    def produce_SHO_preds(G, t):
        x_raw = G(t)

        # adjust for initial conditions on x and dx_dt
        x_adj = x0 + (1 - torch.exp(-t)) * dx_dt0 + ((1 - torch.exp(-t))**2) * x_raw

        dx_dt = diff(x_adj, t)

        d2x_dt2 = diff(dx_dt, t)

        return x_adj, dx_dt, d2x_dt2

    def produce_SHO_preds_system(G, t):
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

    # def basic_SHO_pred(G, t):
    #     pred = G(t)
    #     dxdt = diff(pred, t)
    #     d2xdt = diff(dxdt, t)
    #     return pred, dxdt, d2xdt

    _pred_fn = produce_SHO_preds_system if systemOfODE else produce_SHO_preds
    # _pred_fn = basic_SHO_pred

    for epoch in range(num_epochs):

        ## =========
        ##  TRAIN G
        ## =========

        for i in range(G_iters):

            t = get_batch(perturb=perturb)

            # analytic = analytic_oscillator(t)
            # if cuda:
            #   analytic.cuda()

            x_adj, dx_dt, d2x_dt2 = _pred_fn(G, t)

            # FOR CONDITIONAL GAN:
            # input t with real/fake (to discriminator)
            if conditionalGAN:
              x_adj = torch.cat((x_adj, t), 1)
              d2x_dt2 = torch.cat((d2x_dt2, t), 1)
              # analytic = torch.cat((analytic, t), 1)

            # calculate loss for generator
            # ss_loss = 0.

            # if real_data:
            #   real = analytic
            #   fake = x_adj
            # else:
            #   real = analytic
            fake = x_adj
            fake2 = -(m/k) * d2x_dt2

            # # save previous params
            # prev_params = G.parameters()

            optiG.zero_grad() # zero grad before backprop

            # first loss is used in the typical GAN sense where "real" is actually the x pred from G
            # second loss is used in our GAN sense
            # the generator wants to fool D both with X and X''
            g_loss1 = criterion(D(fake), real_label_vec)
            g_loss2 = criterion(D2(fake2), real_label_vec)
            g_loss = g_loss1 + d2_hyper * g_loss2
            # Below: trying MSE loss
            # g_loss = mse_loss(fake, real) + d2_hyper * criterion(D2(fake2), real_label_vec)
            g_loss.backward(retain_graph=True)
            optiG.step()

            # # store the exponential moving average of parameters in prev_params
            # # see here: https://discuss.pytorch.org/t/set-model-weights-to-preset-tensor-with-torch/35051/2
            # if weight_average:
            #     curr_params = G.parameters()
            #     # first step, just set estimate to current params
            #     if epoch == weight_average_start:
            #         avg_params = []
            #         for p in curr_params:
            #             avg_params.append(p.data)
            #         average_params = [deepcopy(avg_params) for _ in range(len(ema_params))]
            #     # future steps, compute exponential moving average
            #     elif epoch > weight_average_start:
            #         # loop over ema params
            #         for e in range(len(ema_params)):
            #             for i, (a, c) in enumerate(zip(average_params[e], curr_params)):
            #                 new_ema = exponential_weight_average(a, c.data, beta=ema_params[e])
            #                 average_params[e][i].data = new_ema
            #     else:
            #         pass

        ## =========
        ##  TRAIN D
        ## =========

        for i in range(D_iters):

            # Real Data
            # t_real = t_torch[observers, :]
            analytic = analytic_oscillator(t_torch) # deterministic location
            analytic = torch.cat((analytic, t_torch), 1)
            if cuda:
              analytic.cuda()

            # Limited
            real = analytic[observers, :]

            if wgan:
                norm_penalty = calc_gradient_penalty(D, real, fake[observers, :], gp_hyper, cuda=cuda)
                norm_penalty2 = calc_gradient_penalty(D2, fake, fake2, gp_hyper, cuda=cuda) # can be D/D2 with real/fake2 or fake/fake2
            else:
                norm_penalty = null_norm_penalty
                norm_penalty2 = null_norm_penalty

            # zero gradients
            optiD.zero_grad()
            optiD2.zero_grad()

            # D1: discriminating between true and pred
            real_loss = criterion(D(real), real_label_vec[observers, :])
            fake_loss = criterion(D(fake), fake_label_vec)
            d_loss1 = real_loss + fake_loss + norm_penalty
            d_loss1.backward(retain_graph=True)
            optiD.step()

            # D2: discriminating between pred and X''
            real_loss = criterion(D2(fake), real_label_vec) # can use real or fake here.
            fake_loss = criterion(D2(fake2), fake_label_vec)
            d_loss2 = real_loss + fake_loss + norm_penalty2
            d_loss2.backward(retain_graph=True)
            optiD2.step()

        # Update learning rates
        if lr_schedule:
          lr_scheduler_G.step()
          lr_scheduler_D.step()
          lr_scheduler_D2.step()

        ## ========
        ## Logging
        ## ========

        D_losses.append((d_loss1.item(), d_loss2.item()))
        G_losses.append((g_loss1.item(), g_loss2.item()))

        if realtime_plot and (epoch % check_every) == 0:
            # either every time or on last epoch, show plots
            # if savefig is True, the figure will be saved
            # (only on last epoch), because we make sure both are not true
            plot_SHO(G_losses, D_losses, t, analytic, G, _pred_fn, savefig=savefig, fname=fname, clear=True)

    if weight_average:
        for e in range(len(ema_params)):
            print('Setting to EMA params for beta={}'.format(ema_params[e]))
            # set G's params to the average params
            for p, a in zip(G.parameters(), average_params[e]):
                p.data = a.data
            # plot final result
            plot_SHO(G_losses, D_losses, t, analytic, G, _pred_fn, savefig=savefig, fname=fname, clear=False)

    return {'G': G, 'D': D, 'G_loss': G_losses, 'D_loss': D_losses, 't': t_torch, 'analytic': analytic_oscillator}

if __name__ == '__main__':
    res = train_GAN_SHO(
        # Architecture
        num_epochs=100000,
        activation=nn.Tanh(),
        g_hidden_units=30,
        d_hidden_units=30,
        g_hidden_layers=7,
        d_hidden_layers=3,
        d_lr=0.0002, # change optimizer params
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
        real_data=False,
        residual=True,
        wgan=True,
        gp_hyper=1.,
        d2_hyper=1.,
        outputTan=True,
        systemOfODE=True,
        conditionalGAN=True,
        lr_schedule=True,
        decay_start_epoch=5000,
        weight_average=False,
        weight_average_start=5000,
        # Inspect
        savefig=False,
        fname=None,
        device=None,
        check_every=500,
        logging=False,
        realtime_plot=True)
