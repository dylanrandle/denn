import torch
import torch.nn as nn
import os

from denn.utils import LambdaLR, plot_results, calc_gradient_penalty, handle_overwrite

def train_GAN(G, D, problem, method='unsupervised', niters=100,
    g_lr=2e-4, g_betas=(0.0, 0.9), d_lr=1e-3, d_betas=(0.0, 0.9),
    lr_schedule=True, obs_every=1, d1=1., d2=1.,
    G_iters=1, D_iters=1, wgan=True, gp=0.1, conditional=True,
    plot=True, save=False, fname='train_GAN.png'):
    """
    Train/test GAN method: supervised/semisupervised/unsupervised
    """
    assert method in ['supervised', 'semisupervised', 'unsupervised'], f'Method {method} not understood!'

    if save:
        handle_overwrite(fname)

    t = problem.get_grid()
    y = problem.get_solution(t)
    observers = torch.arange(0, len(t), obs_every)
    t_obs = t[observers, :]
    y_obs = y[observers, :]

    # labels
    real_label = 1
    fake_label = -1 if wgan else 0
    real_labels = torch.full((len(t),), real_label).reshape(-1,1)
    fake_labels = torch.full((len(t),), fake_label).reshape(-1,1)

    # optimization
    optiG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)
    optiD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    wass = lambda y_true, y_pred: torch.mean(y_true * y_pred)
    criterion = wass if wgan else bce

    if lr_schedule:
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optiG, lr_lambda=LambdaLR(niters, 0, 0).step)
        lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optiD, lr_lambda=LambdaLR(niters, 0, 0).step)

    null_norm_penalty = torch.zeros(1)
    losses = {'G': [], 'D': []}

    for epoch in range(niters):
        # Train Generator
        for p in D.parameters():
            p.requires_grad = False # turn off computation for D

        for i in range(G_iters):
            if method == 'unsupervised':
                t_samp = problem.get_grid_sample()
                xhat = G(t_samp)
                residuals = problem.get_equation(xhat, t_samp)

                if conditional:
                    # concat "real" (all zeros) with t (conditional GAN)
                    # concat "fake" (residuals) with t (conditional GAN)
                    real = torch.cat((torch.zeros_like(t_samp), t_samp), 1)
                    fake = torch.cat((residuals, t_samp), 1)
                else:
                    real = torch.zeros_like(t_samp)
                    fake = residuals

                g_loss = criterion(D(fake), real_labels)
                optiG.zero_grad()
                g_loss.backward(retain_graph=True)
                optiG.step()

            elif method == 'semisupervised':
                raise NotImplementedError()

            else: # supervised
                xhat = G(t)
                xadj = problem.adjust(xhat, t)[0]

                if conditional:
                    # concat "real" (y) with t (conditional GAN)
                    # concat "fake" (xadj) with t (conditional GAN)
                    real = torch.cat((y, t), 1)
                    fake = torch.cat((xadj, t), 1)
                else:
                    real = y
                    fake = xadj

                g_loss = criterion(D(fake), real_labels)
                optiG.zero_grad()
                g_loss.backward(retain_graph=True)
                optiG.step()

        # Train Discriminator
        for p in D.parameters():
            p.requires_grad = True # turn on computation for D

        for i in range(D_iters):
            norm_penalty = calc_gradient_penalty(D, real, fake, gp, cuda=False) if wgan else null_norm_penalty
            real_loss = criterion(D(real), real_labels)
            fake_loss = criterion(D(fake), fake_labels)
            d_loss = real_loss + fake_loss + norm_penalty
            optiD.zero_grad()
            d_loss.backward(retain_graph=True)
            optiD.step()

        if lr_schedule:
          lr_scheduler_G.step()
          lr_scheduler_D.step()

        losses['D'].append(d_loss.item())
        losses['G'].append(g_loss.item())

        # if realtime_plot and (epoch % check_every) == 0:
            # plot_NLO_GAN(G_losses, D_losses, t_torch, y_num, G, _pred_fn, savefig=False, clear=True)

    if plot:
        loss_dict = {}
        loss_dict['$D$'] = losses['D']
        loss_dict['$G$'] = losses['G']
        pred_dict, diff_dict = problem.get_plot_dicts(G(t), t, y)
        plot_results(loss_dict, t.detach(), pred_dict, diff_dict=diff_dict,
            save=save, fname=fname, logloss=False, alpha=0.8)

    xhat = G(t)
    xadj = problem.adjust(xhat, t)[0]
    final_mse = mse(xadj, y).item()
    print(f'Final MSE {final_mse}')
    return {'final_mse': final_mse, 'model': G}

def train_L2(model, problem, method='unsupervised', niters=100,
    lr=2e-4, betas=(0, 0.9), lr_schedule=True, obs_every=1, d1=1, d2=1,
    plot=True, save=False, fname='train_L2.png'):
    """
    Train/test Lagaris method: supervised/semisupervised/unsupervised
    """
    assert method in ['supervised', 'semisupervised', 'unsupervised'], f'Method {method} not understood!'

    if save:
        handle_overwrite(fname)

    t = problem.get_grid()
    y = problem.get_solution(t)
    observers = torch.arange(0, len(t), obs_every)
    t_obs = t[observers, :]
    y_obs = y[observers, :]

    # optimizers & loss functions
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    mse = torch.nn.MSELoss()
    # lr scheduler
    if lr_schedule:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=LambdaLR(niters, 0, 0).step)

    loss_trace = []
    for i in range(niters):
        if method == 'unsupervised':
            t_samp = problem.get_grid_sample()
            xhat = model(t_samp)
            residuals = problem.get_equation(xhat, t_samp)
            loss = mse(residuals, torch.zeros_like(residuals))
            loss_trace.append(loss.item())

        elif method == 'semisupervised':
            # supervised part
            xhat = model(t_obs)
            xadj = problem.adjust(xhat, t_obs)[0]
            loss1 = mse(xadj, y_obs)

            # unsupervised part
            t_samp = problem.get_grid_sample()
            xhat = model(t_samp)
            residuals = problem.get_equation(xhat, t_samp)
            loss2 = mse(residuals, torch.zeros_like(residuals))

            # combine together
            loss = d1 * loss1 + d2 * loss2
            loss_trace.append((loss1.item(), loss2.item()))

        else: # supervised
            xhat = model(t_obs)
            xadj = problem.adjust(xhat, t_obs)[0]
            loss = mse(xadj, y_obs)
            loss_trace.append(loss.item())

        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()
        if lr_schedule:
            lr_scheduler.step()

    if plot:
        loss_dict = {}
        if method == 'supervised':
            loss_dict['$L_S$'] = loss_trace
        elif method == 'semisupervised':
            loss_dict['$L_S$'] = [l[0] for l in loss_trace]
            loss_dict['$L_U$'] = [l[1] for l in loss_trace]
        else:
            loss_dict['$L_U$'] = loss_trace

        pred_dict, diff_dict = problem.get_plot_dicts(model(t), t, y)
        plot_results(loss_dict, t.detach(), pred_dict, diff_dict=diff_dict,
            save=save, fname=fname, logloss=True, alpha=0.8)

    xhat = model(t)
    xadj = problem.adjust(xhat, t)[0]
    final_mse = mse(xadj, y).item()
    print(f'Final MSE {final_mse}')
    return {'final_mse': final_mse, 'model': model}
