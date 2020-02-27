import torch
import torch.nn as nn
import os

from ray.tune import track

from denn.utils import LambdaLR, plot_results, calc_gradient_penalty, handle_overwrite
from denn.config.config import write_config

this_dir = os.path.dirname(os.path.abspath(__file__))

def train_GAN(G, D, problem, method='unsupervised', niters=100,
    g_lr=1e-3, g_betas=(0.0, 0.9), d_lr=1e-3, d_betas=(0.0, 0.9),
    lr_schedule=True, gamma=0.999, obs_every=1, d1=1., d2=1.,
    G_iters=1, D_iters=1, wgan=True, gp=0.1, conditional=True,
    log=True, plot=True, save=False, dirname='train_GAN',
    config=None, **kwargs):
    """
    Train/test GAN method: supervised/semisupervised/unsupervised
    """
    assert method in ['supervised', 'semisupervised', 'unsupervised'], f'Method {method} not understood!'

    dirname = os.path.join(this_dir, '../experiments/runs', dirname)
    if plot and save:
        handle_overwrite(dirname)

    # training: full grid/solution (t/y)
    grid = problem.get_grid()
    soln = problem.get_solution(grid)

    # # validation
    # val_grid = problem.get_grid_sample()
    # val_soln = problem.get_solution(val_grid)

    # observer mask and masked grid/solution (t_obs/y_obs)
    observers = torch.arange(0, len(grid), obs_every)
    grid_obs = grid[observers, :]
    soln_obs = soln[observers, :]

    # labels
    real_label = 1
    fake_label = -1 if wgan else 0
    real_labels = torch.full((len(grid),), real_label).reshape(-1,1)
    fake_labels = torch.full((len(grid),), fake_label).reshape(-1,1)
    # masked label vectors
    real_labels_obs = real_labels[observers, :]
    fake_labels_obs = fake_labels[observers, :]

    # optimization
    optiG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)
    optiD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)
    if lr_schedule:
        lr_scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer=optiG, gamma=gamma)
        lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer=optiD, gamma=gamma)

    # losses
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    wass = lambda y_true, y_pred: torch.mean(y_true * y_pred)
    criterion = wass if wgan else bce

    # history
    losses = {'G': [], 'D': []}
    mses = {'train': []}

    for epoch in range(niters):
        # Train Generator
        for p in D.parameters():
            p.requires_grad = False # turn off computation for D

        for i in range(G_iters):
            if method == 'unsupervised':
                grid_samp = problem.get_grid_sample()
                pred = G(grid_samp)
                residuals = problem.get_equation(pred, grid_samp)

                real = torch.zeros_like(residuals)
                fake = residuals

                if conditional:
                    real = torch.cat((real, grid_samp), 1)
                    fake = torch.cat((fake, grid_samp), 1)

                optiG.zero_grad()
                g_loss = criterion(D(fake), real_labels)
                g_loss.backward(retain_graph=True)
                optiG.step()

            elif method == 'semisupervised':
                # unsupervised part (use GAN)
                grid_samp = problem.get_grid_sample()
                pred = G(grid_samp)
                residuals = problem.get_equation(pred, grid_samp)

                real = torch.zeros_like(residuals)
                fake = residuals

                if conditional:
                    real = torch.cat((real, grid_samp), 1)
                    fake = torch.cat((fake, grid_samp), 1)

                g_loss1 = criterion(D(fake), real_labels)

                # supervised part (use L2)
                pred = G(grid_obs)
                pred_adj = problem.adjust(pred, grid_obs)[0]
                g_loss2 = mse(pred_adj, soln_obs)

                # combine losses
                g_loss = d1 * g_loss1 + d2 * g_loss2
                optiG.zero_grad()
                g_loss.backward(retain_graph=True)
                optiG.step()

            else: # supervised
                # @note: Why removed for now?
                # the discriminator below uses real_labels/fake_labels
                # and NOT real_labels_obs/fake_labels_obs such that it is
                # compatible both with unsupervised and semi-supervised (in
                # the case where the GAN is used for the unsupervised portion)
                raise NotImplementedError()

                # xhat = G(t_obs)
                # xadj = problem.adjust(xhat, t_obs)[0]
                # residuals = y_obs - xadj
                #
                # if conditional:
                #     # concat "real" (y) with t (conditional GAN)
                #     # concat "fake" (xadj) with t (conditional GAN)
                #     real = torch.cat((torch.zeros_like(residuals), t_obs), 1)
                #     fake = torch.cat((residuals, t_obs), 1)
                # else:
                #     real = torch.zeros_like(residuals)
                #     fake = residuals
                #
                # g_loss = criterion(D(fake), real_labels_obs)
                # optiG.zero_grad()
                # g_loss.backward(retain_graph=True)
                # optiG.step()

        # Train Discriminator
        for p in D.parameters():
            p.requires_grad = True # turn on computation for D

        for i in range(D_iters):
            if wgan:
                norm_penalty = calc_gradient_penalty(D, real, fake, gp, cuda=False)
            else:
                norm_penalty = torch.zeros(1)

            real_loss = criterion(D(real), real_labels)
            fake_loss = criterion(D(fake), fake_labels)

            optiD.zero_grad()
            d_loss = (real_loss + fake_loss)/2 + norm_penalty
            d_loss.backward(retain_graph=True)
            optiD.step()

        losses['D'].append(d_loss.item())
        losses['G'].append(g_loss.item())

        if lr_schedule:
          lr_scheduler_G.step()
          lr_scheduler_D.step()

        # track current MSE on ground truth (train)
        train_pred = G(grid)
        train_pred_adj = problem.adjust(pred, grid)[0]
        train_mse = mse(train_pred_adj, soln).item()
        mses['train'].append(train_mse)
        # try:
        #     track.log(mean_squared_error=train_mse)
        # except:
        #     pass

        # # track current MSE on ground truth (val)
        # val_pred = G(val_grid)
        # val_pred_adj = problem.adjust(val_pred, val_grid)[0]
        # val_mse = mse(val_pred_adj, val_soln).item()
        # mses['val'].append(val_mse)

        if log:
            print(f'Step {epoch} Train MSE {train_mse}')

    if plot:
        pred_dict, diff_dict = problem.get_plot_dicts(G(grid), grid, soln)
        plot_results(mses, losses, grid.detach(), pred_dict, diff_dict=diff_dict,
            save=save, dirname=dirname, logloss=False, alpha=0.7)

    if save:
        write_config(config, os.path.join(dirname, 'config.yaml'))

    return {'mses': mses, 'model': G, 'losses': losses}

def train_L2(model, problem, method='unsupervised', niters=100,
    lr=1e-3, betas=(0, 0.9), lr_schedule=True, gamma=0.999,
    obs_every=1, d1=1, d2=1, log=True, plot=True, save=False,
    dirname='train_L2', config=None, loss_fn=None, **kwargs):
    """
    Train/test Lagaris method: supervised/semisupervised/unsupervised
    """
    assert method in ['supervised', 'semisupervised', 'unsupervised'], f'Method {method} not understood!'

    dirname = os.path.join(this_dir, '../experiments/runs', dirname)
    if plot and save:
        handle_overwrite(dirname)

    # training: full grid/solution (t/y)
    grid = problem.get_grid()
    sol = problem.get_solution(grid)

    # # validation
    # val_grid = problem.get_grid_sample()
    # val_sol = problem.get_solution(val_grid)

    observers = torch.arange(0, len(grid), obs_every)
    grid_obs = grid[observers, :]
    sol_obs = sol[observers, :]

    # optimizers & loss functions
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    if loss_fn:
        mse = eval(f"torch.nn.{loss_fn}()")
    else:
        mse = torch.nn.MSELoss()
    # lr scheduler
    if lr_schedule:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=gamma)

    loss_trace = []
    mses = {'train': []}

    for i in range(niters):
        if method == 'unsupervised':
            grid_samp = problem.get_grid_sample()
            pred = model(grid_samp)
            residuals = problem.get_equation(pred, grid_samp)
            loss = mse(residuals, torch.zeros_like(residuals))
            loss_trace.append(loss.item())

        elif method == 'semisupervised':
            # supervised part
            pred = model(grid_obs)
            pred_adj = problem.adjust(pred, grid_obs)[0]
            loss1 = mse(pred_adj, sol_obs)

            # unsupervised part
            grid_samp = problem.get_grid_sample()
            pred = model(grid_samp)
            residuals = problem.get_equation(pred, grid_samp)
            loss2 = mse(residuals, torch.zeros_like(residuals))

            # combine together
            loss = d1 * loss1 + d2 * loss2
            loss_trace.append((loss1.item(), loss2.item()))

        else: # supervised
            pred = model(grid_obs)
            pred_adj = problem.adjust(pred, grid_obs)[0]
            loss = mse(pred_adj, sol_obs)
            loss_trace.append(loss.item())

        # track current MSE on ground truth (train)
        train_pred = model(grid)
        train_pred_adj = problem.adjust(pred, grid)[0]
        train_mse = mse(train_pred_adj, sol).item()
        mses['train'].append(train_mse)

        # # track current MSE on ground truth (val)
        # val_pred = model(val_grid)
        # val_pred_adj = problem.adjust(val_pred, val_grid)[0]
        # val_mse = mse(val_pred_adj, val_sol).item()
        # mses['val'].append(val_mse)

        if log:
            print(f'Step {i} Train MSE {train_mse}')

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

        save_to = os.path.join(this_dir, '../experiments/runs', dirname)

        pred_dict, diff_dict = problem.get_plot_dicts(model(grid), grid, sol)
        plot_results(mses, loss_dict, grid.detach(), pred_dict, diff_dict=diff_dict,
            save=save, dirname=dirname, logloss=True, alpha=0.7)

    if save:
        write_config(config, os.path.join(dirname, 'config.yaml'))

    return {'mses': mses, 'model': model, 'losses': loss_trace}
