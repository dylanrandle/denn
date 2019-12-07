import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from denn.models import MLP
from denn.utils import LambdaLR, plot_results
from denn.problems import SimpleOscillator

def train_MSE(model, problem, method='unsupervised', seed=0, niters=10000,
    lr=0.001, betas=(0, 0.9), lr_schedule=True, obs_every=1, d1=1, d2=1,
    plot=True, save=False, fname='train_MSE_plot.png'):
    """
    Train/test Lagaris method: supervised/semisupervised/unsupervised
    """
    assert method in ['supervised', 'semisupervised', 'unsupervised']

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
        if method == 'supervised':
            xhat = model(t_obs)
            xadj = problem.adjust(xhat, t_obs)[0]
            loss = mse(xadj, y_obs)
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

        else: # unsupervised
            t_samp = problem.get_grid_sample()
            xhat = model(t_samp)
            residuals = problem.get_equation(xhat, t_samp)
            loss = mse(residuals, torch.zeros_like(residuals))
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
