import os
from matplotlib import lines
import torch
from torch import autograd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from IPython.display import clear_output
import pandas as pd 

# global plot params
plt.rc('axes', titlesize=18, labelsize=18)
plt.rc('legend', fontsize=16)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
#plt.rcParams['text.usetex'] = True

def diff(x, t, order=1):
    """The derivative of a variable with respect to another.
    :param x: The :math:`x` in :math:`\\displaystyle\\frac{\\partial x}{\\partial t}`.
    :type x: `torch.tensor`
    :param t: The :math:`t` in :math:`\\displaystyle\\frac{\\partial x}{\\partial t}`.
    :type t: `torch.tensor`
    :param order: The order of the derivative, defaults to 1.
    :type order: int
    :returns: The derivative.
    :rtype: `torch.tensor`
    """
    ones = torch.ones_like(x)
    der, = autograd.grad(x, t, create_graph=True, grad_outputs=ones)
    for i in range(1, order):
        ones = torch.ones_like(der)
        der, = autograd.grad(der, t, create_graph=True, grad_outputs=ones)
    return der

def plot_results(mse_dict, loss_dict, grid, pred_dict, diff_dict=None, clear=False,
    save=False, dirname=None, logloss=False, alpha=0.8, plot_sep_curves=False, 
    dims=None, plot_1d_curves=False):
    """ helpful plotting function """

    plt.rc('axes', titlesize=18, labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rcParams['text.usetex'] = True

    if clear:
      clear_output(True)

    if save and not dirname:
        raise RuntimeError('Please provide a directory name `dirname` when `save=True`.')

    if plot_sep_curves:
        n_curves = int(len(pred_dict.keys())/2)
        fig, ax = plt.subplots(2, n_curves+1, figsize=(4*(n_curves+1), 8))
    elif plot_1d_curves:
        fig, ax = plt.subplots(2, 4, figsize=(16, 8))
        ax = ax.ravel()
    else:
        if diff_dict:   # add derivatives plot
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        else:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']*3
    linewidth = 2
    alphas = [alpha]*10
    colors = ['crimson', 'blue', 'skyblue', 'limegreen',
        'aquamarine', 'violet', 'black', 'brown', 'pink', 'gold']

    # MSEs (Pred vs Actual)
    for i, (k, v) in enumerate(mse_dict.items()):
        if plot_sep_curves:
            ax[0][0].plot(np.arange(len(v)), v, label=k,
                alpha=alphas[i], linewidth=linewidth, color=colors[i],
                linestyle=linestyles[i])
        else:
            ax[0].plot(np.arange(len(v)), v, label=k,
                alpha=alphas[i], linewidth=linewidth, color=colors[i],
                linestyle=linestyles[i])

    if plot_sep_curves:
        if len(mse_dict.keys()) > 1: # only add legend if > 1 curves
            ax[0][0].legend(loc='upper right')
        ax[0][0].set_ylabel('Mean Squared Error')
        ax[0][0].set_xlabel('Iteration')
        ax[0][0].set_yscale('log')
    else:
        if len(mse_dict.keys()) > 1: # only add legend if > 1 curves
            ax[0].legend(loc='upper right')
        ax[0].set_ylabel('Mean Squared Error')
        ax[0].set_xlabel('Iteration')
        ax[0].set_yscale('log')

    # Losses
    for i, (k, v) in enumerate(loss_dict.items()):
        if plot_sep_curves:
            ax[1][0].plot(np.arange(len(v)), v, label=k,
                alpha=alphas[i], linewidth=linewidth, color=colors[i],
                linestyle=linestyles[i])
        else:
            ax[1].plot(np.arange(len(v)), v, label=k,
                alpha=alphas[i], linewidth=linewidth, color=colors[i],
                linestyle=linestyles[i])

    if plot_sep_curves:
        if len(loss_dict.keys()) > 1: # only add legend if > 1 curves
            ax[1][0].legend(loc='upper right')
        ax[1][0].set_xlabel('Iteration')
        ax[1][0].set_ylabel('Loss')
        if logloss:
            ax[1][0].set_yscale('log')
    else:
        if len(loss_dict.keys()) > 1: # only add legend if > 1 curves
            ax[1].legend(loc='upper right')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Loss')
        if logloss:
            ax[1].set_yscale('log')

    # Predictions
    if grid.shape[1] == 2: # PDE
        x, y = grid[:, 0], grid[:, 1]
        xdim, ydim = dims.values()
        xx, yy = x.reshape((xdim, ydim)), y.reshape((xdim, ydim))
        for i, (k, v) in enumerate(pred_dict.items()):
            v = v.reshape((xdim, ydim))
            cf = ax[2].contourf(xx, yy, v, cmap='Reds')
            cb = fig.colorbar(cf, format='%.0e', ax=ax[2])
            break
        xlab, ylab = dims.keys()
        ax[2].set_xlabel(f'${xlab}$')
        ax[2].set_ylabel(f'${ylab}$')
    else: # ODE
        if plot_sep_curves:
            for i, (k, v) in enumerate(pred_dict.items()):
                if i%2 == 0:
                    plot_id = int((i/2)+1)
                    style_id = 0
                else:
                    style_id = 1
                ax[0][plot_id].plot(grid, v, label=k, 
                alpha=alphas[style_id], linestyle=linestyles[style_id], 
                linewidth=linewidth, color=colors[style_id])
                ax[0][plot_id].set_xlabel('$t$')
                ax[0][plot_id].set_ylabel(k)
                ax[0][plot_id].legend()
        else:
            for i, (k, v) in enumerate(pred_dict.items()):
                ax[2].plot(grid, v, label=k,
                    alpha=alphas[i], linestyle=linestyles[i],
                    linewidth=linewidth, color=colors[i])
            ax[2].set_xlabel('$t$')
            ax[2].set_ylabel('$x$')
            if len(pred_dict.keys()) > 1:
                ax[2].legend(loc='upper right')

    # 1-dimensional curves for PDE
    if plot_1d_curves:
        xvals = grid[:, 0].reshape((xdim, ydim))
        tvals = grid[:, 1].reshape((xdim, ydim))
        t_ids = [0, int(ydim/3), int(2*ydim/3), ydim-1]
        for i in range(4, 8):
            for j, (k, v) in enumerate(pred_dict.items()):
                v = v.reshape((xdim, ydim))
                ax[i].plot(xvals[:, 0], v[:, t_ids[i-4]], label=k, 
                        alpha=alphas[j], linestyle=linestyles[j], 
                        linewidth=linewidth, color=colors[j])
            ax[i].set_xlabel(f'${xlab}$')
            ax[i].set_ylabel('$u$')
            t = float(tvals[0, :][t_ids][i-4])
            ax[i].set_title(f'$t={t:.3f}$')
            if len(pred_dict.keys()) > 1:
                ax[i].legend()

    # Derivatives
    if diff_dict:
        if grid.shape[1] == 2: # PDE
            x, y = grid[:, 0], grid[:, 1]
            xdim, ydim = dims.values()
            xx, yy = x.reshape((xdim, ydim)), y.reshape((xdim, ydim))
            for i, (k, v) in enumerate(diff_dict.items()):
                v = v.reshape((xdim, ydim))
                cf = ax[3].contourf(xx, yy, v, cmap='Reds')
                cb = fig.colorbar(cf, format='%.0e', ax=ax[3])
            ax[3].set_xlabel(f'${xlab}$')
            ax[3].set_ylabel(f'${ylab}$')
        else:
            if plot_sep_curves:
                for i, (k, v) in enumerate(diff_dict.items()):
                    plot_id = i+1
                    ax[1][plot_id].plot(grid, v, label=k, 
                    alpha=alphas[i], linestyle=linestyles[0], 
                    linewidth=linewidth, color=colors[0])
                    ax[1][plot_id].legend(loc='upper right')
                    ax[1][plot_id].set_xlabel('$t$')
                    ax[1][plot_id].set_ylabel('$F$')
                    ax[1][plot_id].set_yscale('log')
            else:
                for i, (k, v) in enumerate(diff_dict.items()):
                    ax[3].plot(grid, v, label=k,
                        alpha=alphas[i], linestyle=linestyles[i],
                        linewidth=linewidth, color=colors[i])
                ax[3].legend(loc='upper right')
                ax[3].set_xlabel('$t$')
                ax[3].set_ylabel('$F$')
                ax[3].set_yscale('log')
    plt.tight_layout()

    if save:
        print(f'Saving results to {dirname}')
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        plt.savefig(os.path.join(dirname, 'plot.png'), dpi=300)
        np.save(os.path.join(dirname, "grid"), grid)
        for k, v in mse_dict.items():
            np.save(os.path.join(dirname, f"{k}_mse"), v)
        for k, v in loss_dict.items():
            np.save(os.path.join(dirname, f"{k}_loss"), v)
        # commented out because saving pred_dict and diff_dict currently throw errors
        #for k, v in pred_dict.items():
        #    np.save(os.path.join(dirname, f"{k}_pred"), v)
        #if diff_dict:
        #    for k, v in diff_dict.items():
        #        np.save(os.path.join(dirname, f"{k}_diff"), v)
    else:
        plt.show()

def plot_multihead(mse_dict, loss_dict, resids_dict, save=False, dirname=None, alpha=0.8):

    plt.rc('axes', titlesize=18, labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rcParams['text.usetex'] = True

    if save and not dirname:
        raise RuntimeError('Please provide a directory name `dirname` when `save=True`.')

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']*3
    linewidth = 2
    alphas = [alpha]*10
    colors = ['crimson', 'blue', 'skyblue', 'limegreen',
        'aquamarine', 'violet', 'black', 'brown', 'pink', 'gold']

    # MSEs (Pred vs Actual)
    for i, (k, v) in enumerate(mse_dict.items()):
        ax[0].plot(np.arange(len(v)), v, label=k,
            alpha=alphas[i], linewidth=linewidth, color=colors[i],
            linestyle=linestyles[i])

    if len(mse_dict.keys()) > 1:
        ax[0].legend(loc='upper right')
    ax[0].set_ylabel('Mean Squared Error')
    ax[0].set_xlabel('Iteration')
    ax[0].set_yscale('log')

    # GAN Losses
    for i, (k, v) in enumerate(loss_dict.items()):
        if k != 'LHS':
            ax[1].plot(np.arange(len(v)), v, label=k,
            alpha=alphas[i], linewidth=linewidth, color=colors[i],
            linestyle=linestyles[i])
    if len(loss_dict.keys()) > 1: 
        ax[1].legend(loc='upper right')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Loss')

    # L2 Residuals
    resid_vectors = resids_dict['resid']
    resid_l2s = [np.square(r_vec).mean() for r_vec in resid_vectors]
    ax[2].plot(np.arange(len(resid_l2s)), resid_l2s, alpha=alphas[0], 
        linewidth=linewidth, color=colors[0], linestyle=linestyles[0])
    ax[2].set_ylabel('Residuals ($L_2$ norm)')
    ax[2].set_xlabel('Iteration')
    ax[2].set_yscale('log')

    plt.tight_layout()

    if save:
        print(f'Saving results to {dirname}')
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        plt.savefig(os.path.join(dirname, 'plot_multihead.png'), dpi=300)
        for k, v in mse_dict.items():
            np.save(os.path.join(dirname, f"{k}_mse"), v)
    else:
        plt.show()

def plot_3D(grid, pred_dict, view=[35, -55], dims=None, save=False, dirname=None):
    """ 3D plotting function for PDEs """

    plt.rc('axes', titlesize=18, labelsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rcParams['text.usetex'] = True

    fig = plt.figure(figsize=(14,9))
    ax = fig.add_subplot(projection='3d')
    grid = grid.numpy()
    x, y = grid[:, 0], grid[:, 1]
    xdim, ydim = dims.values()
    xx, yy = x.reshape((xdim, ydim)), y.reshape((xdim, ydim))
    for v in pred_dict.values():
        v = v.numpy().reshape((xdim, ydim))
        break
    xlab, ylab = dims.keys()
    ax.set_xlabel(f'${xlab}$')
    ax.set_ylabel(f'${ylab}$')
    ax.set_zlabel('$u$')
    ax.plot_surface(xx, yy, v, cmap=cm.coolwarm, rcount=500, ccount=500, alpha=0.8)
    ax.view_init(elev=view[0], azim=view[1])
    if save:
        plt.savefig(os.path.join(dirname, 'plot3D.png'), dpi=300)
    else:
        plt.show()

def plot_reps_results(arrs_dict,
    linewidth=2, alpha_line=0.8, alpha_shade=0.4, figsize=(12,8),
    pctiles = (2.5, 97.5), window=10, fname=None):

    plt.rc('axes', titlesize=24, labelsize=24)
    plt.rc('legend', fontsize=20)
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    plt.rcParams['text.usetex'] = True

    linestyles = ['solid', 'solid', 'solid', 'solid']
    colors = ['crimson', 'blue', 'skyblue', 'limegreen',
        'aquamarine', 'violet', 'black', 'brown', 'pink', 'gold']

    plt.figure(figsize=figsize)
    plt.yscale('log')

    arrs = list(arrs_dict.values())

    steps = np.arange(arrs[0].shape[1])

    for i, (k, a) in enumerate(arrs_dict.items()):
        a = pd.DataFrame(data=a).rolling(window, axis=1).mean().values
        plt.plot(steps, np.median(a, axis=0), label=k,
                 color=colors[i], linestyle=linestyles[i], linewidth=linewidth, alpha=alpha_line)
        lqt, upt = np.percentile(a, pctiles, axis=0)
        plt.fill_between(steps, lqt, upt, alpha=alpha_shade, color=colors[i])

    plt.legend(loc='lower left')
    # plt.legend(loc='upper right')
    # plt.xticks([0, 5000, 10000, 15000, 20000])
    plt.xlabel('Iteration')
    plt.ylabel('Mean squared error')

    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def handle_overwrite(fname):
    """ helper to handle case where we might overwrite """
    if os.path.exists(fname):
        owrite = check_overwrite(fname)
        if not owrite:
            print('Quitting to prevent overwriting.')
            exit(0)

def check_overwrite(fname):
    """ helper function to get user input for overwriting """
    print(f'File found at {fname} and save=True.')
    resp = input('Overwrite (y/n)? ').strip().lower()
    if resp == 'y':
        return True
    else:
        return False

class LambdaLR():
    """ Simple linear decay schedule """
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        # max(0,_) ensures never < 0
        # min(0.9999,_) ensures never > 0.9999
        # ==> 1e-4 < 1 - min(1,max(0,_)) < 1
        return 1.0 - min(0.9999, max(0., epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch))

def calc_gradient_penalty(disc, real_data, generated_data, gp_lambda, cuda=False):
    """ helper method for gradient penalty (WGAN-GP) """
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_data)
    if cuda:
      alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = autograd.Variable(interpolated, requires_grad=True)
    if cuda:
      interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = disc(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                          grad_outputs=torch.ones(prob_interpolated.size()).cuda() if cuda else torch.ones(
                                prob_interpolated.size()),
                          create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    # self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_lambda * ((gradients_norm - 1) ** 2).mean()

def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def exponential_weight_average(prev_weights, curr_weights, beta=0.999):
    """ returns exponential moving average of prev_weights and curr_weights
        (beta=0 => no averaging) """
    return beta*prev_weights + (1-beta)*curr_weights

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.

    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality

    :example2:
        # from pde_nn.utils import draw_neural_net
        # fig = plt.figure(figsize=(12, 12))
        # ax = fig.gca()
        # ax.axis('off')
        # draw_neural_net(ax, .1, .9, .1, .9, [1, 20, 20, 1])
        # fig.savefig('nn.png')
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)

def shake_weights(m, std=1):
    '''
    Adds normal noise to the weights of a model m
    '''
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn(p.size()) * std)

def plot_grads(params, ax, logscale=False):
    '''
    Plots the gradients in the layers of a network given
    its named parameters and a matplotlib axis object.
    '''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in params:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    ax[0].plot(ave_grads, alpha=0.3, color="b")
    ax[0].hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    ax[0].set_xticks(range(0,len(ave_grads), 1))
    ax[0].set_xticklabels(layers, rotation="vertical")
    ax[0].set_xlim(xmin=0, xmax=len(ave_grads))
    if logscale:
        ax[0].set_yscale("log")
    ax[0].set_xlabel("Layers")
    ax[0].set_ylabel("Gradient")
    ax[0].grid(True)
    ax[1].bar(np.arange(0.5, len(max_grads)+0.5), max_grads, alpha=0.1, lw=1, color="c")
    ax[1].bar(np.arange(0.5, len(max_grads)+0.5), ave_grads, alpha=0.1, lw=1, color="b")
    ax[1].hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    ax[1].set_xticks(np.arange(0.5, len(ave_grads)+0.5))
    ax[1].set_xticklabels(layers, rotation="vertical")
    ax[1].set_xlim(left=0, right=len(ave_grads))
    ax[1].set_ylim(bottom = -0.001, top=0.02)
    ax[1].set_xlabel("Layers")
    ax[1].grid(True)
    ax[1].legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'], loc='upper center')   