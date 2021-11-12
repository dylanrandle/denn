import os
import torch
from torch import autograd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  
from IPython.display import clear_output
import pandas as pd

# global plot params
plt.rc('axes', titlesize=15, labelsize=15)
plt.rc('legend', fontsize=15)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)

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
    save=False, dirname=None, logloss=False, alpha=0.8):
    """ helpful plotting function """

    plt.rc('axes', titlesize=15, labelsize=15)
    plt.rc('legend', fontsize=15)
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)

    if clear:
      clear_output(True)

    if save and not dirname:
        raise RuntimeError('Please provide a directory name `dirname` when `save=True`.')

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
        ax[0].plot(np.arange(len(v)), v, label=k,
            alpha=alphas[i], linewidth=linewidth, color=colors[i],
            linestyle=linestyles[i])

    if len(mse_dict.keys()) > 1: # only add legend if > 1 curves
        ax[0].legend(loc='upper right')
    ax[0].set_ylabel('Mean Squared Error')
    ax[0].set_xlabel('Step')
    ax[0].set_yscale('log')

    # Losses
    for i, (k, v) in enumerate(loss_dict.items()):
        ax[1].plot(np.arange(len(v)), v, label=k,
            alpha=alphas[i], linewidth=linewidth, color=colors[i],
            linestyle=linestyles[i])

    if len(loss_dict.keys()) > 1: # only add legend if > 1 curves
        ax[1].legend(loc='upper right')
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('Loss')
    if logloss:
        ax[1].set_yscale('log')

    # Predictions
    if grid.shape[1] == 2: # PDE
        x, y = grid[:, 0], grid[:, 1]
        xdim, ydim = int(np.sqrt(len(x))), int(np.sqrt(len(y)))
        xx, yy = x.reshape((xdim, ydim)), y.reshape((xdim, ydim))
        for i, (k, v) in enumerate(pred_dict.items()):
            v = v.reshape((xdim, ydim))
            cf = ax[2].contourf(xx, yy, v, cmap='Reds')
            cb = fig.colorbar(cf, format='%.0e', ax=ax[2])
        ax[2].set_xlabel('$x$')
        ax[2].set_ylabel('$y$')
    else: # ODE
        for i, (k, v) in enumerate(pred_dict.items()):
            ax[2].plot(grid, v, label=k,
                alpha=alphas[i], linestyle=linestyles[i],
                linewidth=linewidth, color=colors[i])
        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel('$x$')
    if len(pred_dict.keys()) > 1:
        ax[2].legend(loc='upper right')

    # Derivatives
    if diff_dict:
        if grid.shape[1] == 2: # PDE
            x, y = grid[:, 0], grid[:, 1]
            xdim, ydim = int(np.sqrt(len(x))), int(np.sqrt(len(y)))
            xx, yy = x.reshape((xdim, ydim)), y.reshape((xdim, ydim))
            for i, (k, v) in enumerate(diff_dict.items()):
                v = v.reshape((xdim, ydim))
                cf = ax[3].contourf(xx, yy, v, cmap='Reds')
                cb = fig.colorbar(cf, format='%.0e', ax=ax[3])
            ax[3].set_xlabel('$x$')
            ax[3].set_ylabel('$y$')
        else:
            for i, (k, v) in enumerate(diff_dict.items()):
                ax[3].plot(grid, v, label=k,
                    alpha=alphas[i], linestyle=linestyles[i],
                    linewidth=linewidth, color=colors[i])
            ax[3].legend(loc='upper right')
            ax[3].set_xlabel('$t$')
            # ax[3].set_ylabel('$x$')
            ax[3].set_ylabel('$F$')
            ax[3].set_yscale('log')

    plt.tight_layout()
    if save:
        print(f'Saving results to {dirname}')
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        plt.savefig(os.path.join(dirname, 'plot.png'))
        np.save(os.path.join(dirname, "grid"), grid)
        for k, v in mse_dict.items():
            np.save(os.path.join(dirname, f"{k}_mse"), v)
        for k, v in loss_dict.items():
            np.save(os.path.join(dirname, f"{k}_loss"), v)
        # Blake comment: commented out because saving pred_dict and diff_dict currently throw errors
        #for k, v in pred_dict.items():
        #    np.save(os.path.join(dirname, f"{k}_pred"), v)
        #if diff_dict:
        #    for k, v in diff_dict.items():
        #        np.save(os.path.join(dirname, f"{k}_diff"), v)
    else:
        plt.show()

def plot_reps_results(arrs_dict,
    linewidth=2, alpha_line=0.8, alpha_shade=0.4, figsize=(12,8),
    pctiles = (2.5, 97.5), window=10, fname=None):

    plt.rc('axes', titlesize=20, labelsize=20)
    plt.rc('legend', fontsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']*3
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
    plt.grid()

    if fname:
        plt.savefig(fname)
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