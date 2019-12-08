import torch
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# global plot params
plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=12)
plt.rc('legend', fontsize=12)

def diff(x, t):
    """ wraps autograd to perform differentiation """
    dx_dt, = autograd.grad(x, t,
                           grad_outputs=x.data.new(x.shape).fill_(1),
                           create_graph=True)
    return dx_dt

def plot_results(loss_dict, grid, pred_dict, diff_dict=None, clear=False,
    save=False, fname=None, logloss=False, alpha=0.8):
    """ helpful plotting function """
    if clear:
      clear_output(True)

    if save and not fname:
        raise RuntimeError('Please provide a file name `fname` when `save=True`.')

    if pred_dict and diff_dict:   # plot losses, preds, and derivatives
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    else:                         # plot losses and preds only
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Losses
    for k, v in loss_dict.items():
        ax[0].plot(np.arange(len(v)), v, label=k, alpha=alpha)
    ax[0].legend(loc='upper right')
    ax[0].set_title('Losses')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    if logloss:
        ax[0].set_yscale('log')

    # Predictions
    for k, v in pred_dict.items():
        ax[1].plot(grid, v, label=k, alpha=alpha)
    ax[1].legend(loc='upper right')
    ax[1].set_title('Predictions')
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('$x$')

    # Derivatives
    if diff_dict:
        for k, v in diff_dict.items():
            ax[2].plot(grid, v, label=k, alpha=alpha)
        ax[2].legend(loc='upper right')
        ax[2].set_title('Derivatives')
        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel('$x$')

    plt.tight_layout()
    if save:
        print(f'Saving plot to {fname}')
        plt.savefig(fname)
    else:
        plt.show()

class LambdaLR():
    """ Simple linear decay schedule """
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

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
