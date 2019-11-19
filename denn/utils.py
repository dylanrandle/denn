import pandas as pd
import torch
import torch.nn as nn
from torch import tensor, autograd
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from IPython.display import clear_output
import itertools

# Global plot params
plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=12)
plt.rc('legend', fontsize=12)

def diff(x, t):
    """ wraps autograd to perform differentiation """
    dx_dt, = autograd.grad(x, t,
                           grad_outputs=x.data.new(x.shape).fill_(1),
                           create_graph=True)
    return dx_dt

class Swish(torch.nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, input):
        return input * torch.sigmoid(self.beta * input)

    def extra_repr(self):
        return 'beta={}'.format(self.beta)

class TorchSin(torch.nn.Module):
    """
    Sin activation function
    """
    def __init__(self):
        super(TorchSin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class ResidualBlock(nn.Module):
    # Most basic residual block
    # https://arxiv.org/pdf/1512.03385.pdf
    # ^ Equation #1

    def __init__(self, n_units, activation):
        super(ResidualBlock, self).__init__()

        self.activation = activation
        self.l1 = nn.Linear(n_units, n_units)
        self.l2 = nn.Linear(n_units, n_units)

    def forward(self, x):
        return self.activation(
            self.l2(self.activation(self.l1(x))) + x
        )

class Generator(nn.Module):
    """
    Generalized generator function for MLP
    """
    def __init__(self, in_dim=1, out_dim=1, n_hidden_units=20, n_hidden_layers=2,
        activation=nn.Tanh(), output_tan=True, residual=False):

        super(Generator, self).__init__()

        # input
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, n_hidden_units))
        self.layers.append(activation)

        # hidden
        for l in range(n_hidden_layers):
            if residual:
                self.layers.append(ResidualBlock(n_hidden_units, activation))
            else:
                self.layers.append(nn.Linear(n_hidden_units, n_hidden_units))
                self.layers.append(activation)

        # output
        self.layers.append(nn.Linear(n_hidden_units, out_dim))
        if output_tan:
            self.layers.append(nn.Tanh())

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class Discriminator(nn.Module):
    """ Generalized discriminator """
    def __init__(self, in_dim=1, out_dim=1, n_hidden_units=20, n_hidden_layers=2,
        activation=nn.Tanh(), unbounded=False, residual=False):

        super(Discriminator, self).__init__()

        # input
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, n_hidden_units))
        self.layers.append(activation)

        # hidden
        for l in range(n_hidden_layers):
            if residual:
                self.layers.append(ResidualBlock(n_hidden_units, activation))
            else:
                self.layers.append(nn.Linear(n_hidden_units, n_hidden_units))
                self.layers.append(activation)

        # output
        self.layers.append(nn.Linear(n_hidden_units, out_dim))
        if not unbounded:
            # unbounded for WGAN (no sigmoid)
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        # x = x.reshape(1,-1)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def calc_gradient_penalty(disc, real_data, generated_data, gp_lambda, cuda=False):
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
