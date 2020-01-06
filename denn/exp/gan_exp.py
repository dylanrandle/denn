"""
Implementation of GAN for Unsupervised deep learning of differential equations

Equation:
x' + Lx = 0

Analytic Solution:
x = exp(-Lt)
"""
import torch
import torch.nn as nn
from torch import tensor, autograd
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('legend', fontsize=12)    # fontsize of the legend

class Generator(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, n_hidden_units=20, n_hidden_layers=2, activation=nn.Tanh(), x0=1,
                output_tan=True):
        super(Generator, self).__init__()

        # initial condition
        self.x0 = x0

        layers = [('lin1', nn.Linear(in_dim, n_hidden_units)), ('act1', activation)]
        for i in range(n_hidden_layers):
            layer_id = i+2
            layers.append(('lin{}'.format(layer_id), nn.Linear(n_hidden_units, n_hidden_units)))
            layers.append(('act{}'.format(layer_id), activation))
        layers.append(('linout', nn.Linear(n_hidden_units, out_dim)))
        if output_tan:
            layers.append(('actout', nn.Tanh()))

        layers = OrderedDict(layers)
        self.main = nn.Sequential(layers)

    def forward(self, x):
        output = self.main(x)
        return output

    def predict(self, t):
        x_pred = self(t)
        # use Marios adjustment for initial condition
        x_adj = self.x0 + (1 - torch.exp(-t)) * x_pred
        return x_adj

class Discriminator(nn.Module):
    def __init__(self, vec_dim=1, n_hidden_units=20, n_hidden_layers=2, activation=nn.Tanh(), unbounded=False):
        super(Discriminator, self).__init__()

        layers = [('lin1', nn.Linear(vec_dim, n_hidden_units)), ('act1', activation)]
        for i in range(n_hidden_layers):
            layer_id = i+2
            layers.append(('lin{}'.format(layer_id), nn.Linear(n_hidden_units, n_hidden_units)))
            layers.append(('act{}'.format(layer_id), activation))
        layers.append(('linout', nn.Linear(n_hidden_units, vec_dim)))
        if not unbounded:
            # unbounded used for WGAN (no sigmoid)
            layers.append(('actout', nn.Sigmoid()))

        layers = OrderedDict(layers)
        self.main = nn.Sequential(layers)

    def forward(self, x):
        output = self.main(x)
        return output

def plot_loss(G_loss, D_loss, ax, legend=True):
    epochs=np.arange(len(G_loss))
    ax.plot(epochs, G_loss, label='G Loss', scaley='log')
    ax.plot(epochs, D_loss, label='D Loss', scaley='log')
    ax.set_title('Loss Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # ax.set_yscale("log") # really only need log-loss for MSE-based
    if legend:
        ax.legend()
    return ax

def plot_preds(G, t, analytic, ax):
    ax.plot(t, analytic(t), label='x')
    t_torch = tensor(t, dtype=torch.float, requires_grad=True).reshape(-1,1)
    pred = G.predict(t_torch)
    ax.plot(t, pred.detach().numpy().flatten(), '--', label='$\hat{x}$')
    ax.set_title('Prediction and Analytic Solution')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend()
    return ax

def plot_derivatives(G, t, ax):
    t_torch = tensor(t, dtype=torch.float, requires_grad=True).reshape(-1,1)
    pred = G.predict(t_torch)
    dxdt, = autograd.grad(pred, t_torch,
                          grad_outputs=pred.data.new(pred.shape).fill_(1),
                          create_graph=True)
    ax.plot(t, pred.detach().numpy().flatten(), label='$\hat{x}$')
    ax.plot(t, dxdt.detach().numpy().flatten(), '--', label="$\hat{x}'$")
    ax.set_title('Prediction and Derivative')
    ax.set_xlabel('$t$')
    ax.set_ylabel("$x$")
    ax.legend()
    return ax

def plot_losses_and_preds(G_loss, D_loss, G, t, analytic, figsize=(15,5), savefig=False, fname=None, legend=True):
    fig, ax = plt.subplots(1,3,figsize=figsize)
    ax1 = plot_loss(G_loss, D_loss, ax[0], legend=legend)
    ax2 = plot_preds(G, t, analytic, ax[1])
    ax3 = plot_derivatives(G, t, ax[2])
    plt.tight_layout()
    if not savefig:
       plt.show()
    else:
       plt.savefig(fname)
    return ax1, ax2, ax3

def train_GAN(num_epochs,
          L=1,
          g_hidden_units=10,
          d_hidden_units=10,
          g_hidden_layers=2,
          d_hidden_layers=2,
          d_lr=0.001,
          g_lr=0.001,
          g_betas=(0.9, 0.999),
          d_betas=(0.9, 0.999),
          t_low=0,
          t_high=10,
          n=100,
          real_label=1,
          fake_label=0,
          logging=True,
          G_iters=1,
          D_iters=1,
          seed=42):
    """
    function to perform training of generator and discriminator for num_epochs
    equation: dx_dt = lambda * x
    """
    if seed:
        torch.manual_seed(seed)

    # initialize nets
    G = Generator(in_dim=1,
                  n_hidden_units=g_hidden_units,
                  n_hidden_layers=g_hidden_layers,
                  activation=nn.LeakyReLU())

    D = Discriminator(vec_dim=1,
                      n_hidden_units=d_hidden_units,
                      n_hidden_layers=d_hidden_layers,
                      activation=nn.LeakyReLU())

    # grid
    t = torch.linspace(t_low, t_high, n, dtype=torch.float, requires_grad=True).reshape(-1,1)

    # perturb grid
    delta_t = t[1]-t[0]
    def get_batch():
        return t + delta_t * torch.randn_like(t) / 3

    # labels
    real_label_vec = torch.full((n,), real_label).reshape(-1,1)
    fake_label_vec = torch.full((n,), fake_label).reshape(-1,1)

    # optimization
    cross_entropy = nn.BCELoss()
    optiD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=d_betas)
    optiG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=g_betas)

    # logging
    D_losses = []
    G_losses = []

    for epoch in range(num_epochs):

        ## =========
        ##  TRAIN G
        ## =========

        for p in D.parameters():
            p.requires_grad = False # turn off computation for D

        t = get_batch()

        for i in range(G_iters):

            x_pred = G.predict(t)
            real = - L * x_pred

            # compute dx/dt
            fake, = autograd.grad(x_pred, t,
                                  grad_outputs=real.data.new(real.shape).fill_(1),
                                  create_graph=True)

            optiG.zero_grad()
            g_loss = cross_entropy(D(fake), real_label_vec)
            g_loss.backward(retain_graph=True)
            optiG.step()

        ## =========
        ##  TRAIN D
        ## =========

        for p in D.parameters():
            p.requires_grad = True # turn on computation for D

        for i in range(D_iters):

            real_loss = cross_entropy(D(real), real_label_vec)
            fake_loss = cross_entropy(D(fake), fake_label_vec)

            optiD.zero_grad()
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward(retain_graph=True)
            optiD.step()

        ## ========
        ## Logging
        ## ========

        if logging:
            print('[%d/%d] D_Loss : %.4f Loss_G: %.4f' % (epoch, num_epochs, d_loss.item(), g_loss.item()))

        D_losses.append(d_loss.item())
        G_losses.append(g_loss.item())

    return G, D, G_losses, D_losses

def train_Lagaris(num_epochs,
                  L=1,
                  g_hidden_units=10,
                  g_hidden_layers=2,
                  g_lr=0.001,
                  t_low=0,
                  t_high=10,
                  G_iters=1,
                  n=100):
    """
    function to perform Lagaris-style training
    """

    # initialize net
    G = Generator(in_dim=1,
                  n_hidden_units=g_hidden_units,
                  n_hidden_layers=g_hidden_layers,
                  activation=nn.LeakyReLU())

    # grid
    t = torch.linspace(t_low, t_high, n, dtype=torch.float, requires_grad=True).reshape(-1,1)

    # perturb grid
    delta_t = t[1]-t[0]
    def get_batch():
        return t + delta_t * torch.randn_like(t) / 3

    mse = nn.MSELoss()
    optiG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.9, 0.999))

    # logging
    D_losses = []
    G_losses = []

    for epoch in range(num_epochs):

        t = get_batch()

        for i in range(G_iters):

            x_pred = G.predict(t)
            lam_x = - L * x_pred

            # compute dx/dt
            dx_dt, = autograd.grad(x_pred, t,
                                  grad_outputs=lam_x.data.new(lam_x.shape).fill_(1),
                                  create_graph=True)

            optiG.zero_grad()
            g_loss = mse(lam_x, dx_dt)
            g_loss.backward(retain_graph=True)
            optiG.step()

        G_losses.append(g_loss.item())

    return G, G_losses

if __name__ == "__main__":
    L = 1
    n = 100
    analytic = lambda t: np.exp(-L*t)
    t = np.linspace(0,10,n)
    G,D,G_loss,D_loss = train_GAN(500,
                          L=L,
                          g_hidden_units=20,
                          g_hidden_layers=3,
                          d_hidden_units=10,
                          d_hidden_layers=2,
                          logging=False,
                          G_iters=10,
                          D_iters=1,
                          n=n)
    plot_losses_and_preds(G_loss, D_loss, G, t, analytic)
