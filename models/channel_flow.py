import numpy as np
import torch
from torch.autograd import grad
import tqdm
import copy

class Chanflow(torch.nn.Module):
    """ Basic neural network to approximate the solution of the stationary channel flow PDE """

    def __init__(self, in_dim=1, out_dim=1, num_units=10, num_layers=5):
        """ initializes architecture """
        super().__init__()
        self.activation=torch.nn.Tanh()
        self.layers=torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0: # input layer
                self.layers.append(torch.nn.Linear(in_dim, num_units))
            elif i == num_layers-1: # output layer
                self.layers.append(torch.nn.Linear(num_units, out_dim))
            else: # hidden layer(s)
                self.layers.append(torch.nn.Linear(num_units, num_units))

        ## TODO ##
        # - should include reynolds stress function when initializing the model
        # - should save the nu associated with each Re_tau number

    def forward(self, x):
        """ implements a forward pass """
        for i in range(len(self.layers)-1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x) # last layer is just linear (regression)

    def predict(self, y, ymin=-1, ymax=1):
        """  implements prediction (using analytical adjustment for boundary conditions) """
        u_bar = self(y)
        # adjust for BV conditions
        boundary=torch.tensor(np.array([ymin, ymax]).reshape(-1,1), dtype=torch.float)
        boundary_condition=torch.zeros_like(boundary)
        y_0 = boundary[0,0]
        y_f = boundary[1,0]
        u_0 = boundary_condition[0,0]
        u_f = boundary_condition[1,0]
        u_bar = u_0 + (u_f-u_0)*(y - y_0)/(y_f - y_0) + (y - y_0)*(y - y_f)*u_bar
        return u_bar

    def compute_diffeq(self, u_bar, y_batch, reynolds_stress_fn, nu, rho, dp_dx):
        # compute d(\bar{u})/dy
        du_dy, = grad(u_bar, y_batch,
                      grad_outputs=u_bar.data.new(u_bar.shape).fill_(1),
                      retain_graph=True,
                      create_graph=True)

        # compute d^2(\bar{u})/dy^2
        d2u_dy2, = grad(du_dy, y_batch,
                        grad_outputs=du_dy.data.new(du_dy.shape).fill_(1),
                        retain_graph=True,
                        create_graph=True)

        # compute d<uv>/dy
        re = reynolds_stress_fn(y_batch, du_dy)
        dre_dy, = grad(re, y_batch,
                       grad_outputs=re.data.new(re.shape).fill_(1),
                       retain_graph=True,
                       create_graph=True)

        diffeq = nu * d2u_dy2 - dre_dy - (1/rho) * dp_dx
        return diffeq

    # def compute_coupled_diffeq(self, u_bar, y_batch, reynolds_stress_fn, nu, rho, dp_dx):
    #     # compute du/dy
    #     du_dy, = grad(u_bar, y_batch,
    #                   grad_outputs=u_bar.data.new(u_bar.shape).fill_(1),
    #                   retain_graph=True,
    #                   create_graph=True)
    #
    #     # compute d<uv>/dy
    #     re = reynolds_stress_fn(y_batch, du_dy)
    #     dre_dy, = grad(re, y_batch,
    #                    grad_outputs=re.data.new(re.shape).fill_(1),
    #                    retain_graph=True,
    #                    create_graph=True)
    #
    #     # diffeq1 =

    def train(self, ymin, ymax, reynolds_stress_fn,
              nu=1., dp_dx=-1., rho=1., batch_size=1000,
              epochs=500, lr=0.001, momentum=0.9, weight_decay=0,
              device=torch.device('cpu'), disable_status=False):

        """ implements training
        args:
        ymin - y coordinate of the lower plate
        ymax - y coordinate of the upper plate
        reynolds_stress_fn - a function that accepts as arguments (y, du_dy) and returns a reynolds stress
        """

        optimizer=torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        losses=[]
        best_model=None
        best_loss=1e8

        with tqdm.trange(epochs, disable=disable_status) as t:
            for e in t:
                # sample y_batch
                y_batch = ymin + (ymax-ymin)*torch.rand((batch_size,1), requires_grad=True, device=device)

                # predict on y_batch (does BV adjustment)
                u_bar = self.predict(y_batch)

                # compute diffeq and loss
                diffeq = self.compute_diffeq(u_bar, y_batch, reynolds_stress_fn, nu, rho, dp_dx)
                loss = torch.mean(torch.pow(diffeq, 2))

                if loss.data.numpy() < best_loss:
                    best_model=copy.deepcopy(self)
                    best_loss=loss.data.numpy()

                # zero grad, backprop, step
                optimizer.zero_grad()
                loss.backward(retain_graph=False, create_graph=False)
                optimizer.step()

                # do some bookkeeping
                loss = loss.data.numpy()
                losses.append(loss)
                t.set_postfix(loss=np.round(loss, 2))

        return losses, best_model

def loss_vs_distance(ax, ymin, ymax, model, hypers, reynolds_stress):
    y = torch.tensor(torch.linspace(ymin,ymax,1000).reshape(-1,1), requires_grad=True)
    u_bar = model.predict(y)
    axial_eqn = model.compute_diffeq(u_bar, y, reynolds_stress, hypers['nu'], hypers['rho'], hypers['dp_dx'])
    ax.plot(y.detach().numpy(), np.power(axial_eqn.detach().numpy(), 2), 'o', markersize=2, lw=0.5, label='square')
    ax.set_title('Loss as a function of distance on ({}, {})'.format(ymin, ymax))
    ax.set_ylabel('Loss (f^2 or |f|)')
    ax.set_xlabel('position (y)')
    ax.legend()

def calc_renot(u_bar, delta, nu):
    n = u_bar.shape[0]
    U_0 = u_bar[n//2][0]
    renot = U_0 * delta / nu
    return renot

def calc_renum(u_bar, ygrid, delta, nu):
    n = u_bar.shape[0]
    U_bar = (1/delta) * np.trapz(u_bar[:n//2], x=ygrid[:n//2], axis=0)[0] # integrate from wall to center
    renum = 2 * delta * U_bar / nu
    return renum

def calc_retau(delta, dp_dx, rho, nu):
    tau_w = -delta * dp_dx
    u_tau = np.sqrt(tau_w / rho)
    re_tau = u_tau * delta / nu
    return re_tau

def convert_dns(delta, hypers, dns):
    tau_w = -delta * hypers['dp_dx']
    u_tau = np.sqrt(tau_w / hypers['rho'])
    h_v = hypers['nu'] / u_tau
    half_y = dns[['y+,']]*h_v
    half_u = dns[['u_1,']]
    # full_u = np.concatenate([half_u, half_u], axis=0)
    # full_y = np.concatenate([half_y, -half_y+2*delta], axis=0)
    return half_u,  half_y

def plot_dns(handle, half_u, half_y, delta):
    handle.plot(half_u, half_y-1, color='red', label='DNS')
    handle.plot(half_u, -half_y+2*delta-1, color='red')
    handle.legend()

def get_hyperparams(dp_dx=-1.0, nu=0.001, rho=1.0, k=0.41, num_units=50,
                    num_layers=5, batch_size=1000, lr=0.001, num_epochs=1000,
                    ymin=-1, ymax=1, weight_decay=0):
    """
    dp_dx - pressure gradient
    nu - kinematic viscosity
    rho - density
    k - karman constant (mixing length model), see: https://en.wikipedia.org/wiki/Von_K%C3%A1rm%C3%A1n_constant
    """
    return dict(dp_dx=dp_dx, nu=nu, rho=rho, k=k, num_units=num_units, num_layers=num_layers,
                batch_size=batch_size, lr=lr, num_epochs=num_epochs, ymin=ymin, ymax=ymax, weight_decay=weight_decay)

def get_yspace(n=1000, ymin=-1, ymax=1):
    ygrid = torch.linspace(ymin, ymax, n).reshape(-1,1)
    return torch.autograd.Variable(ygrid, requires_grad=True)

def get_mixing_len_model(k, delta, dp_dx, rho, nu):
    return lambda y, du_dy: -1*((k*(torch.abs(y)-delta))**2)*torch.abs(du_dy)*du_dy

    # def model(y, du_dy):
    #     tau_w = -delta * dp_dx
    #     u_tau = np.sqrt(tau_w / rho)
    #     h_v = nu / u_tau
    #     y = y / h_v
    #     return -1*((k*(torch.abs(y)-delta))**2)*torch.abs(du_dy)*du_dy
    # return model

def make_plots(ax, losses,  model, hypers, retau, dns_u=None, dns_y=None):
    """ plot loss and prediction of model at retau """
    # losses
    ax[0].loglog(np.arange(len(losses)), losses, color='blue')
    ax[0].set_title('Log mean loss vs. log epoch at Retau={}'.format(retau))
    ax[0].set_xlabel('log( epoch )')
    ax[0].set_ylabel('log( mean loss )')
    # preds
    y_space = torch.linspace(hypers['ymin'], hypers['ymax'], 1000).reshape(-1,1)
    preds = model.predict(y_space).detach().numpy()
    ax[1].plot(preds, y_space.detach().numpy(), alpha=1, color='blue', label='NN')
    if dns_u is not None and dns_y is not None:
        ax[1].plot(dns_u, dns_y, alpha=1, color='red', label='DNS')
    ax[1].set_title('Predicted $<u>$ at Retau={}'.format(retau))
    ax[1].set_ylabel('y')
    ax[1].set_xlabel('$<u>$')
    ax[1].legend()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import os
    print('Testing channel flow NN...')

    # hyperparams
    hypers = get_hyperparams(ymin=-1, ymax=1, num_epochs=1000, lr=0.0001, num_layers=4, num_units=40, batch_size=1000, weight_decay=.1)
    delta = (hypers['ymax']-hypers['ymin'])/2
    reynolds_stress = get_mixing_len_model(hypers['k'], delta, hypers['dp_dx'], hypers['rho'], hypers['nu'])

    # test training
    hypers['nu']=0.005555555555
    retau=calc_retau(delta, hypers['dp_dx'], hypers['rho'], hypers['nu'])
    print('Training at Retau={}'.format(retau))
    pdenn = Chanflow(num_units=hypers['num_units'], num_layers=hypers['num_layers'])
    losses, pdenn_best = pdenn.train(hypers['ymin'], hypers['ymax'],
                                   reynolds_stress,
                                   nu=hypers['nu'],
                                   dp_dx=hypers['dp_dx'],
                                   rho=hypers['rho'],
                                   batch_size=hypers['batch_size'],
                                   epochs=hypers['num_epochs'],
                                   lr=hypers['lr'],
                                   weight_decay=hypers['weight_decay'])

    # test saving everything from the run
    y=np.linspace(-1,1,1000)
    preds = pdenn_best.predict(torch.tensor(y.reshape(-1,1), dtype=torch.float)).detach().numpy()
    timestamp=time.time()
    retau=np.round(retau, decimals=2)
    os.mkdir('data/{}'.format(timestamp))
    np.save('data/{}/mixlen_preds_u{}.npy'.format(timestamp, retau), preds)
    np.save('data/{}/mixlen_loss_u{}.npy'.format(timestamp, retau), np.array(losses))
    np.save('data/{}/mixlen_hypers_u{}.npy'.format(timestamp, retau), hypers)
    torch.save(pdenn_best.state_dict(), 'data/{}/mixlen_model_u{}.pt'.format(timestamp, retau))

    # test plots
    fig, ax = plt.subplots(1, 2, figsize=(20,10))
    make_plots(ax, losses1000, pdenn1000, hypers, retau)
    plt.show()
