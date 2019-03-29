import numpy as np
import torch
from torch.autograd import grad
import tqdm
import copy
import time
import os
import utils

class Chanflow(torch.nn.Module):
    """ Basic neural network to approximate the solution of the stationary channel flow PDE """

    def __init__(self, **kwargs):
        """
        initializes architecture, sets hyper parameters, and sets reynolds stress model
        """
        super().__init__()
        self.activation=torch.nn.Tanh()
        self.layers=torch.nn.ModuleList()

        self.set_hyperparams(**kwargs)
        num_units=self.hypers['num_units']
        num_layers=self.hypers['num_layers']
        in_dim=self.hypers['in_dim']
        out_dim=self.hypers['out_dim']
        self.set_reynolds_stress_fn()

        activation=self.hypers['activation']
        if activation == 'swish':
            self.activation=utils.Swish()

        # build architecture
        self.layers.append(torch.nn.Linear(in_dim, num_units)) # input layer
        for i in range(num_layers):
            self.layers.append(torch.nn.Linear(num_units, num_units)) # hidden layer
        self.layers.append(torch.nn.Linear(num_units, out_dim)) # output layer

    def set_hyperparams(self, dp_dx=-1.0, nu=0.0055555555, rho=1.0, k=0.41, num_units=40,
                        num_layers=2, batch_size=1000, lr=0.0001, num_epochs=1000,
                        ymin=-1, ymax=1, n=1000, weight_decay=0, in_dim=1, out_dim=1,
                        delta=1, retau=180, sampling='grid', activation='tanh'):
        """
        dp_dx - pressure gradient
        nu - kinematic viscosity
        rho - density
        k - karman constant (mixing length model),
            see: https://en.wikipedia.org/wiki/Von_K%C3%A1rm%C3%A1n_constant
        """
        # delta = np.abs(ymax-ymin)/2
        self.hypers = dict(dp_dx=dp_dx, nu=nu, rho=rho, k=k,
                    num_units=num_units, num_layers=num_layers,
                    batch_size=batch_size, lr=lr, num_epochs=num_epochs,
                    ymin=ymin, ymax=ymax, weight_decay=weight_decay,
                    delta=delta, n=1000, in_dim=1, out_dim=1, retau=retau,
                    sampling=sampling, activation=activation)

    def forward(self, x):
        """ implements a forward pass """
        for i in range(len(self.layers)-1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x) # last layer is just linear (regression)

    def predict(self, y):
        """  implements prediction (using analytical adjustment for boundary conditions) """
        ymin = self.hypers['ymin']
        ymax = self.hypers['ymax']
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

    def set_reynolds_stress_fn(self):
        k=self.hypers['k']
        delta=self.hypers['delta']
        self.reynolds_stress_fn = lambda y, du_dy: -((k * (torch.abs(y)-delta)/(2*delta)) ** 2) * torch.abs(du_dy) * du_dy

    def compute_diffeq(self, u_bar, y_batch):
        dp_dx=self.hypers['dp_dx']
        rho=self.hypers['rho']
        nu=self.hypers['nu']
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
        re = self.reynolds_stress_fn(y_batch, du_dy)
        dre_dy, = grad(re, y_batch,
                       grad_outputs=re.data.new(re.shape).fill_(1),
                       retain_graph=True,
                       create_graph=True)

        diffeq = nu * d2u_dy2 - dre_dy - (1/rho) * dp_dx
        return diffeq

    def train(self, device=torch.device('cpu'), disable_status=False, save_run=False):

        """ implements training
        device - which device (cpu / gpu) to put model on
        disable_status - turns off TQDM status bar
        """
        print('Training with hyperparameters: ')
        print(self.hypers)
        ymin=self.hypers['ymin']
        ymax=self.hypers['ymax']
        batch_size=self.hypers['batch_size']
        epochs=self.hypers['num_epochs']
        lr=self.hypers['lr']
        weight_decay=self.hypers['weight_decay']
        sampling=self.hypers['sampling']
        self.set_reynolds_stress_fn() # in case hypers has changed since initialization (TODO: check if this is necessary)

        optimizer=torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        train_losses, val_losses=[], []
        best_model=None
        best_loss=1e8

        if sampling == 'grid':
            grid = torch.linspace(ymin, ymax, batch_size, requires_grad=True, device=device).reshape(-1,1)
            get_batch = lambda i: grid # just returns grid every time

        elif sampling == 'uniform':
            get_batch = lambda i: ymin + (ymax-ymin)*torch.rand((batch_size, 1), requires_grad=True, device=device)

        elif sampling == 'perturb':
            grid = torch.linspace(ymin, ymax, batch_size).reshape(-1,1)
            delta_y = grid[1]-grid[0]
            def get_batch(i):
                noise = delta_y * torch.randn_like(grid) / 3 # ensure 3 x sigma is within delta_y
                pts = grid + noise
                return torch.tensor(pts, requires_grad=True, device=device)

        elif sampling == 'boundary':
            lhs = np.geomspace(0.02, 1, num=500) - 1.02
            rhs = np.flip(-lhs, axis=0)
            geomgrid = torch.tensor(np.concatenate([lhs, rhs]), requires_grad=True, device=device, dtype=torch.float).reshape(-1, 1)
            get_batch = lambda i: geomgrid

        else:
            raise Exception('Encountered unexpected sampling type: {}'.format(sampling))

        with tqdm.trange(epochs, disable=disable_status) as t:
            for e in t:
                # get a batch
                y_batch = get_batch(e)

                # predict on y_batch (does BV adjustment)
                u_bar = self.predict(y_batch)

                # compute diffeq
                diffeq = self.compute_diffeq(u_bar, y_batch)

                # compute loss
                loss = torch.mean(torch.pow(diffeq, 2))

                # zero grad, backprop, step
                optimizer.zero_grad()
                loss.backward(retain_graph=False, create_graph=False)
                optimizer.step()

                # record history
                loss_np = loss.data.cpu().numpy()
                train_losses.append(loss_np)
                t.set_postfix(loss=np.round(loss_np, 2))

                if e % 100 == 0: # run validation
                    val_batch = ymin + (ymax-ymin)*torch.rand((batch_size, 1), requires_grad=True, device=device)
                    u_val = self.predict(val_batch)
                    diffeq_val = self.compute_diffeq(u_val, val_batch)
                    val_loss = torch.mean(torch.pow(diffeq_val, 2))
                    val_loss_np = val_loss.data.cpu().numpy()
                    val_losses.append(val_loss_np)
                    if val_loss_np < best_loss: # record best so far
                        best_model=copy.deepcopy(self)
                        best_loss=val_loss_np

                if e > 0 and disable_status and e % 1000 == 0: # use very light logging when disable_status is true
                    print('Epoch {}: Loss = {}'.format(e, loss_np))

        run_dict = dict(train_loss=train_losses, val_loss=val_losses, best_model=best_model)

        if save_run:
            self.save_run(run_dict)

        return run_dict

    def save_run(self, run_dict, top_dir='data/'):
        """ saves everything from a training run in data/timestamp """
        # objects to save
        pdenn_best = run_dict['best_model']
        train_loss = np.array(run_dict['train_loss']).reshape(-1,1)
        val_loss = np.array(run_dict['val_loss']).reshape(-1,1)
        y = np.linspace(self.hypers['ymin'],self.hypers['ymax'],self.hypers['n'])
        preds = pdenn_best.predict(torch.tensor(y.reshape(-1,1), dtype=torch.float)).detach().numpy()
        timestamp=time.time()
        retau=np.round(utils.calc_retau(self.hypers['delta'], self.hypers['dp_dx'], self.hypers['rho'], self.hypers['nu']), decimals=2)
        hypers=self.hypers
        hypers['retau']=retau
        # saving them
        os.mkdir(top_dir+'{}'.format(timestamp))
        np.save(top_dir+'{}/preds.npy'.format(timestamp), preds)
        np.save(top_dir+'{}/train_loss.npy'.format(timestamp), train_loss)
        np.save(top_dir+'{}/val_loss.npy'.format(timestamp), val_loss)
        np.save(top_dir+'{}/hypers.npy'.format(timestamp), hypers)
        torch.save(pdenn_best.state_dict(), 'data/{}/model.pt'.format(timestamp, retau))
        print('Successfully saved at '+top_dir+'{}/'.format(timestamp))

if __name__ == '__main__':
    print('Testing channel flow NN')
    pdenn = Chanflow()
    run_dict = pdenn.train()
    print('Saving run dict')
    pdenn.save_run(run_dict)
