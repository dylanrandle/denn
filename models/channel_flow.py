import numpy as np
import torch
from torch.autograd import grad

class Chanflow(torch.nn.Module):
    def __init__(self, in_dim=2, out_dim=2, num_units=10, num_layers=5):
        super().__init__()
        self.layers=torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(torch.nn.Linear(in_dim, num_units))
            elif i == num_layers-1:
                self.layers.append(torch.nn.Linear(num_units, out_dim))
            else:
                self.layers.append(torch.nn.Linear(num_units, num_units))

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = torch.tanh(self.layers[i](x))
        return self.layers[-1](x) # last layer is just linear (regression)

    def train(self, y_space, reynolds_stress_fn, boundary=(-1,1), kinematic_viscosity=1.0, pressure_gradient=1.0, batch_size=100, epochs=500, lr=0.001, C=1.0):
        optimizer=torch.optim.Adam(self.parameters(), lr=lr)
        n = y_space.shape[0]
        num_batches = n//batch_size
        data_mask = np.arange(0, n)
        curves=[]
        losses=[]

        for e in range(epochs):
            accum_loss=0
            np.random.shuffle(data_mask) # shuffle data

            for b in range(num_batches):
                # sample batch of y points
                batch_mask = data_mask[b*batch_size:(b+1)*batch_size]
                y_batch = y_space[batch_mask, :]

                # compute \bar{u} predictions
                u_bar = self(y_batch)

                # compute d(\bar{u})/dy
                du_dy, = grad(u_bar, y_space,
                              grad_outputs=u_bar.data.new(u_bar.shape).fill_(1),
                              create_graph=True)

                # compute d^2(\bar{u})/dy^2
                d2u_dy2, = grad(du_dy, y_space,
                                grad_outputs=du_dy.data.new(du_dy.shape).fill_(1),
                                create_graph=True)

                # compute d<uv>/dy
                re = reynolds_stress_fn(y_batch, du_dy)
                dre_dy, = grad(re, y_space,
                               grad_outputs=re.data.new(re.shape).fill_(1),
                               create_graph=True)

                # compute loss!
                axial_eqn = kinematic_viscosity * d2u_dy2 - dre_dy + pressure_gradient
                lower_boundary = u_bar[np.where(y_batch == boundary[0])]
                upper_boundary = u_bar[np.where(y_batch == boundary[1])]
                loss = torch.mean(torch.pow(axial_eqn, 2)) + C * torch.mean(torch.pow(lower_boundary + upper_boundary, 2))
                accum_loss+=loss

                # zero grad, backprop, step
                optimizer.zero_grad()
                loss.backward(create_graph=True)
                optimizer.step()

            curves.append((y_batch, u_bar))
            epoch_loss=accum_loss/num_batches
            losses.append(epoch_loss)
            if e % 10 == 0:
                print('Mean loss for epoch {}: {}'.format(e, epoch_loss))

        return losses, curves

if __name__ == '__main__':
    print('Testing channel flow NN...')
    # super-simple model for reynolds stress
    pressure_gradient = 1.0
    kinematic_viscosity = 1.0
    karman_const = 0.4 # https://en.wikipedia.org/wiki/Von_K%C3%A1rm%C3%A1n_constant
    reynolds_stress = lambda y, du_dy: -1*((karman_const*y)**2)*torch.abs(du_dy)*du_dy
    # create grid of points
    n = 1000
    ymin, ymax = -1, 1
    ygrid = torch.linspace(ymin, ymax, n).reshape(-1,1)
    y_space = torch.autograd.Variable(ygrid, requires_grad=True)
    # initialize the PDE net
    num_units=100
    num_layers=7
    pde_nn_chanflow = Chanflow(in_dim=1, out_dim=1, num_units=num_units, num_layers=num_layers)
    num_epochs=100
    batch_size=1000
    lr=0.001
    losses, curves = pde_nn_chanflow.train(y_space, reynolds_stress,
                        kinematic_viscosity=kinematic_viscosity, pressure_gradient=pressure_gradient,
                         batch_size=batch_size, epochs=num_epochs, lr=lr)
