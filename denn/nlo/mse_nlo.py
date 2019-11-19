import torch
import matplotlib.pyplot as plt
import numpy as np
from denn.utils import Generator
from denn.nlo.nlo_utils import produce_SHO_preds_system, 

omega = 1
epsilon = .1
beta = .1
phi = 1
F = .1
forcing = lambda x: F * np.sin(x)
nlo = lambda d2x, dx, x: d2x + 2 * beta * dx + (omega ** 2) * x + phi * (x ** 2) + epsilon * (x ** 3) - forcing(x)

def train_MSE(model, method='semisupervised', niters=10000, x0=0, dx_dt0=0.5, seed=0, n=100,
                    tmax=4*np.pi, perturb=True, lr=0.001, betas=(0, 0.9), observe_every=1,
                    d1=1, d2=1, make_plot=False):
    """
    Train/test Lagaris method (MSE loss) fully supervised
    """
    assert method in ['supervised', 'semisupervised', 'unsupervised']
    torch.manual_seed(seed)
    numerical_solution =
    t_torch = torch.linspace(0, tmax, n, requires_grad=True).reshape(-1, 1)
    y = analytic_oscillator(t_torch)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    mse = torch.nn.MSELoss()
    loss_trace = []
    delta_t = t_torch[1]-t_torch[0]
    # this generates an index mask for our "observers"
    observers = torch.arange(0, n, observe_every)
    t_observers = t_torch[observers, :]
    y_observers = analytic_oscillator(t_observers)
    # batch getter
    def get_batch(perturb=False):
        """ perturb grid """
        if perturb:
          return t_torch + delta_t * torch.randn_like(t_torch) / 3
        else:
          return t_torch

    for i in range(niters):
        opt.zero_grad()

        if method == 'supervised':
            xadj, dxdt, d2xdt2 = produce_SHO_preds_system(model, t_observers, x0=x0, dx_dt0=dx_dt0)
            loss = mse(xadj, y_observers)
            loss_trace.append(loss.item())

        elif method == 'semisupervised':
            # supervised part
            xadj, dxdt, d2xdt2 = produce_SHO_preds_system(model, t_observers, x0=x0, dx_dt0=dx_dt0)
            loss1 = mse(xadj, y_observers)
            # unsupervised part
            t = get_batch(perturb=perturb)
            xadj, dxdt, d2xdt2 = produce_SHO_preds_system(model, t, x0=x0, dx_dt0=dx_dt0)
            loss2 = mse(xadj, -d2xdt2)
            # combined
            loss = d1 * loss1 + d2 * loss2
            loss_trace.append((loss1.item(), loss2.item()))

        else: # unsupervised
            t = get_batch(perturb=perturb)
            xadj, dxdt, d2xdt2 = produce_SHO_preds_system(model, t, x0=x0, dx_dt0=dx_dt0)
            loss = mse(xadj, -d2xdt2) # equation : x = -x''
            loss_trace.append(loss.item())

        loss.backward(retain_graph=True)
        opt.step()

    if make_plot:
        fig, ax = plt.subplots(1,3,figsize=(15,5))

        t = get_batch(perturb=False)

        if not method == 'semisupervised':
            ax[0].plot(np.arange(niters), loss_trace)
        else:
            ax[0].plot(np.arange(niters), [l[0] for l in loss_trace], label='$L_{S}$')
            ax[0].plot(np.arange(niters), [l[1] for l in loss_trace], label='$L_{U}$')
            ax[0].legend()
        ax[0].set_yscale('log')
        ax[0].set_title('Loss Curve')
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")

        ax[1].plot(t.detach().numpy(), y.detach().numpy(), label='x')
        xadj, dxdt, d2xdt2 = produce_SHO_preds_system(model, t, x0=x0, dx_dt0=dx_dt0)
        ax[1].plot(t.detach().numpy(), xadj.detach().numpy(), '--', label="$\hat{x}$")
        ax[1].set_title('Prediction And Analytic Solution')
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$x$')
        ax[1].legend()

        ax[2].plot(t.detach().numpy(), xadj.detach().numpy(), label="$\hat{x}$")
        ax[2].plot(t.detach().numpy(), d2xdt2.detach().numpy(), '--', label="$\hat{x}''$")
        ax[2].set_title('Prediction And Second Derivative')
        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel("$x$")
        ax[2].legend()

        plt.tight_layout()
        plt.show()

    xadj, dxdt, d2xdt2 = produce_SHO_preds_system(model, t_torch, x0=x0, dx_dt0=dx_dt0)
    final_mse = mse(xadj, y).item()
    return {'final_mse': final_mse, 'model': model}

if __name__=="__main__":
    resgnet = Generator(n_hidden_units=32, n_hidden_layers=4, residual=True)
    train_MSE(resgnet, method='semisupervised', niters=500)
