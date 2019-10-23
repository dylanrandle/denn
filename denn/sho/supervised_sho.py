import torch
import matplotlib.pyplot as plt
import numpy as np
from denn.utils import Generator
from denn.sho.sho_utils import produce_SHO_preds_system

def train_supervised(model, niters=1000, x0=0, dx_dt0=0.5, seed=42, n=100,
                    tmax=4*np.pi, perturb=False, lr=0.0002, betas=(0, 0.9)):
    """
    Train/test Lagaris method (MSE loss) fully supervised
    """
    torch.manual_seed(seed)
    analytic_oscillator = lambda t: x0*torch.cos(t) + dx_dt0*torch.sin(t)

    t_torch = torch.linspace(0,tmax,n, requires_grad=True).reshape(-1,1)
    y = analytic_oscillator(t_torch)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    mse = torch.nn.MSELoss()
    loss_trace=[]
    delta_t = t_torch[1]-t_torch[0]
    # batch getter
    def get_batch(perturb=False):
        """ perturb grid """
        if perturb:
          return t_torch + delta_t * torch.randn_like(t_torch) / 3
        else:
          return t_torch

    for i in range(niters):
        opt.zero_grad()
        t = get_batch(perturb=perturb)
        xadj, dxdt, d2xdt2 = produce_SHO_preds_system(model, t, x0=x0, dx_dt0=dx_dt0)
        loss = mse(xadj, y)
        loss.backward(retain_graph=True)
        opt.step()
        loss_trace.append(loss.item())

    fig, ax = plt.subplots(1,3,figsize=(15,5))

    t = get_batch(perturb=False)

    ax[0].plot(np.arange(niters), loss_trace)
    ax[0].set_yscale('log')
    ax[0].set_title('Loss Curve')
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")


    ax[1].plot(t.detach().numpy(), y.detach().numpy(), label='x')
    xadj, dxdt, d2xdt2 = produce_SHO_preds_system(model, t, x0=x0, dx_dt0=dx_dt0)
    ax[1].plot(t.detach().numpy(), xadj.detach().numpy(), '--', label="$\hat{x}$")
    ax[1].set_title('Prediction And Analytic Solution')
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('$x$, $\hatx$')
    ax[1].legend()

    ax[2].plot(t.detach().numpy(), xadj.detach().numpy(), label="$\hat{x}$")
    ax[2].plot(t.detach().numpy(), d2xdt2.detach().numpy(), '--', label="$\hat{x}''$")
    ax[2].set_title('Prediction And Second Derivative')
    ax[2].set_xlabel('$t$')
    ax[2].set_ylabel("$\hat{x}$, $\hat{x''}$")
    ax[2].legend()

    plt.tight_layout()
    plt.show()

def train_supervised_GAN():
    """ TODO: implement this from Notebook code (already implemented there) """
    raise NotImplementedError()

if __name__=="__main__":
    resgnet = Generator(n_hidden_units=30, n_hidden_layers=20, residual=True)
    train_supervised(resgnet, niters=500)
