import torch
import matplotlib.pyplot as plt
import numpy as np
from denn.utils import Generator
from denn.sho_utils import produce_SHO_preds_system

def train_supervised(model, niters=1000, n=100, x0=0, dx_dt0=0.5, tmax=4*np.pi):
    analytic_oscillator = lambda t: x0*torch.cos(t) + dx_dt0*torch.sin(t)

    t = torch.linspace(0, tmax, n, requires_grad=True).reshape(-1,1)
    y = analytic_oscillator(t)
    opt = torch.optim.Adam(model.parameters())
    mse = torch.nn.MSELoss()

    loss_trace=[]

    for i in range(niters):
        opt.zero_grad()
        xadj, dxdt, d2xdt2 = produce_SHO_preds_system(model, t, x0=x0, dx_dt0=dx_dt0)
        loss = mse(xadj, y)
        loss.backward(retain_graph=True)
        opt.step()

        loss_trace.append(loss.item())

    fig, ax = plt.subplots(1,2,figsize=(15,6))

    ax[0].plot(np.arange(niters), loss_trace)
    ax[0].set_yscale('log')
    ax[0].set_title('learning curve')
    ax[0].set_xlabel("iter")
    ax[0].set_ylabel("loss")


    ax[1].plot(t.detach().numpy(), y.detach().numpy(), label='true')
    xadj, dxdt, d2xdt2 = produce_SHO_preds_system(model, t, x0=x0, dx_dt0=dx_dt0)
    ax[1].plot(t.detach().numpy(), xadj.detach().numpy(), '--', label="x")
    ax[1].plot(t.detach().numpy(), -d2xdt2.detach().numpy(), '--', label="-x''")
    ax[1].set_title('preds')
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('x')
    ax[1].legend()
    plt.show()

gnet = Generator(n_hidden_units=30, n_hidden_layers=20, residual=False)
resgnet = Generator(n_hidden_units=30, n_hidden_layers=20, residual=True)

train_supervised(gnet, niters=500)
train_supervised(resgnet, niters=500)
