import torch
import matplotlib.pyplot as plt
import numpy as np
from denn.utils import Generator
from denn.sho.sho_utils import produce_SHO_preds_system

def unsupervised_SHO(model, niters=1000, n=100, x0=0, dx_dt0=0.5, tmax=4*np.pi):
    """
    simple unsupervised learning case with MSE loss
    """
    analytic_oscillator = lambda t: x0*torch.cos(t) + dx_dt0*torch.sin(t)

    t = torch.linspace(0, tmax, n, requires_grad=True).reshape(-1,1)
    y = analytic_oscillator(t)
    opt = torch.optim.Adam(model.parameters())
    mse = torch.nn.MSELoss()

    loss_trace=[]

    for i in range(niters):
        opt.zero_grad()
        xadj, dxdt, d2xdt2 = produce_SHO_preds_system(model, t, x0=x0, dx_dt0=dx_dt0)
        loss = mse(xadj, -d2xdt2) # equation : x = -x''
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

def unsupervised_SHO_GAN():
    """ GAN method unsupervised"""
    raise NotImplementedError()

if __name__=="__main__":
    resgnet = Generator(n_hidden_units=50, n_hidden_layers=20, residual=True)
    unsupervised_SHO(resgnet, niters=500)
