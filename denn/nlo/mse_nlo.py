import torch
import matplotlib.pyplot as plt
import numpy as np
from denn.utils import Generator, LambdaLR
from denn.nlo.nlo_utils import produce_preds, produce_preds_system, numerical_solution, nlo_eqn

def train_MSE(model, method='semisupervised', niters=10000, seed=0, n=100, system_of_ODE=False,
                    nperiods=4, perturb=True, lr=0.001, betas=(0, 0.9), observe_every=1,
                    d1=1, d2=1, make_plot=False, lr_schedule=True, decay_start_epoch=0):
    """
    Train/test Lagaris method (MSE loss) fully supervised
    """
    assert method in ['supervised', 'semisupervised', 'unsupervised']

    torch.manual_seed(seed)
    t_torch = torch.linspace(0, nperiods * np.pi, n, requires_grad=True).reshape(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    mse = torch.nn.MSELoss()
    zeros = torch.zeros_like(t_torch)

    # compute the numerical solution
    y_num = numerical_solution(t_torch.detach().numpy().reshape(-1))
    y_num = torch.tensor(y_num, dtype=torch.float).reshape(-1, 1)

    # this generates an index mask for our "observers"
    observers = torch.arange(0, n, observe_every)
    t_observers = t_torch[observers, :]
    y_observers = y_num[observers, :]

    # batch getter
    delta_t = t_torch[1]-t_torch[0]
    def get_batch(perturb=False):
        """ perturb grid """
        if perturb:
          return t_torch + delta_t * torch.randn_like(t_torch) / 3
        else:
          return t_torch

    # lr schedulers
    start_epoch = 0
    if lr_schedule:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=LambdaLR(niters, start_epoch, decay_start_epoch).step)

    _pred_fn = produce_preds_system if system_of_ODE else produce_preds

    # train model
    loss_trace = []
    for i in range(niters):
        opt.zero_grad()

        if method == 'supervised':
            xadj, dxdt, d2xdt2 = _pred_fn(model, t_observers)
            loss = mse(xadj, y_observers)
            loss_trace.append(loss.item())

        elif method == 'semisupervised':
            # supervised part
            xadj, dxdt, d2xdt2 = _pred_fn(model, t_observers)
            loss1 = mse(xadj, y_observers)
            # unsupervised part
            t = get_batch(perturb=perturb)
            xadj, dxdt, d2xdt2 = _pred_fn(model, t)
            loss2 = mse(nlo_eqn(d2xdt2, dxdt, xadj), zeros)
            # combined
            loss = d1 * loss1 + d2 * loss2
            loss_trace.append((loss1.item(), loss2.item()))

        else: # unsupervised
            t = get_batch(perturb=perturb)
            xadj, dxdt, d2xdt2 = _pred_fn(model, t)
            loss = mse(nlo_eqn(d2xdt2, dxdt, xadj), zeros)
            loss_trace.append(loss.item())

        loss.backward(retain_graph=True)
        opt.step()

        if lr_schedule:
            lr_scheduler.step()

    if make_plot:
        fig, ax = plt.subplots(1,2,figsize=(10,5))

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

        xadj, dxdt, d2xdt2 = _pred_fn(model, t)
        ax[1].plot(t.detach().numpy(), y_num.detach().numpy(), label='x')
        ax[1].plot(t.detach().numpy(), xadj.detach().numpy(), '--', label="$\hat{x}$")
        ax[1].set_title('Prediction And Analytic Solution')
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$x$')
        ax[1].legend()

        plt.tight_layout()
        plt.show()

    xadj, dxdt, d2xdt2 = _pred_fn(model, t_torch)
    final_mse = mse(xadj, y_num).item()
    print(f'Final MSE: {final_mse}')
    return {'final_mse': final_mse, 'model': model}

if __name__=="__main__":
    resgnet = Generator(n_hidden_units=32, n_hidden_layers=4, residual=True)
    train_MSE(resgnet, method='supervised', niters=1000, make_plot=True)
