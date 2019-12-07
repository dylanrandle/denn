from algos import train_MSE
from models import MLP
from problems import SimpleOscillator
import torch
import torch.nn as nn

def mse_experiment_SHO(method, niters=1000, hidden_units=32, hidden_layers=4, obs_every=1, seed=0):
    torch.manual_seed(seed)
    problem = SimpleOscillator()
    model = MLP(in_dim=1, out_dim=1, n_hidden_units=hidden_units,
        n_hidden_layers=hidden_layers, activation=nn.Tanh(),
        residual=True, regress=True)
    res = train_MSE(model, problem, method=method,
        obs_every=obs_every, niters=niters,
        plot=True, save=True, fname=f'mse_{method}_SHO.png',)
    return res

if __name__ == '__main__':
    mse_experiment_SHO('supervised')
    mse_experiment_SHO('semisupervised')
    mse_experiment_SHO('unsupervised')
