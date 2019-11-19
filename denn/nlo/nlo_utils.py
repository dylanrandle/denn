import torch
from denn.utils import diff
import numpy as np
from scipy import optimize

NLO_PARAMS = {
    'omega': 1,
    'epsilon': .1,
    'beta': .1,
    'phi': 1,
    'F': .1,
    'forcing': lambda t: np.cos(t),
    'x0': 0,
    'dx0': 0.5
}

def nlo_eqn(d2x, dx, x):
    eqn = d2x + 2 * NLO_PARAMS['beta'] * dx + (NLO_PARAMS['omega'] ** 2) * x + NLO_PARAMS['phi'] * (x ** 2) \
        + NLO_PARAMS['epsilon'] * (x ** 3) # - NLO_PARAMS['F'] * NLO_PARAMS['forcing'](t)
    return eqn

def numerical_solution(t):
    """ uses scipy to solve NLO """
    dt = t[1] - t[0]
    guess = np.ones_like(t)

    def get_diff(x):
        x[0] = NLO_PARAMS['x0']
        dx = np.gradient(x, dt, edge_order = 2)
        dx[0] = NLO_PARAMS['dx0']
        d2x = np.gradient(dx, dt, edge_order = 2)
        return nlo_eqn(d2x, dx, x)

    opt_sol = optimize.root(get_diff, guess, method='lm')
    print(f'Numerical solution succes: {opt_sol.success}')
    return opt_sol.x

def produce_preds(G, t):
    """ produce sho preds that satisfy system, without u adjustment """
    x_hat = G(t)
    # x condition
    x_adj = NLO_PARAMS['x0'] + (1 - torch.exp(-t)) * NLO_PARAMS['dx0'] + ((1 - torch.exp(-t))**2) * x_hat
    dx = diff(x_adj, t)
    d2x = diff(dx, t)
    return x_adj, dx, d2x

def produce_preds_system(G, t):
    """ produce preds that satisfy system of equations """
    x_hat = G(t)
    dx = diff(x_hat, t)
    # x condition
    x_adj = NLO_PARAMS['x0'] + (1 - torch.exp(-t)) * NLO_PARAMS['dx0'] + ((1 - torch.exp(-t))**2) * x_hat
    # dx condition guarantees that dx_dt = u (first equation in system)
    u_adj = torch.exp(-t) * NLO_PARAMS['dx0'] + 2 * (1 - torch.exp(-t)) * torch.exp(-t) * x_hat \
        + (1 - torch.exp(-t)) * dx
    # compute du_dt = d2x_dt2
    d2x = diff(u_adj, t)
    return x_adj, u_adj, d2x
