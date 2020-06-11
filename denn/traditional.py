import argparse
import numpy as np
import torch
from denn.config.config import get_config
from denn.rk4 import rk4
from denn.fd import fd
from denn.problems import NonlinearOscillator, CoupledOscillator, SIRModel

def exp_deriv(t, x):
    """
    dxdt = -x
    """
    rhs = -x
    return rhs

def solve_exp(params):
    t, sol = rk4(exp_deriv, [0, 10], 1, 100)
    sol = sol[:, 0]
    true = np.exp(-t)
    mse = np.mean((true-sol)**2)
    print(f"MSE: {mse}")
    return t, sol, true

def sho_deriv(t, xz):
    """
    dxdt = z
    dzdt = -x
    """
    x = xz[0]
    z = xz[1]
    rhs = np.array([z, -x])
    return rhs

def solve_sho(params):
    t, sol = rk4(sho_deriv, [0, 6.28], [0,1], 400)
    sol = sol[:,0]
    true = np.sin(t)
    mse = np.mean((true-sol)**2)
    print(f"MSE: {mse}")
    return t, sol, true

def nlo_deriv(t, xz):
    """
    $$ \ddot{x} + 2 \beta \dot{x} + \omega^{2} x + \phi x^{2} + \epsilon x^{3} = f(t) $$
    dxdt = z
    dzdt = -2 beta z - omega^2 x - phi x^2 - eps x^3

    x0 = 0
    self.omega = 1
    self.epsilon = .1
    self.beta = .1
    self.phi = 1

    n: 400
    t_max: 12.56
    dx_dt0: 0.5
    """
    b = 0.1
    e = 0.1
    o = 1
    p = 1

    x = xz[0]
    z = xz[1]
    rhs = np.array([z, -2*b*z - o*o*x - p*x*x - e*(x**3)])
    return rhs

def solve_nlo(params):
    t, sol = rk4(nlo_deriv, [0, 12.56], [0, 0.5], 1000)
    sol = sol[:,0]
    nlo = NonlinearOscillator(dx_dt0=0.5, n=1000)
    true = nlo.get_solution(t).numpy()
    true = true[:,0]
    mse = np.mean((true-sol)**2)
    print(f"MSE: {mse}")
    return t, sol, true

def coo_deriv(t, xy):
    """
    dxdt = -ty
    dydt = tx
    """
    x, y = xy[0], xy[1]

    rhs1 = -t*y
    rhs2 = t*x
    return np.array([rhs1, rhs2])
    return rhs

def solve_coo(params):
    t, sol = rk4(coo_deriv, [0, 6.28], [1, 0], 800)
    true = CoupledOscillator(x0=1, y0=0, n=800).get_solution(torch.tensor(t))
    mse = np.mean( (sol - true.numpy())**2 )
    print(f"MSE: {mse}")
    return t, sol, true

def sir_deriv(t, sir):
    """
    dxdt = z
    dzdt = -x
    """
    S, I, R = sir[0], sir[1], sir[2]

    beta = 3
    N = 1
    gamma = 1

    rhs1 = -beta*I*S/N
    rhs2 = (beta*I*S/N) - gamma*I
    rhs3 = gamma*I
    return np.array([rhs1, rhs2, rhs3])

def solve_sir(params):
    t, sol = rk4(sir_deriv, [0, 10], [0.99, 0.01, 0.00], 800)
    true = SIRModel(S0=0.99, I0=0.01, R0=0.00, beta=3, gamma=1, n=800).get_solution(t)
    mse = np.mean( (sol - true.numpy())**2 )
    print(f"MSE: {mse}")
    return t, sol, true

def solve_pos(params):
    X, Y, sol = fd()
    true = X*(1-X)*Y*(1-Y)*np.exp(X-Y)
    mse = np.mean( (sol - true)**2 )
    print(f"MSE: {mse}")
    return X, Y, sol, true

def solve(pkey, params):
    """ helper to parse problem key and return appropriate problem
    """
    pkey = pkey.lower().strip()
    if pkey == 'exp':
        solve_exp(params)
    elif pkey == 'sho':
        solve_sho(params)
    elif pkey == 'nlo':
        solve_nlo(params)
    elif pkey == 'pos':
        solve_pos(params)
    elif pkey == 'sir':
        solve_sir(params)
    elif pkey == 'coo':
        solve_coo(params)
    else:
        raise RuntimeError(f'Did not understand problem key (pkey): {pkey}')

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--pkey', type=str, default='EXP',
        help='problem to run (exp=Exponential, sho=SimpleOscillator, nlo=NonlinearOscillator)')
    args = args.parse_args()
    params = get_config(args.pkey)
    res = solve(args.pkey, params)
