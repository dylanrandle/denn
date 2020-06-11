import numpy as np
import torch
from scipy.integrate import odeint, solve_ivp
from denn.utils import diff
from denn.rans.numerical import solve_rans_scipy_solve_bvp
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class Problem():
    """ parent class for all problems
    """
    def __init__(self, n = 100, perturb = True):
        """
        n: number of points on grid
        perturb: boolean indicator for perturbed sampling of grid points
        """
        self.n = n
        self.perturb = perturb

    def sample_grid(self, grid, spacing, tau=3):
        """ return perturbed samples from the grid
            grid is the torch tensor representing the grid
            d is the inter-point spacing
        """
        if self.perturb:
            return grid + spacing * torch.randn_like(grid) / tau
        else:
            return grid

    def get_grid(self):
        """ return base grid """
        raise NotImplementedError()

    def get_grid_sample(self):
        """ sample from grid (if perturb=False, returns grid) """
        raise NotImplementedError()

    def get_solution(self):
        """ return solution to problem """
        raise NotImplementedError()

    def get_equation(self, *args):
        """ return equation output (i.e. residuals s.t. solved iff == 0) """
        raise NotImplementedError()

    def adjust(self, *args):
        """ adjust a pred according to some IV/BC conditions
            should return all components needed for equation
            as they might be needed to be adjusted as well
            e.g. adjusting dx_dt for SimpleOscillator as a system
        """
        raise NotImplementedError()

    def get_plot_dicts(self, *args):
        """ return pred_dict and optionall diff_dict (or None) to be used for plotting
            depending on the problem we may want to plot different things, which is why
            this method exists (and is required)
        """
        raise NotImplementedError()

class Exponential(Problem):
    """
    Equation:
    x' + Lx = 0

    Analytic Solution:
    x = exp(-Lt)
    """
    def __init__(self, t_min = 0, t_max = 10, x0 = 1., L = 1, **kwargs):
        """
        inputs:
            - t_min: start time
            - t_max: end time
            - x0: initial condition on x
            - L: rate of decay constant
            - kwargs: keyword args passed to Problem.__init__()
        """
        super().__init__(**kwargs)

        self.t_min = t_min
        self.t_max = t_max
        self.x0 = x0
        self.L = L
        self.grid = torch.linspace(
            t_min,
            t_max,
            self.n,
            dtype=torch.float,
            requires_grad=True
        ).reshape(-1,1)
        self.spacing = self.grid[1, 0] - self.grid[0, 0]

    def get_grid(self):
        return self.grid

    def get_grid_sample(self):
        return self.sample_grid(self.grid, self.spacing)

    def get_solution(self, t):
        """ return the analytic solution @ t for this problem """
        return torch.exp(-self.L * t)

    def get_equation(self, x, t):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(x, t)
        x, dx = adj['pred'], adj['dx']
        return dx + self.L * x

    def adjust(self, x, t):
        """ perform initial value adjustment """
        x_adj = self.x0 + (1 - torch.exp(-t)) * x
        dx_dt = diff(x_adj, t)
        return {'pred': x_adj, 'dx': dx_dt}

    def get_plot_dicts(self, x, t, y):
        """ return appropriate pred_dict and diff_dict used for plotting """
        adj = self.adjust(x, t)
        xadj, dx = adj['pred'], adj['dx']
        pred_dict = {'$\hat{x}$': xadj.detach(), '$x$': y.detach()}
        # diff_dict = {'$\hat{x}$': xadj.detach(), '$-\hat{\dot{x}}$': (-dx).detach()}
        residual = self.get_equation(x, t)
        diff_dict = {'$|\hat{F}|$': np.abs(residual.detach())}
        return pred_dict, diff_dict

class SimpleOscillator(Problem):
    """ simple harmonic oscillator problem """
    def __init__(self, t_min = 0, t_max = 4 * np.pi, dx_dt0 = 1., **kwargs):
        """
        inputs:
            - t_min: start time
            - t_max: end time
            - dx_dt0: initial condition on dx_dt
            - kwargs: keyword args passed to Problem.__init__()
        """
        super().__init__(**kwargs)

        # ======
        # TODO:
        x0 = 0
        # ======

        self.t_min = t_min
        self.t_max = t_max
        self.x0 = x0
        self.dx_dt0 = dx_dt0
        self.grid = torch.linspace(
            t_min,
            t_max,
            self.n,
            dtype=torch.float,
            requires_grad=True
        ).reshape(-1,1)
        self.spacing = self.grid[1, 0] - self.grid[0, 0]

    def get_grid(self):
        return self.grid

    def get_grid_sample(self):
        return self.sample_grid(self.grid, self.spacing)

    def get_solution(self, t):
        """ return the analytic solution @ t for this problem """
        return self.dx_dt0 * torch.sin(t)

    def get_equation(self, x, t):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(x, t)
        x, dx, d2x = adj['pred'], adj['dx'], adj['d2x']
        return d2x + x

    def adjust(self, x, t):
        """ perform initial value adjustment """
        x_adj = self.x0 + (1 - torch.exp(-t)) * self.dx_dt0 + ((1 - torch.exp(-t))**2) * x
        dx_dt = diff(x_adj, t)
        d2x_dt2 = diff(dx_dt, t)
        return {'pred': x_adj, 'dx': dx_dt, 'd2x': d2x_dt2}

    def get_plot_dicts(self, x, t, y):
        """ return appropriate pred_dict and diff_dict used for plotting """
        adj = self.adjust(x, t)
        xadj, dx, d2x = adj['pred'], adj['dx'], adj['d2x']
        pred_dict = {'$\hat{x}$': xadj.detach(), '$x$': y.detach()}
        residual = self.get_equation(x, t)
        # diff_dict = {'$\hat{x}$': xadj.detach(), '$-\hat{\ddot{x}}$': (-d2x).detach()}
        diff_dict = {'$|\hat{F}|$': np.abs(residual.detach())}
        return pred_dict, diff_dict

class NonlinearOscillator(Problem):
    """
    Nonlinear Oscillator Problem:

    $$ \ddot{x} + 2 \beta \dot{x} + \omega^{2} x + \phi x^{2} + \epsilon x^{3} = f(t) $$
    """
    def __init__(self, t_min = 0, t_max = 4 * np.pi, dx_dt0 = 1., **kwargs):
        """
        inputs:
            - t_min: start time
            - t_max: end time
            - dx_dt0: initial condition on dx_dt
            - kwargs: keyword args passed to Problem.__init__()
        """
        super().__init__(**kwargs)

        # ======
        # TODO
        x0 = 0
        self.omega = 1
        self.epsilon = .1
        self.beta = .1
        self.phi = 1
        # self.F = .1
        # self.forcing = lambda t: torch.cos(t)
        # ======

        self.t_min = t_min
        self.t_max = t_max
        self.x0 = x0
        self.dx_dt0 = dx_dt0
        self.grid = torch.linspace(
            t_min,
            t_max,
            self.n,
            dtype=torch.float,
            requires_grad=True
        ).reshape(-1, 1)
        self.spacing = self.grid[1, 0] - self.grid[0, 0]

        atol = 1e-8
        rtol = 1e-8
        self.sol = solve_ivp(
            self._nlo_system,
            t_span=(t_min, t_max),
            y0=[self.x0, self.dx_dt0],
            dense_output=True,
            atol=atol,
            rtol=rtol,
        )

    def get_grid(self):
        return self.grid

    def get_grid_sample(self):
        return self.sample_grid(self.grid, self.spacing)

    def get_solution(self, t, atol=1e-8, rtol=1e-8):
        """ uses scipy to solve NLO """
        try:
            t = t.detach().numpy() # if torch tensor, convert to numpy
        except:
            pass

        t = t.reshape(-1)

        sol = self.sol.sol(t)
        return torch.tensor(sol[0,:], dtype=torch.float).reshape(-1, 1)

    def _nlo_system(self, t, z):
        """ NLO decomposed as system of first order equations """
        x, y = z   # y = x'
        return np.array([y, -(2 * self.beta * y + (self.omega**2) * x + self.phi * (x**2) + self.epsilon * (x**3))])

    def _nlo_eqn(self, x, dx, d2x):
        return d2x + 2 * self.beta * dx + (self.omega ** 2) * x + self.phi * (x ** 2) \
            + self.epsilon * (x ** 3) # - self.F * self.forcing(t)

    def get_equation(self, x, t):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(x, t)
        x, dx, d2x = adj['pred'], adj['dx'], adj['d2x']
        return self._nlo_eqn(x, dx, d2x)

    def adjust(self, x, t):
        """ perform initial value adjustment """
        x_adj = self.x0 + (1 - torch.exp(-t)) * self.dx_dt0 + ((1 - torch.exp(-t))**2) * x
        dx = diff(x_adj, t)
        d2x = diff(dx, t)
        return {'pred': x_adj, 'dx': dx, 'd2x': d2x}

    def get_plot_dicts(self, x, t, y):
        """ return appropriate pred_dict and diff_dict used for plotting """
        adj = self.adjust(x, t)
        xadj, dx, d2x = adj['pred'], adj['dx'], adj['d2x']
        pred_dict = {'$\hat{x}$': xadj.detach(), '$x$': y.detach()}
        residual = self._nlo_eqn(xadj, dx, d2x)
        diff_dict = {'$|\hat{F}|$': np.abs(residual.detach())}
        return pred_dict, diff_dict

class ReynoldsAveragedNavierStokes(Problem):
    """
    RANS Equations for 1-Dimensional Channel Flow
    """
    def __init__(self, ymin = -1, ymax = 1, bc = [0, 0],
        kappa=0.41/4, rho=1.0, nu=0.0055555555, dp_dx = -1,
        **kwargs):
        """
        ymin - min y-coordinate
        ymax - max y-coordinate
        bc - boundary condation as [u(ymin), y(ymax)]
        kwargs - keyword args passed to `Problem`
        """
        super().__init__(**kwargs)
        self.ymin = ymin
        self.ymax = ymax
        self.bc = bc
        self.kappa = kappa
        self.rho = rho
        self.nu = nu
        self.dp_dx = dp_dx
        self.delta = 1
        self.grid = torch.linspace(
            ymin,
            ymax,
            self.n,
            dtype=torch.float,
            requires_grad=True
        ).reshape(-1, 1)
        self.spacing = self.grid[1, 0] - self.grid[0, 0]

    def get_grid(self):
        return self.grid

    def get_grid_sample(self):
        return self.sample_grid(self.grid, self.spacing)

    def get_solution(self, y, max_nodes=1000, tol=1e-3):
        try:
            y = y.detach().numpy() # if torch tensor, convert to numpy
        except:
            pass

        y = y.reshape(-1)

        res = solve_rans_scipy_solve_bvp(y, k=self.kappa, nu=self.nu, rho=self.rho,
            dpdx=self.dp_dx, delta=self.delta, max_nodes=max_nodes, tol=tol)
        soln = res.sol(y)[0]
        return torch.tensor(soln, dtype=torch.float).reshape(-1,1)

    def _reynolds_stress(self, y, du_dy):
        a = self.kappa * (torch.abs(y)-self.delta) # / (2*self.delta)
        return -(a ** 2) * torch.abs(du_dy) * du_dy

    def _rans_eqn(self, dre, d2u):
        return self.nu * d2u - dre - (1/self.rho) * self.dp_dx

    def adjust(self, y, u):
        a = self.bc[0]
        b = (self.bc[1]-self.bc[0]) * (y - self.ymin)
        c = self.ymax - self.ymin
        d = (y - self.ymin)*(y - self.ymax) * u
        u_adj = a + b/c + d
        du = diff(u_adj, y)
        dre = diff(self._reynolds_stress(y, du), y)
        d2u = diff(du, y)
        return {'pred': u_adj, 'dre': dre, 'd2u': d2u}

    def get_equation(self, y, u):
        adj = self.adjust(y, u)
        uadj, dre, d2u = adj['pred'], adj['dre'], adj['d2u']
        return self._rans_eqn(dre, d2u)

    def get_plot_dicts(self, u, y, sol):
        adj = self.adjust(y, u)
        uadj, dre, d2u = adj['pred'], adj['dre'], adj['d2u']
        pred_dict = {'$\hat{u}$': uadj.detach(), '$u$': sol.detach()}
        diff_dict = None
        return pred_dict, diff_dict

# make a function to set global state
FLOAT_DTYPE=torch.float32

def set_default_dtype(dtype):
    """Set the default `dtype` of `torch.tensor` used in `neurodiffeq`.
    :param dtype: `torch.float`, `torch.double`, etc
    """
    global FLOAT_DTYPE
    FLOAT_DTYPE=dtype
    torch.set_default_dtype(FLOAT_DTYPE)

set_default_dtype(FLOAT_DTYPE)

class PoissonEquation(Problem):
    """
    Poisson Equation:

    $$ \nabla^{2} \varphi = f $$

    -Laplace(u) = f    in the unit square
              u = u_D  on the boundary

    u_D = 0
      f = 1

    NOTE: reduce to Laplace (f=0) for Analytical Solution

    thanks to Feiyu Chen for this:
    https://github.com/odegym/neurodiffeq/blob/master/neurodiffeq/pde.py

    d2u_dx2 + d2u_dy2 = 0
    with (x, y) in [0, 1] x [0, 1]

    Boundary conditions:
    u(x,y) | x=0 : sin(pi * y)
    u(x,y) | x=1 : 0
    u(x,y) | y=0 : 0
    u(x,y) | y=1 : 0

    Solution:
    u(x,y) = sin(pi * y) * sinh( pi * (1 - x) ) / sinh(pi)
    """
    def __init__(self, nx=32, ny=32, xmin=0, xmax=1, ymin=0, ymax=1, batch_size=100, **kwargs):
        super().__init__(**kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nx = nx
        self.ny = ny
        self.batch_size = batch_size
        self.pi = torch.tensor(np.pi)
        self.hx = (xmax - xmin) / nx
        self.hy = (ymax - ymin) / ny
        self.noise_xstd = self.hx / 4.0
        self.noise_ystd = self.hy / 4.0

        xgrid = torch.linspace(xmin, xmax, nx, requires_grad=True)
        ygrid = torch.linspace(ymin, ymax, ny, requires_grad=True)

        grid_x, grid_y = torch.meshgrid(xgrid, ygrid)
        self.grid_x, self.grid_y = grid_x.reshape(-1,1), grid_y.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x, self.grid_y)

    def get_grid_sample(self):
        x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
        y_noisy = torch.normal(mean=self.grid_y, std=self.noise_ystd)
        return (x_noisy, y_noisy)

    def get_solution(self, x, y):
        sol = x * (1-x) * y * (1-y) * torch.exp(x - y)
        return sol

    def _poisson_eqn(self, u, x, y):
        return diff(u, x, order=2) + diff(u, y, order=2) - 2*x * (y - 1) * (y - 2*x + x*y + 2) * torch.exp(x - y)

    def get_equation(self, u, x, y):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(u, x, y)
        u_adj = adj['pred']
        return self._poisson_eqn(u_adj, x, y)

    def adjust(self, u, x, y):
        """ perform boundary value adjustment

        thanks to Feiyu Chen for this:
        https://github.com/odegym/neurodiffeq/blob/master/neurodiffeq/pde.py
        """
        self.x_min_val = lambda y: 0
        self.x_max_val = lambda y: 0
        self.y_min_val = lambda x: 0
        self.y_max_val = lambda x: 0

        x_tilde = (x-self.xmin) / (self.xmax-self.xmin)
        y_tilde = (y-self.ymin) / (self.ymax-self.ymin)

        Axy = (1-x_tilde)*self.x_min_val(y) + x_tilde*self.x_max_val(y) + \
              (1-y_tilde)*( self.y_min_val(x) - ((1-x_tilde)*self.y_min_val(self.xmin * torch.ones_like(x_tilde))
                                                  + x_tilde *self.y_min_val(self.xmax * torch.ones_like(x_tilde))) ) + \
                 y_tilde *( self.y_max_val(x) - ((1-x_tilde)*self.y_max_val(self.xmin * torch.ones_like(x_tilde))
                                                  + x_tilde *self.y_max_val(self.xmax * torch.ones_like(x_tilde))) )

        u_adj = Axy + x_tilde*(1-x_tilde)*y_tilde*(1-y_tilde)*u

        return {'pred': u_adj}

    def get_plot_dicts(self, pred, x, y, sol):
        """ return appropriate pred_dict / diff_dict used for plotting """
        adj = self.adjust(pred, x, y)
        pred_adj = adj['pred']
        pred_dict = {'$\hat{u}$': pred_adj.detach()}

        resid = self.get_equation(pred, x, y)
        diff_dict = {'$|\hat{F}|$': np.abs(resid.detach())}
        return pred_dict, diff_dict

class SIRModel(Problem):
    """ SIR model for epidemiological spread of disease
        three outputs: S, I, R
        three equations: minimize residual sum
    """
    def __init__(self, t_min = 0, t_max = 10, S0 = 0.7,
        I0 = 0.3, R0 = 0, beta = 1, gamma = 1, **kwargs):
        """
        inputs:
            - t_min: start time
            - t_max: end time
            - dx_dt0: initial condition on dx_dt
            - kwargs: keyword args passed to Problem.__init__()
        """
        super().__init__(**kwargs)

        self.t_min = t_min
        self.t_max = t_max
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.N = S0 + I0 + R0
        self.beta = beta
        self.gamma = gamma
        self.grid = torch.linspace(
            t_min,
            t_max,
            self.n,
            dtype=torch.float,
            requires_grad=True
        ).reshape(-1, 1)
        self.spacing = self.grid[1, 0] - self.grid[0, 0]

        atol = 1e-8
        rtol = 1e-8
        self.sol = solve_ivp(
            self._sir_system,
            t_span = (t_min, t_max),
            y0 = [self.S0, self.I0, self.R0],
            dense_output=True,
            atol=atol,
            rtol=rtol,
        )

    def get_grid(self):
        return self.grid

    def get_grid_sample(self):
        return self.sample_grid(self.grid, self.spacing)

    def get_solution(self, t):
        """ uses scipy to solve """
        try:
            t = t.detach().numpy() # if torch tensor, convert to numpy
        except:
            pass

        t = t.reshape(-1)

        sol = self.sol.sol(t)
        return torch.tensor(sol.T, dtype=torch.float)

    def _sir_system(self, t, x):
        S, I, R = x[0], x[1], x[2]
        rhs1 = -self.beta*I*S/self.N
        rhs2 = (self.beta*I*S/self.N) - self.gamma*I
        rhs3 = self.gamma*I
        return np.array([rhs1, rhs2, rhs3])

    def _sir_eqn(self, t, x_adj):
        S_adj, I_adj, R_adj = x_adj[:,0], x_adj[:,1], x_adj[:,2]
        S_adj, I_adj, R_adj = S_adj.reshape(-1,1), I_adj.reshape(-1,1), R_adj.reshape(-1,1)

        eqn1 = diff(S_adj, t) + self.beta * I_adj * S_adj / self.N
        eqn2 = diff(I_adj, t) - (self.beta * I_adj * S_adj / self.N) + self.gamma * I_adj
        eqn3 = diff(R_adj, t) - self.gamma * I_adj
        return eqn1, eqn2, eqn3

    def get_equation(self, x, t):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(x, t)
        x_adj = adj['pred']
        eqn1, eqn2, eqn3 = self._sir_eqn(t, x_adj)
        # it's important to return concat here and NOT the sum
        # works much better (for point-wise loss)
        return torch.cat((eqn1, eqn2, eqn3), axis=1)

    def adjust(self, x, t):
        """ perform initial value adjustment """
        S, I, R = x[:, 0], x[:, 1], x[:, 2]

        S_adj = self.S0 + (1 - torch.exp(-t)) * S.reshape(-1,1)
        I_adj = self.I0 + (1 - torch.exp(-t)) * I.reshape(-1,1)
        R_adj = self.R0 + (1 - torch.exp(-t)) * R.reshape(-1,1)

        # the other problem classes return multiple elements here,
        # (e.g. x, dx, d2x) add a None here to mimic that,
        # although we don't need it per-se
        return {'pred': torch.cat((S_adj, I_adj, R_adj), axis=1)}

    def get_plot_dicts(self, x, t, y):
        """ return appropriate pred_dict and diff_dict used for plotting """
        adj = self.adjust(x, t)
        x_adj = adj['pred']
        S_adj, I_adj, R_adj = x_adj[:,0], x_adj[:,1], x_adj[:,2]
        S_adj, I_adj, R_adj = S_adj.reshape(-1,1), I_adj.reshape(-1,1), R_adj.reshape(-1,1)
        S_true, I_true, R_true = y[:, 0], y[:, 1], y[:, 2]
        pred_dict = {'$\hat{S}$': S_adj.detach(), '$S$': S_true.detach(),
                     '$\hat{I}$': I_adj.detach(), '$I$': I_true.detach(),
                     '$\hat{R}$': R_adj.detach(), '$R$': R_true.detach(),}
        # diff_dict = None
        residuals = self.get_equation(x, t)
        r1, r2, r3 = residuals[:,0], residuals[:,1], residuals[:,2]
        diff_dict = {'$|\hat{F_1}|$': np.abs(r1.detach()),
                     '$|\hat{F_2}|$': np.abs(r2.detach()),
                     '$|\hat{F_3}|$': np.abs(r3.detach())}
        return pred_dict, diff_dict

class CoupledOscillator(Problem):
    """
    1) dx_dt = -ty
    2) dy_dt = tx

    Soln:
    x = cos(t^2 / 2)
    y = sin(t^2 / 2)
    """
    def __init__(self, t_min = 0, t_max = 6, x0 = 1, y0 = 0, **kwargs):
        """
        inputs:
            - t_min: start time
            - t_max: end time
            - x0: x initial condition
            - y0: y initial condition
            - kwargs: keyword args passed to Problem.__init__()
        """
        super().__init__(**kwargs)

        self.t_min = t_min
        self.t_max = t_max
        self.x0 = x0
        self.y0 = y0
        self.grid = torch.linspace(
            t_min,
            t_max,
            self.n,
            dtype=torch.float,
            requires_grad=True
        ).reshape(-1, 1)
        self.spacing = self.grid[1, 0] - self.grid[0, 0]

    def get_grid(self):
        return self.grid

    def get_grid_sample(self):
        return self.sample_grid(self.grid, self.spacing)

    def get_solution(self, t):
        """ use analytical solution """
        t = t.reshape(-1, 1)
        xsol = torch.cos((t**2) / 2)
        ysol = torch.sin((t**2) / 2)
        return torch.cat((xsol, ysol), axis=1)

    def _co_eqn(self, t, sol_adj):
        """ compute residuals LHS """
        x_adj, y_adj = sol_adj[:, 0].reshape(-1,1), sol_adj[:, 1].reshape(-1,1)
        eqn1 = diff(x_adj, t) + t * y_adj
        eqn2 = diff(y_adj, t) - t * x_adj
        return eqn1, eqn2

    def get_equation(self, sol, t):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(sol, t)
        pred_adj = adj['pred']
        eqn1, eqn2 = self._co_eqn(t, pred_adj)
        # it's important to return concat here and NOT the sum
        # works much better (for point-wise loss)
        return torch.cat((eqn1, eqn2), axis=1)

    def adjust(self, sol, t):
        """ perform initial value adjustment """
        x, y = sol[:, 0], sol[:, 1]

        x_adj = self.x0 + (1 - torch.exp(-t)) * x.reshape(-1,1)
        y_adj = self.y0 + (1 - torch.exp(-t)) * y.reshape(-1,1)

        # the other problem classes return multiple elements here,
        # (e.g. x, dx, d2x) add a None here to mimic that,
        # although we don't need it per-se
        return {'pred': torch.cat((x_adj, y_adj), axis=1)}

    def get_plot_dicts(self, sol, t, true):
        """ return appropriate pred_dict and diff_dict used for plotting """
        adj = self.adjust(sol, t)['pred']
        x_adj, y_adj = adj[:,0].reshape(-1, 1), adj[:,1].reshape(-1, 1)
        x_true, y_true = true[:, 0], true[:, 1]
        pred_dict = {'$\hat{x}$': x_adj.detach(), '$x$': x_true.detach(),
                     '$\hat{y}$': y_adj.detach(), '$y$': y_true.detach(),}
        # diff_dict = None
        residuals = self.get_equation(sol, t)
        r1, r2 = residuals[:,0], residuals[:,1]
        diff_dict = {'$|\hat{F_1}|$': np.abs(r1.detach()),
                     '$|\hat{F_2}|$': np.abs(r2.detach())}
        return pred_dict, diff_dict

if __name__ == "__main__":
    import denn.utils as ut
    import matplotlib.pyplot as plt
    print("Testing CoupledOscillator")
    co = CoupledOscillator()
    t = co.get_grid()
    s = co.get_solution(t)
    adj = co.adjust(s, t)['pred']
    res = co.get_equation(adj, t)

    plt_dict = co.get_plot_dicts(adj, t, s)

    t = t.detach()
    s = s.detach()
    adj = adj.detach()
    res = res.detach()

    plt.plot(t, s[:,0])
    plt.plot(t, s[:,1])
    plt.plot(t, adj[:, 0])
    plt.plot(t, adj[:, 1])
    plt.plot(t, res[:,0])
    plt.plot(t, res[:,1])
    plt.show()

# if __name__ == '__main__':
#     import denn.utils as ut
#     import matplotlib.pyplot as plt
#     print('Testing SIR Model')
#     for i, b in enumerate(np.linspace(0.5,4,20)):
#         sir = SIRModel(n=100, S0=0.99, I0=0.01, R0=0.0, beta=b, gamma=1)
#         t = sir.get_grid()
#         sol = sir.get_solution(t)
#
#         # plot
#         t = t.detach()
#         sol = sol.detach()
#         a=0.5
#         if i == 0:
#             plt.plot(t, sol[:,0], alpha=a, label='Susceptible', color='crimson')
#             plt.plot(t, sol[:,1], alpha=a, label='Infected', color='blue')
#             plt.plot(t, sol[:,2], alpha=a, label='Recovered', color='aquamarine')
#         else:
#             plt.plot(t, sol[:,0], alpha=a, color='crimson')
#             plt.plot(t, sol[:,1], alpha=a, color='blue')
#             plt.plot(t, sol[:,2], alpha=a, color='aquamarine')
#
#     plt.title('Flattening the Curve')
#     plt.xlabel('Time')
#     plt.ylabel('Proportion of Population')
#
#     plt.axhline(0.3, label='Capacity', color='k', linestyle='--', alpha=a)
#     plt.legend()
#     plt.show()
