import numpy as np
import torch
from denn.problem import Problem
from scipy.integrate import solve_ivp
from denn.utils import diff
from denn.rans.numerical import solve_rans_scipy_solve_bvp
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class Exponential(Problem):
    """
    Equation:
    x' + Lx = 0

    Analytic Solution:
    x = exp(-Lt)
    """
    def __init__(self, t_min = 0, t_max = 10, x0 = 1., L = 1, shuheng = False, **kwargs):
        """
        inputs:
            - t_min: start time
            - t_max: end time
            - x0: initial condition on x
            - L: rate of decay constant
            - shuheng: whether or not to use Shuheng's reparameterization
            - kwargs: keyword args passed to Problem.__init__()
        """
        super().__init__(**kwargs)

        self.t_min = t_min
        self.t_max = t_max
        self.x0 = x0
        self.L = L
        self.shuheng = shuheng
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

    def get_equation(self, x, t, G=None):
        """ return value of residuals of equation (i.e. LHS) """
        if self.shuheng:
            adj = self.adjust(x, t, G)
        else:
            adj = self.adjust(x, t)
        x, dx = adj['pred'], adj['dx']
        return dx + self.L * x

    def adjust(self, x, t, G=None):
        """ perform initial value adjustment """
        if self.shuheng:
            t_0 = - torch.log(torch.FloatTensor([self.x0])) / self.L
            x_adj = self.x0 + x - G(t_0)
        else:
            x_adj = self.x0 + (1 - torch.exp(-t)) * x
        dx_dt = diff(x_adj, t)
        return {'pred': x_adj, 'dx': dx_dt}

    def get_plot_dicts(self, x, t, y, G):
        """ return appropriate pred_dict and diff_dict used for plotting """
        if self.shuheng:
            adj = self.adjust(x, t, G)
            residual = self.get_equation(x, t, G)
        else:
            adj = self.adjust(x, t)
            residual = self.get_equation(x, t)
        xadj, dx = adj['pred'], adj['dx']
        pred_dict = {'$\hat{x}$': xadj.detach(), '$x$': y.detach()}
        # diff_dict = {'$\hat{x}$': xadj.detach(), '$-\hat{\dot{x}}$': (-dx).detach()}
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

    def get_equation(self, x, t, G=None):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(x, t)
        x, dx, d2x = adj['pred'], adj['dx'], adj['d2x']
        return d2x + x

    def adjust(self, x, t, G=None):
        """ perform initial value adjustment """
        x_adj = self.x0 + (1 - torch.exp(-t)) * self.dx_dt0 + ((1 - torch.exp(-t))**2) * x
        dx_dt = diff(x_adj, t)
        d2x_dt2 = diff(dx_dt, t)
        return {'pred': x_adj, 'dx': dx_dt, 'd2x': d2x_dt2}

    def get_plot_dicts(self, x, t, y, G):
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

    def get_equation(self, x, t, G=None):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(x, t)
        x, dx, d2x = adj['pred'], adj['dx'], adj['d2x']
        return self._nlo_eqn(x, dx, d2x)

    def adjust(self, x, t, G=None):
        """ perform initial value adjustment """
        x_adj = self.x0 + (1 - torch.exp(-t)) * self.dx_dt0 + ((1 - torch.exp(-t))**2) * x
        dx = diff(x_adj, t)
        d2x = diff(dx, t)
        return {'pred': x_adj, 'dx': dx, 'd2x': d2x}

    def get_plot_dicts(self, x, t, y, G):
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

    def adjust(self, y, u, G=None):
        a = self.bc[0]
        b = (self.bc[1]-self.bc[0]) * (y - self.ymin)
        c = self.ymax - self.ymin
        d = (y - self.ymin)*(y - self.ymax) * u
        u_adj = a + b/c + d
        du = diff(u_adj, y)
        dre = diff(self._reynolds_stress(y, du), y)
        d2u = diff(du, y)
        return {'pred': u_adj, 'dre': dre, 'd2u': d2u}

    def get_equation(self, y, u, G=None):
        adj = self.adjust(y, u)
        uadj, dre, d2u = adj['pred'], adj['dre'], adj['d2u']
        return self._rans_eqn(dre, d2u)

    def get_plot_dicts(self, u, y, sol, G):
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

    def get_equation(self, x, t, G=None):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(x, t)
        x_adj = adj['pred']
        eqn1, eqn2, eqn3 = self._sir_eqn(t, x_adj)
        # it's important to return concat here and NOT the sum
        # works much better (for point-wise loss)
        return torch.cat((eqn1, eqn2, eqn3), axis=1)

    def adjust(self, x, t, G=None):
        """ perform initial value adjustment """
        S, I, R = x[:, 0], x[:, 1], x[:, 2]

        S_adj = self.S0 + (1 - torch.exp(-t)) * S.reshape(-1,1)
        I_adj = self.I0 + (1 - torch.exp(-t)) * I.reshape(-1,1)
        R_adj = self.R0 + (1 - torch.exp(-t)) * R.reshape(-1,1)

        # the other problem classes return multiple elements here,
        # (e.g. x, dx, d2x) add a None here to mimic that,
        # although we don't need it per-se
        return {'pred': torch.cat((S_adj, I_adj, R_adj), axis=1)}

    def get_plot_dicts(self, x, t, y, G):
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

    def get_equation(self, sol, t, G=None):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(sol, t)
        pred_adj = adj['pred']
        eqn1, eqn2 = self._co_eqn(t, pred_adj)
        # it's important to return concat here and NOT the sum
        # works much better (for point-wise loss)
        return torch.cat((eqn1, eqn2), axis=1)

    def adjust(self, sol, t, G=None):
        """ perform initial value adjustment """
        x, y = sol[:, 0], sol[:, 1]

        x_adj = self.x0 + (1 - torch.exp(-t)) * x.reshape(-1,1)
        y_adj = self.y0 + (1 - torch.exp(-t)) * y.reshape(-1,1)

        # the other problem classes return multiple elements here,
        # (e.g. x, dx, d2x) add a None here to mimic that,
        # although we don't need it per-se
        return {'pred': torch.cat((x_adj, y_adj), axis=1)}

    def get_plot_dicts(self, sol, t, true, G):
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

class EinsteinEquations(Problem):
    """ Hu-Sawicky f(R) motified Einstein equations
        five outputs: x, y, v, Om, r
        five equations: minimize residual sum

        We use the variable change z' = (-z + z_0)/z_0 proposed
        by Augusto Chantada to obtain the system below.

        dx/dz = -z_0/(-z_0(z-1) + 1) (-Om -2v + x + 4y + xv + x^2)
        dy/dz = -z_0/(-z_0(z-1) + 1) (vx Gam(r) - xy + 4y - 2yv)
        dv/dz = z_0 v/(-z_0(z-1) + 1) (x Gam(r) + 4 - 2v)
        dOm/dz = z_0 Om/(-z_0(z-1) + 1) (-1 + 2v + x)
        dr/dz = -z_0 r Gam(r) x / (-z_0(z-1) + 1)
    """
    def __init__(self, z_0 = 10, Om_m_0 = 0.15, b = 5, **kwargs):
        """
        inputs:
        """
        super().__init__(**kwargs)
        self.z_0 = z_0
        self.Om_m_0 = Om_m_0
        self.b = b
        self.Om_L_0 = 1 - Om_m_0
        self.z_prime_0 = self._z_to_z_prime(z_0)
        self.z_prime_f = self._z_to_z_prime(0)
        self.dz_prime_dz = -1/z_0
        self.grid = torch.linspace(
            self.z_prime_0,
            self.z_prime_f,
            self.n,
            dtype=torch.float,
            requires_grad=True
        ).reshape(-1, 1)
        self.spacing = self.grid[1, 0] - self.grid[0, 0]

        atol = 1e-16
        rtol = 1e-11
        self.sol = solve_ivp(
            self._hu_sawicky_system,
            t_span = (self.z_0, 0),
            y0 = [self.x_0_condition(self.z_prime_0), 
                self.y_0_condition(self.z_prime_0), 
                self.v_0_condition(self.z_prime_0), 
                self.Om_0_condition(self.z_prime_0), 
                self._r_prime_to_r(self.r_prime_0_condition(self.z_prime_0))],
            dense_output=True,
            atol=atol,
            rtol=rtol,
        )

    def get_grid(self):
        return self.grid

    def get_grid_sample(self):
        return self.sample_grid(self.grid, self.spacing)

    def get_solution(self, z):
        """ uses scipy to solve """
        try:
            z = z.detach().numpy() # if torch tensor, convert to numpy
        except:
            pass

        z = self._z_prime_to_z(z).reshape(-1)
        x_num, y_num, v_num, Om_num, r_num = [s[::-1] for s in self.sol.sol(z)]
        r_prime_num = self._r_to_r_prime(r_num)
        return torch.tensor([x_num, y_num, v_num, Om_num, r_prime_num], dtype=torch.float).T

    def _z_to_z_prime(self, z):
        return 1 - z/self.z_0

    def _z_prime_to_z(self, z_prime):
        return self.z_0*(1 - z_prime)

    def _r_to_r_prime(self, r):
        if isinstance(r, torch.Tensor):
            r_prime = torch.log(r)
        else:
            r_prime = np.log(r)
        return r_prime

    def _r_prime_to_r(self, r_prime):
        if isinstance(r_prime, torch.Tensor):
            r = torch.exp(r_prime)
        else:
            r = np.exp(r_prime)
        return r

    def x_0_condition(self, z_prime):
        return 0.0
    
    def y_0_condition(self, z_prime):
        z = self._z_prime_to_z(z_prime)
        return (self.Om_m_0*((1 + z)**3) + 2*self.Om_L_0)/(2*(self.Om_m_0*((1 + z)**3) + self.Om_L_0))

    def v_0_condition(self, z_prime):
        z = self._z_prime_to_z(z_prime)
        return (self.Om_m_0*((1 + z)**3) + 4*self.Om_L_0)/(2*(self.Om_m_0*((1 + z)**3) + self.Om_L_0))

    def Om_0_condition(self, z_prime):
        z = self._z_prime_to_z(z_prime)
        return self.Om_m_0*((1 + z)**3)/((self.Om_m_0*((1 + z)**3) + self.Om_L_0))

    def r_prime_0_condition(self, z_prime):
        z = self._z_prime_to_z(z_prime)
        r_0 =  (self.Om_m_0*((1 + z)**3) + 4*self.Om_L_0)/self.Om_L_0
        r_prime_0 = self._r_to_r_prime(r_0)
        return r_prime_0

    def _hu_sawicky_system(self, z, vars):
        x, y, v, Om, r = vars
        Gamma = (r + self.b)*(((r + self.b)**2) - 2*self.b) / (4*r*self.b)
        s0 = (-Om + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
        s1 = (- (v*x*Gamma - x*y + 4*y - 2*y*v)) / (z+1)
        s2 = (-v * (x*Gamma + 4 - 2*v)) / (z+1)
        s3 = (Om * (-1 + x + 2*v)) / (z+1)
        s4 = (-(x * r * Gamma)) / (1+z)
        return [s0, s1, s2, s3, s4]

    def _hu_sawicky_eqn(self, z_prime, u_adj):
        x_adj, y_adj, v_adj, Om_adj, r_prime_adj = u_adj[:,0], u_adj[:,1], u_adj[:,2], u_adj[:,3], u_adj[:,4]
        x_adj, y_adj, v_adj, Om_adj, r_prime_adj = x_adj.reshape(-1,1), y_adj.reshape(-1,1), v_adj.reshape(-1,1), Om_adj.reshape(-1,1), r_prime_adj.reshape(-1,1)
        #Gamma = (torch.exp(r_prime_adj) + self.b)*(((torch.exp(r_prime_adj) + self.b)**2) - 2*self.b)/(4*self.b*torch.exp(r_prime_adj))
        #z = self._z_prime_to_z(z_prime)
        r_adj = self._r_prime_to_r(r_prime_adj)
        Gamma = (r_adj + self.b)*(((r_adj + self.b)**2) - 2*self.b)/(4*r_adj*self.b)
        
        #eqn1 = diff(x_adj, z_prime) * self.dz_prime_dz - (-Om_adj - 2*v_adj + x_adj + 4*y_adj + x_adj*v_adj + x_adj**2)/(z + 1)
        #eqn2 = diff(y_adj, z_prime) * self.dz_prime_dz + (v_adj*x_adj*Gamma - x_adj*y_adj + 4*y_adj - 2*y_adj*v_adj)/(z + 1)
        #eqn3 = diff(v_adj, z_prime) * self.dz_prime_dz + v_adj*(x_adj*Gamma + 4 - 2*v_adj)/(z + 1)
        #eqn4 = diff(Om_adj, z_prime) * self.dz_prime_dz - Om_adj*(-1 + 2*v_adj + x_adj)/(z + 1)
        #eqn5 = diff(r_prime_adj, z_prime) * self.dz_prime_dz + (Gamma*x_adj)/(z + 1)

        eqn1 = diff(x_adj, z_prime) + self.z_0*(-Om_adj - 2*v_adj + x_adj + 4*y_adj + x_adj*v_adj + x_adj**2)/(-self.z_0*(z_prime - 1) + 1)
        eqn2 = diff(y_adj, z_prime) + self.z_0*(v_adj*x_adj*Gamma - x_adj*y_adj + 4*y_adj - 2*y_adj*v_adj)/(-self.z_0*(z_prime - 1) + 1)
        eqn3 = diff(v_adj, z_prime) - self.z_0*v_adj*(x_adj*Gamma + 4 - 2*v_adj)/(-self.z_0*(z_prime - 1) + 1)
        eqn4 = diff(Om_adj, z_prime) + self.z_0*Om_adj*(-1 + 2*v_adj + x_adj)/(-self.z_0*(z_prime - 1) + 1)
        eqn5 = diff(r_prime_adj, z_prime) + self.z_0*(Gamma*x_adj)/(-self.z_0*(z_prime - 1) + 1)
        return eqn1, eqn2, eqn3, eqn4, eqn5

    def get_equation(self, u, z_prime, G=None):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(u, z_prime)
        u_adj = adj['pred']
        eqn1, eqn2, eqn3, eqn4, eqn5 = self._hu_sawicky_eqn(z_prime, u_adj)
        return torch.cat((eqn1, eqn2, eqn3, eqn4, eqn5), axis=1)

    def adjust(self, u, z_prime, G=None):
        """ perform initial value adjustment """
        x, y, v, Om, r_prime = u[:, 0], u[:, 1], u[:, 2], u[:, 3], u[:, 4]

        x_adj = self.x_0_condition(z_prime) + (1 - torch.exp(-z_prime + self.z_prime_0))*x.reshape(-1,1)*(1 - torch.exp(-torch.tensor((1/30)*self.b)))
        y_adj = self.y_0_condition(z_prime) + (1 - torch.exp(-z_prime + self.z_prime_0))*y.reshape(-1,1)*(1 - torch.exp(-torch.tensor((1/30)*self.b)))
        v_adj = self.v_0_condition(z_prime) + (1 - torch.exp(-z_prime + self.z_prime_0))*v.reshape(-1,1)*(1 - torch.exp(-torch.tensor((1/30)*self.b)))
        Om_adj = self.Om_0_condition(z_prime) + (1 - torch.exp(-z_prime + self.z_prime_0))*Om.reshape(-1,1)*(1 - torch.exp(-torch.tensor((1/30)*self.b)))
        r_prime_adj = self.r_prime_0_condition(z_prime) + (1 - torch.exp(-z_prime + self.z_prime_0))*r_prime.reshape(-1,1)*(1 - torch.exp(-torch.tensor((1/30)*self.b)))

        return {'pred': torch.cat((x_adj, y_adj, v_adj, Om_adj, r_prime_adj), axis=1)}

    def get_plot_dicts(self, u, z_prime, sol, G):
        """ return appropriate pred_dict and diff_dict used for plotting """
        adj = self.adjust(u, z_prime)
        u_adj = adj['pred']
        x_adj, y_adj, v_adj, Om_adj, r_prime_adj = u_adj[:,0], u_adj[:,1], u_adj[:,2], u_adj[:,3], u_adj[:,4]
        x_adj, y_adj, v_adj, Om_adj, r_prime_adj = x_adj.reshape(-1,1), y_adj.reshape(-1,1), v_adj.reshape(-1,1), Om_adj.reshape(-1,1), r_prime_adj.reshape(-1,1)
        #r_adj = self._r_prime_to_r(r_prime_adj)
        x_true, y_true, v_true, Om_true, r_prime_true = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
        pred_dict = {'$\hat{x}$': x_adj.detach(), '$x$': x_true.detach(),
                     '$\hat{y}$': y_adj.detach(), '$y$': y_true.detach(),
                     '$\hat{v}$': v_adj.detach(), '$v$': v_true.detach(),
                     '$\hat{\Omega}$': Om_adj.detach(), '$\Omega$': Om_true.detach(),
                     "$\hat{r'}$": r_prime_adj.detach(), "$r'$": r_prime_true.detach()}
        residuals = self.get_equation(u, z_prime)
        r1, r2, r3, r4, r5 = residuals[:,0], residuals[:,1], residuals[:,2], residuals[:,3], residuals[:,4]
        diff_dict = {'$|\hat{F_1}|$': np.abs(r1.detach()),
                     '$|\hat{F_2}|$': np.abs(r2.detach()),
                     '$|\hat{F_3}|$': np.abs(r3.detach()),
                     '$|\hat{F_4}|$': np.abs(r4.detach()),
                     '$|\hat{F_5}|$': np.abs(r5.detach())}
        return pred_dict, diff_dict

if __name__ == "__main__":
    import denn.utils as ut
    import matplotlib.pyplot as plt

    print('Testing Hu-Sawicky Model')
    model = EinsteinEquations(n=100, z_0 = 10, Om_m_0 = 0.15, b = 5)
    z_prime = model.get_grid()
    sol = model.get_solution(z_prime)
    adj = model.adjust(sol, z_prime)['pred']
    res = model.get_equation(adj, z_prime)
    pred_dict, diff_dict = model.get_plot_dicts(adj, z_prime, sol, None)
    z_prime = z_prime.detach()
    sol = sol.detach()
    fig, axs = plt.subplots(1, 5, figsize=(14,5))
    for i in range(5):
        axs[i].plot(z_prime, sol[:,i])
    plt.show()

    #x, t = np.linspace(0,1,32), np.linspace(0,1,32)
    #xx, tt = np.meshgrid(x, t)
    #sol = be.get_solution(xx, tt)
    #fig, ax = plt.subplots(figsize=(10,7))
    #ax.contourf(xx, tt, sol, cmap="Reds")
    #plt.show()

# if __name__ == '__main__':
# print("Testing CoupledOscillator")
# co = CoupledOscillator()
# t = co.get_grid()
# s = co.get_solution(t)
# adj = co.adjust(s, t)['pred']
# res = co.get_equation(adj, t)

# plt_dict = co.get_plot_dicts(adj, t, s)

# t = t.detach()
# s = s.detach()
# adj = adj.detach()
# res = res.detach()

# plt.plot(t, s[:,0])
# plt.plot(t, s[:,1])
# plt.plot(t, adj[:, 0])
# plt.plot(t, adj[:, 1])
# plt.plot(t, res[:,0])
# plt.plot(t, res[:,1])
# plt.show()
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