import numpy as np
from scipy.integrate._ivp.radau import P
import torch
from denn.problem import Problem
from denn.utils import diff
from denn.burgers.fft_burgers import fft_burgers
from denn.allen_cahn.fft_allen_cahn import fft_allen_cahn

class PoissonEquation(Problem):
    """
    Poisson Equation:

    $$ \nabla^{2} \varphi = f $$

    -Laplace(u) = f    in the unit square
              u = u_D  on the boundary

    u_D = 0
      f = 2x * (y-1) * (y-2x+x*y+2) * exp(x-y)

    NOTE: reduce to Laplace (f=0) for Analytical Solution

    thanks to Feiyu Chen for this:
    https://github.com/odegym/neurodiffeq/blob/master/neurodiffeq/pde.py

    d2u_dx2 + d2u_dy2 = 0
    with (x, y) in [0, 1] x [0, 1]

    Boundary conditions:
    u(x,y) | x=0 : 0
    u(x,y) | x=1 : 0
    u(x,y) | y=0 : 0
    u(x,y) | y=1 : 0

    Solution:
    u(x,y) = x * (1-x) * y * (1-y) * exp(x-y)
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

        grid_x, grid_y = torch.meshgrid(xgrid, ygrid, indexing='ij')
        self.grid_x, self.grid_y = grid_x.reshape(-1,1), grid_y.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x, self.grid_y)

    def get_grid_sample(self, grid, resid, resid_delta):
        x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
        y_noisy = torch.normal(mean=self.grid_y, std=self.noise_ystd)
        return (x_noisy, y_noisy)

    def get_plot_grid(self):
        return self.get_grid()

    def get_plot_dims(self):
        return {'x': self.nx, 'y': self.ny}

    def get_solution(self, x, y):
        sol = x * (1-x) * y * (1-y) * torch.exp(x - y)
        return sol

    def get_plot_solution(self, x, y):
        return self.get_solution(x, y)

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

class WaveEquation(Problem):
    """
    Wave Equation:

    $$ u_{tt} = c^2 u_{xx} $$

    Boundary conditions:
    u(x,t)     | x=0 : 0
    u(x,t)     | x=1 : 0

    Initial conditions:
    u(x,t)     | t=0 : sin(pi*x)
    du_dt(x,t) | t=0 : 0

    Solution:
    u(x,t) = cos(c*pi*t) sin(pi*x)
    """
    def __init__(self, nx=32, nt=32, c=1, xmin=0, xmax=1, tmin=0, tmax=1, **kwargs):
        super().__init__(**kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.tmin = tmin
        self.tmax = tmax
        self.nx = nx
        self.nt = nt
        self.c = c
        self.pi = torch.tensor(np.pi)
        self.hx = (xmax - xmin) / nx
        self.ht = (tmax - tmin) / nt
        self.noise_xstd = self.hx / 4.0
        self.noise_tstd = self.ht / 4.0

        xgrid = torch.linspace(xmin, xmax, nx, requires_grad=True)
        tgrid = torch.linspace(tmin, tmax, nt, requires_grad=True)

        grid_x, grid_t = torch.meshgrid(xgrid, tgrid, indexing='ij')
        self.grid_x, self.grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x, self.grid_t)

    def get_grid_sample(self, grid, resid, resid_delta):
        x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
        t_noisy = torch.normal(mean=self.grid_t, std=self.noise_tstd)
        return (x_noisy, t_noisy)

    def get_plot_grid(self):
        return self.get_grid()

    def get_plot_dims(self):
        return {'x': self.nx, 't': self.nt}

    def get_solution(self, x, t):
        sol = torch.cos(self.c * self.pi * t) * torch.sin(self.pi * x)
        return sol

    def get_plot_solution(self, x, t):
        return self.get_solution(x, t)

    def _wave_eqn(self, u, x, t):
        return diff(u, t, order=2) - (self.c**2)*diff(u, x, order=2)

    def get_equation(self, u, x, t):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(u, x, t)
        u_adj = adj['pred']
        return self._wave_eqn(u_adj, x, t)

    def adjust(self, u, x, t):
        """ perform boundary value adjustment """
        x_tilde = (x-self.xmin) / (self.xmax-self.xmin)
        t_tilde = (t-self.tmin) / (self.tmax-self.tmin)
        Axt = torch.sin(self.pi * x)

        u_adj = Axt + x_tilde*(1-x_tilde)*(1 - torch.exp(-t_tilde**2))*u

        return {'pred': u_adj}

    def get_plot_dicts(self, pred, x, t, sol):
        """ return appropriate pred_dict / diff_dict used for plotting """
        adj = self.adjust(pred, x, t)
        pred_adj = adj['pred']
        pred_dict = {'$\hat{u}$': pred_adj.detach()}

        resid = self.get_equation(pred, x, t)
        diff_dict = {'$|\hat{F}|$': np.abs(resid.detach())}
        return pred_dict, diff_dict

class BurgersEquation(Problem):
    """
    Inviscid Burgers Equation:

    u_t + u u_x = 0

    with u(x,t=0) = ax + b

    Solution:
    u(x,t) = (ax + b) / (at + 1)
    """
    def __init__(self, nx=32, nt=32, a=1, b=1, xmin=0, xmax=1, tmin=0, tmax=1, **kwargs):
        super().__init__(**kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.tmin = tmin
        self.tmax = tmax
        self.nx = nx
        self.nt = nt
        self.a = a
        self.b = b
        self.hx = (xmax - xmin) / nx
        self.ht = (tmax - tmin) / nt
        self.noise_xstd = self.hx / 4.0
        self.noise_tstd = self.ht / 4.0

        xgrid = torch.linspace(xmin, xmax, nx, requires_grad=True)
        tgrid = torch.linspace(tmin, tmax, nt, requires_grad=True)

        grid_x, grid_t = torch.meshgrid(xgrid, tgrid, indexing='ij')
        self.grid_x, self.grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x, self.grid_t)

    def get_grid_sample(self):
        x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
        t_noisy = torch.normal(mean=self.grid_t, std=self.noise_tstd)
        return (x_noisy, t_noisy)

    def get_plot_grid(self):
        return self.get_grid()

    def get_plot_dims(self):
        return {'x': self.nx, 't': self.nt}

    def get_solution(self, x, t):
        sol = (self.a * x + self.b) / (self.a * t + 1)
        return sol

    def get_plot_solution(self, x, t):
        return self.get_solution(x, t)

    def _burgers_eqn(self, u, x, t):
        return diff(u, t, order=1) + u*diff(u, x, order=1)

    def get_equation(self, u, x, t):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(u, x, t)
        u_adj = adj['pred']
        return self._burgers_eqn(u_adj, x, t)

    def adjust(self, u, x, t):
        """ perform boundary value adjustment """
        x_tilde = (x-self.xmin) / (self.xmax-self.xmin)
        t_tilde = (t-self.tmin) / (self.tmax-self.tmin)
        Axt = self.a * x + self.b

        u_adj = Axt + (1 - torch.exp(-t_tilde))*u

        return {'pred': u_adj}

    def get_plot_dicts(self, pred, x, t, sol):
        """ return appropriate pred_dict / diff_dict used for plotting """
        adj = self.adjust(pred, x, t)
        pred_adj = adj['pred']
        pred_dict = {'$\hat{u}$': pred_adj.detach()}

        resid = self.get_equation(pred, x, t)
        diff_dict = {'$|\hat{F}|$': np.abs(resid.detach())}
        return pred_dict, diff_dict

class BurgersViscous(Problem):
    """
    Burgers Equation with viscocity:

    u_t + u u_x - \nu u_{xx} = 0

    Boundary conditions:
    u(x,t)     | x=-5 : 0
    u(x,t)     | x=5  : 0

    Initial condition:
    u(x,t)     | t=0   : 1/cosh(x)
    """
    def __init__(self, nx=64, nt=64, nu=0.001, xmin=-5, xmax=5, tmin=0, tmax=2.5,
        xmin_p=-5, xmax_p=5, tmin_p=0, tmax_p=2.5, nx_p=1000, nt_p=100, **kwargs):
        super().__init__(**kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.tmin = tmin
        self.tmax = tmax
        self.nx = nx
        self.nt = nt
        self.nu = nu
        self.pi = torch.tensor(np.pi)
        self.hx = (xmax - xmin) / nx
        self.ht = (tmax - tmin) / nt
        self.noise_xstd = self.hx / 4.0
        self.noise_tstd = self.ht / 4.0
        self.xgrid = torch.linspace(xmin, xmax, nx+1, requires_grad=True)
        self.tgrid = torch.linspace(tmin, tmax, nt+1, requires_grad=True)
        grid_x, grid_t = torch.meshgrid(self.xgrid, self.tgrid, indexing='ij')
        self.grid_x, self.grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)

        self.xmin_p = xmin_p
        self.xmax_p = xmax_p
        self.tmin_p = tmin_p
        self.tmax_p = tmax_p
        self.nx_p = nx_p
        self.nt_p = nt_p
        self.xgrid_p = torch.linspace(xmin_p, xmax_p, self.nx_p, requires_grad=True)
        self.tgrid_p = torch.linspace(tmin_p, tmax_p, self.nt_p, requires_grad=True)
        grid_x_p, grid_t_p = torch.meshgrid(self.xgrid_p, self.tgrid_p, indexing='ij')
        self.grid_x_p, self.grid_t_p = grid_x_p.reshape(-1,1), grid_t_p.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x, self.grid_t)

    def get_grid_sample(self, t=None, resid=None, resid_delta=None):
        x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
        t_noisy = torch.normal(mean=self.grid_t, std=self.noise_tstd)
        return (x_noisy, t_noisy)

    def get_plot_grid(self):
        return (self.grid_x_p.float(), self.grid_t_p.float())

    def get_plot_dims(self):
        return {'x': self.nx_p, 't': self.nt_p}

    def get_solution(self, x, t):
        """ use FFT method """
        x = np.linspace(self.xmin, self.xmax, (self.nx*8)+1)
        t = np.linspace(self.tmin, self.tmax, self.nt+1)
        sol = fft_burgers(x, t, self.nu)
        sol = sol[:,::8]
        sol = sol.T
        sol = torch.tensor(sol.reshape(-1,1))
        return sol

    def get_plot_solution(self, x, t):
        x = self.xgrid_p.detach().numpy()
        t = self.tgrid_p.detach().numpy()
        sol = fft_burgers(x, t, self.nu)
        sol = sol.T
        sol = torch.tensor(sol.reshape(-1,1))
        return sol

    def _burgers_eqn(self, u, x, t):
        return diff(u, t, order=1) + u*diff(u, x, order=1) - self.nu*diff(u, x, order=2)

    def get_equation(self, u, x, t):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(u, x, t)
        u_adj = adj['pred']
        return self._burgers_eqn(u_adj, x, t)

    def adjust(self, u, x, t):
        """ perform boundary value adjustment """
        x_tilde = (x-self.xmin) / (self.xmax-self.xmin)
        t_tilde = (t-self.tmin) / (self.tmax-self.tmin)
        Axt = 1/torch.cosh(x)

        u_adj = Axt + x_tilde*(1-x_tilde)*(1 - torch.exp(-t_tilde))*u

        return {'pred': u_adj}

    def get_plot_dicts(self, pred, x, t, sol):
        """ return appropriate pred_dict / diff_dict used for plotting """
        adj = self.adjust(pred, x, t)
        pred_adj = adj['pred']
        pred_dict = {'$\hat{u}$': pred_adj.detach(), '$u$': sol.detach()}

        resid = self.get_equation(pred, x, t)
        diff_dict = {'$|\hat{F}|$': np.abs(resid.detach())}
        return pred_dict, diff_dict

class HeatEquation(Problem):
    """
    Heat Equation:

    $$ u_t - c^2 u_{xx} = 0 $$

    Boundary conditions:
    u(x,t)     | x=0 : 0
    u(x,t)     | x=1 : 0

    Initial condition:
    u(x,t)     | t=0 : sin(pi*x)

    Solution:
    u(x,t) = exp(-c^2 pi^2 t) sin(pi x)
    """
    def __init__(self, nx=32, nt=32, c=1, xmin=0, xmax=1, tmin=0, tmax=0.2, **kwargs):
        super().__init__(**kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.tmin = tmin
        self.tmax = tmax
        self.nx = nx
        self.nt = nt
        self.c = c
        self.pi = torch.tensor(np.pi)
        self.hx = (xmax - xmin) / nx
        self.ht = (tmax - tmin) / nt
        self.noise_xstd = self.hx / 4.0
        self.noise_tstd = self.ht / 4.0

        xgrid = torch.linspace(xmin, xmax, nx, requires_grad=True)
        tgrid = torch.linspace(tmin, tmax, nt, requires_grad=True)

        grid_x, grid_t = torch.meshgrid(xgrid, tgrid, indexing='ij')
        self.grid_x, self.grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x, self.grid_t)

    def get_grid_sample(self, grid, resid, resid_delta):
        x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
        t_noisy = torch.normal(mean=self.grid_t, std=self.noise_tstd)
        return (x_noisy, t_noisy)

    def get_plot_grid(self):
        return self.get_grid()

    def get_plot_dims(self):
        return {'x': self.nx, 't': self.nt}

    def get_solution(self, x, t):
        sol = torch.exp((-self.c**2)*(self.pi**2)*t) * torch.sin(self.pi * x)
        return sol

    def get_plot_solution(self, x, t):
        return self.get_solution(x, t)

    def _heat_eqn(self, u, x, t):
        return diff(u, t, order=1) - (self.c**2)*diff(u, x, order=2)

    def get_equation(self, u, x, t):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(u, x, t)
        u_adj = adj['pred']
        return self._heat_eqn(u_adj, x, t)

    def adjust(self, u, x, t):
        """ perform boundary value adjustment """
        x_tilde = (x-self.xmin) / (self.xmax-self.xmin)
        t_tilde = (t-self.tmin) / (self.tmax-self.tmin)
        Axt = torch.sin(self.pi * x)

        u_adj = Axt + x_tilde*(1-x_tilde)*(1 - torch.exp(-t_tilde))*u

        return {'pred': u_adj}

    def get_plot_dicts(self, pred, x, t, sol):
        """ return appropriate pred_dict / diff_dict used for plotting """
        adj = self.adjust(pred, x, t)
        pred_adj = adj['pred']
        pred_dict = {'$\hat{u}$': pred_adj.detach()}

        resid = self.get_equation(pred, x, t)
        diff_dict = {'$|\hat{F}|$': np.abs(resid.detach())}
        return pred_dict, diff_dict

class AllenCahn(Problem):
    """
    Allen-Cahn equation:

    u_t - \epsilon u_{xx} - u + u^3 = 0

    Boundary conditions:
    u(x,t)     | x=0    : 0
    u(x,t)     | x=2*pi : 0

    Initial condition:
    u(x,t)     | t=0  : 0.25*sin(x)
    """
    def __init__(self, nx=64, nt=64, epsilon=0.001, xmin=0, xmax=2*np.pi, tmin=0, tmax=5, 
        xmin_p=0, xmax_p=2*np.pi, tmin_p=0, tmax_p=5, nx_p=1000, nt_p=100, lam=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.xmin = xmin
        self.xmax = xmax
        self.tmin = tmin
        self.tmax = tmax
        self.nx = nx
        self.nt = nt
        self.epsilon = epsilon
        self.pi = torch.tensor(np.pi)
        self.hx = (xmax - xmin) / nx
        self.ht = (tmax - tmin) / nt
        self.noise_xstd = self.hx / 4.0
        self.noise_tstd = self.ht / 4.0
        self.xgrid = torch.linspace(xmin, xmax, nx+1, requires_grad=True)
        self.tgrid = torch.linspace(tmin, tmax, nt+1, requires_grad=True)
        grid_x, grid_t = torch.meshgrid(self.xgrid, self.tgrid, indexing='ij')
        self.grid_x, self.grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)

        self.xmin_p = xmin_p
        self.xmax_p = xmax_p
        self.tmin_p = tmin_p
        self.tmax_p = tmax_p
        self.nx_p = nx_p
        self.nt_p = nt_p
        self.xgrid_p = torch.linspace(xmin_p, xmax_p, self.nx_p, requires_grad=True)
        self.tgrid_p = torch.linspace(tmin_p, tmax_p, self.nt_p, requires_grad=True)
        grid_x_p, grid_t_p = torch.meshgrid(self.xgrid_p, self.tgrid_p, indexing='ij')
        self.grid_x_p, self.grid_t_p = grid_x_p.reshape(-1,1), grid_t_p.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x.float(), self.grid_t.float())

    def get_grid_sample(self, grid, resid, resid_delta):
        if self.sampling == 'fixed':
            return (self.grid_x.float(), self.grid_t.float())
        elif self.sampling == 'perturb':
            x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
            t_noisy = torch.normal(mean=self.grid_t, std=self.noise_tstd)
            return (x_noisy.float(), t_noisy.float())
        elif self.sampling == 'gradient':
            # x, t = grid[:, 0].reshape(-1,1), grid[:, 1].reshape(-1,1)
            # xresid_delta, tresid_delta = resid_delta[:, 0].reshape(-1,1), resid_delta[:, 1].reshape(-1,1)
            # #tresid_delta = self._grid_reorder(tresid_delta.detach().numpy())
            # xgrads = self._resid_grad(resid, xresid_delta)
            # tgrads = self._resid_grad(resid, tresid_delta)
            # xgrad_mean = np.mean(np.abs(xgrads))
            # tgrad_mean = np.mean(np.abs(tgrads))
            # eta_x = 1 / xgrad_mean if xgrad_mean != 0 else 1
            # eta_t = 1 / tgrad_mean if tgrad_mean != 0 else 1
            # grad_clip_x = self.hx / eta_x
            # grad_clip_t = self.ht / eta_t
            # xgrads[xgrads >= 2*grad_clip_x] = 2*grad_clip_x
            # tgrads[tgrads >= 2*grad_clip_t] = 2*grad_clip_t
            # x_new = x.detach().numpy() + eta_x*xgrads
            # t_new = t.detach().numpy() + eta_t*tgrads
            # x_new[x_new < self.xmin] = self.xmin + x_new[x_new < self.xmin] % self.xmax
            # x_new[x_new > self.xmax] = self.xmin + x_new[x_new > self.xmax] % self.xmax
            # t_new[t_new < self.tmin] = self.tmin + t_new[t_new < self.tmin] % self.tmax
            # t_new[t_new > self.tmax] = self.tmin + t_new[t_new > self.tmax] % self.tmax
            # x_new = np.sort(x_new, axis=0)
            # #print("UNSORTED:", t_new[:65])
            # t_new = self._sort_t(t_new)
            # #t_new = np.sort(t_new, axis=0)
            # x_new[:self.nx+1] = self.xmin
            # x_new[-(self.nx+1):] = self.xmax
            # t_new[::self.nt+1] = self.tmin
            # t_new[self.nt::self.nt+1] = self.tmax
            # #print("SORTED:", t_new[:65])
            # x_new = torch.tensor(x_new, requires_grad=True).reshape(-1,1)
            # t_new = torch.tensor(t_new, requires_grad=True).reshape(-1,1)
            # return (x_new.float(), t.float())
            xgrid = torch.cat((torch.linspace(0, 0.8, 20, requires_grad=True), torch.linspace(0.8, 3.14, 13, requires_grad=True), torch.linspace(3.14, 5.48, 12, requires_grad=True), torch.linspace(5.48, 6.28, 20)))
            tgrid = torch.linspace(0, 5, 65, requires_grad=True)
            grid_x, grid_t = torch.meshgrid(xgrid, tgrid, indexing='ij')
            grid_x, grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)
            x_noisy = torch.normal(mean=grid_x, std=self.noise_xstd)
            t_noisy = torch.normal(mean=grid_t, std=self.noise_tstd)
            return (x_noisy.float(), t_noisy.float())

    def _grid_reorder(self, l):
        reordered_l = []
        dim = self.nt+1
        for i in range(dim):
            for j in range(dim):
                reordered_l.append(l[i+dim*j])
        return np.array(reordered_l).reshape(-1,1)

    def _sort_t(self, t):
        t_new = np.array([])
        dim = self.nt+1
        for i in range(dim):
            col = t[i*dim:i*dim+dim]
            t_new = np.append(t_new, np.sort(col, axis=0))
        return t_new.reshape(-1,1)

    def _resid_grad(self, resid, resid_delta):
        """ get the gradient of the residuals at the grid points """
        try:
            resid = resid.detach().numpy()
            resid_delta = resid_delta.detach().numpy()
        except:
            pass
        delta = 1e-5
        grads = (np.abs(resid_delta) - np.abs(resid)) / delta
        return grads

    def get_plot_grid(self):
        return (self.grid_x_p.float(), self.grid_t_p.float())

    def get_plot_dims(self):
        return {'x': self.nx_p, 't': self.nt_p}

    def get_solution(self, x, t):
        """ use FFT method """
        x = np.linspace(self.xmin, self.xmax, (self.nx*16)+1)
        t = np.linspace(self.tmin, self.tmax, self.nt+1)
        sol = fft_allen_cahn(x, t, self.epsilon)
        sol = sol[:,::16]
        sol = sol.T
        sol = torch.tensor(sol.reshape(-1,1))
        return sol

    def get_plot_solution(self, x, t):
        x = self.xgrid_p.detach().numpy()
        t = self.tgrid_p.detach().numpy()
        plot_sol = fft_allen_cahn(x, t, self.epsilon)
        plot_sol = plot_sol.T
        plot_sol = torch.tensor(plot_sol.reshape(-1,1))
        return plot_sol

    def _allen_cahn_eqn(self, u, x, t):
        return diff(u, t, order=1) - self.epsilon*diff(u, x, order=2) - u + u**3

    def get_equation(self, u, x, t, plot=False):
        """ return value of residuals of equation (i.e. LHS) """
        adj = self.adjust(u, x, t)
        u_adj = adj['pred']
        # resid = self._allen_cahn_eqn(u_adj, x, t)
        # if plot is True:
        #     u_adj = u_adj.reshape((self.nx_p, self.nt_p))
        #     x = x.reshape((self.nx_p, self.nt_p))
        #     ic = u_adj[:, 0] - 0.25*torch.sin(x[:, 0])
        # else:
        #     u_adj = u_adj.reshape((self.nx+1, self.nt+1))
        #     x = x.reshape((self.nx+1, self.nt+1))
        #     ic = u_adj[:, 0] - 0.25*torch.sin(x[:, 0])
        # ic, bc1, bc2 = ic.reshape(-1, 1), u_adj[0, :].reshape(-1, 1), u_adj[-1, :].reshape(-1, 1)
        # ic, bc1, bc2 = ic*self.lam, bc1*self.lam, bc2*self.lam
        # return torch.cat((resid, ic, bc1, bc2))
        return self._allen_cahn_eqn(u_adj, x, t)*torch.exp(-t/self.lam)

    def adjust(self, u, x, t):
        """ perform boundary value adjustment """
        x_tilde = (x-self.xmin) / (self.xmax-self.xmin)
        t_tilde = (t-self.tmin) / (self.tmax-self.tmin)
        Axt = 0.25*torch.sin(x)

        u_adj = Axt + x_tilde*(1-x_tilde)*(1 - torch.exp(-t_tilde))*u
        return {'pred': u_adj}

    def get_plot_dicts(self, pred, x, t, sol):
        """ return appropriate pred_dict / diff_dict used for plotting """
        adj = self.adjust(pred, x, t)
        pred_adj = adj['pred']
        pred_dict = {'$\hat{u}$': pred_adj.detach(), '$u$': sol.detach()}

        resid = self.get_equation(pred, x, t, plot=True)
        resid_len = self.nx_p*self.nt_p
        diff_dict = {'$|\hat{F}|$': np.abs(resid.detach()[:resid_len])}
        return pred_dict, diff_dict
        

if __name__ == "__main__":
    import denn.utils as ut
    import matplotlib.pyplot as plt

    ac = AllenCahn()
    plot_x, plot_y = ac.get_plot_grid()
    plot_grid = torch.cat((plot_x, plot_y), 1)
    plot_soln = ac.get_plot_solution(plot_x, plot_y)
    resid = ac.get_equation(plot_soln, plot_x, plot_y)
    #plot_grid = plot_grid.detach()
    #x, t = plot_grid[:, 0], plot_grid[:, 1]
    #xdim, tdim = ac.get_plot_dims().values()
    #xx, tt = x.reshape((xdim, tdim)), t.reshape((xdim, tdim))
    #plot_soln = plot_soln.reshape((xdim, tdim))
    #fig, ax = plt.subplots(figsize=(6,5))
    #cf = ax.contourf(xx, tt, plot_soln, cmap='Reds')
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
