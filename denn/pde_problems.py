import numpy as np
import torch
from denn.problem import Problem
from denn.utils import diff
from denn.burgers.numerical import burgers_viscous_time_exact1
from denn.burgers.fft_burgers import fft_burgers

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

class WaveEquation(Problem):
    """
    Wave Equation:

    $$ u_{tt} = c^2 u_{xx} $$

    Boundary conditions:
    u(x,t)     | x=0 : 0
    u(x,t)     | x=1 : 0
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

        grid_x, grid_t = torch.meshgrid(xgrid, tgrid)
        self.grid_x, self.grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x, self.grid_t)

    def get_grid_sample(self):
        x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
        t_noisy = torch.normal(mean=self.grid_t, std=self.noise_tstd)
        return (x_noisy, t_noisy)

    def get_solution(self, x, t):
        sol = torch.cos(self.c * self.pi * t) * torch.sin(self.pi * x)
        return sol

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

        grid_x, grid_t = torch.meshgrid(xgrid, tgrid)
        self.grid_x, self.grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x, self.grid_t)

    def get_grid_sample(self):
        x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
        t_noisy = torch.normal(mean=self.grid_t, std=self.noise_tstd)
        return (x_noisy, t_noisy)

    def get_solution(self, x, t):
        sol = (self.a * x + self.b) / (self.a * t + 1)
        return sol

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

    u_t + u u_x - \nu u_xx = 0

    Boundary conditions:
    u(x,t)     | x=-10 : 0
    u(x,t)     | x=10  : 0
    u(x,t)     | t=0   : 1/cosh(x)
    """
    def __init__(self, nx=64, nt=64, nu=0.001, xmin=-5, xmax=5, tmin=0, tmax=2.5, **kwargs):
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

        self.xgrid = torch.linspace(xmin, xmax, nx, requires_grad=True)
        self.tgrid = torch.linspace(tmin, tmax, nt, requires_grad=True)

        grid_x, grid_t = torch.meshgrid(self.xgrid, self.tgrid)
        self.grid_x, self.grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x, self.grid_t)

    def get_grid_sample(self):
        x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
        t_noisy = torch.normal(mean=self.grid_t, std=self.noise_tstd)
        return (x_noisy, t_noisy)

    def get_solution(self, x, t):
        """ use FFT method """
        try:
            x = self.xgrid.detach().numpy()
            t = self.tgrid.detach().numpy()
        except:
            pass

        sol = fft_burgers(self.nu, self.nx, x, self.nt, t)

        #sol = burgers_viscous_time_exact1(self.nu, self.nx, x, self.nt, t)
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
        #t_0 = self.tmin*torch.ones_like(t, requires_grad=True)
        #Axt = -torch.sin(self.pi*x) + \
        #      x_tilde*(torch.zeros_like(t) - torch.zeros_like(t_0)) + \
        #      (1 - x_tilde)*(torch.zeros_like(t) - torch.zeros_like(t_0))
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

        grid_x, grid_t = torch.meshgrid(xgrid, tgrid)
        self.grid_x, self.grid_t = grid_x.reshape(-1,1), grid_t.reshape(-1,1)

    def get_grid(self):
        return (self.grid_x, self.grid_t)

    def get_grid_sample(self):
        x_noisy = torch.normal(mean=self.grid_x, std=self.noise_xstd)
        t_noisy = torch.normal(mean=self.grid_t, std=self.noise_tstd)
        return (x_noisy, t_noisy)

    def get_solution(self, x, t):
        sol = torch.exp((-self.c**2)*(self.pi**2)*t) * torch.sin(self.pi * x)
        return sol

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

if __name__ == "__main__":
    import denn.utils as ut
    import matplotlib.pyplot as plt

    be = BurgersViscous()
    #xgrid, tgrid = be.xgrid, be.tgrid
    #grid_x, grid_t = torch.meshgrid(xgrid, tgrid)
    x, y = be.get_grid()
    print(float(y.detach()[0]), float(y.detach()[-1]))
    grid = torch.cat((x, y), 1)
    grid = grid.detach()
    soln = be.get_solution(x, y)
    x, y = grid[:, 0], grid[:, 1]
    xdim, ydim = int(np.sqrt(len(x))), int(np.sqrt(len(y)))
    soln = soln.reshape((xdim, ydim))
    print(xdim, ydim)
    print(soln.shape)
    fig, ax = plt.subplots(1, 4, figsize=(16,4))
    ax[0].plot(be.xgrid.detach(), soln[:, 0])
    ax[1].plot(be.xgrid.detach(), soln[:, 15])
    ax[2].plot(be.xgrid.detach(), soln[:, 31])
    ax[3].plot(be.xgrid.detach(), soln[:, 63])
    plt.show()
    #fig, ax = plt.subplots(figsize=(10,7))
    #cf = ax.contourf(grid_x, grid_t, soln, cmap="Reds")
    #cb = fig.colorbar(cf, format='%.0e', ax=ax)
    #ax.set_xlabel('x')
    #ax.set_ylabel('t')
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