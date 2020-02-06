import numpy as np
import torch
from scipy.integrate import odeint
from denn.utils import diff
from denn.poisson.poisson import compute_solution as poisson_compute_solution

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

    def sample_grid(self, grid, spacing):
        """ return perturbed samples from the grid
            grid is the torch tensor representing the grid
            d is the inter-point spacing
        """
        if self.perturb:
            return grid + spacing * torch.randn_like(grid) / 3
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
        x, dx = self.adjust(x, t)
        return dx + self.L * x

    def adjust(self, x, t):
        """ perform initial value adjustment """
        x_adj = self.x0 + (1 - torch.exp(-t)) * x
        dx_dt = diff(x_adj, t)
        return x_adj, dx_dt

    def get_plot_dicts(self, x, t, y):
        """ return appropriate pred_dict and diff_dict used for plotting """
        xadj, dx = self.adjust(x, t)
        pred_dict = {'$\hat{x}$': xadj.detach(), '$x$': y.detach()}
        diff_dict = {'$\hat{x}$': xadj.detach(), '$-\hat{\dot{x}}$': (-dx).detach()}
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
        x, dx, d2x = self.adjust(x, t)
        return d2x + x

    def adjust(self, x, t):
        """ perform initial value adjustment """
        x_adj = self.x0 + (1 - torch.exp(-t)) * self.dx_dt0 + ((1 - torch.exp(-t))**2) * x
        dx_dt = diff(x_adj, t)
        d2x_dt2 = diff(dx_dt, t)
        return x_adj, dx_dt, d2x_dt2

    # def adjust(self, x, t):
    #     """ perform initial value adjustment using coupled equations """
    #     dx_dt = diff(x, t)
    #
    #     x_adj = self.x0 + (1 - torch.exp(-t)) * self.dx_dt0 + ((1 - torch.exp(-t))**2) * x
    #
    #     nn_adj = (1 - torch.exp(-t)) * dx_dt + 2 * torch.exp(-t) * x
    #     dx_dt_adj = self.dx_dt0 + (1 - torch.exp(-t)) * nn_adj
    #
    #     d2x_dt2 = diff(dx_dt_adj, t)
    #     return x_adj, dx_dt_adj, d2x_dt2

    def get_plot_dicts(self, x, t, y):
        """ return appropriate pred_dict and diff_dict used for plotting """
        xadj, dx, d2x = self.adjust(x, t)
        pred_dict = {'$\hat{x}$': xadj.detach(), '$x$': y.detach()}
        diff_dict = {'$\hat{x}$': xadj.detach(), '$-\hat{\ddot{x}}$': (-d2x).detach()}
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

    def get_grid(self):
        return self.grid

    def get_grid_sample(self):
        return self.sample_grid(self.grid, self.spacing)

    def get_solution(self, t):
        """ uses scipy to solve NLO """
        try:
            t = t.detach().numpy() # if torch tensor, convert to numpy
        except:
            pass

        t = t.reshape(-1)
        sol = odeint(self._nlo_system, [self.x0, self.dx_dt0], t, tfirst=True)
        return torch.tensor(sol[:,0], dtype=torch.float).reshape(-1, 1)

    def _nlo_system(self, t, z):
        """ NLO decomposed as system of first order equations """
        x, y = z   # y = x'
        return np.array([y, -(2 * self.beta * y + (self.omega**2) * x + self.phi * (x**2) + self.epsilon * (x**3))])

    def _nlo_eqn(self, x, dx, d2x):
        return d2x + 2 * self.beta * dx + (self.omega ** 2) * x + self.phi * (x ** 2) \
            + self.epsilon * (x ** 3) # - self.F * self.forcing(t)

    def get_equation(self, x, t):
        """ return value of residuals of equation (i.e. LHS) """
        x, dx, d2x = self.adjust(x, t)
        return self._nlo_eqn(x, dx, d2x)

    def adjust(self, x, t):
        """ perform initial value adjustment """
        x_adj = self.x0 + (1 - torch.exp(-t)) * self.dx_dt0 + ((1 - torch.exp(-t))**2) * x
        dx = diff(x_adj, t)
        d2x = diff(dx, t)
        return x_adj, dx, d2x

    def get_plot_dicts(self, x, t, y):
        """ return appropriate pred_dict and diff_dict used for plotting """
        xadj, dx, d2x = self.adjust(x, t)
        pred_dict = {'$\hat{x}$': xadj.detach(), '$x$': y.detach()}
        diff_dict = None     # the derivatives here are less meaningful, so dont plot them
        return pred_dict, diff_dict

class PoissonEquation(Problem):
    """
    Poisson Equation:

    $$ \nabla^{2} \varphi = f $$

    -Laplace(u) = f    in the unit square
              u = u_D  on the boundary

    u_D = 0
      f = 1
    """
    def __init__(self, nx=100, ny=100, xmin=0, xmax=1, ymin=0, ymax=1, f=1, **kwargs):
        super().__init__(**kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nx = nx
        self.ny = ny
        self.f = f
        self.xgrid = torch.linspace(
            xmin,
            xmax,
            self.nx,
            dtype=torch.float,
            requires_grad=True
        ).reshape(-1)
        self.ygrid = torch.linspace(
            ymin,
            ymax,
            self.ny,
            dtype=torch.float,
            requires_grad=True
        ).reshape(-1)
        # set up our grid as just the set of all point tuples
        # self.grid = torch.cartesian_prod(self.xgrid, self.ygrid)
        self.grid = torch.cartesian_prod(self.xgrid, self.ygrid)
        # to match Fenics Mesh order
        self.grid = torch.flip(self.grid, [1])
        # take minimum spacing, which is equal to spacing along an axis
        self.spacing = self.xgrid[1] - self.xgrid[0]

    def get_grid(self):
        return self.grid

    def get_grid_sample(self):
        return self.sample_grid(self.grid, self.spacing)

    def get_solution(self, grid):
        """ use Fenics (finite elements) to compute solution
            TODO: calculate at `grid` instead of the current fixed mesh
        """
        u_np, mesh_np = poisson_compute_solution(self.nx, self.ny, self.f)
        return torch.tensor(u_np, dtype=torch.float).reshape(-1,1) #, mesh_np

    def _poisson_eqn(self, d2x, d2y):
        """ return RHS of equation (should equal 0) """
        return d2x + d2y + self.f

    def get_equation(self, u, grid):
        """ return value of residuals of equation (i.e. LHS) """
        u_adj, d2x, d2y = self.adjust(u, grid)
        return self._poisson_eqn(d2x, d2y)

    def adjust(self, u, grid):
        """ perform boundary value adjustment

        thanks to Feiyu Chen for this:
        https://github.com/odegym/neurodiffeq/blob/master/neurodiffeq/pde.py
        """
        u = u.reshape(-1)
        x, y = grid[:,0], grid[:, 1]
        x_tilde = (x-self.xmin) / (self.xmax-self.xmin)
        y_tilde = (y-self.ymin) / (self.ymax-self.ymin)

        # TODO: generalize the boundary conditions to some functions
        # @note: here we have just 0 everywhere
        self.x_min_val = lambda y: 0
        self.x_max_val = lambda y: 0
        self.y_min_val = lambda x: 0
        self.y_max_val = lambda x: 0

        Axy = (1-x_tilde)*self.x_min_val(y) + x_tilde*self.x_max_val(y) + \
              (1-y_tilde)*( self.y_min_val(x) - ((1-x_tilde)*self.y_min_val(self.xmin * torch.ones_like(x_tilde))
                                                  + x_tilde *self.y_min_val(self.xmax * torch.ones_like(x_tilde))) ) + \
                 y_tilde *( self.y_max_val(x) - ((1-x_tilde)*self.y_max_val(self.xmin * torch.ones_like(x_tilde))
                                                  + x_tilde *self.y_max_val(self.xmax * torch.ones_like(x_tilde))) )
        u_adj = Axy + x_tilde*(1-x_tilde)*y_tilde*(1-y_tilde)*u
        dx = diff(u_adj, x)
        dy = diff(u_adj, y)
        d2x = diff(dx, x)
        d2y = diff(dy, y)
        return u_adj.reshape(-1,1), d2x.reshape(-1,1), d2y.reshape(-1,1)

    def get_plot_dicts(self, pred, grid, sol):
        """ return appropriate pred_dict / diff_dict used for plotting """
        pred_adj, d2x, d2y = self.adjust(pred, grid)
        pred_adj = pred_adj.reshape(-1)
        sol = sol.reshape(-1)
        pred_dict = {'$\hat{u}$': pred_adj.detach()}
        diff_dict = None
        return pred_dict, diff_dict


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print('testing poisson object')
    ps = PoissonEquation(n=100)
    samp = ps.get_grid_sample()
    print('samp')
    print(samp)
    print(samp.shape)
    sol = ps.get_solution(samp)
    print('sol')
    print(sol)
    print(sol.shape)
    # print('mesh')
    # print(mesh)
    # print(mesh.shape)
    grid = ps.get_grid()
    print('grid')
    print(grid)
    print(grid.shape)
    grid = grid.detach().numpy()
    print('grid np')
    print(grid)
    print(grid.shape)

    # assert np.allclose(grid, mesh)
    plt.scatter(grid[:,0], grid[:,1], c=sol.reshape(-1))
    plt.show()

    print('sol_adj')
    grid = ps.get_grid()
    sol = torch.tensor(sol)
    sol_adj = ps.adjust(sol, grid)
    print(sol_adj)
