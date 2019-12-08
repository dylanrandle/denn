import numpy as np
import torch
from scipy import optimize

from denn.utils import diff

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
    def __init__(self, t_min = 0, t_max = 4 * np.pi, x0 = 1., L = 1, **kwargs):
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
        diff_dict = {'$\hat{x}$': xadj.detach(), '$\hat{\dot{x}}$': dx.detach()}
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

    def get_plot_dicts(self, x, t, y):
        """ return appropriate pred_dict and diff_dict used for plotting """
        xadj, dx, d2x = self.adjust(x, t)
        pred_dict = {'$\hat{x}$': xadj.detach(), '$x$': y.detach()}
        diff_dict = {'$\hat{x}$': xadj.detach(), '$\hat{\ddot{x}}$': d2x.detach()}
        return pred_dict, diff_dict

class NonlinearOscillator(Problem):
    """
    Nonlinear Oscillator Problem:

    $$ \ddot{x} + 2 \beta \dot{x} + \omega^{2} x + \phi x^{2} + \epsilon x^{3} = f(t) $$
    """
    def __init__(self, t_min = 0, t_max = 4 * np.pi, dx_dt0 = .5, **kwargs):
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
        dt = t[1] - t[0]
        guess = np.ones_like(t)

        def get_diff(x):
            x[0] = self.x0
            dx = np.gradient(x, dt, edge_order = 2)
            dx[0] = self.dx_dt0
            d2x = np.gradient(dx, dt, edge_order = 2)
            return self._nlo_eqn(x, dx, d2x)

        opt_sol = optimize.root(get_diff, guess, method='lm')
        print(f'Numerical solution success: {opt_sol.success}')
        return torch.tensor(opt_sol.x, dtype=torch.float).reshape(-1, 1)

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
