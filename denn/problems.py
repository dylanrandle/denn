import numpy as np
import torch
from denn.utils import diff

class Problem():
    """ parent class for all problems
        e.g. simple harmonic oscillator
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

    def get_solution(self):
        raise NotImplementedError()

    def get_equation(self, *args):
        raise NotImplementedError()

    def adjust(self, pred):
        raise NotImplementedError()

class SimpleOscillator(Problem):
    """ simple harmonic oscillator problem """
    def __init__(self, t_min = 0, t_max = 4 * np.pi, dx_dt0 = 1., **kwargs):
        """
        inputs:
            - t_min: start time
            - t_max: end time
            - x0: initial condition on x
            - dx_dt0: initial condition on dx_dt
            - m: mass of object
            - k: spring constant
            - kwargs: keyword args passed to Problem.__init__()
        """
        super().__init__(**kwargs)

        # ======
        # TODO: changing x0 / m / k
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
