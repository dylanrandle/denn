import numpy as np
import torch
import os

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
        """ return pred_dict and optional diff_dict (or None) to be used for plotting
            depending on the problem we may want to plot different things, which is why
            this method exists (and is required)
        """
        raise NotImplementedError()
