# Neural Networks for Differential Equations

Our main goal is to train neural networks (once) that can accurately approximate the solution to a complex
differential equation of choice with high resolution and generalization. We want to be able to use and re-use
this network to model our dynamical system.

## Channel Flow

We first examine a set of equations in Fluid Dynamics called the Navier Stokes equations. In the simple case of
a 1-dimensional channel, the Reynolds-Averaged Navier Stokes equations are written as:

$$ \nu \frac{d^2u}{dy^2} - \frac{d\overline{uu}}{dy} - \frac{1}{\rho} \frac{dp}{dx} = \nu \frac{d^2u}{dy^2} - \frac{d(\kappa y)^{2} \left|\frac{du}{dy}\right| \frac{du}{dy}}{dy} - \frac{1}{\rho} \frac{dp}{dx} = 0 $$

Where we have substituted the "mixing length" model for the Reynolds stress tensor to close the equations.

Please investigate this repository and its plethora of IPython notebooks with many graphs and experiments to learn more.

## Code

The code is organized as follows:

- `channel_flow.py`: contains the neural network class as well as all methods used for training it

- `utils.py`: contains useful utilities for calculating physical quantities and for plotting

- `train_chanflow.py`: demonstrates a simple case of training the Chanflow model

- `numerical.py`: implements the finite-difference Newton's method for solving the equation

- `cv_kappa.py` / `diff_sampling.py`: are ideal snippets of code that use the latest abstractions for training
