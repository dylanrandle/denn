# Neural Networks for Differential Equations

Solutions to differential equations are of significant scientific and engineering interest. However,
many differential equations admit no analytic solutions and must be numerically approximated. Classic
numerical algorithms solve the equation at a set of points (called a grid or mesh); if we want results
at new points, we are forced to re-run the entire analysis.

Neural networks, on the other hand, have emerged as universal function-approximators. Since the solution
to differential equations *are functions*, we are motivated to apply neural networks to solving differential
equations.

Our main goal is to train neural networks (once) that can accurately approximate the solution to a complex
differential equation of choice with high resolution and generalization. We surmise that having access to an
accurate *functional form* will provide many uses in the scientific and engineering communities.

## Contents of this Repository

This repo is a collection of research code that was developed by Dylan Randle at Harvard University. The core
routines are contained in `denn`. There are a few modules to explain:
1. `rans`: Module implementing unsupervised methods for solving Reynolds-Averaged Navier Stokes equations
2. `exp`: Module implementing unsupervised Generative Adversarial Network methods for solving exponential decay; \
also includes the simpler mean-square error method for comparison
3. `sho`: Module for Simple Harmonic Oscillator; includes updated approach using semi-supervised learning

`notebooks` contains a large collection of Jupyter notebooks with experimental results, as well as some additional nuggets of insight and small implementations. Particular highlights:
- `channel_flow_nb.ipynb`: initial experiments on the RANS equations
- `analytical_bcs.ipynb`: use of analytical (physics-constrained) transformation for boundary conditions
- `CV_Kappa.ipynb`: investigates the effect of mixing length in the RANS model
- `Analysis_of_Sampling.ipynb`: shows the effect of different grid sampling strategies
- `GAN.ipynb`: shows the setup and results of GANs and MSE-based method on exponential decay
- `GAN_Paper_Results.ipynb`: shows concise results for examples in paper
