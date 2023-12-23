# Unsupervised Neural Networks for Solving Differential Equations

This repository implements DEQGAN, a generative adversarial network method for solving ordinary and partial differential equations. Our [paper](https://arxiv.org/abs/2209.07081) appeared at the AI4Science workshop at ICML 2022.

## Minimal Installation

Using Anaconda:
1. `conda env create -f environment_minimal.yml`
2. `conda activate denn_minimal`
3. `python setup.py develop`

The minimal installation includes the dependencies required to reproduce the results of experiments. Additional dependencies include:

- [`ray`](https://ray.io/) for hyperparameter tuning (`ray_tune.py`)
- [`plotly`](https://plotly.com/) for parallel plots
- [`fenics`](https://fenicsproject.org/) for finite element methods

The full list of dependencies can be installed via the `environment.yml` file.

## Reproducing Experimental Results

Substitute `{key}` with the appropriate problem key (e.g. `exp`, `sho`, `nlo`, etc.) and follow instructions for each method below. See the table below for the full list of problem keys.

DEQGAN:
- `python denn/experiments.py --pkey {key} --gan`

L1 / L2 / Huber:
- Define PyTorch loss in `denn/config/{key}.yaml` under `training.loss_fn` (L1=`L1Loss`, L2=`MSELoss`, Huber=`SmoothL1Loss`)
- `python denn/experiments.py --pkey {key}`

RK4 / FD:
- `python denn/traditional.py --pkey {key}`

## Summary of Experiments

This table details the currently available differential equations and corresponding problem keys.

| Key   | Equation                    | Class | Order | Linear |
|-------|-----------------------------|-------|-------|--------|
| `exp` | Exponential Decay           | ODE   | 1st   | Yes    |
| `sho` | Simple Harmonic Oscillator  | ODE   | 2nd   | Yes    |
| `nlo` | Damped Nonlinear Oscillator | ODE   | 2nd   | No     |
| `coo` | Coupled Oscillators         | ODE   | 1st   | Yes    |
| `sir` | SIR Epidemiological Model   | ODE   | 1st   | No     |
| `ham` | Hamiltonian System          | ODE   | 1st   | No     |
| `ein` | Einstein's Gravity System   | ODE   | 1st   | No     |
| `pos` | Poisson Equation            | PDE   | 2nd   | Yes    |
| `hea` | Heat Equation               | PDE   | 2nd   | Yes    |
| `wav` | Wave Equation               | PDE   | 2nd   | Yes    |
| `bur` | Burgers' Equation           | PDE   | 2nd   | No     |
| `aca` | Allen-Cahn Equation         | PDE   | 2nd   | No     |

## Citing DEQGAN

If you would like to reference our work, please use the following BibTeX citation!

```
@misc{bullwinkel2022deqgan,
      title={DEQGAN: Learning the Loss Function for PINNs with Generative Adversarial Networks}, 
      author={Blake Bullwinkel and Dylan Randle and Pavlos Protopapas and David Sondak},
      year={2022},
      eprint={2209.07081},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
