# Unsupervised Neural Networks for Solving Differential Equations

Please read the paper (under review, link forthcoming) for detailed information.

## Minimal Installation

Using Anaconda 4.8.3:
- `conda env create -f environment_minimal.yml`
- `conda activate denn_minimal`
- `python setup.py develop`

The minimal installation includes the dependencies required to reproduce the results of experiments. Additional dependencies include:

- [`ray`](https://ray.io/) for hyperparameter tuning (`ray_tune.py`)
- [`plotly`](https://plotly.com/) for parallel plots

## Reproducing Experimental Results

Substitute `{key}` with the appropriate problem key (e.g. "exp", "sho", "nlo", etc.) and follow instructions for each method below.

DEQGAN:
- `python denn/experiments.py --pkey {key} --gan`

$L_1$ / $L_2$ / Huber:
- Define PyTorch loss in `denn/config/{key}.yaml` under `training.loss_fn` ($L_1$="L1Loss", $L_2$="MSELoss", Huber="SmoothL1Loss")
- `python denn/experiments.py --pkey {key}`

RK4 / FD:
- See `notebooks/runs/{key}.ipynb` individually. Note that we must set NumPy to use 32-bit precision to be comparable to PyTorch (NumPy's default is 64)
