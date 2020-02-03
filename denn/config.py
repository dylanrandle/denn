import torch.nn as nn
import numpy as np
import denn.problems as pb
import os

FIG_DIR = '/Users/dylanrandle/Documents/Harvard/research/denn/experiments/figs/'
CSV_DIR = '/Users/dylanrandle/Documents/Harvard/research/denn/experiments/csvs/'

# ==========================
# Problem
# ==========================

## EXP PARAMS
## n=100 | perturb = True | t_max = 10

## SHO PARAMS
## n=100 | perturb = True | t_max = 4 * np.pi

## NLO PARAMS
## n=1000 | perturb = True | t_max = 8 * np.pi

exp_problem = pb.Exponential(n=100, perturb=True, t_max=10)
sho_problem = pb.SimpleOscillator(n=100, perturb=True, t_max=4*np.pi)
nlo_problem = pb.NonlinearOscillator(n=1000, perturb=True, t_max=8*np.pi)
pos_problem = pb.PoissonEquation(nx=40, ny=40, f=10, perturb=True)

# ==========================
# GAN
# ==========================

## EXP PARAMS (unsupervised)
## niters: 500 | Gen: 20x2 | Disc: 20x2
## g_lr: 1e-2 | d_lr: 1e-2 | g_betas = d_betas = (0.9, 0.999)
## G:D iters : 1:1 | wgan = False | conditional = False
## lr_schedule = True | activations : Tanh
## residual connections = True

## SHO PARAMS
## niters = 10K | Gen = 64x2 | Disc = 64x10
## wgan = True | conditional = True | g_lr = d_lr = 1e-3
## g_betas = d_betas = (0., 0.9) | lr_schedule = True | gamma = 0.999
## G:D iters = 1:1 | gp = 0.1 | activations : Tanh
## residual connections = True

## NLO PARAMS
## niters: 50K | Gen: 64x12 | Disc: 64x16
## wgan = True | conditional = True | g_lr = d_lr = 1e-3
## g_betas = d_betas = (0., 0.9) | lr_schedule = True | gamma = 0.9999
## G:D iters = 1:1 | gp = 0.1 | activations : Tanh
## residual connections = True

# GAN Algorithm
gan_kwargs = dict(
    method='unsupervised',
    niters=1000,
    g_lr=2e-4,
    g_betas=(0., 0.9),
    d_lr=2e-4,
    d_betas=(0., 0.9),
    lr_schedule=True,
    gamma=0.995,
    obs_every=1,
    d1=1.,
    d2=1.,
    G_iters=1,
    D_iters=1,
    wgan=True,
    gp=0.1,
    conditional=True,
    plot=True,
    save=True,
    fname=os.path.join(FIG_DIR, 'train_GAN.png'),
)

# Generator MLP
gen_kwargs = dict(
    in_dim=2,
    out_dim=1,
    n_hidden_units=64,
    n_hidden_layers=4,
    activation=nn.Tanh(),
    residual=True,
    regress=True,
    spectral_norm=False, # included for completeness, but not used; in general
)                        # spectral norm method is only on discriminator

# Discriminator MLP
disc_kwargs = dict(
    in_dim=3,
    out_dim=1,
    n_hidden_units=64,
    n_hidden_layers=4,
    activation=nn.Tanh(),
    residual=True,
    regress=True,        # true for WGAN, false otherwise
    spectral_norm=False, # true for spectral norm method
)

# ==========================
# L2 (a.k.a. Lagaris)
# ==========================

# == PARAMS ==
# model: same as GENERATOR from associated GAN
# learning: same as associated GAN (where applicable)

# L2 Algorithm
L2_kwargs = dict(
    method='unsupervised',
    niters=10,
    lr=1e-3,
    betas=(0., 0.9),
    lr_schedule=True,
    gamma=0.999,
    obs_every=1,
    d1=1,
    d2=1,
    plot=True,
    save=True,
    fname=os.path.join(FIG_DIR, 'train_L2.png'),
)

# L2 MLP
L2_mlp_kwargs = dict(
    in_dim=2,
    out_dim=1,
    n_hidden_units=64,
    n_hidden_layers=2,
    activation=nn.Tanh(),
    residual=True,
    regress=True,
    spectral_norm=False, # included for completeness, but not used
)

# ==========================
# Hyper tuning
# ==========================

# GAN SHO (model specs)
gan_sho_hyper_space = dict(
      # gan_kwargs
      gan_niters = [100],
      gan_d_lr = [1e-3, 2e-4],
      gan_g_lr = [1e-3, 2e-4],
      gan_gamma = [0.999],
      # disc_kwargs
      disc_n_hidden_units = [64],
      disc_n_hidden_layers = [4, 6, 8, 10],
      # gen_kwargs
      gen_n_hidden_units = [64],
      gen_n_hidden_layers = [2, 4, 6, 8],
)

# GAN SHO (niters)
gan_sho_niters = dict(
     # gan_kwargs
     gan_niters = [10, 20, 50, 100, 200],
     gan_gamma = [0.999],
     # disc_kwargs
     disc_n_hidden_units = [64],
     disc_n_hidden_layers = [10],
     # gen_kwargs
     gen_n_hidden_units = [64],
     gen_n_hidden_layers = [2],
)

# L2 SHO (niters)
L2_sho_niters = dict(
    # model_kwargs
    model_n_hidden_units=[64],
    model_n_hidden_layers=[2],
    # train_kwargs
    train_niters=[1000, 2000, 5000, 10000, 20000],
    train_gamma=[0.999],
)

# GAN NLO (model specs)
gan_nlo_hyper_space = dict(
     # gan_kwargs
     gan_niters = [250],
     gan_d_lr = [1e-3],
     gan_g_lr = [1e-3],
     gan_gamma = [0.9999],
     # disc_kwargs
     disc_n_hidden_units = [64],
     disc_n_hidden_layers = [12, 14, 16, 18, 20],
     # gen_kwargs
     gen_n_hidden_units = [64],
     gen_n_hidden_layers = [6, 8, 10, 12],
)

# GAN NLO (niters)
gan_nlo_niters = dict(
     # gan_kwargs
     gan_niters = [10000, 25000, 50000, 75000, 100000],
     # disc_kwargs
     disc_n_hidden_units = [64],
     disc_n_hidden_layers = [16],
     # gen_kwargs
     gen_n_hidden_units = [64],
     gen_n_hidden_layers = [12],
)

# L2 NLO (model specs)
L2_nlo_hyper_space = dict(
   # model_kwargs
   model_n_hidden_units=[32, 64],
   model_n_hidden_layers=[4, 6, 8],
   # train_kwargs
   train_niters=[10000, 50000, 100000],
)

# L2 NLO (niters)
L2_nlo_niters = dict(
   # model_kwargs
   model_n_hidden_units=[64],
   model_n_hidden_layers=[12],
   # train_kwargs
   train_niters=[10000, 25000, 50000, 75000, 100000],
)

gan_pos_hyper_space = dict(
    # gan_kwargs
    gan_niters = [1000],
    gan_d_lr = [1e-3],
    gan_g_lr = [1e-3],
    gan_gamma = [0.999],
    # disc_kwargs
    disc_n_hidden_units = [64],
    disc_n_hidden_layers = [4, 8, 12, 16],
    # gen_kwargs
    gen_n_hidden_units = [64],
    gen_n_hidden_layers = [2, 4, 8, 12],
)

gan_pos_niters = dict(
     # gan_kwargs
     gan_niters = [10, 25, 50, 75, 100],
     # disc_kwargs
     disc_n_hidden_units = [64],
     disc_n_hidden_layers = [4],
     # gen_kwargs
     gen_n_hidden_units = [64],
     gen_n_hidden_layers = [2],
)

L2_pos_niters = dict(
     # gan_kwargs
     train_niters = [10, 25, 50, 75, 100],
     # model_kwargs
     model_n_hidden_units=[64],
     model_n_hidden_layers=[2],
)
