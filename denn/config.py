import torch.nn as nn
import numpy as np
import denn.problems as pb

# ==========================
# Setup of the problem
# ==========================

## EXP PARAMS
## n=100 | perturb = True | t_max = 10
exp_problem = pb.Exponential(n=100, perturb=True, t_max=10)

## SHO PARAMS
## n=100 | perturb = True | t_max = 4 * np.pi
sho_problem = pb.SimpleOscillator(n=100, perturb=True, t_max=4*np.pi)

## NLO PARAMS
## n=1000 | perturb = True | t_max = 8 * np.pi
nlo_problem = pb.NonlinearOscillator(n=1000, perturb=True, t_max=8*np.pi)

# ==========================
# GAN
# ==========================

## EXP PARAMS
## niters: 500 | Gen: 20x3 | Disc: 10x2
## G_iters = 9 | wgan = False | conditional = False

## SHO PARAMS
## niters: 10K | Gen: 32x2 | Disc: 24x4
## wgan = True | conditional = True

## NLO PARAMS
## niters: 10K | Gen: 64x12 | Disc: 64x14
## wgan = True | conditional = True

# GAN Algorithm
gan_kwargs = dict(
    method='unsupervised',
    niters=10000,
    g_lr=1e-3,
    g_betas=(0., 0.9),
    d_lr=1e-3,
    d_betas=(0., 0.9),
    lr_schedule=True,
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
    fname='train_GAN.png',
)

# Generator MLP
gen_kwargs = dict(
    in_dim=1,
    out_dim=1,
    n_hidden_units=64,
    n_hidden_layers=12,
    activation=nn.Tanh(),
    residual=True,
    regress=True,
)

# Discriminator MLP
disc_kwargs = dict(
    in_dim=2,
    out_dim=1,
    n_hidden_units=64,
    n_hidden_layers=14,
    activation=nn.Tanh(),
    residual=True,
    regress=True, # true for WGAN, false otherwise
)

# Discriminator MLP #2 (for semi-supervised with two Ds)
disc_kwargs_2 = dict(
    in_dim=2,
    out_dim=1,
    n_hidden_units=16,
    n_hidden_layers=2,
    activation=nn.Tanh(),
    residual=True,
    regress=True, # true for WGAN, false otherwise
)

# ==========================
# L2 (Lagaris)
# ==========================

## SHO PARAMS
## niters: 10K | MLP: 32x2

## NLO PARAMS
## niters: 10K | MLP: 64x12

# L2 Algorithm
L2_kwargs = dict(
    method='unsupervised',
    niters=10000,
    lr=1e-3,
    betas=(0., 0.9),
    lr_schedule=True,
    obs_every=1,
    d1=1,
    d2=1,
    plot=True,
    save=True,
    fname='train_L2.png',
)

# L2 MLP
L2_mlp_kwargs = dict(
    in_dim=1,
    out_dim=1,
    n_hidden_units=64,
    n_hidden_layers=12,
    activation=nn.Tanh(),
    residual=True,
    regress=True,
)

# ==========================
# Hyper tuning
# ==========================

# GAN SHO
gan_sho_hyper_space = dict(
     # gan_kwargs
     gan_niters = [10000],
     # disc_kwargs
     disc_n_hidden_units = [16, 24, 32],
     disc_n_hidden_layers = [2, 3, 4, 5],
     # gen_kwargs
     gen_n_hidden_units = [16, 24, 32],
     gen_n_hidden_layers = [2, 3, 4, 5],
)

# GAN NLO
gan_nlo_hyper_space = dict(
     # gan_kwargs
     gan_niters = [10000],
     # disc_kwargs
     disc_n_hidden_units = [32, 64],
     disc_n_hidden_layers = [6, 8, 10, 12, 14],
     # gen_kwargs
     gen_n_hidden_units = [32, 64],
     gen_n_hidden_layers = [6, 8, 10, 12, 14],
)

# L2 NLO
L2_nlo_hyper_space = dict(
   # model_kwargs
   model_n_hidden_units=[32, 64],
   model_n_hidden_layers=[4, 6, 8],
   # train_kwargs
   train_niters=[10000, 50000, 100000],
)
