import torch.nn as nn
import numpy as np

# ==========================
# Setup of the problem
# ==========================

## SHO PARAMS
## n=100 | perturb = True | t_max = 4 * np.pi

## NLO PARAMS
## n=1000 | perturb = True | t_max = 8 * np.pi

problem_kwargs = dict(
    n=1000,
    perturb=True,
    t_max=8*np.pi
)

# ==========================
# GAN
# ==========================

## SHO PARAMS
## niters: 10K | Gen: 16x3 | Disc: 32x2

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
    n_hidden_units=16,
    n_hidden_layers=3,
    activation=nn.Tanh(),
    residual=True,
    regress=True,
)
# Discriminator MLP
disc_kwargs = dict(
    in_dim=2,
    out_dim=1,
    n_hidden_units=32,
    n_hidden_layers=2,
    activation=nn.Tanh(),
    residual=True,
    regress=True, # true for WGAN, false otherwise
)

# ==========================
# L2 (Lagaris)
# ==========================


## NLO PARAMS
## niters: 200K | MLP: 32x8

# L2 Algorithm
L2_kwargs = dict(
    method='unsupervised',
    niters=200000,
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
    n_hidden_units=32,
    n_hidden_layers=8,
    activation=nn.Tanh(),
    residual=True,
    regress=True,
)

# ==========================
# Hyper tuning
# ==========================

# GAN
hyper_space = dict(
     # gan_kwargs
     gan_niters = [100000],
     # disc_kwargs
     disc_n_hidden_units = [32],
     disc_n_hidden_layers = [4, 6, 8, 10],
     # gen_kwargs
     gen_n_hidden_units = [32],
     gen_n_hidden_layers = [4, 6, 8, 10],
)

# L2
# hyper_space = dict(
#    # model_kwargs
#    model_n_hidden_units=[32, 64],
#    model_n_hidden_layers=[4, 6, 8],
#    # train_kwargs
#    train_niters=[10000, 50000, 100000],
#)
