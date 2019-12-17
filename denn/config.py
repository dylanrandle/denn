import torch.nn as nn
import numpy as np

# ==========================
# Setup of the problem
# ==========================

problem_kwargs = dict(
    n=1000,
    perturb=True,
    t_max=8*np.pi
)

# ==========================
# GAN
# ==========================

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

## Settings for SHO
## Gen: 16x3 | Disc: 32x2

# Generator MLP
gen_kwargs = dict(
    in_dim=1,
    out_dim=1,
    n_hidden_units=32,
    n_hidden_layers=3,
    activation=nn.Tanh(),
    residual=True,
    regress=True,
)
# Discriminator MLP
disc_kwargs = dict(
    in_dim=2,
    out_dim=1,
    n_hidden_units=16,
    n_hidden_layers=3,
    activation=nn.Tanh(),
    residual=True,
    regress=True, # true for WGAN, false otherwise
)

# ==========================
# L2 (Lagaris)
# ==========================

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
    n_hidden_units=32,
    n_hidden_layers=4,
    activation=nn.Tanh(),
    residual=True,
    regress=True,
)

# ==========================
# Hyper tuning
# ==========================

# GAN
# hyper_space = dict(
#     # disc_kwargs
#     disc_n_hidden_units = [32],
#     disc_n_hidden_layers = [2, 3, 4, 5],
#     # gen_kwargs
#     gen_n_hidden_units = [32],
#     gen_n_hidden_layers = [2, 3, 4, 5],
# )

# L2
hyper_space = dict(
    # model_kwargs
    model_n_hidden_units=[32, 64],
    model_n_hidden_layers=[4, 6, 8],
    # train_kwargs
    train_niters=[10000, 50000, 100000],
)
