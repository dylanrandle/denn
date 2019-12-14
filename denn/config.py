import torch.nn as nn

# ==========================
# Setup of the problem
# ==========================

problem_kwargs = dict(
    n=100,
    perturb=True
)

# ==========================
# GAN
# ==========================

# GAN Algorithm
gan_kwargs = dict(
    method='unsupervised',
    niters=10000,
    g_lr=1e-3,
    g_betas=(0.0, 0.9),
    d_lr=1e-3,
    d_betas=(0.0, 0.9),
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
    save=False,
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

# L2 Algorithm
L2_kwargs = dict(
    method='unsupervised',
    niters=10000,
    lr=1e-3,
    betas=(0, 0.9),
    lr_schedule=True,
    obs_every=1,
    d1=1,
    d2=1,
    plot=True,
    save=False,
    fname='train_L2.png',
)
# L2 MLP
L2_mlp_kwargs = dict(
    in_dim=1,
    out_dim=1,
    n_hidden_units=16,
    n_hidden_layers=3,
    activation=nn.Tanh(),
    residual=True,
    regress=True,
)
