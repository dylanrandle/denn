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

gan_kwargs = dict(
    niters=10000,
    g_lr=2e-4,
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
    plot=True,
    save=True,
)
gen_kwargs = dict(
    in_dim=1,
    out_dim=1,
    n_hidden_units=64,
    n_hidden_layers=6,
    activation=nn.Tanh(),
    residual=True,
    regress=True,
)
disc_kwargs = dict(
    in_dim=2,
    out_dim=1,
    n_hidden_units=64,
    n_hidden_layers=4,
    activation=nn.Tanh(),
    residual=True,
    regress=True, # true for WGAN
)


# ==========================
# L2 (Lagaris)
# ==========================
L2_kwargs = dict(
    niters=10000,
    lr=2e-4,
    betas=(0, 0.9),
    lr_schedule=True,
    obs_every=1,
    d1=1,
    d2=1,
    plot=True,
    save=True,
)
L2_mlp_kwargs = dict(
    in_dim=1,
    out_dim=1,
    n_hidden_units=64,
    n_hidden_layers=6,
    activation=nn.Tanh(),
    residual=True,
    regress=True,
)
