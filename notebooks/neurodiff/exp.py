import numpy as np
import matplotlib.pyplot as plt
from neurodiffeq import diff
from neurodiffeq.ode import solve
from neurodiffeq.conditions import IVP

exponential = lambda u, t: diff(u, t) + u
init_val_ex = IVP(t_0=0.0, u_0=1.0)

# solve the ODE
solution_ex, loss_ex = solve(
    ode=exponential, condition=init_val_ex, t_min=0.0, t_max=10.0
)

ts = np.linspace(0, 10.0, 100)
u_net = solution_ex(ts, to_numpy=True)
u_ana = np.exp(-ts)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
ax1.plot(ts, u_net, label='neurodiffeq solution', c='r')
ax1.plot(ts, u_ana, label='analytical solution', c='b', linestyle='dashed')
ax1.set_ylabel('u')
ax1.set_xlabel('t')
ax1.set_title('exponential decay equation')
ax1.legend()
ax2.plot(loss_ex['train_loss'], label='training loss')
ax2.plot(loss_ex['valid_loss'], label='validation loss')
ax2.set_yscale('log')
ax2.set_title('loss during training')
ax2.legend()
plt.show()
