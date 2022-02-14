import numpy as np
from neurodiffeq import diff
from neurodiffeq.conditions import IVP, BaseCondition
from neurodiffeq.solvers import Solver1D
from neurodiffeq.networks import FCNN
import torch
import torch.nn as nn
from denn.models import MLP
from functools import partial
from scipy import constants
import matplotlib.pyplot as plt

def get_pretrained_fcnn(pkey, save=False, pretrained=False):

    if pkey.lower().strip() == "eins":

        class CombinedModel(torch.nn.Module):
            def __init__(self, nets):
                super().__init__()
                self.nets = torch.nn.ModuleList(nets)
                
            def forward(self, t):
                outputs = [net(t) for net in self.nets]
                return torch.cat(outputs, dim=1)

        if save:
            c = constants.c/1000
            Om_m_0 = 0.15
            z_0 = 10.0
            b = 5
            H_0 = 1
            Lambda = 3*((H_0/c)**2)*(1 - Om_m_0)
            z_rescale = z_0
            transform_z = True
            transform_r = True
            custom_loss = True
            custom_reparam = True
            w_b = 1/30

            def z_to_z_prime(z):
                if transform_z:
                    z_prime = (z_0/z_rescale) - (z/z_rescale)
                else:
                    z_prime = z

                return z_prime

            def z_prime_to_z(z_prime):

                if transform_z:
                    z = z_rescale * ((z_0/z_rescale) - z_prime)
                else:
                    z = z_prime
                return z

            def r_prime_to_r(r_prime):
                if transform_r:
                    if isinstance(r_prime, torch.Tensor):
                        r = torch.exp(r_prime)
                    else:
                        r = np.exp(r_prime)
                else:
                    r = r_prime
                return r

            def r_to_r_prime(r):
                if transform_r:
                    if isinstance(r, torch.Tensor):
                        r_prime = torch.log(r)
                    else:
                        r_prime = np.log(r)
                else:
                    r_prime = r
                return r_prime

            Om_L_0 = 1 - Om_m_0
            z_prime_0 = z_to_z_prime(z_0)
            z_prime_f = z_to_z_prime(0.0)

            def x_0(z_prime):
                return 0.0

            def y_0(z_prime):
                z = z_prime_to_z(z_prime)
                return (Om_m_0*((1 + z)**3) + 2*Om_L_0)/(2*(Om_m_0*((1 + z)**3) + Om_L_0))

            def v_0(z_prime):
                z = z_prime_to_z(z_prime)
                return (Om_m_0*((1 + z)**3) + 4*Om_L_0)/(2*(Om_m_0*((1 + z)**3) + Om_L_0))

            def Om_0(z_prime):
                z = z_prime_to_z(z_prime)
                return Om_m_0*((1 + z)**3)/((Om_m_0*((1 + z)**3) + Om_L_0))

            def r_prime_0(z_prime):
                z = z_prime_to_z(z_prime)
                r_0 =  (Om_m_0*((1 + z)**3) + 4*Om_L_0)/Om_L_0
                r_prime_0 = r_to_r_prime(r_0)
                return r_prime_0

            conditions = [x_0, y_0, v_0, Om_0, r_prime_0]

            def reparam(output_network, z_prime, func):
                out = func(z_prime) + (1 - torch.exp(-z_prime + z_prime_0)) * output_network * (1 - torch.exp(-torch.tensor(w_b*b)))
                return out

            if custom_reparam:
                base = [BaseCondition() for i in range(len(conditions))]
                for i, func in enumerate(conditions):
                    base[i].parameterize = partial(reparam, func=func)
                init_vals_f_R_sys = base
            else:
                init_vals_f_R_sys = [IVP(t_0=z_prime_0, u_0=func(z_prime_0)) for func in conditions]

            def mass1_r(z_prime, y, Om, r_prime):
                z = z_prime_to_z(z_prime)
                const = ((c**2)/(6*(H_0 ** 2)))
                r = r_prime_to_r(r_prime)
                f_r = Lambda*(r - 2*(1-(b/(r+b))))

                mass = (const*Om*f_r)/(y*((z + 1) ** 3))
                return mass

            def mass2_r(z_prime, v, Om, r_prime):
                z = z_prime_to_z(z_prime)
                const = Lambda*((c**2)/(6*(H_0 ** 2)))
                r = r_prime_to_r(r_prime)
                f_R_r = 1 - 2*b/((r + b)**2)

                mass = (const*r*Om*f_R_r)/(v*((z + 1) ** 3))
                return mass

            def eq(x, y, v, Om):
                return Om + v - x - y

            def custom_loss_func(res, f, t):
                z_prime = t[0]
                cl = (res ** 2)
                x = f[0]
                y = f[1]
                v = f[2]
                Om = f[3]
                r_prime = f[4]
                w = 2
                cl = torch.exp(-w * (z_prime - z_prime_0)) * cl
                mass1_0 = Om_m_0
                mass1z = mass1_r(z_prime, y, Om, r_prime)
                cl += (((mass1_0 - mass1z)/mass1_0) ** 2)
                mass2_0 = Om_m_0
                mass2z = mass2_r(z_prime, v, Om, r_prime)
                cl += (((mass2_0 - mass2z)/mass2_0) ** 2)
                eq_0 = 1
                eqz = eq(x, y, v, Om)
                cl += (((eq_0 - eqz)/eq_0) ** 2)
                clm = torch.mean(cl, dim=0).sum()
                return clm

            dz_prime_dz = -1/z_rescale if transform_z else 1

            def f_R_sys_r(x, y, v, Om, r_prime, z_prime):
                r = r_prime_to_r(r_prime)
                z = z_prime_to_z(z_prime)

                Gamma = (r + b)*(((r+b)**2) - 2*b)/(4*r*b)

                if transform_r:
                    A = (diff(x, z_prime) * dz_prime_dz) - (-Om - 2*v + x + 4*y + x*v + x**2)/(z + 1)
                    B = (diff(y, z_prime) * dz_prime_dz) + (v*x*Gamma - x*y + 4*y - 2*y*v)/(z + 1)
                    C = (diff(v, z_prime) * dz_prime_dz) + v*(x*Gamma + 4 - 2*v)/(z + 1)
                    D = (diff(Om, z_prime) * dz_prime_dz) - Om*(-1 + 2*v + x)/(z + 1)
                    E = (diff(r_prime, z_prime) * dz_prime_dz) + (Gamma*x)/(z + 1)
                else:
                    A = (diff(x, z_prime) * dz_prime_dz) - (-Om - 2*v + x + 4*y + x*v + x**2)/(z + 1)
                    B = (diff(y, z_prime) * dz_prime_dz) + (v*x*Gamma - x*y + 4*y - 2*y*v)/(z + 1)
                    C = (diff(v, z_prime) * dz_prime_dz) + v*(x*Gamma + 4 - 2*v)/(z + 1)
                    D = (diff(Om, z_prime) * dz_prime_dz) - Om*(-1 + 2*v + x)/(z + 1)
                    E = (diff(r, z_prime) * dz_prime_dz) + (r*Gamma*x)/(z + 1)
                return [A, B, C, D, E]

            z_prime_max = max(z_prime_0, z_prime_f)
            z_prime_min = min(z_prime_0, z_prime_f)

            # The common front layers that will be shared by all networks
            common_front = FCNN(n_input_units=1, n_output_units=128, hidden_units=(32, 32))

            # Different last layers
            end1 = torch.nn.Linear(128, 1)
            end2 = torch.nn.Linear(128, 1)
            end3 = torch.nn.Linear(128, 1)
            end4 = torch.nn.Linear(128, 1)
            end5 = torch.nn.Linear(128, 1)

            net1 = torch.nn.Sequential(common_front, torch.nn.Tanh(), end1)
            net2 = torch.nn.Sequential(common_front, torch.nn.Tanh(), end2)
            net3 = torch.nn.Sequential(common_front, torch.nn.Tanh(), end3)
            net4 = torch.nn.Sequential(common_front, torch.nn.Tanh(), end4)
            net5 = torch.nn.Sequential(common_front, torch.nn.Tanh(), end5)

            solver_single = Solver1D(f_R_sys_r, conditions=init_vals_f_R_sys, t_min=z_prime_min, t_max=z_prime_max, criterion=custom_loss_func, nets=[net1, net2, net3, net4, net5])
            solver_single.fit(max_epochs=20000)

            combined_model = CombinedModel(solver_single.nets)
            torch.save(net1.state_dict(), 'config/pretrained_fcnn/eins_net1.pth')
            torch.save(net2.state_dict(), 'config/pretrained_fcnn/eins_net2.pth')
            torch.save(net3.state_dict(), 'config/pretrained_fcnn/eins_net3.pth')
            torch.save(net4.state_dict(), 'config/pretrained_fcnn/eins_net4.pth')
            torch.save(net5.state_dict(), 'config/pretrained_fcnn/eins_net5.pth')

        else:
            initialized_models = [
                torch.nn.Sequential(
                    FCNN(1, 128, hidden_units=(32, 32)),
                    torch.nn.Tanh(),
                    torch.nn.Linear(128, 1),
                )
                for _ in range(5)
            ]
            if pretrained:
                for i, net in enumerate(initialized_models):
                    net.load_state_dict(torch.load(f'C:/Users/Blake Bullwinkel/Documents/Harvard/denn/denn/config/pretrained_fcnn/eins_net{i+1}.pth')) # /n/home01/bbullwinkel/denn/denn/config/pretrained_fcnn
            combined_model = CombinedModel(initialized_models)

        return combined_model

    elif pkey.lower().strip() == "aca":

        class pretrained_aca(MLP):
            def __init__(self):
                super().__init__(in_dim=2, out_dim=1, n_hidden_units=30, 
                n_hidden_layers=5, residual=True, regress=True)

            def forward(self, x):
                for i in range(len(self.layers)):
                    x = self.layers[i](x)
                return x

        model = pretrained_aca()
        if pretrained:
            model.load_state_dict(torch.load(f'C:/Users/Blake Bullwinkel/Documents/Harvard/denn/denn/config/pretrained_fcnn/aca_gen.pth'))

        return model
    
    else:
        raise NotImplementedError(f"Pretrained FCNN not implemented for problem {pkey}.")

if __name__ == "__main__":
    eins_pretrained = get_pretrained_fcnn("eins")
    zs_prime = np.linspace(1, 0, 1000)
    zs = np.linspace(0, 10, 1000)
    pred = eins_pretrained(torch.tensor(zs_prime).reshape(-1,1))
    fig, axs = plt.subplots(5, 1, figsize=(4,20))
    axs = axs.ravel()
    for i in range(5):
        axs[i].plot(zs, pred[:,i].detach().numpy())
    plt.show()
