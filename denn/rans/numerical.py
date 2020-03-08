import denn.rans.channel_flow as chan
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

def solve_rans_scipy_solve_bvp(y, k=0.41/4, nu=0.0055555555, rho=1,
    dpdx=-1, max_nodes=100000, tol=1e-6, delta=1):
    """ use scipy solve_bvp to solve RANS equation """

    def fun(y, u):
        """ solves the equation as system """
        du_sig = np.sign(u[1])
        a = 4 * ((y**2) - delta) * y * du_sig * u[1]**2
        b = ((y**2) - delta)**2
        return np.vstack([
            u[1],
            (-(k**2)*a + dpdx/rho)/( nu + (k**2) * b * (du_sig * u[1] + np.abs(u[1])))
        ])
    def bc(ua, ub):
        """ boundary residuals (just zero) """
        return np.array([ua[0], ub[0]])

    u0 = np.zeros((2, y.size))
    return solve_bvp(fun, bc, y, u0, max_nodes=max_nodes, tol=tol)

def handle_boundary(u, i, n, U_0=0, U_L=0):
    # careful with boundaries. contains {i-2, i-1, i, i+1, i+2}
    # if we step on or beyond the boundary, just set to the boundary (check this assumption!)
    # returns u[i-2], u[i-1], u[i], u[i+1], u[i+2] while respecting boundary conditions

    if i==0: # far left
        return U_0, U_0, u[i], u[i+1], u[i+2]
    elif i==1: # second from left
        return U_0, u[i-1], u[i], u[i+1], u[i+2]
    elif i==n-2: # second from right
        return u[i-2], u[i-1], u[i], u[i+1], U_L
    elif i==n-1: # far right
        return u[i-2], u[i-1], u[i], U_L, U_L
    else: # no boundaries
        return u[i-2], u[i-1], u[i], u[i+1], u[i+2]

def Gf(u):
    n = u.shape[0]
    gvec = np.zeros(n)
    for i in range(n):

        uim2, uim1, ui, uip1, uip2 = handle_boundary(u, i, n)
        yim1, yip1 = y[i] - delta_y, y[i] + delta_y

        a = (hypers['nu'] / (delta_y ** 2)) * (uip1 - 2*ui + uim1)
        fplus = (uip2 - ui) / (2*delta_y)
        fminus = (ui - uim2) / (2*delta_y)
        b = (1 / (2 * delta_y)) * ((hypers['k'] * np.abs(yip1 - delta)) ** 2) * np.abs(fplus) * fplus
        c = (-1 / (2 * delta_y)) * ((hypers['k'] * np.abs(yim1 - delta)) ** 2) * np.abs(fminus) * fminus
        d = (-1 / hypers['rho']) * hypers['dp_dx']

        gvec[i] = a + b + c + d

    return gvec

def dGi_ui(u, i, eps=1e-8):
    n = u.shape[0]
    uim2, uim1, ui, uip1, uip2 = handle_boundary(u, i, n)
    yim1, yip1 = y[i] - delta_y, y[i] + delta_y

    a = -2*hypers['nu'] / (delta_y**2)
    b = (hypers['k']**2) / (2*delta_y)

    fplus = ((uip2 - ui)/(2*delta_y)) + eps
    fminus = ((ui - uim2)/(2*delta_y)) + eps

    c1 = (np.abs(yip1-delta)**2)
    c2 = (fplus**2) * (-1/(2*delta_y)) / np.abs(fplus)
    c3 = np.abs(fplus) * (-1/(2*delta_y))
    c = c1*(c2+c3)

    d1 = -(np.abs(yim1-delta)**2)
    d2 = (fminus**2) * (1/(2*delta_y)) / np.abs(fminus)
    d3 = np.abs(fminus) * (1/(2*delta_y))
    d = d1*(d2+d3)

    return a + b * (c + d)

def dGi_uim2(u, i, eps=1e-8):
    n = u.shape[0]
    uim2, uim1, ui, uip1, uip2 = handle_boundary(u, i, n)
    yim1, yip1 = y[i] - delta_y, y[i] + delta_y

    a = -((hypers['k'] * np.abs(yim1 - delta))**2) / (2*delta_y)
    fminus = ((ui - uim2) / (2*delta_y)) + eps
    b = (fminus**2) * (-1/(2*delta_y)) / np.abs(fminus)
    c = np.abs(fminus) * (-1/(2*delta_y))

    return a * (b + c)

def dGi_uip2(u, i, eps=1e-8):
    n = u.shape[0]
    uim2, uim1, ui, uip1, uip2 = handle_boundary(u, i, n)
    yim1, yip1 = y[i] - delta_y, y[i] + delta_y

    a = -((hypers['k'] * np.abs(yip1 - delta))**2) / (2*delta_y)
    fplus = ((uip2 - ui) / (2*delta_y)) + eps
    b = (fplus**2) * (1/(2*delta_y)) / np.abs(fplus)
    c = np.abs(fplus) * (1/(2*delta_y))

    return a * (b + c)

def jacobian(u, eps=1e-8):
    # returns an entire column of jacobian
    # careful with boundaries. contains {i-2, ..}
    n = u.shape[0]
    jac = np.zeros((n,n))
    for i in range(n):
        if i == 0:
            jac[i,i]=dGi_ui(u, i, eps)
            jac[i+1, i]=hypers['nu']/(delta_y**2)
            jac[i+2, i]=dGi_uip2(u, i, eps)
        elif i==1:
            jac[i-1, i] = hypers['nu']/(delta_y**2)
            jac[i, i] = dGi_ui(u, i, eps)
            jac[i+1, i] = hypers['nu']/(delta_y**2)
            jac[i+2, i] = dGi_uip2(u, i, eps)
        elif i==n-2:
            jac[i-2, i] = dGi_uim2(u, i, eps)
            jac[i-1, i] = hypers['nu']/(delta_y**2)
            jac[i, i] = dGi_ui(u, i, eps)
            jac[i+1, i] = hypers['nu']/(delta_y**2)
        elif i==n-1:
            jac[i-2, i] = dGi_uim2(u, i, eps)
            jac[i-1, i] = hypers['nu']/(delta_y**2)
            jac[i, i] = dGi_ui(u, i, eps)
        else:
            jac[i-2, i] = dGi_uim2(u, i, eps)
            jac[i-1, i] = hypers['nu']/(delta_y**2)
            jac[i, i] = dGi_ui(u, i, eps)
            jac[i+1, i] = hypers['nu']/(delta_y**2)
            jac[i+2, i] = dGi_uip2(u, i, eps)

    return jac

def newton(u0, tol=1e-14, max_iter=int(1e6), eps=1e-8):
    u = u0
    for it in range(max_iter):
        G = Gf(u)
        jac = jacobian(u, eps=eps)
        delta_u = np.linalg.solve(jac, -G)
        u = u + delta_u
        max_du = np.max(delta_u)
        if it % 100 == 0:
            print('iter {}/{}: max delta_u = {}'.format(it, max_iter, max_du))
        if max_du < tol:
            break
    return u
