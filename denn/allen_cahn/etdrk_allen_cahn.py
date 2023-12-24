import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt
from matplotlib import cm

# Code adapted from: https://ora.ox.ac.uk/objects/uuid:223cd334-15cb-4436-8b77-d30408f684c5/download_file?safe_filename=NA-03-14.pdf&file_format=application%2Fpdf&type_of_work=Report

def etdrk_allen_cahn(nx, nt):

    def cheb(nx):
        if nx == 0:
            return (0, 1)
        x = np.cos(np.pi*np.arange(0, nx+1)/nx)
        c = -1*(np.array((nx-1)*[1]))
        c = np.array([2]+[c_i**2 if i%2==1 else c_i for i, c_i in enumerate(c)]+[2]).reshape(-1,1)
        XT = np.tile(x,(nx+1,1))
        X = XT.T
        dX = X-XT
        D = (c*(1/c).T) / (dX+np.eye(nx+1))
        D = D - np.diag(np.sum(D.T, axis=0))
        return (D, x.reshape(-1,1))

    D, x = cheb(nx)
    x = x[1:nx]
    w = 0.53*x + 0.47*np.sin(-1.5*np.pi*x) - x
    u = w+x
    u = np.insert(u,0,1)
    u = np.append(u,-1).reshape(-1,1)

    h = 0.25
    M = 32
    r = 15*np.exp(1j*np.pi*(np.arange(1,M+1)-0.5)/M)
    L = np.matmul(D,D)
    L = 0.01*L[1:nx,1:nx]
    A = h*L
    E = sp.expm(A)
    E2 = sp.expm(A/2)
    I = np.eye(nx-1)
    Z = np.zeros((nx-1, nx-1))
    f1 = Z
    f2 = Z
    f3 = Z
    Q = Z

    for j in range(1, M+1):
        z = r[j-1]
        zIA = sp.inv(z*I-A)
        Q = Q + h*zIA*(np.exp(z/2)-1)
        f1 = f1 + h*zIA*(-4-z+np.exp(z)*(4-3*z+(z**2)))/(z**2)
        f2 = f2 + h*zIA*(2+z+np.exp(z)*(z-2))/(z**2)
        f3 = f3 + h*zIA*(-4-3*z-(z**2)+np.exp(z)*(4-z))/(z**2)
        
    f1 = np.real(f1/M)
    f2 = np.real(f2/M)
    f3 = np.real(f3/M)
    Q = np.real(Q/M)

    uu = u
    tt = [0]
    tmax = nt
    nmax = int(np.round(tmax/h))
    nplt = int(np.floor((tmax/nt)/h))

    for n in range(1, nmax+1):
        t = n*h
        Nu = (w+x) - (w+x)**3
        a = E2@w + Q@Nu
        Na = a + x - (a+x)**3
        b = E2@w + Q@Na
        Nb = b + x - (b+x)**3
        c = E2@a + Q@(2*Nb-Nu)
        Nc = c + x - (c+x)**3
        w = E@w + f1@Nu + 2*f2@(Na+Nb) + f3@Nc
        if n % nplt == 0:
            u = w+x
            u = np.insert(u,0,1)
            u = np.append(u,-1).reshape(-1,1)
            uu = np.append(uu, u, axis=1)
            tt.append(t)
    tt = np.array(tt)
    tt /= 70
    x = np.insert(x.flatten(),0,1)
    x = np.append(x, -1)

    return x, tt, uu

if __name__ == "__main__":
    nx, nt = 100, 70
    xgrid, tgrid, uu = etdrk_allen_cahn(nx, nt)
    x_grid, t_grid = np.meshgrid(xgrid, tgrid)
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(x_grid, t_grid, uu.T, cmap=cm.coolwarm, rcount=500, ccount=500)
    ax.view_init(elev=50, azim=-135)
    plt.show()