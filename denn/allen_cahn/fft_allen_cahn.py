import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Code adapted from: https://open.umich.edu/sites/default/files/downloads/2012-parallel-spectral-numerical-methods.pdf 

def fft_allen_cahn(x, t, epsilon):
    
    nx, nt = len(x), len(t)
    dt=0.001
    v=0.25*np.sin(x)
    k = np.arange(0,nx/2)
    k = np.append(k, 0)
    k = np.concatenate((k, np.arange(-nx/2+1,0)))
    k = k*1j
    k2=np.real(np.square(k))
    tmax=np.max(t)
    tplot=tmax/(nt-1)
    plotgap=int(np.round(tplot/dt))
    nplots=int(np.round(tmax/tplot))
    data = np.zeros((nplots+1,nx))
    data[0,:] = v

    for i in np.arange(1,nplots+1):
        for n in np.arange(1,plotgap+1):
            v_hat = np.fft.fft(v) # converts to Fourier space
            vv = np.power(v, 3) # computes nonlinear term in real space
            vv = np.fft.fft(vv) # converts nonlinear term to Fourier space
            v_hat = np.divide(v_hat*(1/dt+1)-vv, 1/dt-k2*epsilon) # Implicit / Explicit
            v = np.fft.ifft(v_hat)
        data[i,:] = np.real(v)
    
    return data

if __name__ == "__main__":
    xgrid = np.linspace(0, 2*np.pi, 80)
    tgrid = np.linspace(0, 5, 26)
    uu = fft_allen_cahn(xgrid, tgrid, 0.001)
    x_grid, t_grid = np.meshgrid(xgrid, tgrid)
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(x_grid, t_grid, uu, cmap=cm.coolwarm, rcount=500, ccount=500)
    ax.view_init(elev=35, azim=-135)
    plt.show()