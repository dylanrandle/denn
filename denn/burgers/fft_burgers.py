import numpy as np
from scipy.integrate import odeint

def fft_burgers(nu, nx, x, nt, t):

    xmin, xmax = np.min(x), np.max(x)

    dx = (xmax - xmin)/nx

    kappa = 2*np.pi*np.fft.fftfreq(nx, d=dx)

    #u0 = -np.sin(np.pi*x)
    u0 = 1/np.cosh(x)

    def rhsBurgers(u, t, kappa, nu):
        uhat = np.fft.fft(u)
        d_uhat = (1j)*kappa*uhat
        dd_uhat = -np.power(kappa, 2)*uhat
        d_u = np.fft.ifft(d_uhat)
        dd_u = np.fft.ifft(dd_uhat)
        du_dt = -u*d_u + nu*dd_u
        return du_dt.real

    u = odeint(rhsBurgers, u0, t, args=(kappa, nu))

    return u

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    u = fft_burgers(0.001, 1000, np.linspace(-5,5,1000), 100, np.linspace(0,2.5,100))
    print(u.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u_plot = u[0:-1:10,:]
    for j in range(u_plot.shape[0]):
        ys = j*np.ones(u_plot.shape[1])
        ax.plot(np.linspace(-5,5,1000),ys,u_plot[j,:])

    plt.figure()
    plt.imshow(np.flipud(u), aspect=8)
    plt.axis('off')
    plt.show()