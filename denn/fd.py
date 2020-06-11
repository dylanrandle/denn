
from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm


# def rhs_func(x, y, M):
#     ###----- Element-wise multiplication -----###
#     g = (20 * np.multiply(np.cos(3*np.pi*X[1:-1,1:-1]), np.sin(2*np.pi*Y[1:-1,1:-1]))).flatten()
#     f = [g[i::M-2] for i in range(M-2)] # Extracts only the inner values
#     f = np.asarray(f).flatten() # Flattens into a ((M-2)**2, ) array
#     return f

def rhs_func(x, y, M):
    ###----- Element-wise multiplication -----###
    x, y = x[1:-1,1:-1], y[1:-1,1:-1]
    g = -(2*x*(y-1)*(y - 2*x + x*y + 2)*np.exp(x-y)).flatten()
    f = [g[i::M-2] for i in range(M-2)] # Extracts only the inner values
    f = np.asarray(f).flatten() # Flattens into a ((M-2)**2, ) array
    return f


def bc_dirichlet(x, y, M):
    ###----- Initializes boundary condition values -----###
    lBC = np.zeros(M) # Y[:,0]**2
    leftBC = lBC[1:M-1]

    rBC = np.zeros(M) # np.ones((M,1)).flatten()
    rightBC = rBC[1:M-1]

    tBC = np.zeros(M) #X[0,:]**3
    topBC = tBC[1:M-1]

    bBC = np.zeros(M) #np.ones((1,M)).flatten()
    bottomBC = bBC[1:M-1]

    ###----- Creates a ((M-2)**2, ) array of zeros -----###
    g1 = np.zeros(((M-2)**2, 1)).flatten()

    ###----- Fills in the top BC (red circles on p. 21) -----###
    for i in range(M-2):
        g1[(M-2)*i] = topBC[i]

    ###----- Fills in the bottom BC (blue circles on p. 21) -----###
    for j in range(M-2):
        g1[(M-2)*(j+1)-1] = bottomBC[j]

    ###----- Fills in the left BC (top orange circle on p. 21) -----###
    k1 = np.zeros((len(leftBC),1))
    k1[0] = 1.0
    leftBCk = sparse.kron(k1,leftBC).toarray().flatten()

    ###----- Fills in the right BC (bottom orange circle on p. 21) -----###
    k2 = np.zeros((len(rightBC),1))
    k2[-1] = 1.0
    rightBCk = sparse.kron(k2,rightBC).toarray().flatten()

    ###----- Collects all -----###
    g = g1 + leftBCk + rightBCk

    return [g, lBC, tBC, rBC, bBC]


def generate_lhs_matrix(M, hx, hy):

    alpha = hx**2/hy**2

    main_diag = 2 * (1 + alpha) * np.ones((M - 2, 1)).ravel()
    off_diag = -1 * np.ones((M - 2, 1)).ravel()

    a = main_diag.shape[0]

    diagonals = [main_diag, off_diag, off_diag]

    B = sparse.diags(diagonals, [0, -1, 1], shape=(a, a)).toarray()

    C = sparse.diags([-1*np.ones((M+1, 1)).ravel()], [0], shape=(a,a)).toarray()

    e1 = sparse.eye(M-2).toarray()

    A1 = sparse.kron(e1,B).toarray()

    e2 = sparse.diags([1*np.ones((M, 1)).ravel(),1*np.ones((M, 1)).ravel()], [-1,1], shape=(M-2,M-2)).toarray()

    A2 = sparse.kron(e2,C).toarray()

    mat = A1 + A2

    return mat


###========================================###

def fd():
    M = 32
    (x0, xf) = (0.0, 1.0)
    (y0, yf) = (0.0, 1.0)

    hx = (xf - x0)/(M-1)
    hy = (yf - y0)/(M-1)

    x1 = np.linspace(x0, xf, M)
    y1 = np.linspace(y0, yf, M)

    ###----- Generates a grid ----###
    X, Y = np.meshgrid(x1, y1)

    ###----- The right hand side function ----###
    frhs = rhs_func(X, Y, M)

    ###----- Boundary conditions ----###
    fbc = bc_dirichlet(X, Y, M)

    rhs = frhs*(hx**2) + fbc[0]

    A = generate_lhs_matrix(M, hx, hy)

    ###----- Solves A*x=b --> x=A\b ----###
    V = np.linalg.solve(A,rhs)

    ###----- Reshapes the 1D array into a 2D array -----###
    V = V.reshape((M-2, M-2)).T

    ###----- Fills in boundary values for Dirichlet BC -----###
    U = np.zeros((M,M))

    U[1:M-1, 1:M-1] = V
    U[:,0] = fbc[1]
    U[0,:] = fbc[2]
    U[:,M-1] = fbc[3]
    U[M-1,:] = fbc[4]

    return X, Y, U

# ###----- Plots -----###
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# surf = ax.plot_surface(X, Y, U, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# ###----- Static image -----###
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('U')
# plt.tight_layout()
# ax.view_init(20, -106)
# plt.show()


## #----- Rotate the axes and update
## for angle in range(0, 360):
##    ax.view_init(20, angle)
##    plt.draw()
##    plt.pause(.001)
#
