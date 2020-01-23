"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary

  e.g.:
    u_D = 1 + x^2 + 2y^2
    f = -6
"""

from __future__ import print_function
from fenics import *

def compute_solution(nx, ny, f):
    # Create mesh and define function space
    mesh = UnitSquareMesh(nx-1, ny-1)
    V = FunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    # u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    u_D = Expression('0', degree=0)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(f)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Plot solution and mesh
    # plot(u)
    # plot(mesh)

    # Convert to Numpy
    u_np = u.compute_vertex_values(mesh)
    mesh_np = mesh.coordinates()

    return u_np, mesh_np


if __name__ == '__main__':
    u_np, mesh_np = compute_solution(100, 100, 1)

    # import numpy as np
    # np.save('poisson_2D_f1_uD0_n100_x0to1_y0to1', u_np)

    import matplotlib.pyplot as plt
    plt.scatter(mesh_np[:,0], mesh_np[:,1], c=u_np)
    plt.colorbar()
    plt.show()
