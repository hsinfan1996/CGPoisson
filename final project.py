#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class PDE:
    def _evolve(self,*, print_err, **kwargs):
        self._scheme(**kwargs)

        #self.u_ref = self.ref_func(self.x, self.x)

        self.steps += 1
        if print_err:
            print(self.err)


    def run(self, scheme, BC, source=None, steps=None, terminate=1e-15, print_err=False, **kwargs):
        """
        For Advection:
        steps: number of steps to iterate
        scheme: "upwind" or "Lax-Wendroff" (Default: "Lax-Wendroff")
        IC: initial condition
        """
        self._set_scheme(scheme)
        self._set_boundary_cond(BC=BC)
        self._set_source(source=source)

        self.steps = 0
        self.err = 1.

        if steps is None:
            while self.err>terminate:
                self._evolve(print_err=print_err, **kwargs)
        else:
            for step in range(steps):
                self._evolve(print_err=print_err, **kwargs)


class Poisson2D(PDE):
    """
    L: computational domain size for each dimension (default: 1.0)
    N: number of computing cells in each dimension (default: 100)
    """
    def __init__(self, L=1.0, N=100):
        self.name="Poisson2D"

        # assigned constants
        self.L = L
        self.N = N

        # derived constants
        #self.dx = self.L / self.N                 # spatial resolution
        self.x, self.dx = np.linspace(0, self.L, self.N, endpoint=False, retstep=True)
        self.x += self.dx/2

        self.u = np.zeros((self.N+2, self.N+2), dtype=float)
        self.source = np.zeros((self.N+2, self.N+2), dtype=float)


    def _set_boundary_cond(self, BC):
        self.u = BC
        self.u[1:-1, 1:-1] = np.zeros((self.N, self.N), dtype=float)


    def _set_source(self, source):
        if source is not None:
            self.source[1:-1, 1:-1] = source


    def _set_scheme(self, scheme):
        if scheme=="Jacobi":
            self.scheme_name="Jacobi"
            self._scheme=self._scheme_Jacobi

        elif scheme=="Gauss-Seidel":
            self.scheme_name="Gauss-Seidel"
            self._scheme=self._scheme_GS

        elif scheme=="SOR":
            self.scheme_name="SOR"
            self._scheme=self._scheme_SOR

        elif scheme == "CG":
            self.scheme_name = "CG"
            self._scheme = self._scheme_CG

        else:
            raise ValueError("No scheme found")


    def _scheme_Jacobi(self):
        self.u_old = np.copy(self.u)

        for i in range(1, self.N+1):
            for j in range(1, self.N+1):
                self.u[i, j] = (self.u_old[i+1, j]
                               +self.u_old[i-1, j]
                               +self.u_old[i, j+1]
                               +self.u_old[i, j-1]
                               -self.source[i, j]*self.dx**2)/4

        self.err = np.abs(self.u[1:-1, 1:-1] - self.u_old[1:-1, 1:-1]).sum()/self.N**2



    def _scheme_GS(self):
        self.u_old = np.copy(self.u)

        for i in range(1, self.N+1):
            for j in range(1, self.N+1):
                self.u[i, j] = (self.u_old[i+1, j]
                               +self.u[i-1, j]
                               +self.u_old[i, j+1]
                               +self.u[i, j-1]
                               -self.source[i, j]*self.dx**2)/4

        self.err = np.abs(self.u[1:-1, 1:-1] - self.u_old[1:-1, 1:-1]).sum()/self.N**2
    
    def _scheme_CG(self):
        """
        Conjugate Gradient (CG) method to solve the 2D Poisson equation.
        """
        # Compute the vector 'b' from the source term and the finite difference
        # discretization of the Poisson equation
        b = self.source[1:-1, 1:-1].flatten()
        b *= -self.dx**2

        # Flatten the initial guess for the solution 'u'
        x = self.u[1:-1, 1:-1].flatten()

        # Initial residual is b - Ax, where A is the Laplacian matrix
        r = b - self._apply_laplacian(x)
        p = r.copy()
        rs_old = np.dot(r, r)

        tol = 1e-10
        max_iterations = 1000
        for _ in range(max_iterations):
            Ap = self._apply_laplacian(p)
            alpha = rs_old / np.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            rs_new = np.dot(r, r)
            if np.sqrt(rs_new) < tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        # Reshape the solution back to the matrix form and update 'u'
        self.u[1:-1, 1:-1] = x.reshape((self.N, self.N))
        self.err = np.sqrt(rs_new)

    def _apply_laplacian(self, x):
        """
        Applies the discrete Laplacian to vector 'x', assuming 'x' is a flattened
        version of the N x N grid.
        """
        N = self.N
        L = x.shape[0]
        y = np.zeros_like(x)

        # Apply the finite difference Laplacian operator
        for i in range(N):
            for j in range(N):
                index = i * N + j
                y[index] = -4 * x[index]
                if i > 0:
                    y[index] += x[index - N]
                if i < N - 1:
                    y[index] += x[index + N]
                if j > 0:
                    y[index] += x[index - 1]
                if j < N - 1:
                    y[index] += x[index + 1]
        
        return y / (self.dx ** 2)


    def _scheme_SOR(self, w=1):
        self.u_old = np.copy(self.u)
        residual = np.zeros_like(self.u, dtype=float)

        for i in range(1, self.N+1):
            for j in range(1, self.N+1):
                residual[i, j] = (self.u_old[i+1, j]
                                  +self.u[i-1, j]
                                  +self.u_old[i, j+1]
                                  +self.u[i, j-1]
                                  -self.source[i, j]*self.dx**2
                                  -4*self.u_old[i, j])/4

                self.u[i, j] = self.u_old[i, j] + w*residual[i, j]

        self.err = np.abs(residual[1:-1, 1:-1]/self.u[1:-1, 1:-1]).sum()/self.N**2
        '''
        if np.isnan(residual/self.u).any():
            self.err = 1.
        else:
        '''
        #self.err = np.abs(self.u - self.u_old).sum()/self.N**2


# Run this file directly to validate against demos
if __name__ == "__main__":
    periods = 2

    adv = Poisson2D().run("Lax-Wendroff",'IC')
    anim = adv.animation(periods, scheme="Lax-Wendroff", nstep_per_image=10)
    #anim = adv.animation(periods, scheme="upwind", nstep_per_image=10)
    plt.show()
    print(f"{adv.t*adv.t_scale}, {adv.N}, {adv.err:.6e}")

    dif = Poisson2D().run("Lax-Wendroff",'IC')
    anim = dif.animation(periods, scheme="BTCS", nstep_per_image=10)
    plt.show()
    print(f"{dif.t/dif.t_scale}, {dif.N}, {dif.err:.6e}")