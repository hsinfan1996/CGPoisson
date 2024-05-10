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
        A = lambda u, i, j: (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4 * u[i, j]) / self.dx**2

        # Initialize the source term and solution vectors
        r = np.zeros_like(self.u)  # Residual
        r[1:-1, 1:-1] = self.source[1:-1, 1:-1] - np.array([[A(self.u, i, j) for j in range(1, self.N+1)] for i in range(1, self.N+1)])

        d = np.copy(r)  # Direction vector
        u_new = np.copy(self.u)  # New solution vector

        for k in range(self.N**2):
            # Calculate alpha
            alpha_num = np.sum(r[1:-1, 1:-1] * r[1:-1, 1:-1])
            alpha_den = np.sum([d[i, j] * A(d, i, j) for i in range(1, self.N+1) for j in range(1, self.N+1)])
            alpha = alpha_num / alpha_den if alpha_den != 0 else 0  # to avoid division by zero

            # Update solution
            u_new[1:-1, 1:-1] += alpha * d[1:-1, 1:-1]

            # Calculate new residual
            r_new = np.zeros_like(r)
            r_new[1:-1, 1:-1] = r[1:-1, 1:-1] - alpha * np.array([[A(d, i, j) for j in range(1, self.N+1)] for i in range(1, self.N+1)])

            # Calculate beta
            beta_num = np.sum(r_new[1:-1, 1:-1] * r_new[1:-1, 1:-1])
            beta_den = np.sum(r[1:-1, 1:-1] * r[1:-1, 1:-1])
            beta = beta_num / beta_den if beta_den != 0 else 0  # to avoid division by zero

            # Update direction vector
            d[1:-1, 1:-1] = r_new[1:-1, 1:-1] + beta * d[1:-1, 1:-1]

            # Update the old residual and solution
            r = np.copy(r_new)
            self.u = np.copy(u_new)

            # Check for convergence
            self.err = np.linalg.norm(r[1:-1, 1:-1], ord=np.inf)
            if self.err < 1e-12:  # Termination condition
                break





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