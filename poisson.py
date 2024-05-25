#!/usr/bin/env python
# coding: utf-8
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch

class PDE:
    def _evolve(self,*, print_err, **kwargs):
        self._scheme(**kwargs)
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
    def __init__(self, L=1.0, N=100, device="cuda"):
        self.name = "Poisson2D"
        self.L = L
        self.N = N

        self._set_device(device)

        self.x = torch.linspace(0, self.L * (1 - 1 / self.N), self.N, device=self.device)
        self.dx = self.L / self.N
        self.x += self.dx / 2

        self.u = torch.zeros((self.N+2, self.N+2), dtype=torch.float, device=self.device)
        self.source = torch.zeros((self.N+2, self.N+2), dtype=torch.float, device=self.device)


    def _set_device(self, device):
        if not torch.cuda.is_available():
            if device!="cpu":
                warnings.warn("Fall back to cpu for PyTorch")
            self.device = "cpu"
        else:
            self.device = "cuda"


    def _set_boundary_cond(self, BC):
        self.u = BC.clone().to(self.device)
        self.u[1:-1, 1:-1] = torch.zeros((self.N, self.N), dtype=torch.float, device=self.device)


    def _set_source(self, source):
        if source is not None:
            self.source[1:-1, 1:-1] = source.to(self.device)


    def _set_scheme(self, scheme):
        if scheme == "Jacobi":
            self._scheme = self._scheme_Jacobi
        elif scheme == "Gauss-Seidel":
            self._scheme = self._scheme_GS
        elif scheme == "SOR":
            self._scheme = self._scheme_SOR
        elif scheme == "CG":
            self._scheme = self._scheme_CG
        elif scheme == "CG-CPU":
            self._scheme = self._scheme_CG_CPU
        else:
            raise ValueError("No scheme found")


    def _scheme_Jacobi(self):
        u_old = self.u.clone()
        self.u[1:-1, 1:-1] = (u_old[2:, 1:-1] + u_old[:-2, 1:-1] +
                               u_old[1:-1, 2:] + u_old[1:-1, :-2] -
                               self.source[1:-1, 1:-1] * self.dx**2) / 4

        self.err = torch.norm(self.u[1:-1, 1:-1] - u_old[1:-1, 1:-1], p=1) / self.N**2


    def _scheme_GS(self):
        u_old = self.u.clone()

        for i in range(1, self.N+1):
            for j in range(1, self.N+1):

                self.u[i, j] = (self.u[i+1, j] + self.u[i-1, j] +
                                self.u[i, j+1] + self.u[i, j-1] -
                                self.source[i, j] * self.dx**2) / 4


        self.err = torch.norm(self.u[1:-1, 1:-1] - u_old[1:-1, 1:-1], p=1) / self.N**2


    def _scheme_SOR(self, w=1.0):
        u_old = self.u.clone()
        for i in range(1, self.N+1):
            for j in range(1, self.N+1):
                self.u[i, j] = (1 - w) * u_old[i, j] + w/4 * (self.u[i+1, j] + self.u[i-1, j] +
                                                             self.u[i, j+1] + self.u[i, j-1] -
                                                             self.source[i, j] * self.dx**2)
        self.err = torch.norm(self.u - u_old, p=1) / self.N**2


    def _scheme_CG(self):
        # Define A for calculation
        def A(u):
            return (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / self.dx**2

        r = self.source.clone()
        r[1:-1, 1:-1] -= A(self.u)
        d = r.clone()
        u_new = self.u.clone()

        Ad = torch.zeros_like(self.u)
        Ad[1:-1, 1:-1] = A(d)
        alpha_num = torch.sum(r[1:-1, 1:-1]**2)
        alpha_den = torch.sum(d[1:-1, 1:-1] * Ad[1:-1, 1:-1])
        alpha = alpha_num / alpha_den if alpha_den != 0 else 0

        u_new[1:-1, 1:-1] += alpha * d[1:-1, 1:-1]
        r_new = r.clone()
        r_new[1:-1, 1:-1] = r[1:-1, 1:-1] - alpha * Ad[1:-1, 1:-1]
        beta_num = torch.sum(r_new[1:-1, 1:-1]**2)
        beta_den = torch.sum(r[1:-1, 1:-1]**2)
        beta = beta_num / beta_den if beta_den != 0 else 0

        d[1:-1, 1:-1] = r_new[1:-1, 1:-1] + beta * d[1:-1, 1:-1]
        self.u, r = u_new, r_new

        self.err = torch.norm(r[1:-1, 1:-1], p=np.inf)


    def _scheme_CG_CPU(self):
        A = lambda u, i, j: (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4 * u[i, j]) / self.dx**2

        # Initialize the source term and solution vectors
        r = np.zeros_like(self.u)  # Residual
        r[1:-1, 1:-1] = self.source[1:-1, 1:-1] - np.array([[A(self.u, i, j) for j in range(1, self.N+1)] for i in range(1, self.N+1)])

        d = np.copy(r)  # Direction vector
        u_new = np.copy(self.u)  # New solution vector

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


# Run this file directly to validate against demo
if __name__ == "__main__":
    periods = 2

    adv = Poisson2D().run()
    anim = adv.animation(periods, scheme="Lax-Wendroff", nstep_per_image=10)
    #anim = adv.animation(periods, scheme="upwind", nstep_per_image=10)
    plt.show()
    print(f"{adv.t*adv.t_scale}, {adv.N}, {adv.err:.6e}")

    dif = Poisson2D().run()
    anim = dif.animation(periods, scheme="BTCS", nstep_per_image=10)
    plt.show()
    print(f"{dif.t/dif.t_scale}, {dif.N}, {dif.err:.6e}")
