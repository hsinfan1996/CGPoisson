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
        if self.scheme == "CG":
            self.x = torch.from_numpy(self.x).to(self.device)
            self.u = torch.from_numpy(self.u).to(self.device)
            self.source = torch.from_numpy(self.source).to(self.device)

        self.steps = 0
        self.err = 1.

        if steps is None:
            while self.err>terminate:
                self._evolve(print_err=print_err, **kwargs)
        else:
            for step in range(steps):
                self._evolve(print_err=print_err, **kwargs)

        if self.scheme == "CG":
            self.x = self.x.cpu().numpy()
            self.u = self.u.cpu().numpy()
            self.source = self.source.cpu().numpy()
            self.err = self.err.cpu().numpy()


class Poisson2D(PDE):
    def __init__(self, L=1.0, N=100, device="cuda"):
        self.name = "Poisson2D"
        self.L = L
        self.N = N

        self._set_device(device)

        self.x, self.dx = np.linspace(0, self.L, self.N, endpoint=False, retstep=True)
        self.x += self.dx / 2

        self.u = np.zeros((self.N+2, self.N+2), dtype=float)
        self.source = np.zeros((self.N+2, self.N+2), dtype=float)


    def _set_device(self, device):
        if device=="cuda":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                warnings.warn("Fall back to cpu for PyTorch")
                self.device = "cpu"
        else:
            self.device = "cpu"


    def _set_boundary_cond(self, BC):
        '''
        if self.device == "cuda":
            self.u = torch.from_numpy(BC).to(self.device)
            self.u[1:-1, 1:-1] = torch.zeros((self.N, self.N), dtype=torch.float, device=self.device)
        else:
        '''
        self.u = BC
        self.u[1:-1, 1:-1] = np.zeros((self.N, self.N))


    def _set_source(self, source):
        if source is not None:
            '''
            if self.device == "cuda":
                self.source[1:-1, 1:-1] = torch.from_numpy(source).to(self.device)
            else:
            '''
            self.source[1:-1, 1:-1] = source


    def _set_scheme(self, scheme):
         if scheme == "Jacobi":
             self.scheme = "Jacobi"
             self._scheme = self._scheme_Jacobi
         elif scheme == "Gauss-Seidel":
             self.scheme = "Gauss-Seidel"
             self._scheme = self._scheme_GS
         elif scheme == "SOR":
             self.scheme = "SOR"
             self._scheme = self._scheme_SOR
         elif scheme == "CG":
             self.scheme = "CG"
             self._scheme = self._scheme_CG
         elif scheme == "CG-CPU":
             self._set_device("cpu")
             self.scheme = "CG-CPU"
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

        if self.steps==0:
            self.r = self.source.clone()
            self.r[1:-1, 1:-1] -= A(self.u)
            self.d = self.r.clone()

        Ad = torch.zeros_like(self.u)
        Ad[1:-1, 1:-1] = A(self.d)

        alpha_num = torch.sum(self.r[1:-1, 1:-1]**2)
        alpha_den = torch.sum(self.d[1:-1, 1:-1] * Ad[1:-1, 1:-1])
        alpha = alpha_num / alpha_den if alpha_den != 0 else 0

        self.u[1:-1, 1:-1] += alpha * self.d[1:-1, 1:-1]
        self.r[1:-1, 1:-1] -= alpha * Ad[1:-1, 1:-1]
        beta_num = torch.sum(self.r[1:-1, 1:-1]**2)
        beta_den = alpha_num
        beta = beta_num / beta_den if beta_den != 0 else 0
        self.d[1:-1, 1:-1] = self.r[1:-1, 1:-1] + beta * self.d[1:-1, 1:-1]

        self.err = torch.norm(self.r[1:-1, 1:-1], p=1)/self.N**2


    def _scheme_CG_CPU(self):
        A = lambda u, i, j: (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4 * u[i, j]) / self.dx**2

        if self.steps==0:
            # Initialize the source term and solution vectors
            self.r = np.zeros_like(self.u)  # Residual
            self.r[1:-1, 1:-1] = self.source[1:-1, 1:-1] - np.array([[A(self.u, i, j) for j in range(1, self.N+1)] for i in range(1, self.N+1)])

            self.d = np.copy(self.r)  # Direction vector

        # Calculate alpha
        alpha_num = np.sum(self.r[1:-1, 1:-1] * self.r[1:-1, 1:-1])
        alpha_den = np.sum([self.d[i, j] * A(self.d, i, j) for i in range(1, self.N+1) for j in range(1, self.N+1)])
        alpha = alpha_num / alpha_den if alpha_den != 0 else 0  # to avoid division by zero

        # Update solution
        self.u[1:-1, 1:-1] += alpha * self.d[1:-1, 1:-1]

        # Calculate new residual
        self.r[1:-1, 1:-1] -= alpha * np.array([[A(self.d, i, j) for j in range(1, self.N+1)] for i in range(1, self.N+1)])

        # Calculate beta
        beta_num = np.sum(self.r[1:-1, 1:-1] * self.r[1:-1, 1:-1])
        beta_den = alpha_num
        beta = beta_num / beta_den if beta_den != 0 else 0  # to avoid division by zero

        # Update direction vector
        self.d[1:-1, 1:-1] = self.r[1:-1, 1:-1] + beta * self.d[1:-1, 1:-1]

        # Check for convergence
        self.err = np.abs(self.r[1:-1, 1:-1]).sum()/self.N**2
