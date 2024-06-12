import warnings

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import torch


class PDE:
    def _evolve(self,*, print_err, **kwargs):
        self._scheme(**kwargs)
        self.steps += 1

        if print_err:
            print(self.err)


    def run(self, scheme, BC, source=None, steps=None, terminate=1e-15, print_err=False, **kwargs):
        """
        scheme (str): Can be "Jacobi", "Gauss-Seidel", "SOR", "CG", "CG-CPU".
        BC (N+2 by N+2 np.ndarray): Boundary condition
        source (N by N np.ndarray): Source term. Optional (Default: None)
        steps (int): Maximum number of steps to iteraate, Optional (Default: None)
        terminate (float): Terminate iteration when average residual is less than this number. Optional (Default: 1e-12)
        print_err (bool): Print out residual for every step. Optional (Default: False)
        kwargs (dict): Additional keyword arguments to be passed to `_scheme()`
        """
        self._set_scheme(scheme)

        self._set_boundary_cond(BC=BC)
        self._set_source(source=source)
        if self.scheme == "CG":
            self._numpy_to_torch("x")
            self._numpy_to_torch("u")
            self._numpy_to_torch("source")

        self.steps = 0
        self.err = 1.

        if steps is None:
            while self.err>terminate:
                self._evolve(print_err=print_err, **kwargs)
        else:
            for step in range(steps):
                self._evolve(print_err=print_err, **kwargs)

        if self.scheme == "CG":
            self._torch_to_numpy("x")
            self._torch_to_numpy("u")
            self._torch_to_numpy("source")
            self._torch_to_numpy("err")


class Poisson2D(PDE):
    def __init__(self, L=1.0, N=100, device="cuda"):
        '''
        L (float): Domain length. Optional (Default: 1.0)
        N (int): Spatial resolution in each dimension. Optional (Default: 100)
        device (str): Can be "cuda" or "cpu". Only effective when using CG with PyTorch. Optional (Default: "cuda")
        '''
        self.name = "Poisson2D"
        self.L = L
        self.N = N

        self._set_device(device)

        self.x, self.dx = np.linspace(0, self.L, self.N+2, retstep=True)
        self.x = self.x[1:-1]

        self.u = np.zeros((self.N+2, self.N+2), dtype=float)
        self.source = np.zeros((self.N+2, self.N+2), dtype=float)


    def _set_device(self, device):
        if device=="cuda":
            if torch.cuda.is_available():
                self.device = "cuda"
                self.blocks = (self.N + 3) // 4
                self.threads = 4
                print(f"CUDA blocks set to: {self.blocks}, threads set to: {self.threads}")
            else:
                warnings.warn("Fall back to cpu for PyTorch")
                self.device = "cpu"
                torch.set_num_threads(1)
        elif device == "cpu_1thread":
            self.device = "cpu"
            torch.set_num_threads(1)
            print(f"Number of threads set to: {torch.get_num_threads()}")
        else:
            self.device = "cpu"
            torch.set_num_threads(4)
            print(f"Number of threads set to: {torch.get_num_threads()}")


    def _numpy_to_torch(self, quantity):
        setattr(self, quantity, torch.from_numpy(getattr(self, quantity)).to(self.device))


    def _torch_to_numpy(self, quantity):
        setattr(self, quantity, getattr(self, quantity).cpu().numpy())


    def _set_boundary_cond(self, BC):
        self.u = BC
        self.u[1:-1, 1:-1] = np.zeros((self.N, self.N))


    def _set_source(self, source):
        if source is not None:
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
         elif scheme == 'CG-MPI':
             self.scheme = "CG-MPI"
             self._scheme = self._scheme_CG_mpi
         else:
             raise ValueError("No scheme found")


    def _scheme_Jacobi(self):
        u_old = self.u.copy()
        self.u[1:-1, 1:-1] = (u_old[2:, 1:-1] + u_old[:-2, 1:-1]
                            + u_old[1:-1, 2:] + u_old[1:-1, :-2]
                            - self.source[1:-1, 1:-1] * self.dx**2) / 4

        self.err = np.abs(self.u[1:-1, 1:-1] - u_old[1:-1, 1:-1]).sum() / self.N**2

    def _scheme_GS(self):
        u_old = self.u.copy()

        for i in range(1, self.N+1):
            for j in range(1, self.N+1):
                self.u[i, j] = (self.u[i+1, j] + self.u[i-1, j] +
                                self.u[i, j+1] + self.u[i, j-1] -
                                self.source[i, j] * self.dx**2) / 4

        self.err = np.abs(self.u[1:-1, 1:-1] - u_old[1:-1, 1:-1]).sum() / self.N**2


    def _scheme_SOR(self, w=1):
        self.u_old = np.copy(self.u)
        residual = np.zeros_like(self.u, dtype=float)

        for i in range(1, self.N+1):
            for j in range(1, self.N+1):
                residual[i, j] = (self.u[i+1, j]
                                  +self.u[i-1, j]
                                  +self.u[i, j+1]
                                  +self.u[i, j-1]
                                  -self.source[i, j]*self.dx**2
                                  -4*self.u[i, j])/4

                self.u[i, j] = self.u_old[i, j] + w*residual[i, j]

        self.err = np.abs(residual[1:-1, 1:-1]/self.u[1:-1, 1:-1]).sum()/self.N**2


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


    def _scheme_CG_mpi(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Define A for calculation
        def A(u):
            return (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / self.dx**2

        # Initialize residuals and direction vectors
        if self.steps == 0:
            self.r = self.source.copy()
            self.r[1:-1, 1:-1] -= A(self.u)
            self.d = self.r.copy()

        # Split data for each process
        u_local = np.array_split(self.u, size, axis=0)[rank]
        d_local = np.array_split(self.d, size, axis=0)[rank]
        r_local = np.array_split(self.r, size, axis=0)[rank]

        Ad_local = np.zeros_like(u_local)
        Ad_local[1:-1, 1:-1] = A(d_local)

        # Calculate alpha
        alpha_num_local = np.sum(r_local[1:-1, 1:-1]**2)
        alpha_den_local = np.sum(d_local[1:-1, 1:-1] * Ad_local[1:-1, 1:-1])

        alpha_num = comm.allreduce(alpha_num_local, op=MPI.SUM)
        alpha_den = comm.allreduce(alpha_den_local, op=MPI.SUM)
        alpha = alpha_num / alpha_den if alpha_den != 0 else 0

        # Update solution
        u_local[1:-1, 1:-1] += alpha * d_local[1:-1, 1:-1]

        # Calculate new residual
        r_local[1:-1, 1:-1] -= alpha * Ad_local[1:-1, 1:-1]

        # Calculate beta
        beta_num_local = np.sum(r_local[1:-1, 1:-1]**2)
        beta_num = comm.allreduce(beta_num_local, op=MPI.SUM)
        beta_den = alpha_num
        beta = beta_num / beta_den if beta_den != 0 else 0

        # Update direction vector
        d_local[1:-1, 1:-1] = r_local[1:-1, 1:-1] + beta * d_local[1:-1, 1:-1]

        # Gather the updated local arrays
        u_parts = comm.gather(u_local, root=0)
        r_parts = comm.gather(r_local, root=0)
        d_parts = comm.gather(d_local, root=0)

        if rank == 0:
            self.u = np.vstack(u_parts)
            self.r = np.vstack(r_parts)
            self.d = np.vstack(d_parts)

        self.err = np.abs(self.r[1:-1, 1:-1]).sum() / self.N**2
