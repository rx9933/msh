import dolfin as dl
import ufl
import numpy as np
from hippylib.modeling.prior import _Prior
from hippylib.algorithms.linSolvers import PETScKrylovSolver, PETScLUSolver
from hippylib.algorithms.linalg import amg_method

class TVPrior:
    """Custom Total Variation regularization prior."""
    def __init__(self, Vh, alpha=0.1, beta=0.001, epsilon=1e-6, mean=None):
        self.Vh = Vh
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
        # Mass matrix
        trial = dl.TrialFunction(Vh)
        test = dl.TestFunction(Vh)
        self.M = dl.assemble(dl.inner(trial, test) * dl.dx)
        self.Msolver = hp.Operator2Solver(self.M)
        
        # Mean
        self.mean = mean
        if self.mean is None:
            self.mean = dl.Vector(self.M.mpi_comm())
            self.M.init_vector(self.mean, 0)
        
        # Simple solver for compatibility
        self.R = self.M
        self.Rsolver = self.Msolver
    
    def init_vector(self, x, dim):
        if dim == "noise":
            self.M.init_vector(x, 1)
        else:
            self.M.init_vector(x, dim)
    
    def cost(self, m):
        """Compute TV regularization cost."""
        m_func = dl.Function(self.Vh, m)
        grad_m = dl.grad(m_func)
        tv_term = dl.sqrt(dl.inner(grad_m, grad_m) + self.epsilon**2)
        tv_energy = dl.assemble(self.alpha * tv_term * dl.dx)
        l2_energy = 0.5 * self.beta * m.inner(m)
        return tv_energy + l2_energy
    
    def grad(self, m, out):
        """Compute gradient of TV regularization."""
        m_func = dl.Function(self.Vh, m)
        test = dl.TestFunction(self.Vh)
        grad_m = dl.grad(m_func)
        tv_term = dl.sqrt(dl.inner(grad_m, grad_m) + self.epsilon**2)
        d_tv = self.alpha * dl.inner(grad_m, dl.grad(test)) / tv_term * dl.dx
        d_l2 = self.beta * dl.inner(m_func, test) * dl.dx
        dl.assemble(d_tv + d_l2, tensor=out) 
        
    def sample(self, noise, s, add_mean=True):
        self.M.mult(noise, s)
        if add_mean:
            s.axpy(1., self.mean)
    
    def getHessianPreconditioner(self):
        return self.Rsolver