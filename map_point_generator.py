# map_point_generator.py
"""
MAP Point Generator for Multi-frequency Helmholtz Inverse Problem

This module generates MAP points and their Jacobians for the multi-frequency
Helmholtz problem. It includes:
1. MAP point generation with random prior samples
2. Full and low-rank Jacobian computation
3. Finite difference validation
4. Eigenvalue spectrum analysis
5. Active subspace computation
"""

import os
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
import hippylib as hp
import hippyflow as hf
from copy import deepcopy
import time
import psutil

# Import your multi-frequency model
from helmholtz_model import MultiFrequencyHelmholtzModel, helmholtz_settings
from main import load_true_parameter, create_smooth_prior_mean, Config

class MAPGenerator:
    """
    Generates MAP points by sampling the prior and solving the inverse problem.
    Also generates the Jacobian of the MAP point with respect to data perturbation.
    
    (Noisy data, MAP parameter, MAP Jacobian)
    
    Run in the following sequence:
    1. MAPGenerator.sample_map_point(n_data, seed)
    2. MAPGenerator.sample_posa_jacobian(n_data) to generate full Jacobian
       or MAPGenerator.sample_posa_low_rank_jacobian(n_data, rank, seed, oversampling)
    3. MAPGenerator.check_generated_jacobian(i_data, low_rank=False or True)
    """
    
    def __init__(self, model, base_dir, rel_noise=0.01, rng=hp.Random(seed=1)):
        """
        Initialize MAP Generator.
        
        Parameters:
        - model: MultiFrequencyHelmholtzModel instance
        - base_dir: Base directory for saving data
        - rel_noise: Relative noise level for synthetic data
        - rng: Random number generator
        """
        self.model = model
        self.base_dir = base_dir
        self.rel_noise = rel_noise
        self.rng = rng
        
        # Get the internal model and components
        self.problem = self.model.model.problem
        self.prior = self.model.prior
        self.misfit = self.model.misfit
        self.B = self.misfit.B
        self.Vh = self.model.Vh
        self.mpi_comm = self.Vh[hp.STATE].mesh().mpi_comm()
        
        # Initialize vectors
        self.data_vec = dl.Vector(self.mpi_comm)
        self.B.init_vector(self.data_vec, 0)
        
        self.prior_noise = dl.Vector(self.mpi_comm)
        self.prior.init_vector(self.prior_noise, "noise")
        self.x = self.model.model.generate_vector("ALL")
        
        self.noise_std_dev = None
        self.initialize_noise_std_dev()

    def initialize_noise_std_dev(self):
        """Initialize noise variance by generating example observation."""
        print("Initializing noise variance by generating example obs")
        
        # Make a reference version of the solution
        self.rng.normal(1.0, self.prior_noise)
        self.prior.sample(self.prior_noise, self.x[hp.PARAMETER])
        
        self.model.model.solveFwd(self.x[hp.STATE], self.x)
        self.misfit.B.mult(self.x[hp.STATE], self.misfit.d)
        
        MAX = self.misfit.d.norm("linf")
        self.noise_std_dev = self.rel_noise * MAX
        self.misfit.noise_variance = self.noise_std_dev * self.noise_std_dev
        self.rng.normal_perturb(self.noise_std_dev, self.misfit.d)
        
        solver = self.make_inverse_solver()
        x_map = self.model.model.generate_vector("ALL")
        x_map[hp.PARAMETER].axpy(1.0, self.x[hp.PARAMETER])
        solver.solve(x_map)
        
        if solver.converged:
            print(f"Converged in {solver.it} iterations")
        else:
            print("Did not converge")
        print(f"Termination reason: {solver.termination_reasons[solver.reason]}")
        print(f"Final gradient norm: {solver.final_grad_norm:.3e}")
        print(f"Final cost: {solver.final_cost:.3e}")
        
        self._plot_inverse_results(x_map)

    def _plot_inverse_results(self, x_map):
        """Plot inverse results for initialization."""
        m_true_fun = hp.vector2Function(self.x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        u_true_fun = hp.vector2Function(self.x[hp.STATE], self.Vh[hp.STATE])
        
        m_map_fun = hp.vector2Function(x_map[hp.PARAMETER], self.Vh[hp.PARAMETER])
        u_map_fun = hp.vector2Function(x_map[hp.STATE], self.Vh[hp.STATE])
        
        plt.figure(figsize=(15, 8))
        
        vmax = np.max(self.x[hp.PARAMETER].get_local())
        vmin = np.min(self.x[hp.PARAMETER].get_local())
        
        # Ground truth
        hp.nb.plot(m_true_fun, mytitle="Parameter", subplot_loc=231, vmin=vmin, vmax=vmax)
        hp.nb.plot(u_true_fun, mytitle="State", subplot_loc=232)
        
        obs_data = self.misfit.d.get_local()
        norm = np.linalg.norm(obs_data.reshape((-1, 2)), axis=1)
        tmp = dl.Vector(dl.MPI.comm_world, len(norm))
        tmp.set_local(norm)
        
        # MAP Point
        hp.nb.plot(m_map_fun, mytitle="MAP param", subplot_loc=233, vmin=vmin, vmax=vmax)
        hp.nb.plot(u_map_fun, mytitle="MAP state", subplot_loc=234)
        
        plt.tight_layout()
        os.makedirs(f"{self.base_dir}/data", exist_ok=True)
        plt.savefig(f'{self.base_dir}/data/example.png')
        plt.close()

    def _sample_ground_truth(self, m0=None):
        """
        Sample parameter, state, and data.
        If m0 is provided, use it as the ground truth parameter.
        """
        self.rng.normal(1.0, self.prior_noise)
        if m0 is not None:
            self.x[hp.PARAMETER].zero()
            self.x[hp.PARAMETER].axpy(1.0, m0)
        else:
            self.prior.sample(self.prior_noise, self.x[hp.PARAMETER])
        
        self.model.model.solveFwd(self.x[hp.STATE], self.x)
        self.B.mult(self.x[hp.STATE], self.misfit.d)
        self.rng.normal_perturb(self.noise_std_dev, self.misfit.d)

    def sample_map_point(self, n_data, seed, m0=None):
        """
        Sample MAP point training data by solving the inverse problem
        for random ground truth parameters and noisy data.
        
        Parameters:
        - n_data: Number of data points to sample
        - seed: Random seed for data directory
        - m0: Optional initial guess for MAP estimation (default: prior sample)
        
        Will save the following files in the data directory:
        - MAP point parameter
        - MAP point state
        - MAP point adjoint
        - Noisy observed data
        - True parameter
        - True state
        """
        x_map = self.model.model.generate_vector("ALL")
        data_dir = f"{self.base_dir}/data"
        os.makedirs(data_dir, exist_ok=True)
        
        i_data = 0
        while i_data < n_data:
            self._sample_ground_truth(m0)
            
            # Use prior sample or provided m0 as initial guess
            if m0 is not None:
                x_map[hp.PARAMETER].zero()
                x_map[hp.PARAMETER].axpy(1.0, m0)
            else:
                x_map[hp.PARAMETER].zero()
                x_map[hp.PARAMETER].axpy(1.0, self.x[hp.PARAMETER])
            
            solver = self.make_inverse_solver()
            solver.solve(x_map)
            
            if solver.converged:
                print(f"Converged in {solver.it} iterations")
                self.save_map_point(data_dir, i_data, x_map, seed)
                i_data += 1
            else:
                print(f"Did not converge (iter={solver.it})")
            
            print(f"Termination reason: {solver.termination_reasons[solver.reason]}")
            print(f"Final gradient norm: {solver.final_grad_norm:.3e}")
            print(f"Final cost: {solver.final_cost:.3e}")

    def sample_posa_jacobian(self, n_data, seed):
        """
        Sample Jacobian of MAP point with respect to data perturbation.
        
        Parameters:
        - n_data: Number of data points to sample
        - seed: Random seed for data directory
        
        Will load the saved MAP point parameter, state, adjoint, and observed data
        generated by sample_map_point method.
        """
        jacobian = MAPJacobian(self.model.model, cg_rel_tol=1e-10)
        data_dir = f"{self.base_dir}/data/seed{seed}"
        os.makedirs(data_dir, exist_ok=True)
        
        x_map = self.model.model.generate_vector("ALL")
        data_test = dl.Vector(self.mpi_comm)
        self.B.init_vector(data_test, 0)
        jacobian_action = self.model.model.generate_vector(hp.PARAMETER)
        
        dim_data = self.misfit.d.get_local().shape[0]
        
        for i_data in range(n_data):
            self.load_map_point(data_dir, i_data, x_map)
            self.misfit.d.set_local(np.load(f"{data_dir}/obs_{i_data}.npy"))
            
            jacobian_np = np.zeros((self.Vh[hp.PARAMETER].dim(), dim_data))
            jacobian.setLinearizationPoint(x_map, False)
            
            for dim_i in range(dim_data):
                ei = np.zeros(dim_data)
                ei[dim_i] = 1.0
                data_test.set_local(ei)
                jacobian.mult(data_test, jacobian_action)
                jacobian_np[:, dim_i] = jacobian_action.get_local()
                
                if dim_i % 100 == 0:
                    print(f"  Column {dim_i}/{dim_data} done")
            
            np.save(f"{data_dir}/map_jacobian_{i_data}.npy", jacobian_np)
            print(f"Saved MAP Jacobian for data point {i_data} with shape {jacobian_np.shape}")

    def sample_posa_low_rank_jacobian(self, n_data, rank, seed, oversampling=10):
        """
        Sample Jacobian of MAP point with respect to data perturbation
        with a low rank approximation of the Hessian at the MAP point.
        
        Parameters:
        - n_data: Number of data points to sample
        - rank: Rank of the low rank approximation
        - seed: Random seed for data directory
        - oversampling: Oversampling factor for the low rank approximation
        """
        jacobian = LowRankMAPJacobian(self.model.model, rank, oversampling=oversampling)
        data_dir = f"{self.base_dir}/data/seed{seed}"
        os.makedirs(data_dir, exist_ok=True)
        
        x_map = self.model.model.generate_vector("ALL")
        data_test = dl.Vector(self.mpi_comm)
        self.B.init_vector(data_test, 0)
        jacobian_action = self.model.model.generate_vector(hp.PARAMETER)
        
        dim_data = self.misfit.d.get_local().shape[0]
        
        for i_data in range(n_data):
            self.load_map_point(data_dir, i_data, x_map)
            self.misfit.d.set_local(np.load(f"{data_dir}/obs_{i_data}.npy"))
            
            jacobian_np = np.zeros((self.Vh[hp.PARAMETER].dim(), dim_data))
            jacobian.setLinearizationPoint(x_map, False)
            
            for dim_i in range(dim_data):
                ei = np.zeros(dim_data)
                ei[dim_i] = 1.0
                data_test.set_local(ei)
                jacobian.mult(data_test, jacobian_action)
                jacobian_np[:, dim_i] = jacobian_action.get_local()
                
                if dim_i % 100 == 0:
                    print(f"  Column {dim_i}/{dim_data} done")
            
            np.save(f"{data_dir}/low_rank_map_jacobian_rank_{rank}_{i_data}.npy", jacobian_np)
            print(f"Saved Low rank MAP Jacobian for data point {i_data} with shape {jacobian_np.shape}")

    def sample_jacobian_at_map(self, n_data, seed):
        """
        Sample Jacobian at MAP points (forward operator from parameters to observations).
        """
        jacobian = JacobianAtMAP(self.model.model)
        data_dir = f"{self.base_dir}/data/seed{seed}"
        os.makedirs(data_dir, exist_ok=True)
        
        x_map = self.model.model.generate_vector("ALL")
        
        for i_data in range(n_data):
            print(f"Processing data point {i_data}")
            self.load_map_point(data_dir, i_data, x_map)
            print("  Loaded MAP point")
            
            jacobian.setLinearizationPoint(x_map, False)
            print("  Set linearization point")
            
            # Compute full Jacobian matrix using adjoint method
            J = jacobian.compute_jacobian_adjoint(x_map)
            print(f"  Computed Jacobian with shape {J.shape}")
            
            np.save(f"{data_dir}/forward_jacobian_{i_data}.npy", J)
            print(f"Saved forward Jacobian for data point {i_data}")

    def check_generated_jacobian(self, i_data, seed, low_rank=False):
        """
        Check the generated Jacobian by comparing with finite difference.
        
        Parameters:
        - i_data: Index of the data point to check
        - seed: Random seed for data directory
        - low_rank: Check low rank Jacobian if True, otherwise full Jacobian
        """
        data_dir = f"{self.base_dir}/data/seed{seed}"
        x_map = self.model.model.generate_vector("ALL")
        self.load_map_point(data_dir, i_data, x_map)
        self.misfit.d.set_local(np.load(f"{data_dir}/obs_{i_data}.npy"))
        
        if low_rank:
            jacobian_np = np.load(f"{data_dir}/low_rank_map_jacobian_{i_data}.npy")
        else:
            jacobian_np = np.load(f"{data_dir}/map_jacobian_{i_data}.npy")
        
        data_test = dl.Vector(self.mpi_comm)
        self.B.init_vector(data_test, 0)
        self.rng.normal(self.noise_std_dev, data_test)
        
        finite_difference_posa_jacobian(self.model.model, x_map, data_test, jacobian_np, 
                                       self.base_dir, i_data)

    def save_map_point(self, data_dir, i_data, x_map, seed):
        """Save MAP point data to files."""
        seed_dir = f"{data_dir}/seed{seed}"
        os.makedirs(seed_dir, exist_ok=True)
        np.save(f"{seed_dir}/map_adjoint_{i_data}.npy", x_map[hp.ADJOINT].get_local())
        np.save(f"{seed_dir}/map_param_{i_data}.npy", x_map[hp.PARAMETER].get_local())
        np.save(f"{seed_dir}/map_state_{i_data}.npy", x_map[hp.STATE].get_local())
        np.save(f"{seed_dir}/obs_{i_data}.npy", self.misfit.d.get_local())
        np.save(f"{seed_dir}/true_param_{i_data}.npy", self.x[hp.PARAMETER].get_local())
        np.save(f"{seed_dir}/true_state_{i_data}.npy", self.x[hp.STATE].get_local())

    def load_map_point(self, data_dir, i_data, x_map):
        """Load MAP point data from files."""
        x_map[hp.ADJOINT].set_local(np.load(f"{data_dir}/map_adjoint_{i_data}.npy"))
        x_map[hp.PARAMETER].set_local(np.load(f"{data_dir}/map_param_{i_data}.npy"))
        x_map[hp.STATE].set_local(np.load(f"{data_dir}/map_state_{i_data}.npy"))

    def make_inverse_solver(self):
        """Create an inverse solver with appropriate parameters."""
        solver = hp.ReducedSpaceNewtonCG(self.model.model)
        solver.parameters["rel_tolerance"] = 1e-8
        solver.parameters["abs_tolerance"] = 1e-12
        solver.parameters["max_iter"] = 1#000
        solver.parameters["GN_iter"] = 1#5
        solver.parameters["globalization"] = "LS"
        solver.parameters["LS"]["c_armijo"] = 1e-4
        solver.parameters["LS"]["max_backtracking_iter"] = 1#800
        solver.parameters["cg_max_iter"] = 3#00
        solver.parameters["cg_coarse_tolerance"] = 0.1
        return solver

    def plot_eigenvalue_spectrum(self, data_idx, rank, seed):
        """Plot eigenvalue spectrum using SpectrumAnalyzer."""
        analyzer = SpectrumAnalyzer(self.model.model, self.base_dir)
        analyzer.plot_eigenvalue_spectrum(data_idx, rank, seed)


class MAPJacobian:
    """
    Jacobian of the MAP point with respect to data perturbation.
    Computes J = d(m_map)/d(data) at the MAP point.
    """
    
    def __init__(self, model, cg_rel_tol=1e-10):
        self.model = model
        self.cg_rel_tol = cg_rel_tol
        
        self.problem = self.model.problem
        self.prior = self.model.prior
        self.misfit = self.model.misfit
        self.B = self.misfit.B
        self.noise_variance = self.misfit.noise_variance
        self.Vh = self.model.problem.Vh
        self.mpi_comm = self.Vh[hp.STATE].mesh().mpi_comm()
        
        self.observable = hf.LinearStateObservable(self.problem, self.B)
        self.jacobian = hf.ObservableJacobian(self.observable)
        
        self.Gamma_data_test = dl.Vector(self.mpi_comm)
        self.B.init_vector(self.Gamma_data_test, 0)
        self.HessApply = None

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """Set the linearization point for the Jacobian."""
        self.model.problem.setLinearizationPoint(x, gauss_newton_approx)
        self.HessApply = hp.ReducedHessian(self.model, misfit_only=False)
        self.HessApply.gauss_newton_approx = False

    def mult(self, data_test, jacobian_action):
        """
        Apply the Jacobian: data_test → jacobian_action
        jacobian_action = H^{-1} * J^T * Gamma_data * data_test
        """
        # Apply covariance inverse to data perturbation
        self.Gamma_data_test.zero()
        self.Gamma_data_test.axpy(1/self.noise_variance, data_test)
        
        # Apply transpose of the Jacobian operator
        hessian_rhs = self.model.generate_vector(hp.PARAMETER)
        self.jacobian.transpmult(self.Gamma_data_test, hessian_rhs)
        
        # Solve Hessian system
        cg_parameters = hp.CGSolverSteihaug_ParameterList()
        cg_parameters['rel_tolerance'] = self.cg_rel_tol
        solver = hp.CGSolverSteihaug(cg_parameters)
        solver.set_operator(self.HessApply)
        solver.set_preconditioner(self.prior.Rsolver)
        solver.solve(jacobian_action, hessian_rhs)


class LowRankMAPJacobian:
    """
    Jacobian of the MAP point with respect to data perturbation
    using a low rank approximation of the Hessian.
    """
    
    def __init__(self, model, rank, oversampling=10):
        self.model = model
        self.rank = rank
        self.oversampling = oversampling
        
        self.problem = self.model.problem
        self.prior = self.model.prior
        self.misfit = self.model.misfit
        self.B = self.misfit.B
        self.noise_variance = self.misfit.noise_variance
        self.Vh = self.model.problem.Vh
        self.mpi_comm = self.Vh[hp.STATE].mesh().mpi_comm()
        
        self.observable = hf.LinearStateObservable(self.problem, self.B)
        self.jacobian = hf.ObservableJacobian(self.observable)
        
        self.Gamma_data_test = dl.Vector(self.mpi_comm)
        self.B.init_vector(self.Gamma_data_test, 0)
        
        self.m_dummy = self.model.generate_vector(hp.PARAMETER)
        self.Omega = hp.MultiVector(self.m_dummy, self.rank + self.oversampling)
        hp.parRandom.normal(1.0, self.Omega)
        
        self.low_rank_hessian = None
        self.eigenvalues = None
        self.eigenvectors = None

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """
        Set the linearization point for the Jacobian and compute low-rank Hessian.
        """
        self.model.problem.setLinearizationPoint(x, gauss_newton_approx)
        Hmisfit = hp.ReducedHessian(self.model, misfit_only=True)
        
        d, U = hp.doublePassG(
            Hmisfit, 
            self.prior.R, 
            self.prior.Rsolver, 
            self.Omega, 
            self.rank, 
            s=1, 
            check=False
        )
        
        self.eigenvalues = d
        self.eigenvectors = U
        self.low_rank_hessian = hp.LowRankHessian(self.prior, d, U)

    def mult(self, data_test, jacobian_action):
        """
        Apply the Jacobian using low-rank Hessian approximation.
        """
        # Apply covariance inverse to data perturbation
        self.Gamma_data_test.zero()
        self.Gamma_data_test.axpy(1/self.noise_variance, data_test)
        
        # Apply transpose of the Jacobian operator
        hessian_rhs = self.model.generate_vector(hp.PARAMETER)
        self.jacobian.transpmult(self.Gamma_data_test, hessian_rhs)
        
        # Solve using low rank Hessian
        self.low_rank_hessian.solve(jacobian_action, hessian_rhs)


class JacobianAtMAP:
    """
    Jacobian of the observable with respect to parameter perturbation at the MAP point.
    Computes J = d(observable)/d(parameter) at the MAP point.
    """
    
    def __init__(self, model):
        self.model = model
        self.problem = self.model.problem
        self.prior = self.model.prior
        self.misfit = self.model.misfit
        self.B = self.misfit.B
        self.Vh = self.model.problem.Vh
        self.mpi_comm = self.Vh[hp.STATE].mesh().mpi_comm()
        
        self.observable = hf.LinearStateObservable(self.problem, self.B)
        self.jacobian = hf.ObservableJacobian(self.observable)
        
        self.x = self.model.generate_vector("ALL")
        self.x[hp.PARAMETER] = self.model.generate_vector(hp.PARAMETER)
        self.x[hp.STATE] = self.model.generate_vector(hp.STATE)
        self.x[hp.ADJOINT] = self.model.generate_vector(hp.ADJOINT)

    def setLinearizationPoint(self, x_map, gauss_newton_approx=False):
        """Set the linearization point (MAP point) for the Jacobian."""
        if x_map[hp.PARAMETER] is not None:
            self.x[hp.PARAMETER].zero()
            self.x[hp.PARAMETER].axpy(1.0, x_map[hp.PARAMETER])
        if x_map[hp.STATE] is not None:
            self.x[hp.STATE].zero()
            self.x[hp.STATE].axpy(1.0, x_map[hp.STATE])
        if x_map[hp.ADJOINT] is not None:
            self.x[hp.ADJOINT].zero()
            self.x[hp.ADJOINT].axpy(1.0, x_map[hp.ADJOINT])
        
        self.model.problem.setLinearizationPoint(self.x, gauss_newton_approx)

    def compute_jacobian_adjoint(self, x_map):
        """
        Compute Jacobian using ObservableJacobian's built-in transpmult method.
        Returns J = d(observable)/d(parameter) as a dense matrix.
        """
        self.problem.setLinearizationPoint(x_map, False)
        
        e_i = dl.Vector(self.mpi_comm)
        self.B.init_vector(e_i, 0)
        
        n_obs = e_i.size()
        n_params = self.problem.Vh[hp.PARAMETER].dim()
        
        J = np.zeros((n_obs, n_params))
        J_row = dl.Function(self.problem.Vh[hp.PARAMETER]).vector()
        
        for i in range(n_obs):
            e_i.zero()
            e_i[i] = 1.0
            J_row.zero()
            self.jacobian.transpmult(e_i, J_row)
            J[i, :] = J_row[:]
            
            if i % 100 == 0:
                print(f"  Column {i}/{n_obs} done")
        
        return J


class IdentityOperator:
    """Simple identity operator for use in output space reduction."""
    
    def __init__(self, size):
        self.size = size
    
    def init_vector(self, x, dim):
        x.init(self.size)
    
    def mult(self, x, y):
        y.zero()
        y.axpy(1.0, x)
    
    def solve(self, y, x):
        x.zero()
        x.axpy(1.0, y)


class SpectrumAnalyzer:
    """Class dedicated to spectral analysis of inverse problems."""
    
    def __init__(self, model, base_dir):
        self.model = model
        self.base_dir = base_dir
        self.Vh = model.problem.Vh
        self.prior = model.prior
        self.misfit = model.misfit
        self.B = model.misfit.B
        self.mpi_comm = self.Vh[hp.STATE].mesh().mpi_comm()
    
    def load_map_point(self, data_dir, i_data, x_map, seed):
        """Helper to load MAP points."""
        seed_dir = f"{data_dir}/seed{seed}"
        x_map[hp.ADJOINT].set_local(np.load(f"{seed_dir}/map_adjoint_{i_data}.npy"))
        x_map[hp.PARAMETER].set_local(np.load(f"{seed_dir}/map_param_{i_data}.npy"))
        x_map[hp.STATE].set_local(np.load(f"{seed_dir}/map_state_{i_data}.npy"))
    
    def plot_eigenvalue_spectrum(self, data_idx, rank, seed):
        """Plot eigenvalue spectrum for various operators."""
        import matplotlib.pyplot as plt
        
        data_dir = f"{self.base_dir}/data"
        
        # Load MAP point
        x_map = self.model.generate_vector("ALL")
        self.load_map_point(data_dir, data_idx, x_map, seed)
        
        # Set linearization point
        self.model.problem.setLinearizationPoint(x_map, False)
        
        print("Creating operators...")
        
        # Hessian operator
        H = hp.ReducedHessian(self.model, misfit_only=False)
        H.gauss_newton_approx = False
        
        # Observable Jacobian operator
        observable = hf.LinearStateObservable(self.model.problem, self.B)
        J_operator = hf.ObservableJacobian(observable)
        
        print("Computing eigenvalues using randomized methods...")
        
        def compute_hessian_eigenvalues(hessian, prior, rank):
            m_dummy = self.model.generate_vector(hp.PARAMETER)
            Omega = hp.MultiVector(m_dummy, rank + 10)
            hp.parRandom.normal(1.0, Omega)
            d, U = hp.doublePassG(hessian, prior.R, prior.Rsolver, Omega, rank, s=1)
            return np.array(d)
        
        def compute_jacobian_singular_values(jacobian_operator, rank):
            param_test = self.model.generate_vector(hp.PARAMETER)
            Omega_param = hp.MultiVector(param_test, min(rank + 10, param_test.size()))
            hp.parRandom.normal(1.0, Omega_param)
            
            data_test = dl.Vector(self.mpi_comm)
            self.B.init_vector(data_test, 0)
            
            Y = hp.MultiVector(data_test, Omega_param.nvec())
            for i in range(Omega_param.nvec()):
                jacobian_operator.mult(Omega_param[i], Y[i])
            
            Y_mat = hf.mv_to_dense(Y)
            s = np.linalg.svd(Y_mat, compute_uv=False)
            return s[:min(rank, len(s))]
        
        # Compute eigenvalues/singular values
        print("Computing Hessian eigenvalues...")
        eigvals_H = compute_hessian_eigenvalues(H, self.prior, rank)
        eigvals_H = np.sort(eigvals_H)[::-1]
        
        print("Computing H_inverse eigenvalues...")
        eigvals_H_inv = 1.0 / eigvals_H
        
        print("Computing Jacobian singular values...")
        eigvals_J = compute_jacobian_singular_values(J_operator, rank)
        
        print("Computing approximate eigenvalues for H_inverse @ J.T...")
        eigvals_H_inv_JT_approx = eigvals_J * np.sqrt(eigvals_H_inv[:len(eigvals_J)])
        
        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Eigenvalue spectra
        axes[0, 0].semilogy(eigvals_H, 'b-', label='H (Hessian)', alpha=0.7, linewidth=2)
        axes[0, 0].semilogy(eigvals_H_inv, 'r-', label='H_inverse', alpha=0.7, linewidth=2)
        axes[0, 0].semilogy(eigvals_J, 'g-', label='J (Jacobian)', alpha=0.7, linewidth=2)
        axes[0, 0].semilogy(eigvals_H_inv_JT_approx, 'm-', label='H_inverse @ J.T (approx)', alpha=0.7, linewidth=2)
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Eigenvalue/Singular Value')
        axes[0, 0].set_title('Eigenvalue/Singular Value Spectra')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative distribution
        def safe_cumsum(vals):
            total = np.sum(vals)
            return np.cumsum(vals) / total if total > 0 else np.cumsum(vals)
        
        axes[0, 1].plot(safe_cumsum(eigvals_H), 'b-', label='H (Hessian)', alpha=0.7, linewidth=2)
        axes[0, 1].plot(safe_cumsum(eigvals_H_inv), 'r-', label='H_inverse', alpha=0.7, linewidth=2)
        axes[0, 1].plot(safe_cumsum(eigvals_J), 'g-', label='J (Jacobian)', alpha=0.7, linewidth=2)
        axes[0, 1].plot(safe_cumsum(eigvals_H_inv_JT_approx), 'm-', label='H_inverse @ J.T (approx)', alpha=0.7, linewidth=2)
        axes[0, 1].set_xlabel('Number of eigenvalues/singular values')
        axes[0, 1].set_ylabel('Cumulative fraction')
        axes[0, 1].set_title('Cumulative Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Normalized spectra
        axes[1, 0].semilogy(eigvals_H / eigvals_H[0], 'b-', label='H (normalized)', alpha=0.7, linewidth=2)
        axes[1, 0].semilogy(eigvals_H_inv / eigvals_H_inv[0], 'r-', label='H_inverse (normalized)', alpha=0.7, linewidth=2)
        axes[1, 0].semilogy(eigvals_J / eigvals_J[0], 'g-', label='J (normalized)', alpha=0.7, linewidth=2)
        axes[1, 0].semilogy(eigvals_H_inv_JT_approx / eigvals_H_inv_JT_approx[0], 'm-', label='H_inverse @ J.T (normalized)', alpha=0.7, linewidth=2)
        axes[1, 0].set_xlabel('Index')
        axes[1, 0].set_ylabel('Normalized eigenvalue/singular value')
        axes[1, 0].set_title('Normalized Spectra')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Condition numbers
        condition_numbers = [
            eigvals_H[0] / eigvals_H[-1] if eigvals_H[-1] > 0 else np.inf,
            eigvals_H_inv[0] / eigvals_H_inv[-1] if eigvals_H_inv[-1] > 0 else np.inf,
            eigvals_J[0] / eigvals_J[-1] if eigvals_J[-1] > 0 else np.inf,
            eigvals_H_inv_JT_approx[0] / eigvals_H_inv_JT_approx[-1] if eigvals_H_inv_JT_approx[-1] > 0 else np.inf
        ]
        
        matrices = ['H', 'H_inv', 'J', 'H_inv@J.T']
        bars = axes[1, 1].bar(matrices, np.log10(np.minimum(condition_numbers, 1e16)))
        axes[1, 1].set_ylabel('log10(Condition number)')
        axes[1, 1].set_title('Matrix Condition Numbers')
        
        for bar, cond in zip(bars, condition_numbers):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{cond:.1e}', ha='center', va='bottom', rotation=0, fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(f"{self.base_dir}/data", exist_ok=True)
        plt.savefig(f'{self.base_dir}/data/eigenvalue_spectrum_data{data_idx}_rank{rank}.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.base_dir}/data/eigenvalue_spectrum_data{data_idx}_rank{rank}.pdf', 
                   bbox_inches='tight')
        plt.close()
        
        # Print summary
        print("\n=== Eigenvalue Spectrum Summary ===")
        print(f"Hessian (H):")
        print(f"  - Min eigenvalue: {eigvals_H[-1]:.3e}")
        print(f"  - Max eigenvalue: {eigvals_H[0]:.3e}")
        print(f"  - Condition number: {condition_numbers[0]:.3e}")
        
        print(f"\nH_inverse:")
        print(f"  - Min eigenvalue: {eigvals_H_inv[-1]:.3e}")
        print(f"  - Max eigenvalue: {eigvals_H_inv[0]:.3e}")
        print(f"  - Condition number: {condition_numbers[1]:.3e}")
        
        print(f"\nJ (Jacobian):")
        print(f"  - Min singular value: {eigvals_J[-1]:.3e}")
        print(f"  - Max singular value: {eigvals_J[0]:.3e}")
        print(f"  - Condition number: {condition_numbers[2]:.3e}")
        
        print(f"\nH_inverse @ J.T (approx):")
        print(f"  - Min eigenvalue: {eigvals_H_inv_JT_approx[-1]:.3e}")
        print(f"  - Max eigenvalue: {eigvals_H_inv_JT_approx[0]:.3e}")
        print(f"  - Condition number: {condition_numbers[3]:.3e}")
        
        # Save data
        np.savez(f'{self.base_dir}/data/eigenvalues_data{data_idx}_rank{rank}.npz',
                H=eigvals_H, H_inv=eigvals_H_inv, J=eigvals_J, 
                H_inv_JT_approx=eigvals_H_inv_JT_approx,
                condition_numbers=condition_numbers)


def finite_difference_posa_jacobian(model, x_map, data_test, jacobian_np, 
                                   base_dir=None, i_data=None):
    """
    Compute finite difference Jacobian for the MAP point.
    
    Parameters:
    - model: The model object
    - x_map: The MAP point
    - data_test: Test direction for finite difference
    - jacobian_np: The numpy array of the Jacobian
    - base_dir: Directory to save plots
    - i_data: Data point index for labeling
    """
    true_data = model.misfit.d.copy()
    
    epsilons = [1e-1, 1e-3, 1e-5]
    fd_errors = []
    ad_np = jacobian_np @ data_test.get_local()
    
    for eps in epsilons:
        perturbed_data = dl.Vector()
        model.misfit.B.init_vector(perturbed_data, 0)
        perturbed_data.axpy(1., true_data)
        perturbed_data.axpy(eps, data_test)
        
        model.misfit.d.zero()
        model.misfit.d.axpy(1., perturbed_data)
        
        # Use the previous MAP point as initial guess
        m_perturbed = x_map[hp.PARAMETER].copy()
        
        solver = hp.ReducedSpaceNewtonCG(model)
        solver.parameters["rel_tolerance"] = 1e-12
        solver.parameters["abs_tolerance"] = 1e-12
        solver.parameters["max_iter"] = 25
        solver.parameters["GN_iter"] = 5
        solver.parameters["globalization"] = "LS"
        solver.parameters["LS"]["c_armijo"] = 1e-4
        x_perturbed = solver.solve([None, m_perturbed, None])
        
        finite_difference_derivative = (x_perturbed[hp.PARAMETER] - x_map[hp.PARAMETER]) / eps
        fd_np = finite_difference_derivative.get_local()
        
        fd_error = np.linalg.norm(fd_np - ad_np) / (np.linalg.norm(ad_np) + 1e-12)
        fd_errors.append(fd_error)
        print(f"  eps={eps:.1e}, FD error={fd_error:.3e}")
        
        # Plot examples
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        fd_fun = dl.Function(model.problem.Vh[hp.PARAMETER])
        fd_fun.vector().set_local(fd_np)
        ax = dl.plot(fd_fun, title="Finite difference", axes=axes[0])
        plt.colorbar(ax, ax=axes[0])
        
        ad_fun = dl.Function(model.problem.Vh[hp.PARAMETER])
        ad_fun.vector().set_local(ad_np)
        ax = dl.plot(ad_fun, title="Inverse Jacobian Action", axes=axes[1])
        plt.colorbar(ax, ax=axes[1])
        
        diff_fun = dl.Function(model.problem.Vh[hp.PARAMETER])
        diff_fun.vector().set_local(fd_np - ad_np)
        ax = dl.plot(diff_fun, title="Derivative error", axes=axes[2])
        plt.colorbar(ax, ax=axes[2])
        
        plt.tight_layout()
        if base_dir is not None:
            os.makedirs(f"{base_dir}/data", exist_ok=True)
            label = f"_data{i_data}" if i_data is not None else ""
            plt.savefig(f"{base_dir}/data/fd_validation_eps_{eps:.1e}{label}.png", 
                       dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot convergence
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(epsilons, fd_errors, '-o', linewidth=2, markersize=8)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('FD Error')
    ax.set_title('Finite Difference Convergence')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if base_dir is not None:
        label = f"_data{i_data}" if i_data is not None else ""
        plt.savefig(f"{base_dir}/data/fd_convergence{label}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return fd_errors


class MAPGeneratorWithCounters(MAPGenerator):
    """
    Extended MAPGenerator that counts forward solves and backtracking iterations.
    """
    
    def __init__(self, model, base_dir, rel_noise=0.01, rng=hp.Random(seed=1)):
        super().__init__(model, base_dir, rel_noise, rng)
        
        self.forward_solve_count = 0
        self.backtracking_iterations = 0
        self.total_iterations = 0
        
        self.original_solveFwd = self.model.model.solveFwd
        self.model.model.solveFwd = self._counted_solveFwd
    
    def _counted_solveFwd(self, state, x):
        """Wrapper that counts forward solves."""
        self.forward_solve_count += 1
        return self.original_solveFwd(state, x)
    
    def reset_counters(self):
        """Reset all counters."""
        self.forward_solve_count = 0
        self.backtracking_iterations = 0
        self.total_iterations = 0
    
    def make_inverse_solver(self):
        """Override to count backtracking iterations."""
        solver = hp.ReducedSpaceNewtonCG(self.model.model)
        solver.parameters["rel_tolerance"] = 1e-8
        solver.parameters["abs_tolerance"] = 1e-12
        solver.parameters["max_iter"] = 1000
        solver.parameters["GN_iter"] = 5
        solver.parameters["globalization"] = "LS"
        solver.parameters["LS"]["c_armijo"] = 1e-4
        solver.parameters["LS"]["max_backtracking_iter"] = 800
        solver.parameters["cg_max_iter"] = 300
        solver.parameters["cg_coarse_tolerance"] = 0.1
        
        # Monkey patch the line search to count backtracking
        original_line_search = solver.line_search
        
        def counted_line_search(*args, **kwargs):
            alpha, n_backtracks = original_line_search(*args, **kwargs)
            self.backtracking_iterations += n_backtracks
            return alpha, n_backtracks
        
        solver.line_search = counted_line_search
        self.current_solver = solver
        return solver
    
    def sample_map_point(self, n_data, seed, m0=None):
        """Override to reset and report counters."""
        self.reset_counters()
        super().sample_map_point(n_data, seed, m0)
        
        print(f"\n=== Solver Statistics ===")
        print(f"Total forward solves: {self.forward_solve_count}")
        print(f"Total backtracking iterations: {self.backtracking_iterations}")
        print(f"Average forward solves per MAP point: {self.forward_solve_count / n_data:.1f}")
        print(f"Average backtracking per MAP point: {self.backtracking_iterations / n_data:.1f}")
        
        return self.forward_solve_count, self.backtracking_iterations


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Setup configuration
    config = Config()
    config.fig_dir = "map_point_data"
    
    # Setup model
    print("=" * 80)
    print("SETTING UP MULTI-FREQUENCY HELMHOLTZ MODEL")
    print("=" * 80)
    
    settings = helmholtz_settings()
    settings['source_frequencies'] = config.source_frequencies
    settings['use_tv_regularization'] = config.use_tv
    
    model = MultiFrequencyHelmholtzModel(settings)
    print(f"State space dimension: {model.Vh[hp.STATE].dim()}")
    print(f"Parameter space dimension: {model.Vh[hp.PARAMETER].dim()}")
    
    # Load true parameter and create prior mean
    m_true = load_true_parameter(model)
    m_smooth = create_smooth_prior_mean(model, m_true)
    model.prior.mean = m_smooth
    
    # Initialize MAP Generator
    base_dir = config.fig_dir
    os.makedirs(base_dir, exist_ok=True)
    
    map_gen = MAPGenerator(model, base_dir, rel_noise=0.01)
    
    # Sample MAP points
    print("\n" + "=" * 80)
    print("SAMPLING MAP POINTS")
    print("=" * 80)
    
    n_data = 1
    seed = 0
    map_gen.sample_map_point(n_data, seed, m0=m_smooth)
    
    # Sample forward Jacobian at MAP points
    print("\n" + "=" * 80)
    print("SAMPLING FORWARD JACOBIAN")
    print("=" * 80)
    
    map_gen.sample_jacobian_at_map(n_data, seed)
    
    # Sample low-rank POSA Jacobian
    print("\n" + "=" * 80)
    print("SAMPLING LOW-RANK POSA JACOBIAN")
    print("=" * 80)
    
    rank = 50
    map_gen.sample_posa_low_rank_jacobian(n_data, rank, seed, oversampling=10)
    
    # Check generated Jacobian
    print("\n" + "=" * 80)
    print("CHECKING GENERATED JACOBIAN")
    print("=" * 80)
    
    map_gen.check_generated_jacobian(0, seed, low_rank=True)
    
    # Plot eigenvalue spectrum
    print("\n" + "=" * 80)
    print("PLOTTING EIGENVALUE SPECTRUM")
    print("=" * 80)
    
    map_gen.plot_eigenvalue_spectrum(0, rank, seed)
    
    print("\n" + "=" * 80)
    print("MAP POINT GENERATION COMPLETE")
    print("=" * 80)
    print(f"Data saved to: {base_dir}/data/")