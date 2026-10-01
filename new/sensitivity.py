# main_multifreq_analysis.py
"""
Multi-frequency Helmholtz inverse problem with sensitivity analysis
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import dolfin as dl
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import gc
import psutil

sys.path.append('../../')
from hippylib import *
import hippylib as hp
import hippyflow as hf
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

np.random.seed(seed=2)

# Import your multi-frequency model
from helmholtz_model import MultiFrequencyHelmholtzModel, helmholtz_settings

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration class for the analysis"""
    def __init__(self):
        self.fig_dir = "multifreq_analysis_figures_sensitivity"
        self.rel_noise = 0.01
        self.max_iter = 1#25
        self.tol = 1e-8
        self.cg_iter = 2000

        
        # Source frequencies per source
        self.source_frequencies = [
            [800],  # Source 1
            [800],  # Source 2
            [800],  # Source 3
            [800],  # Source 4
        ]
        
        # TV regularization flag
        self.use_tv = False
        
        # Solver parameters
        self.solver_params = {
            "rel_tolerance": 1e-6,
            "abs_tolerance": 1e-8,
            "max_iter": 1,
            "GN_iter": 10,
            "globalization": "LS",
            "LS": {"c_armijo": 1e-2, "max_backtracking_iter": 200},
            "cg_max_iter": 2000,
            "cg_coarse_tolerance": 0.01
        }
config = Config()

# Create figure directory
os.makedirs(config.fig_dir, exist_ok=True)
print(f"Saving figures to: {config.fig_dir}/")

# ============================================================================
# MODEL SETUP
# ============================================================================
def setup_model(config):
    """Set up the multi-frequency Helmholtz model"""
    settings = helmholtz_settings()
    settings['source_frequencies'] = config.source_frequencies
    settings['use_tv_regularization'] = config.use_tv
    
    print("=" * 80)
    print("MULTI-FREQUENCY HELMHOLTZ PROBLEM")
    print("=" * 80)
    print(f"Source frequencies: {config.source_frequencies}")
    print(f"Regularization: {'TV (edge-preserving)' if config.use_tv else 'Gaussian (BiLaplacian)'}")
    
    model = MultiFrequencyHelmholtzModel(settings)
    
    Vh = model.Vh
    print(f"State space dimension: {Vh[STATE].dim()}")
    print(f"Parameter space dimension: {Vh[PARAMETER].dim()}")
    print(f"Number of observation points: {len(model.targets)}")
    
    return model, settings

# ============================================================================
# TRUE PARAMETER GENERATION
# ============================================================================
def load_true_parameter(model, vel_file_path='../realistic/model1.npy', slice_idx=100):
    """Load and interpolate true parameter from velocity data"""
    Vh_parameter = model.Vh[PARAMETER]
    
    # Load velocity data
    vel = np.load(vel_file_path)
    vel_2d = vel[slice_idx, :, :, :][0]
    
    # Get dof coordinates
    coords = Vh_parameter.tabulate_dof_coordinates()
    points_xy = coords.reshape(-1, 2)
    
    # Interpolate
    from scipy.interpolate import RegularGridInterpolator
    box = model.settings['box']
    box_pml = model.settings['box_pml']
    
    ny, nx = vel_2d.shape
    x_coords = np.linspace(box_pml[0], box_pml[2], nx)
    y_coords = np.linspace(box_pml[1], box_pml[3], ny)
    
    interpolator = RegularGridInterpolator(
        (y_coords, x_coords),
        vel_2d,
        method='linear',
        bounds_error=False,
        fill_value=np.mean(vel_2d)
    )
    
    points_for_interp = np.column_stack([points_xy[:, 1], points_xy[:, 0]])
    vel_values = interpolator(points_for_interp)
    
    # Transform to parameter
    m_values = vel_values / 10000.0
    
    m_true_func = dl.Function(Vh_parameter)
    m_true_func.vector().set_local(m_values)
    m_true_func.vector().apply("insert")
    m_true = m_true_func.vector()
    
    return m_true

def create_smooth_prior_mean(model, m_true, num_steps=20, dt=0.01, kappa=0.05):
    """Create smooth prior mean using diffusion"""
    Vh_parameter = model.Vh[PARAMETER]
    
    # Try to import smoothing function
    try:
        sys.path.append('../realistic/')
        from smooth_prior import create_smooth_prior_mean as smooth_func
        m_smooth = smooth_func(
            dl.Function(Vh_parameter, m_true),
            Vh_parameter,
            num_steps=num_steps,
            dt=dt,
            kappa=kappa
        )
        return m_smooth.vector()
    except ImportError:
        print("Smooth prior function not found, using mean value")
        m_mean = dl.Vector()
        Vh_parameter.init_vector(m_mean, PARAMETER)
        m_mean.zero()
        m_mean.axpy(1.0, m_true)
        return m_mean

# ============================================================================
# DATA GENERATION
# ============================================================================
def generate_synthetic_data(model, m_true, rel_noise=0.01):
    """Generate synthetic observations for all frequencies"""
    observations = {}
    states = {}
    noise_stds = {}
    
    for freq, info in model.frequency_models.items():
        print(f"Generating data for frequency {freq}Hz...")
        
        # Generate vector
        x = info['model'].generate_vector()
        x[PARAMETER] = m_true
        
        # Solve forward
        info['model'].solveFwd(x[STATE], x)
        
        # Get observations
        d = dl.Vector()
        info['misfit'].B.init_vector(d, 0)
        info['misfit'].B.mult(x[STATE], d)
        
        # Add noise
        MAX = d.norm("linf")
        noise_std_dev = rel_noise * MAX
        if noise_std_dev < 1e-12:
            noise_std_dev = 1e-6
        hp.parRandom.normal_perturb(noise_std_dev, d)
        info['misfit'].d = d
        info['misfit'].noise_variance = noise_std_dev * noise_std_dev
        
        observations[freq] = d
        states[freq] = x[STATE]
        noise_stds[freq] = noise_std_dev
        
        print(f"  Freq {freq}Hz: observations={d.size()}, noise std={noise_std_dev:.4e}")
    
    return observations, states, noise_stds

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_parameter_field(param_vec, Vh, title, fig_dir, filename, vmin=None, vmax=None, cmap='viridis'):
    """Plot a parameter field"""
    param_func = dl.Function(Vh, param_vec)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if vmin is None or vmax is None:
        vmin = param_vec.min()
        vmax = param_vec.max()
    
    cbar = dl.plot(param_func, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(cbar, ax=ax)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

def plot_state_by_source(state_vec, Vh, model, freq, save_dir):
    """Plot state components for active sources at a given frequency"""
    state_space = Vh[STATE]
    state_func = dl.Function(state_space, state_vec)
    
    # Get active sources for this frequency
    freq_to_sources = {}
    for s_idx, freqs in enumerate(model.source_frequencies):
        for f in freqs:
            if f not in freq_to_sources:
                freq_to_sources[f] = []
            freq_to_sources[f].append(s_idx)
    
    active_sources = freq_to_sources.get(freq, [])
    n_active = len(active_sources)
    
    if n_active == 0:
        print(f"  No active sources for freq {freq}Hz")
        return
    
    fig, axes = plt.subplots(2, n_active, figsize=(4*n_active, 8), squeeze=False)
    
    for idx, source_idx in enumerate(active_sources):
        # Extract source component
        source_func = state_func.sub(source_idx)
        real_space = state_space.sub(source_idx).sub(0).collapse()
        imag_space = state_space.sub(source_idx).sub(1).collapse()
        
        real_func = dl.interpolate(source_func.sub(0), real_space)
        imag_func = dl.interpolate(source_func.sub(1), imag_space)
        
        # Plot real
        ax = axes[0, idx]
        cbar = dl.plot(real_func)
        plt.colorbar(cbar, ax=ax)
        ax.set_title(f"Source {source_idx+1} ({freq/1000:.1f}kHz) - Real")
        ax.set_aspect('equal')
        if model.targets is not None:
            ax.scatter(model.targets[:, 0], model.targets[:, 1], 
                      c='red', s=10, alpha=0.5)
        
        # Plot imaginary
        ax = axes[1, idx]
        cbar = dl.plot(imag_func)
        plt.colorbar(cbar, ax=ax)
        ax.set_title(f"Source {source_idx+1} ({freq/1000:.1f}kHz) - Imag")
        ax.set_aspect('equal')
        if model.targets is not None:
            ax.scatter(model.targets[:, 0], model.targets[:, 1], 
                      c='red', s=10, alpha=0.5)
    
    plt.tight_layout()
    filename = f"state_freq_{freq}Hz.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

def plot_observations_by_source(model, observations, fig_dir):
    """Plot observation data for each source and frequency"""
    n_targets = len(model.targets)
    n_sources = model.settings['n_sources']
    
    for freq, d in observations.items():
        d_local = d.get_local()
        d_reshaped = d_local.reshape((n_targets, n_sources, 2))
        
        fig, axes = plt.subplots(1, n_sources, figsize=(5*n_sources, 4))
        if n_sources == 1:
            axes = [axes]
        
        for i in range(n_sources):
            magnitude = np.linalg.norm(d_reshaped[:, i, :], axis=1)
            sc = axes[i].scatter(model.targets[:, 0], model.targets[:, 1],
                                c=magnitude, s=40, cmap='viridis',
                                edgecolors='k', linewidth=0.5)
            plt.colorbar(sc, ax=axes[i], label='Magnitude')
            axes[i].set_title(f'Freq {freq}Hz - Source {i+1}')
            axes[i].set_aspect('equal')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'observations_freq_{freq}Hz.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

# ============================================================================
# MAP ESTIMATION
# ============================================================================
def solve_map_estimation(model, m0, solver_params, fig_dir):
    """Solve the MAP estimation problem"""
    print("\n" + "=" * 80)
    print("SOLVING MAP ESTIMATION")
    print("=" * 80)
    
    # Initialize solver
    solver = ReducedSpaceNewtonCG(model.model)
    
    for key, val in solver_params.items():
        if key == "LS":
            for ls_key, ls_val in val.items():
                solver.parameters[key][ls_key] = ls_val
        else:
            solver.parameters[key] = val
    
    # Prepare initial guess
    x = model.model.generate_vector()
    x[PARAMETER] = m0
    x[STATE] = model.model.generate_vector(STATE)
    x[ADJOINT] = model.model.generate_vector(ADJOINT)
    
    print(f"Initial guess: parameter dim={x[PARAMETER].size()}, "
          f"min={x[PARAMETER].min():.4e}, max={x[PARAMETER].max():.4e}")
    
    # Monitor memory
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / 1e9
    
    start_time = time.time()
    solver.solve(x)
    elapsed_time = time.time() - start_time
    
    mem_end = process.memory_info().rss / 1e9
    
    print(f"\nMAP estimation completed in {elapsed_time:.2f} seconds")
    print(f"Memory used: {mem_end - mem_start:.2f} GB")
    print(f"Converged: {solver.converged}")
    print(f"Iterations: {solver.it}")
    print(f"Final gradient norm: {solver.final_grad_norm:.4e}")
    print(f"Final cost: {solver.final_cost:.4e}")
    
    return x[PARAMETER], solver

# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================
def compute_inverse_jacobian_action(model, x_map, data_test, noise_std, 
                                   use_low_rank=True, rank=1000, oversampling=10):
    """
    Compute inverse Jacobian action for sensitivity analysis.
    Can use either full Hessian or low-rank approximation.
    
    Parameters:
    - model: The multi-frequency model
    - x_map: The MAP point (dictionary with STATE, PARAMETER, ADJOINT)
    - data_test: Random data perturbation vector
    - noise_std: Noise standard deviation
    - use_low_rank: If True, use low-rank Hessian approximation
    - rank: Rank for low-rank approximation
    - oversampling: Oversampling factor for randomized SVD
    
    Returns:
    - inverse_jacobian_action: The result of H^{-1} * J^T * Gamma_data
    - Additional info if low-rank: (eigenvalues, eigenvectors)
    """
    print("\n" + "=" * 80)
    print(f"COMPUTING INVERSE JACOBIAN ACTION ({'LOW-RANK' if use_low_rank else 'FULL'})")
    print("=" * 80)
    
    # Setup observable
    observable = hf.LinearStateObservable(model.model.problem, model.misfit.B)
    model.model.problem.setLinearizationPoint(x_map, gn_approx=False)
    
    jacobian = hf.ObservableJacobian(observable)
    noise_variance = noise_std**2
    
    # Data perturbation with covariance inverse
    Gamma_data_test = dl.Vector()
    model.misfit.B.init_vector(Gamma_data_test, 0)
    Gamma_data_test.zero()
    Gamma_data_test.axpy(1/noise_variance, data_test)
    
    # Apply transpose of Jacobian
    hessian_rhs = model.model.generate_vector(hp.PARAMETER)
    jacobian.transpmult(Gamma_data_test, hessian_rhs)
    
    if use_low_rank:
        # Low-rank Hessian approximation
        print(f"Using low-rank Hessian approximation (rank={rank}, oversampling={oversampling})")
        
        # Create random vectors for randomized SVD
        m_dummy = model.model.generate_vector(hp.PARAMETER)
        Omega = hp.MultiVector(m_dummy, rank + oversampling)
        hp.parRandom.normal(1.0, Omega)
        
        # Compute Hessian misfit operator
        Hmisfit = hp.ReducedHessian(model.model, misfit_only=True)
        
        # Double pass to get low-rank approximation
        d, U = hp.doublePassG(
            Hmisfit, 
            model.prior.R, 
            model.prior.Rsolver, 
            Omega, 
            rank, 
            s=oversampling, 
            check=False
        )
        
        # Create low-rank Hessian
        low_rank_hessian = hp.LowRankHessian(model.prior, d, U)
        
        # Solve using low-rank Hessian
        inverse_jacobian_action = model.model.generate_vector(hp.PARAMETER)
        low_rank_hessian.solve(inverse_jacobian_action, hessian_rhs)
        
        print(f"Low-rank Hessian eigenvalues: {d[:5]}... (showing first 5)")
        print(f"Condition number (approx): {d[0]/d[-1] if len(d) > 1 else 'N/A'}")
        
        return inverse_jacobian_action, d, U
        
    else:
        # Full Hessian solve
        print("Using full Hessian solve")
        
        # Hessian solve using true Hessian
        HessApply = hp.ReducedHessian(model.model, misfit_only=False)
        HessApply.gn_approx = False
        
        # CG solver
        cg_parameters = hp.CGSolverSteihaug_ParameterList()
        cg_parameters['rel_tolerance'] = 1e-12
        solver = hp.CGSolverSteihaug(cg_parameters)
        solver.set_operator(HessApply)
        solver.set_preconditioner(model.prior.Rsolver)
        
        inverse_jacobian_action = model.model.generate_vector(hp.PARAMETER)
        solver.solve(inverse_jacobian_action, hessian_rhs)
        
        print(f"Inverse Jacobian action computed, norm: {inverse_jacobian_action.norm('l2'):.4e}")
        print(f"CG iterations: {solver.iter}")
        
        return inverse_jacobian_action


def finite_difference_validation(model, m_map_original, true_data, data_test, 
                                 inverse_jacobian_action, Vh, fig_dir):
    """Perform finite difference validation"""
    print("\n" + "=" * 80)
    print("FINITE DIFFERENCE VALIDATION")
    print("=" * 80)
    
    # Initialize solver
    solver = ReducedSpaceNewtonCG(model.model)
    solver.parameters["rel_tolerance"] = 1e-12
    solver.parameters["abs_tolerance"] = 1e-12
    solver.parameters["cg_max_iter"] = 1000
    
    epsilons = [1e-3, 1e-4, 1e-5, 1e-6]
    fd_errors = []
    
    for eps in epsilons:
        print(f"\nTesting eps = {eps:.1e}")
        
        # Perturb data
        perturbed_data = dl.Vector()
        model.misfit.B.init_vector(perturbed_data, 0)
        perturbed_data.axpy(1., true_data)
        perturbed_data.axpy(eps, data_test)
        
        model.misfit.d.zero()
        model.misfit.d.axpy(1., perturbed_data)
        
        # Solve perturbed problem
        x_perturbed = model.model.generate_vector("ALL")
        x_perturbed[PARAMETER].axpy(1.0, m_map_original)
        
        solver.solve(x_perturbed)
        
        # Compute finite difference derivative
        fd_derivative = x_perturbed[PARAMETER].copy()
        fd_derivative.axpy(-1.0, m_map_original)
        fd_derivative *= (1.0 / eps)
        
        fd_np = fd_derivative.get_local()
        analytic_np = inverse_jacobian_action.get_local()
        
        # Compute relative error
        fd_error = np.linalg.norm(fd_np - analytic_np) / (np.linalg.norm(analytic_np) + 1e-12)
        fd_errors.append(fd_error)
        print(f"  FD error: {fd_error:.3e}")
        
        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # FD derivative
        fd_func = dl.Function(Vh[PARAMETER])
        fd_func.vector().set_local(fd_np)
        ax = dl.plot(fd_func, title=f"Finite Difference (eps={eps:.1e})")
        plt.colorbar(ax, ax=axes[0])
        
        # Analytic derivative
        ana_func = dl.Function(Vh[PARAMETER])
        ana_func.vector().set_local(analytic_np)
        ax = dl.plot(ana_func, title="Inverse Jacobian Action")
        plt.colorbar(ax, ax=axes[1])
        
        # Error
        diff_func = dl.Function(Vh[PARAMETER])
        diff_func.vector().set_local(fd_np - analytic_np)
        ax = dl.plot(diff_func, title="Derivative Error")
        plt.colorbar(ax, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'fd_validation_eps_{eps:.1e}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot convergence
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(epsilons, fd_errors, '-o', linewidth=2, markersize=8, label='FD error')
    ax.loglog(epsilons, [e for e in epsilons], 'k--', label='O(eps)', alpha=0.5)
    ax.loglog(epsilons, [e**2 for e in epsilons], 'r--', label='O(eps²)', alpha=0.5)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Relative FD Error')
    ax.set_title('Finite Difference Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fd_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFD errors: {fd_errors}")
    return fd_errors

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function"""
    
    # Setup model
    model, settings = setup_model(config)
    Vh = model.Vh
    
    # Load true parameter
    print("\n=== Loading true parameter ===")
    m_true = load_true_parameter(model)
    m_true_func = dl.Function(Vh[PARAMETER], m_true)
    print(f"True parameter stats - Mean: {m_true.min():.4f}, Max: {m_true.max():.4f}")
    
    # Create smooth prior mean
    print("\n=== Creating smooth prior mean ===")
    m_smooth = create_smooth_prior_mean(model, m_true)
    model.prior.mean = m_smooth
    
    # Plot true parameter and prior mean
    plot_parameter_field(m_true, Vh[PARAMETER], "True Parameter", 
                        config.fig_dir, "true_parameter.png")
    plot_parameter_field(m_smooth, Vh[PARAMETER], "Prior Mean", 
                        config.fig_dir, "prior_mean.png")
    
    # Generate synthetic data
    print("\n=== Generating synthetic data ===")
    observations, states, noise_stds = generate_synthetic_data(
        model, m_true, config.rel_noise
    )
    print(f"Generated observations for {len(observations)} frequencies")
    
    # Plot states for each frequency
    print("\n=== Plotting states ===")
    for freq, state in states.items():
        plot_state_by_source(state, Vh, model, freq, config.fig_dir)
    
    # Plot observations
    print("\n=== Plotting observations ===")
    plot_observations_by_source(model, observations, config.fig_dir)
    
    # Initialize for MAP estimation
    m0 = dl.Vector()
    model.prior.init_vector(m0, PARAMETER)
    # m0.axpy(1.0, model.prior.mean.copy())
    # m0.axpy(1.0, map_previous)
    map_previous = np.load('multifreq_figures_TEST/map_estimate.npy')
    m0.set_local(map_previous)
    m0.apply("insert")


    # Solve MAP estimation
    m_map, solver = solve_map_estimation(model, m0, config.solver_params, config.fig_dir)
    
    # Plot results
    print("\n=== Plotting results ===")
    vmin = min(m_true.min(), m_map.min())
    vmax = max(m_true.max(), m_map.max())
    
    plot_parameter_field(m_true, Vh[PARAMETER], "True Parameter", 
                        config.fig_dir, "true_parameter_final.png", vmin, vmax)
    plot_parameter_field(m_map, Vh[PARAMETER], "MAP Estimate", 
                        config.fig_dir, "map_estimate.png", vmin, vmax)
    
    # Difference
    diff_vec = m_map.copy()
    diff_vec.axpy(-1.0, m_true)
    plot_parameter_field(diff_vec, Vh[PARAMETER], "Difference (MAP - True)", 
                        config.fig_dir, "difference.png", 
                        vmin=-0.15, vmax=0.15, cmap='RdBu_r')
    
    # Compute error
    rel_error = diff_vec.norm('l2') / m_true.norm('l2')
    print(f"\nRelative L2 error: {rel_error:.4f}")
    
    # Save results
    np.save(os.path.join(config.fig_dir, 'map_estimate.npy'), m_map.get_local())
    np.save(os.path.join(config.fig_dir, 'true_parameter.npy'), m_true.get_local())
    
    # Sensitivity analysis - prepare data perturbation
    print("\n=== Preparing for sensitivity analysis ===")
    data_test = dl.Vector()
    model.misfit.B.init_vector(data_test, 0)
    hp.parRandom.normal(0.1, data_test)
    
    # Need to solve forward for MAP point
    x_map = model.model.generate_vector()
    x_map[PARAMETER] = m_map
    model.model.solveFwd(x_map[STATE], x_map)
    model.model.solveAdj(x_map[ADJOINT], x_map)
    
    # Compute inverse Jacobian action
    noise_std = list(noise_stds.values())[0]  # Use first frequency's noise
    inv_jac_action, d, u = compute_inverse_jacobian_action(
        model, x_map, data_test, noise_std
    )
    ig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Eigenvalue decay
    axes[0].semilogy(d, 'b-', linewidth=2, label='Eigenvalues')
    axes[0].set_xlabel('Eigenvalue index')
    axes[0].set_ylabel('Eigenvalue magnitude')
    axes[0].set_title('Hessian Eigenvalue Spectrum (Low-rank)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Cumulative energy
    cumsum = np.cumsum(d) / np.sum(d)
    axes[1].plot(cumsum, 'r-', linewidth=2, label='Cumulative energy')
    axes[1].axhline(y=0.95, color='k', linestyle='--', alpha=0.5, label='95% energy')
    axes[1].set_xlabel('Number of eigenvalues')
    axes[1].set_ylabel('Cumulative energy fraction')
    axes[1].set_title('Cumulative Energy Distribution')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.fig_dir, 'hessian_eigenvalues.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    # Plot inverse Jacobian action
    plot_parameter_field(inv_jac_action, Vh[PARAMETER], 
                        "Inverse Jacobian Action", config.fig_dir, 
                        "inverse_jacobian_action.png")
    
    # Finite difference validation
    true_data = observations[list(observations.keys())[0]]
    m_map_original = m_map.copy()
    
    fd_errors = finite_difference_validation(
        model, m_map_original, true_data, data_test,
        inv_jac_action, Vh, config.fig_dir
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All figures saved to: {config.fig_dir}/")
    print(f"Final relative error: {rel_error:.4f}")
    
    return model, m_true, m_map, solver, fd_errors

if __name__ == "__main__":
    model, m_true, m_map, solver, fd_errors = main()