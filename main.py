# main.py 

import os
os.environ['OMP_NUM_THREADS'] ='1'

import dolfin as dl, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../')
from hippylib import *
from hippylib.algorithms.bfgs import RescaledIdentity, BFGS

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

np.random.seed(seed=2)

# Import the multi-frequency model
from helmholtz_model import *
tv_regul = False

from warmstart import * 
warmstart = False

NewtonSolver = True
BFGSSolver = False
data_dir = 'data/'
sys.path.append(data_dir)
m_true_dir = data_dir +'model1.npy'
fig_dir = "multifreq_figures_TEST"
if tv_regul:
    fig_dir += '_TV/'
os.makedirs(fig_dir, exist_ok=True)
print(f"Saving figures to: {fig_dir}/")

# ============================================================================
# MAIN EXECUTION WITH MULTIPLE FREQUENCIES PER SOURCE
# ============================================================================
print("=" * 80)
print("MULTI-FREQUENCY HELMHOLTZ PROBLEM")
print("=" * 80)

# Define frequencies per source
source_frequencies = [
    [800],# 300, 400, 500, 600, 700, 800, 1000],
    [800],#, 300, 400, 500, 600, 700, 800, 1000],
    [800],# 300, 400, 500, 600, 700, 800, 1000],
    [800],# 300, 400, 500, 600, 700, 800, 1000],
]

print('Source Frequencies \n')
print(source_frequencies)

# Create model
settings = helmholtz_settings()
settings['source_frequencies'] = source_frequencies

#=======================================================
# REGULARIZATION 
# ============================================================
settings['use_tv_regularization'] = tv_regul  

print(f"\nRegularization: {'TV (edge-preserving)' if settings['use_tv_regularization'] else 'Gaussian (BiLaplacian)'}")

print(f"Source frequencies: {source_frequencies}")
print(f"Total (source, frequency) pairs: {sum(len(freqs) for freqs in source_frequencies)}")

model = MultiFrequencyHelmholtzModel(settings)

Vh = model.Vh
print(f"State space dimension: {Vh[hp.STATE].dim()}")
print(f"Parameter space dimension: {Vh[hp.PARAMETER].dim()}")

# ============================================================================
# LOAD VELOCITY DATA AND CREATE TRUE PARAMETER
# ============================================================================
print("\n=== Loading velocity data ===")

# Load velocity data
vel = np.load(m_true_dir)
j = 100
vel_2d = vel[j, :, :, :][0]

# Interpolate onto the model's mesh
m_true_func = dl.Function(Vh[hp.PARAMETER])
coords = Vh[hp.PARAMETER].tabulate_dof_coordinates()
points_xy = coords.reshape(-1, 2)

from scipy.interpolate import RegularGridInterpolator

box = settings['box']
box_pml = settings['box_pml']

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

# Transform velocity to wave number parameter
c = vel_values / 10000.0
m_values = c  # m = log(c) for k = wave_number * exp(m)

m_true_func.vector().set_local(m_values)
m_true_func.vector().apply("insert")
m_true = m_true_func.vector()

print(f"Parameter field stats - Mean: {m_true_func.vector().get_local().mean():.4f}, "
      f"Std: {m_true_func.vector().get_local().std():.4f}")

# ============================================================================
# SMOOTH THE PRIOR MEAN
# ============================================================================
print("\n=== Creating smooth prior mean ===")

from smooth_prior import *

m_smooth = create_smooth_prior_mean(m_true_func, Vh[hp.PARAMETER], 
                                   num_steps=20, dt=0.01, kappa=0.05) # ut = kappa laplace u

# Update the prior mean in the model
model.prior.mean = m_smooth.vector()

prior_mean_func = dl.Function(Vh[hp.PARAMETER])
prior_mean_func.vector().axpy(1.0, model.prior.mean)
fig, axes = plt.subplots(1, 1, figsize=(5, 5))

vmin_param =prior_mean_func.vector().min()
vmax_param = prior_mean_func.vector().max()

cbar_prior = dl.plot(prior_mean_func, vmin=vmin_param, vmax=vmax_param)
plt.colorbar(cbar_prior)
plt.title("Prior Mean")
plt.savefig(fig_dir + '/prior_mean.png')

# ============================================================================
# GENERATE SYNTHETIC DATA FOR EACH FREQUENCY
# ============================================================================
print("\n=== Generating synthetic data ===")
rel_noise = 0.01

observations = {}
states = {}

# Generate observations for each frequency
for freq, info in model.frequency_models.items():
    print(f"\nGenerating data for frequency {freq}Hz...")
    
    # Generate vector
    x = info['model'].generate_vector()
    x[hp.PARAMETER] = m_true
    
    # Solve forward
    info['model'].solveFwd(x[hp.STATE], x)
    
    # Get observations
    d = dl.Vector()
    info['misfit'].B.init_vector(d, 0)
    info['misfit'].B.mult(x[hp.STATE], d)
    
    # Add noise
    MAX = d.norm("linf")
    noise_std_dev = rel_noise * MAX
    if noise_std_dev < 1e-12:
        noise_std_dev = 1e-6
    hp.parRandom.normal_perturb(noise_std_dev, d)
    info['misfit'].d = d
    info['misfit'].noise_variance = noise_std_dev * noise_std_dev
    
    observations[freq] = d
    states[freq] = x[hp.STATE]
    
    print(f"  Freq {freq}Hz: state={x[hp.STATE].size()}, observations={d.size()}, "
            f"parameter={x[hp.PARAMETER].size()}, noise std={noise_std_dev:.4e}")

print(f"\nGenerated observations for {len(observations)} frequencies")
print("Factorization cache after data generation:", model.factorization_summary())

def plot_states_by_source(states, model, source_frequencies, save_dir):
    """Plot real and imaginary state components only for sources that have each frequency."""
    state_space = model.Vh[hp.STATE]
    
    # Create mapping of frequency to sources that have it
    freq_to_sources = {}
    for s_idx, freqs in enumerate(source_frequencies):
        for freq in freqs:
            if freq not in freq_to_sources:
                freq_to_sources[freq] = []
            freq_to_sources[freq].append(s_idx)
    
    print(f"Frequency to source mapping: {freq_to_sources}")
    
    for freq, state in states.items():
        state_function = dl.Function(state_space, state)
        active_sources = freq_to_sources.get(freq, [])
        n_active = len(active_sources)
        
        print(f"Frequency {freq}Hz: active sources = {[s+1 for s in active_sources]}")
        
        if n_active == 0:
            print(f"  No active sources for freq {freq}Hz, skipping")
            continue
        
        # Only plot active sources
        fig, axes = plt.subplots(2, n_active, figsize=(4 * n_active, 8), squeeze=False)
        
        for idx, source_index in enumerate(active_sources):
            source_function = state_function.sub(source_index)
            
            # Get real and imaginary parts
            real_space = state_space.sub(source_index).sub(0).collapse()
            imag_space = state_space.sub(source_index).sub(1).collapse()
            real_function = dl.interpolate(source_function.sub(0), real_space)
            imag_function = dl.interpolate(source_function.sub(1), imag_space)
            
            # Plot real part
            real_axis = axes[0, idx]
            plt.sca(real_axis)
            real_plot = dl.plot(real_function)
            real_axis.set_title(f"Source {source_index+1} ({freq/1000:.1f}kHz) - Real")
            real_axis.set_aspect("equal")
            real_axis.set_xlabel("x")
            real_axis.set_ylabel("y")
            fig.colorbar(real_plot, ax=real_axis)
            
            # Plot imaginary part
            imag_axis = axes[1, idx]
            plt.sca(imag_axis)
            imag_plot = dl.plot(imag_function)
            imag_axis.set_title(f"Source {source_index+1} ({freq/1000:.1f}kHz) - Imag")
            imag_axis.set_aspect("equal")
            imag_axis.set_xlabel("x")
            imag_axis.set_ylabel("y")
            fig.colorbar(imag_plot, ax=imag_axis)
            
            # Plot receiver locations if available
            if model.targets is not None:
                real_axis.scatter(
                    model.targets[:, 0], model.targets[:, 1],
                    c="red", s=10, alpha=0.5
                )
                imag_axis.scatter(
                    model.targets[:, 0], model.targets[:, 1],
                    c="red", s=10, alpha=0.5
                )
        
        fig.tight_layout()
        output_path = os.path.join(save_dir, f"state_sources_freq_{freq}Hz.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved state plot to {output_path}")

# ============================================================================
# PLOT STATES FOR EACH FREQUENCY
# ============================================================================
print("\n=== Plotting states ===")

plot_states_by_source(states, model, source_frequencies,  fig_dir)

# ====='=======================================================================
# SOLVE INVERSE PROBLEM
# ============================================================================
print("\n" + "=" * 80)
print("SOLVING INVERSE PROBLEM WITH MULTI-FREQUENCY DATA")
print("=" * 80)

# Start from prior mean
m0 = dl.Vector()
model.prior.init_vector(m0, hp.PARAMETER)
m0.axpy(1.0,model.prior.mean.copy())#m_true_func.vector().copy())# m_smooth.vector())# model.prior.mean)

print(f"Initial guess: parameter dimension={m0.size()}, "
    f"min={min(m0.get_local()):.4e}, max={max(m0.get_local()):.4e}")

# m0 = dl.Vector()
# model.prior.init_vector(m0, hp.PARAMETER)

# # Set to a specific constant value (e.g., 0.25)
# constant_value = 0.25
# m0.zero()
# m0.axpy(1.0, dl.Constant(constant_value) * m0)  # This won't work directly
# ============================================================================
# WARMSTART INITIALIZATION
# ============================================================================
if warmstart:
    print("\n" + "=" * 80)
    print("APPLYING WARM-START INITIALIZATION")
    print("=" * 80)
    
    # Define beta values (these correspond to your regularization parameter)
    # You need to determine what beta represents in your model
    # For Gaussian prior: beta = 1/noise_variance or related to gamma/delta
    # For TV prior: beta = tv_beta
    
    beta_current = 0.1   # Starting regularization parameter
    beta_target = 1.0    # Target regularization parameter
    
    try:
        # Compute improved initialization
        m_init = improved_initialization(model, m_true, beta_current, beta_target)
        
        # Update m0 with warmstart
        m0.zero()
        m0.axpy(1.0, m_init)
        
        # Plot warmstart results
        plot_warmstart(model, m_true, m_init, beta_target, save_dir=fig_dir)
        
        print(f"Warmstart completed: beta_current={beta_current} -> beta_target={beta_target}")
        
    except Exception as e:
        print(f"Warmstart failed: {e}")
        print("Using default initialization")
else:
    print("Warmstart disabled, using default initialization")
class BacktrackingTracker:
    def __init__(self):
        self.backtrack_counts = []
        self.current_backtracks = 0
        
    def callback(self, it, x):
        # This is called at the end of each Newton iteration
        if hasattr(self, 'backtrack_count'):
            print(f"  Newton iteration {it}: {self.backtrack_count} backtracking iterations")
            self.backtrack_counts.append(self.backtrack_count)
            self.backtrack_count = 0
            
    def increment_backtrack(self):
        if not hasattr(self, 'backtrack_count'):
            self.backtrack_count = 0
        self.backtrack_count += 1

# Usage in main.py:
tracker = BacktrackingTracker()

# Wrap the solveFwd to count backtracking
original_solveFwd = model.model.solveFwd
def solveFwd_with_tracking(out, x):
    # This will be called during backtracking
    tracker.increment_backtrack()
    return original_solveFwd(out, x)

# Replace the method temporarily
model.model.solveFwd = solveFwd_with_tracking

if NewtonSolver:
# Set optimization parameters
    solver_params = {
        "rel_tolerance": 1e-6,
        "abs_tolerance": 1e-8,
        "max_iter": 35,
        "GN_iter": 10,
        "globalization": "LS",
        "LS": {"c_armijo": 1e-2, "max_backtracking_iter": 200},
        "cg_max_iter": 2000,
        "cg_coarse_tolerance": 0.01
    }
#     solver_params = {
#     # Newton-CG parameters
#     "rel_tolerance": 1e-6,
#     "abs_tolerance": 1e-8,
#     "max_iter": 25,
#     "GN_iter": 2,  # More GN iterations for stability (default was 5, but 2-3 is good)
#     "globalization": "TR",  # Switch from LS to TR
#     "cg_max_iter": 200,
#     "cg_coarse_tolerance": 0.1,
#     "print_level": 0,
    
#     # Trust Region specific parameters
#     "TR": {
#         "eta": 0.05,  # Default: reject step if actual/predicted < eta
#         # Note: initial_radius is auto-computed from ||H^{-1}g||_R
#         # The radius adapts automatically based on rho_TR
#     }
# }
    print("\nStarting MAP estimation with ReducedSpaceNewtonCG...")
    print(f"Parameters: max_iter={solver_params['max_iter']}, "
        f"GN_iter={solver_params['GN_iter']}, "
        f"cg_max_iter={solver_params['cg_max_iter']}")

    
    solver = ReducedSpaceNewtonCG(model.model)
    
    for key, val in solver_params.items():
        if key == "LS":
            for ls_key, ls_val in val.items():
                solver.parameters[key][ls_key] = ls_val
        else:
            solver.parameters[key] = val
    
    x = model.model.generate_vector()
    x[hp.PARAMETER] = m0
    x[hp.STATE] = model.model.generate_vector(hp.STATE)
    x[hp.ADJOINT] = model.model.generate_vector(hp.ADJOINT)
    print(f"Optimization vectors: state={x[hp.STATE].size()}, parameter={x[hp.PARAMETER].size()}, "
            f"adjoint={x[hp.ADJOINT].size()}")
    
    model.model.solveFwd(x[hp.STATE], x)
    
    import time
    start_time = time.time()
    solver.solve(x)
    elapsed_time = time.time() - start_time
    
    print(f"\nInverse problem solved in {elapsed_time:.2f} seconds")
    print(f"Final vectors: state={x[hp.STATE].size()}, parameter={x[hp.PARAMETER].size()}, "
            f"adjoint={x[hp.ADJOINT].size()}")

elif BFGSSolver:
    bfgs_solver_params = {
        "rel_tolerance": 1e-6,
        "abs_tolerance": 1e-8,
        "max_iter": 250,
        "globalization": "LS",
        "print_level":1,
    }
    solver = BFGS(model.model)
    
    for key, val in bfgs_solver_params.items():
        if key == "LS":
            for ls_key, ls_val in val.items():
                solver.parameters[key][ls_key] = ls_val
        else:
            solver.parameters[key] = val
    
    x = model.model.generate_vector()
    x[hp.PARAMETER] = m0
    x[hp.STATE] = model.model.generate_vector(hp.STATE)
    x[hp.ADJOINT] = model.model.generate_vector(hp.ADJOINT)
    
    model.model.solveFwd(x[hp.STATE], x)
    # Create H0inv (initial inverse Hessian approximation)
    H0inv = RescaledIdentity()
    H0inv.init_vector = model.model.generate_vector 
    

    import time
    start_time = time.time()
    solver.solve(x, H0inv)
    elapsed_time = time.time() - start_time

print("Factorization cache after MAP estimate:", model.factorization_summary())
print(f"\nBacktracking summary: {tracker.backtrack_counts}")
# Extract results
m_map = x[hp.PARAMETER]
m_map_func = dl.Function(Vh[hp.PARAMETER])
m_map_func.vector().axpy(1., m_map)

# Compute error
m_error_func = dl.Function(Vh[hp.PARAMETER])
m_error_vec = m_map.get_local() - m_true.get_local()
m_error_func.vector().set_local(m_error_vec)
m_error_func.vector().apply("insert")

error_norm = np.linalg.norm(m_error_vec)
true_norm = np.linalg.norm(m_true.get_local())
rel_error = error_norm / true_norm if true_norm > 0 else float('inf')

print(f"\nError metrics:")
print(f"  ||m_map - m_true||_2 = {error_norm:.6e}")
print(f"  ||m_true||_2 = {true_norm:.6e}")
print(f"  Relative error = {rel_error:.2%}")
# Plot results
# Plot results
print("\n=== Plotting results ===")

# Get the data values
true_values = m_true_func.vector().get_local()
map_values = m_map_func.vector().get_local()
error_values = m_error_func.vector().get_local()

# Determine scale for True and MAP (full range)
vmin_param = min(true_values.min(), map_values.min())
vmax_param = max(true_values.max(), map_values.max())
vrange_param = vmax_param - vmin_param
vmin_param -= 0.05 * vrange_param
vmax_param += 0.05 * vrange_param

# Symmetric scale for error (centered at 0)
error_max = max(abs(error_values.min()), abs(error_values.max()))
error_max *= 1.05  # Add small buffer

fig, axes = plt.subplots(1, 4, figsize=(24, 5))

# True parameter - full colorbar
plt.sca(axes[0])
cbar1 = dl.plot(m_true_func, vmin=vmin_param, vmax=vmax_param)
plt.colorbar(cbar1, ax=axes[0])
axes[0].set_title("True Parameter")

plt.sca(axes[1])
cbar_prior = dl.plot(prior_mean_func, vmin=vmin_param, vmax=vmax_param)
plt.colorbar(cbar_prior, ax=axes[1])
axes[1].set_title("Prior Mean")


# MAP estimate - full colorbar
plt.sca(axes[2])
cbar2 = dl.plot(m_map_func, vmin=vmin_param, vmax=vmax_param)
plt.colorbar(cbar2, ax=axes[1])
axes[2].set_title("MAP Estimate")

# Difference - symmetric colorbar with 0 as white
plt.sca(axes[3])
cbar3 = dl.plot(m_error_func, cmap='RdBu_r', vmin=-.15, vmax=.15)
plt.colorbar(cbar3, ax=axes[2])
axes[3].set_title("Difference (MAP - True)")

plt.tight_layout()
plt.savefig(fig_dir + '/map_parameter.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nAll figures saved to: {fig_dir}/")

print("\n=== Saving MAP estimate as .npy file ===")

# Extract the MAP estimate values
m_map_values = m_map.get_local()

# Also get the true parameter values for comparison
m_true_values = m_true.get_local()

# Save to npy files
np.save(fig_dir + '/map_estimate.npy', m_map_values)
np.save(fig_dir + '/true_parameter.npy', m_true_values)