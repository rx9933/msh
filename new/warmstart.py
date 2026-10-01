# warmstart.py
import dolfin as dl
import matplotlib.pyplot as plt
import hippylib as hp


def compute_beta_sensitivity(model, mtrue, beta):
    """
    Compute dm/dβ using the implicit function theorem:
    H * dm/dβ = -dg/dβ = -C⁻¹(m - m0)
    
    Based on the MAPGenerator approach for proper linearization
    """
    # Create the full state vector and set parameter to mtrue
    x = model.model.generate_vector()
    x[hp.PARAMETER] = mtrue.copy()
    
    # First, solve the forward problem to get proper state - CRITICAL STEP
    model.model.solveFwd(x[hp.STATE], x)
    
    # Solve adjoint problem - need proper adjoint rhs
    # Create adjoint rhs from data misfit (similar to MAPGenerator)

    
    # Solve adjoint system
    model.model.solveAdj(x[hp.ADJOINT], x)
    
    # Set linearization point with proper state/adjoint - LIKE MAPGenerator
    model.model.problem.setLinearizationPoint(x, False)
    
    # Right-hand side: -dg/dβ = -C⁻¹(m - m0)
    rhs = model.model.generate_vector(hp.PARAMETER)
    rhs.zero()
    
    # For Bilaplacian prior: C⁻¹(m - m0) = R*(m - m0)
    m_minus_m0 = mtrue.copy()
    m_minus_m0.axpy(-1.0, model.prior.mean)
    
    # Apply prior precision operator
    model.prior.R.mult(m_minus_m0, rhs)
    rhs *= -1.0  # -C⁻¹(m - m0)
    
    # Use the Hessian apply operator to solve H * dm_dbeta = rhs
    # This is the key insight from MAPGenerator - use the reduced Hessian
    hessian = hp.ReducedHessian(model.model, misfit_only=False)
    
    # Configure and solve with CG - similar to MAPJacobian
    cg_params = hp.CGSolverSteihaug_ParameterList()
    cg_params['rel_tolerance'] = 1e-10
    cg_params['max_iter'] = 100
    
    solver = hp.CGSolverSteihaug(cg_params)
    solver.set_operator(hessian)
    solver.set_preconditioner(model.prior.Rsolver)
    
    dm_dbeta = model.model.generate_vector(hp.PARAMETER)
    solver.solve(dm_dbeta, rhs)
    
    return dm_dbeta

def improved_initialization(model, mtrue, beta_current, beta_target):
    """
    Compute improved initialization at beta_target
    using first-order sensitivity from beta_current.
    """
    # try:
    dm_dbeta = compute_beta_sensitivity(model, mtrue, beta_current)
    # except: 
    #     dm_dbeta, iter_info = compute_beta_sensitivity(model, mtrue, beta_current)
    delta_beta = beta_target - beta_current
    m_init = mtrue.copy()
    m_init.axpy(delta_beta, dm_dbeta)

    return m_init#, iter_info


def plot_warmstart(model, mtrue, minit, beta_target, save_dir="free_surface"):
    """
    Quick visualization of warm-start initialization.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(18, 6))

    # True parameter
    plt.subplot(131)
    mtrue_func = dl.Function(model.model.problem.Vh[hp.PARAMETER], mtrue)
    cbar = dl.plot(mtrue_func, title="True Parameter")
    plt.colorbar(cbar)

    # Initialized parameter
    plt.subplot(132)
    minit_func = dl.Function(model.model.problem.Vh[hp.PARAMETER], minit)
    cbar = dl.plot(minit_func, title=f"Warm Start Init β={beta_target:.2f}")
    plt.colorbar(cbar)

    # Difference
    plt.subplot(133)
    difference = mtrue.copy()
    difference.axpy(-1.0, minit)
    diff_func = dl.Function(model.model.problem.Vh[hp.PARAMETER], difference)
    cbar = dl.plot(diff_func, title="Difference: True - Init")
    plt.colorbar(cbar)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"warmstart_beta{beta_target:.2f}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Warm start plot saved to {fig_path}")
