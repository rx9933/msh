"""MAP estimation and data-to-MAP sensitivity routines for dataset samples."""

from __future__ import annotations

from typing import Any

import numpy as np


def create_smooth_prior_mean(model: Any, true_parameter: Any, options: dict[str, Any]):
    """Diffuse the true parameter with the heat equation, matching main.py's prior mean."""
    import dolfin as dl
    import hippylib as hp

    steps = int(options.get("num_steps", 20))
    dt = float(options.get("dt", 0.01))
    kappa = float(options.get("kappa", 0.05))
    if steps < 0 or dt <= 0 or kappa < 0:
        raise ValueError("Prior-mean smoothing requires num_steps >= 0, dt > 0, and kappa >= 0")

    parameter_space = model.Vh[hp.PARAMETER]
    smoothed = dl.Function(parameter_space)
    source_vector = true_parameter.vector() if hasattr(true_parameter, "vector") else true_parameter
    source_values = (
        source_vector.get_local()
        if hasattr(source_vector, "get_local")
        else np.asarray(source_vector).reshape(-1)
    )
    if source_values.size != smoothed.vector().size():
        raise ValueError("True parameter size does not match the model parameter space")
    smoothed.vector().set_local(source_values)
    smoothed.vector().apply("insert")
    if steps == 0 or kappa == 0:
        return smoothed

    cache = getattr(model, "_prior_mean_smoothing_cache", {})
    cache_key = (steps, dt, kappa)
    solver_data = cache.get(cache_key)
    if solver_data is None:
        trial = dl.TrialFunction(parameter_space)
        test = dl.TestFunction(parameter_space)
        measure = dl.Measure("dx", parameter_space.mesh())
        mass = dl.assemble(trial * test * measure)
        stiffness = dl.dot(dl.grad(trial), dl.grad(test)) * measure
        operator = dl.assemble((1.0 / dt) * trial * test * measure + kappa * stiffness)
        solver_data = (mass, dl.LUSolver(operator))
        cache[cache_key] = solver_data
        model._prior_mean_smoothing_cache = cache

    mass, solver = solver_data
    rhs = dl.Vector()
    mass.init_vector(rhs, 0)
    next_field = dl.Function(parameter_space)
    for _ in range(steps):
        mass.mult(smoothed.vector(), rhs)
        rhs *= 1.0 / dt
        solver.solve(next_field.vector(), rhs)
        smoothed.assign(next_field)
    return smoothed


def _newton_solver(model: Any, options: dict[str, Any]):
    """Construct hIPPYlib's reduced-space Newton-CG solver from JSON options."""
    import hippylib as hp

    solver = hp.ReducedSpaceNewtonCG(model.model)
    solver.parameters["rel_tolerance"] = float(options.get("rel_tolerance", 1.0e-6))
    solver.parameters["abs_tolerance"] = float(options.get("abs_tolerance", 1.0e-8))
    solver.parameters["max_iter"] = int(options.get("max_iter", 35))
    solver.parameters["GN_iter"] = int(options.get("gauss_newton_iterations", 0))
    solver.parameters["cg_max_iter"] = int(options.get("cg_max_iter", 500))
    solver.parameters["cg_coarse_tolerance"] = float(options.get("cg_coarse_tolerance", 0.1))
    line_search = solver.parameters["LS"]
    line_search["c_armijo"] = float(options.get("line_search_armijo", 1.0e-2))
    line_search["max_backtracking_iter"] = int(options.get("line_search_max_backtracking", 200))
    return solver


def solve_map_point(model: Any, true_parameter: Any, options: dict[str, Any]):
    """Solve for a MAP parameter at the model's current noisy-data observation."""
    import hippylib as hp

    point = model.model.generate_vector("ALL")
    point[hp.PARAMETER].zero()
    initial = options.get("initial_guess", "prior_mean")
    if initial == "prior_mean":
        point[hp.PARAMETER].axpy(1.0, model.prior.mean)
    elif initial == "true_parameter":
        point[hp.PARAMETER].axpy(1.0, true_parameter)
    else:
        raise ValueError("map_solver.initial_guess must be 'prior_mean' or 'true_parameter'")

    model.model.solveFwd(point[hp.STATE], point)
    solver = _newton_solver(model, options)
    solver.solve(point)

    # Refresh state and adjoint at the returned parameter so saved data and
    # subsequent Hessian/Jacobian operations all share the final linearization.
    model.model.solveFwd(point[hp.STATE], point)
    model.model.solveAdj(point[hp.ADJOINT], point)
    converged = bool(getattr(solver, "converged", False))
    return point, {
        "converged": converged,
        "iterations": int(getattr(solver, "it", -1)),
        "final_cost": float(getattr(solver, "final_cost", np.nan)),
        "final_gradient_norm": float(getattr(solver, "final_grad_norm", np.nan)),
    }


def map_data_sensitivities(
    model: Any,
    map_point: Any,
    rng: np.random.Generator,
    options: dict[str, Any],
):
    """Compute notebook-style $H^{-1}J^T\Gamma^{-1}\delta d$ actions at a MAP point."""
    import dolfin as dl
    import hippyflow as hf
    import hippylib as hp

    count = int(options.get("directions", 1))
    perturbation_std = float(options.get("perturbation_std", 0.1))
    if count < 1 or perturbation_std < 0:
        raise ValueError("Sensitivity directions must be positive and perturbation_std non-negative")

    observable = hf.LinearStateObservable(model.model.problem, model.misfit.B)
    jacobian = hf.ObservableJacobian(observable)
    # This initializes both PDE and observation-misfit Hessian blocks.
    model.model.setPointForHessianEvaluations(
        map_point, gauss_newton_approx=bool(options.get("use_gauss_newton", False))
    )
    hessian = hp.ReducedHessian(model.model, misfit_only=False)
    hessian.gauss_newton_approx = bool(options.get("use_gauss_newton", False))

    data_direction = dl.Vector()
    model.misfit.B.init_vector(data_direction, 0)
    weighted_direction = dl.Vector()
    model.misfit.B.init_vector(weighted_direction, 0)
    noise_variance = float(model.misfit.noise_variance)
    if not np.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError(f"Invalid observation noise variance: {noise_variance}")

    parameter_rhs = model.model.generate_vector(hp.PARAMETER)
    sensitivity = model.model.generate_vector(hp.PARAMETER)
    cg_parameters = hp.CGSolverSteihaug_ParameterList()
    cg_parameters["rel_tolerance"] = float(options.get("cg_relative_tolerance", 1.0e-10))
    cg_parameters["max_iter"] = int(options.get("cg_max_iter", 1000))
    cg_solver = hp.CGSolverSteihaug(cg_parameters)
    cg_solver.set_operator(hessian)
    cg_solver.set_preconditioner(model.prior.Rsolver)

    actions = []
    directions = []
    iterations = []
    for direction_index in range(count):
        values = rng.normal(0.0, perturbation_std, size=data_direction.size())
        data_direction.set_local(values)
        data_direction.apply("insert")
        weighted_direction.zero()
        weighted_direction.axpy(1.0 / noise_variance, data_direction)

        parameter_rhs.zero()
        jacobian.transpmult(weighted_direction, parameter_rhs)
        sensitivity.zero()
        cg_solver.solve(sensitivity, parameter_rhs)
        actions.append(sensitivity.get_local().copy())
        directions.append(data_direction.get_local().copy())
        iterations.append(int(getattr(cg_solver, "iter", -1)))
        print(
            f"  MAP sensitivity {direction_index + 1}/{count}: "
            f"CG iterations={iterations[-1]}, norm={sensitivity.norm('l2'):.4e}",
            flush=True,
        )
    return np.asarray(directions), np.asarray(actions), iterations
