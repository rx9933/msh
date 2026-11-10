"""MAP estimation and data-to-MAP sensitivity routines for dataset samples."""

from __future__ import annotations

from typing import Any

import numpy as np


class _IdentityPreconditioner:
    def solve(self, out: Any, rhs: Any) -> None:
        out.zero()
        out.axpy(1.0, rhs)


class _IdentityPreconditionedModel:
    def __init__(self, model: Any):
        self._model = model
        self._preconditioner = _IdentityPreconditioner()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def Rsolver(self) -> _IdentityPreconditioner:
        return self._preconditioner


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

    disable_prior_preconditioner = options.get("disable_prior_preconditioner", False)
    if not isinstance(disable_prior_preconditioner, bool):
        raise ValueError("map_solver.disable_prior_preconditioner must be a boolean")
    optimization_model = (
        _IdentityPreconditionedModel(model.model)
        if disable_prior_preconditioner
        else model.model
    )
    solver = hp.ReducedSpaceNewtonCG(optimization_model)
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
    if options.get("disable_prior_preconditioner", False):
        print("  MAP Newton-CG prior preconditioner disabled", flush=True)
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


def map_data_jacobian(model: Any, map_point: Any, options: dict[str, Any]):
    """Compute the full derivative d(MAP parameter)/d(data) at a MAP point.

    Each column matches MAPJacobian.mult in the supplied reference: apply
    ``J.T / noise_variance`` to one data-space basis vector, then solve the
    exact reduced-Hessian system. The prior preconditioner can be disabled
    with ``disable_prior_preconditioner``.
    """
    import dolfin as dl
    import hippyflow as hf
    import hippylib as hp

    observable = hf.LinearStateObservable(model.model.problem, model.misfit.B)
    jacobian = hf.ObservableJacobian(observable)
    # Match MAPJacobian.setLinearizationPoint(x, False) in the reference.
    model.model.setPointForHessianEvaluations(map_point, gauss_newton_approx=False)
    hessian = hp.ReducedHessian(model.model, misfit_only=False)
    hessian.gauss_newton_approx = False

    data_basis = dl.Vector()
    model.misfit.B.init_vector(data_basis, 0)
    noise_variance = float(model.misfit.noise_variance)
    if not np.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError(f"Invalid observation noise variance: {noise_variance}")

    n_observations = int(data_basis.size())
    n_parameters = int(map_point[hp.PARAMETER].size())
    full_jacobian = np.empty((n_parameters, n_observations), dtype=np.float64)
    parameter_rhs = model.model.generate_vector(hp.PARAMETER)
    cg_parameters = hp.CGSolverSteihaug_ParameterList()
    cg_parameters["rel_tolerance"] = float(options.get("cg_relative_tolerance", 1.0e-10))
    cg_parameters["max_iter"] = int(options.get("cg_max_iter", 1000))
    if cg_parameters["rel_tolerance"] <= 0 or cg_parameters["max_iter"] < 1:
        raise ValueError("MAP Jacobian CG tolerance and iteration limit must be positive")
    disable_prior_preconditioner = options.get("disable_prior_preconditioner", False)
    if not isinstance(disable_prior_preconditioner, bool):
        raise ValueError("map_sensitivity.disable_prior_preconditioner must be a boolean")

    iterations = np.empty(n_observations, dtype=np.int32)
    nonconverged_observations = []
    for observation_index in range(n_observations):
        data_basis.zero()
        data_basis[observation_index] = 1.0 / noise_variance
        data_basis.apply("insert")
        parameter_rhs.zero()
        jacobian.transpmult(data_basis, parameter_rhs)

        sensitivity = model.model.generate_vector(hp.PARAMETER)
        cg_solver = hp.CGSolverSteihaug(cg_parameters)
        cg_solver.set_operator(hessian)
        if not disable_prior_preconditioner:
            cg_solver.set_preconditioner(model.prior.Rsolver)
        sensitivity.zero()
        cg_solver.solve(sensitivity, parameter_rhs)
        full_jacobian[:, observation_index] = sensitivity.get_local()
        iterations[observation_index] = int(getattr(cg_solver, "iter", -1))
        if iterations[observation_index] >= cg_parameters["max_iter"]:
            nonconverged_observations.append(observation_index)
        if (observation_index + 1) % 8 == 0 or observation_index + 1 == n_observations:
            print(
                f"  MAP-data Jacobian columns {observation_index + 1}/{n_observations}; "
                f"latest CG iterations={iterations[observation_index]}",
                flush=True,
            )
    return full_jacobian, iterations.tolist(), nonconverged_observations
