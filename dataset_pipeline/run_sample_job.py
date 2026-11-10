#!/usr/bin/env python3
"""Generate parameter/data/Jacobian samples for one local or array job."""


from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Set thread limits before importing NumPy, DOLFIN, PETSc, or hIPPYlib.
for _name in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
MSH_DIR = HERE.parent
sys.path.insert(0, str(MSH_DIR))
sys.path.insert(0, str(HERE))

from pipeline_config import SampleIndex, discover_model_files, load_config
from inverse_analysis import create_smooth_prior_mean, map_data_jacobian, solve_map_point


def _atomic_save(path: Path, array: np.ndarray) -> None:
    """Write NPY data atomically so interrupted tasks are safe to rerun."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(array))
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _make_file_alias(source: Path, alias: Path) -> None:
    """Create a same-filesystem hard-link alias, falling back to a copy."""
    if alias.exists():
        alias.unlink()
    try:
        os.link(source, alias)
    except OSError:
        import shutil
        shutil.copyfile(source, alias)


def _field_2d(sample: np.ndarray, channel: int) -> np.ndarray:
    """Extract one 2-D field from [H,W] or [C,H,W] sample layouts."""
    sample = np.asarray(sample)
    if sample.ndim == 2:
        return sample
    if sample.ndim == 3:
        if channel < 0 or channel >= sample.shape[0]:
            raise IndexError(f"velocity_channel={channel} invalid for sample shape {sample.shape}")
        return sample[channel]
    raise ValueError(f"Expected a 2-D field or [channels,H,W] sample, got {sample.shape}")


def _parameter_from_velocity(model, velocity: np.ndarray, settings: dict[str, Any], scale: float):
    """Interpolate a source velocity image onto the model parameter space."""
    import dolfin as dl
    from scipy.interpolate import RegularGridInterpolator
    import hippylib as hp

    field = _field_2d(velocity, int(settings.get("velocity_channel", 0)))
    box_pml = settings["box_pml"]
    ny, nx = field.shape
    x_coords = np.linspace(box_pml[0], box_pml[2], nx)
    y_coords = np.linspace(box_pml[1], box_pml[3], ny)
    interpolator = RegularGridInterpolator(
        (y_coords, x_coords), field, method="linear", bounds_error=False,
        fill_value=float(np.mean(field)),
    )

    parameter_space = model.Vh[hp.PARAMETER]
    coordinates = parameter_space.tabulate_dof_coordinates().reshape(-1, 2)
    points_yx = np.column_stack((coordinates[:, 1], coordinates[:, 0]))
    values = interpolator(points_yx) * scale
    parameter = dl.Function(parameter_space).vector()
    parameter.set_local(np.asarray(values, dtype=float))
    parameter.apply("insert")
    return parameter, field


def _compute_forward_jacobian(pde, observation_operator, x, n_observations: int, n_parameters: int):
    """Compute d(observation)/d(parameter), with shape [observation, parameter]."""
    import dolfin as dl
    import hippyflow as hf
    import hippylib as hp

    observable = hf.LinearStateObservable(pde, observation_operator)
    jacobian = hf.ObservableJacobian(observable)
    pde.setLinearizationPoint(x, False)

    basis = dl.Vector()
    observation_operator.init_vector(basis, 0)
    row = dl.Function(pde.Vh[hp.PARAMETER]).vector()
    result = np.empty((n_observations, n_parameters), dtype=np.float32)
    for obs_index in range(n_observations):
        basis.zero()
        basis[obs_index] = 1.0
        row.zero()
        jacobian.transpmult(basis, row)
        result[obs_index, :] = row.get_local()
        if obs_index and obs_index % 64 == 0:
            print(f"  Jacobian adjoints: {obs_index}/{n_observations}", flush=True)
    return result


def _build_model(config):
    """Create one model per task, keeping import-time work out of config modules."""
    import hippylib as hp
    from helmholtz_model import MultiFrequencyHelmholtzModel, helmholtz_settings

    settings = helmholtz_settings()
    settings.update(config.get("model_settings", {}))
    frequencies = [freq for source_freqs in settings["source_frequencies"] for freq in source_freqs]
    if not frequencies or len(set(map(float, frequencies))) != 1:
        raise ValueError(
            "The resolved helmholtz_settings must assign one identical frequency to every source."
        )
    model = MultiFrequencyHelmholtzModel(settings)
    return model


def _process_sample(model, sample, provenance, config, sample_id, rng: np.random.Generator,
                    skip_map: bool = False):
    """Generate and persist all requested quantities for a single input model."""
    import dolfin as dl
    import hippylib as hp

    outputs = config.get("outputs", {})
    make_map_data_sensitivities = (
        outputs.get("map_data_sensitivities", False)
        or outputs.get("map_data_jacobian", False)
        or outputs.get("map_sensitivity", False)
    )
    if skip_map:
        make_map_data_sensitivities = False
    settings = config.get("model_settings", {})
    parameter, raw_field = _parameter_from_velocity(
        model, sample, {**model.settings, "velocity_channel": config.get("velocity_channel", 0)},
        float(config["velocity_scale"]),
    )
    smooth_prior = create_smooth_prior_mean(
        model, parameter, config.get("prior_mean_smoothing", {})
    )
    model.prior.mean = smooth_prior.vector()

    # This model implementation combines all sources in one shared state, so
    # validate_config restricts the configured sources to one shared frequency.
    frequency, info = next(iter(model.frequency_models.items()))
    pde = info["pde"]
    misfit = info["misfit"]
    observation_operator = misfit.B
    x = model.model.generate_vector("ALL")
    x[hp.PARAMETER].zero()
    x[hp.PARAMETER].axpy(1.0, parameter)
    model.model.solveFwd(x[hp.STATE], x)

    clean_vector = dl.Vector()
    observation_operator.init_vector(clean_vector, 0)
    observation_operator.mult(x[hp.STATE], clean_vector)
    clean_data = clean_vector.get_local().copy()
    amplitude = float(np.max(np.abs(clean_data))) if clean_data.size else 0.0
    noise_std = max(float(config.get("noise_floor", 0.0)),
                    float(config["noise_relative_std"]) * amplitude)
    noisy_data = clean_data + rng.normal(0.0, noise_std, size=clean_data.shape)
    noise_variance = noise_std * noise_std
    misfit.d.set_local(noisy_data)
    misfit.d.apply("insert")
    misfit.noise_variance = noise_variance

    jacobian_matrix = None
    sensitivity = None
    fisher_diagonal = None
    map_manifest = None
    map_point = None
    map_jacobian = None
    map_jacobian_iterations = None
    map_jacobian_nonconverged = None
    if outputs.get("forward_jacobian", False) or outputs.get("data_sensitivity", False) or outputs.get("fisher_diagonal", False):
        jacobian_matrix = _compute_forward_jacobian(
            pde, observation_operator, x, clean_data.size, parameter.size()
        )
        residual = clean_data - noisy_data
        if outputs.get("data_sensitivity", False):
            sensitivity = jacobian_matrix.astype(np.float64).T @ residual / noise_variance
        if outputs.get("fisher_diagonal", False):
            fisher_diagonal = np.sum(jacobian_matrix.astype(np.float64) ** 2, axis=0) / noise_variance

    do_map = (
        outputs.get("map_point", False)
        or make_map_data_sensitivities
    ) and not skip_map
    if do_map:
        print(f"  Solving MAP point for sample {sample_id}", flush=True)
        map_point, map_manifest = solve_map_point(
            model, parameter, config.get("map_solver", {})
        )
        print(
            f"  MAP result: converged={map_manifest['converged']}, "
            f"iterations={map_manifest['iterations']}, cost={map_manifest['final_cost']:.6e}",
            flush=True,
        )
        if make_map_data_sensitivities:
            print(f"  Computing full MAP-data Jacobian for sample {sample_id}", flush=True)
            map_jacobian, map_jacobian_iterations, map_jacobian_nonconverged = map_data_jacobian(
                model,
                map_point,
                config.get("map_sensitivity", {}),
            )

    sample_dir = Path(config["output_dir"]) / f"sample_{sample_id:07d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    if outputs.get("prior_mean_parameter", True):
        _atomic_save(sample_dir / "prior_mean_parameter.npy", model.prior.mean.get_local())
    if outputs.get("parameter", True):
        _atomic_save(sample_dir / "true_parameter.npy", parameter.get_local())
    if outputs.get("velocity_field", True):
        _atomic_save(sample_dir / "velocity_field.npy", raw_field)
    if outputs.get("state", True):
        _atomic_save(sample_dir / "true_state.npy", x[hp.STATE].get_local())
    if outputs.get("clean_data", True):
        _atomic_save(sample_dir / "clean_observations.npy", clean_data)
    if outputs.get("noisy_data", True):
        _atomic_save(sample_dir / "noisy_observations.npy", noisy_data)
    if outputs.get("forward_jacobian", False):
        _atomic_save(sample_dir / "forward_jacobian.npy", jacobian_matrix)
    if outputs.get("data_sensitivity", False):
        _atomic_save(sample_dir / "misfit_gradient_at_true.npy", sensitivity)
    if outputs.get("fisher_diagonal", False):
        _atomic_save(sample_dir / "fisher_information_diagonal.npy", fisher_diagonal)
    if do_map and map_point is not None:
        if outputs.get("map_initial_parameter", True):
            initial_guess = config.get("map_solver", {}).get("initial_guess", "prior_mean")
            if initial_guess == "prior_mean":
                initial_parameter = model.prior.mean.get_local()
            elif initial_guess == "true_parameter":
                initial_parameter = parameter.get_local()
            else:
                raise ValueError("map_solver.initial_guess must be 'prior_mean' or 'true_parameter'")
            _atomic_save(sample_dir / "map_initial_parameter.npy", initial_parameter)
        _atomic_save(sample_dir / "map_parameter.npy", map_point[hp.PARAMETER].get_local())
        _atomic_save(sample_dir / "map_state.npy", map_point[hp.STATE].get_local())
        _atomic_save(sample_dir / "map_adjoint.npy", map_point[hp.ADJOINT].get_local())
    if make_map_data_sensitivities and map_jacobian is not None:
        expected_shape = (int(parameter.size()), int(clean_data.size))
        if map_jacobian.shape != expected_shape:
            raise ValueError(
                f"MAP-data sensitivity shape {map_jacobian.shape} does not match "
                f"the transpose layout of forward_jacobian: expected {expected_shape}"
            )
        if jacobian_matrix is not None and map_jacobian.shape != jacobian_matrix.T.shape:
            raise ValueError(
                "MAP-data sensitivity shape does not match forward_jacobian.T: "
                f"{map_jacobian.shape} vs {jacobian_matrix.T.shape}"
            )
        sensitivity_path = sample_dir / "map_Data_Sensitiviites.npy"
        _atomic_save(sensitivity_path, map_jacobian)
        # Keep readable/correctly-spelled names and the old output name as aliases
        # without storing additional copies on filesystems that support hard links.
        _make_file_alias(sensitivity_path, sample_dir / "map_data_sensitivities.npy")
        _make_file_alias(sensitivity_path, sample_dir / "map_data_jacobian.npy")

    manifest = {
        "pipeline_schema_version": 2,
        "status": "complete",
        "sample_id": sample_id,
        **provenance,
        "frequency_hz": float(frequency),
        "parameter_dofs": int(parameter.size()),
        "state_dofs": int(x[hp.STATE].size()),
        "observation_count": int(clean_data.size),
        "noise_std": noise_std,
        "noise_variance": noise_variance,
        "jacobian_shape": list(jacobian_matrix.shape) if jacobian_matrix is not None else None,
        "jacobian_axes": ["observation", "parameter"] if jacobian_matrix is not None else None,
        "velocity_scale": float(config["velocity_scale"]),
        "outputs": outputs,
        "model_settings": model.settings,
        "map_solver_settings": config.get("map_solver", {}),
        "prior_mean_smoothing_settings": config.get("prior_mean_smoothing", {}),
        "map_skipped": bool(skip_map),
        "map": map_manifest,
        "map_sensitivity": {
            "cg_iterations_by_observation": map_jacobian_iterations,
            "nonconverged_observation_indices": map_jacobian_nonconverged,
            "shape": list(map_jacobian.shape) if map_jacobian is not None else None,
            "axes": ["parameter", "observation"] if map_jacobian is not None else None,
            "filename": "map_Data_Sensitiviites.npy" if map_jacobian is not None else None,
            "forward_jacobian_transposed_shape": [int(parameter.size()), int(clean_data.size())],
            "prior_preconditioner_disabled": bool(
                config.get("map_sensitivity", {}).get("disable_prior_preconditioner", False)
            ),
            "method": "exact_reduced_hessian_unit_data_basis",
        } if map_jacobian_iterations is not None else None,
    }
    _atomic_json(sample_dir / "manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config" / "default.json"))
    parser.add_argument("--task-id", type=int, default=0, help="Zero-based job-array task ID")
    parser.add_argument("--num-tasks", type=int, default=1, help="Total number of array tasks")
    parser.add_argument("--sample-count", type=int, help="Override configured total number of samples")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even complete samples")
    parser.add_argument(
        "--skip_map",
        action="store_true",
        help="Skip the MAP point solve and MAP-data sensitivities entirely (no computation, no output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("skip map", args.skip_map)
    config = load_config(args.config)
    if args.sample_count is not None:
        if args.sample_count < 1:
            raise ValueError("--sample-count must be positive")
        config["sample_count"] = args.sample_count
    if args.num_tasks < 1 or args.task_id < 0 or args.task_id >= args.num_tasks:
        raise ValueError("Require 0 <= --task-id < --num-tasks, with --num-tasks >= 1")

    files = discover_model_files(config)
    sample_index = SampleIndex(files) 
    requested = int(config["sample_count"])
    if requested > sample_index.total_samples:
        raise ValueError(
            f"Requested {requested} samples, but {len(files)} matching model files provide "
            f"only {sample_index.total_samples}. Add model*.npy files or lower sample_count."
        )

    indices = list(range(args.task_id, requested, args.num_tasks))
   
    if not indices:
        print(f"Task {args.task_id}/{args.num_tasks}: no samples assigned.", flush=True)
        return

    print(f"Task {args.task_id}/{args.num_tasks}: {len(indices)} sample(s), requested total={requested}", flush=True)
    print(f"Discovered {len(files)} model files / {sample_index.total_samples} fields", flush=True)
    if args.skip_map:
        print("MAP point and MAP-data sensitivities will be skipped.", flush=True)
    model = _build_model(config)
    seed_base = int(config.get("seed", 0))

    for sample_id in indices:
        sample_dir = Path(config["output_dir"]) / f"sample_{sample_id:07d}"
        manifest_path = sample_dir / "manifest.json"
        if manifest_path.exists() and not args.overwrite:
            try:
                old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                outputs = config.get("outputs", {})
                needs_map_sensitivities = (
                    outputs.get("map_data_sensitivities", False)
                    or outputs.get("map_data_jacobian", False)
                    or outputs.get("map_sensitivity", False)
                ) and not args.skip_map
                expected_sensitivity_file = sample_dir / "map_data_sensitivities.npy"
                is_complete = (
                    old_manifest.get("status") == "complete"
                    and (not needs_map_sensitivities or expected_sensitivity_file.is_file())
                )
                if is_complete:
                    print(f"Skipping complete sample {sample_id}", flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        sample, provenance = sample_index.get(sample_id)
        started = time.time()
        sample_rng = np.random.default_rng(seed_base + sample_id)
        print(
        f"Generating sample {sample_id} from "
        f"{provenance['source_file']} "
        f"(global_index={provenance['global_index']}, "
        f"file_sample_index={provenance['file_sample_index']})",
        flush=True,
        ) 
        manifest = _process_sample(model, sample, provenance, config, sample_id, sample_rng,
                                   skip_map=args.skip_map)
        print(
            f"Completed sample {sample_id}: obs={manifest['observation_count']}, "
            f"params={manifest['parameter_dofs']}, elapsed={time.time() - started:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()