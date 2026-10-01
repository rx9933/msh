#!/usr/bin/env python3
"""Plot every NPY output (and manifest summary) for one generated sample.

Run from this directory, for example:
    python visualize_sample.py generated_data/sample_0000000
    python visualize_sample.py generated_data/sample_0000000 -o out.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so figures always save headlessly

import matplotlib.pyplot as plt
import numpy as np


def _display_values(array: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert an array to real values suitable for plotting."""
    if np.iscomplexobj(array):
        return np.abs(array), "magnitude of complex values"
    return array, ""


def _plot_array(axis: Any, path: Path) -> None:
    array = np.load(path, allow_pickle=False)
    values, note = _display_values(array)
    label = f"{path.name}\nshape={array.shape}, dtype={array.dtype}"
    if note:
        label += f" ({note})"

    if array.size == 0:
        axis.text(0.5, 0.5, "Empty array", ha="center", va="center")
        axis.set_title(label)
        axis.set_axis_off()
        return

    if not np.issubdtype(values.dtype, np.number):
        axis.text(0.02, 0.98, np.array2string(values, threshold=40), va="top")
        axis.set_title(label)
        axis.set_axis_off()
        return

    values = np.asarray(values)
    if values.ndim == 0:
        axis.text(0.5, 0.5, f"{values.item():.5g}", ha="center", va="center")
        axis.set_axis_off()
    elif values.ndim == 1:
        axis.plot(np.arange(values.size), values, linewidth=0.9)
        axis.set_xlabel("index")
        axis.grid(alpha=0.2)
    else:
        if values.ndim > 2:
            values = values.reshape(-1, values.shape[-1])
            label += f"\nviewing as {values.shape[0]} × {values.shape[1]}"
        image = axis.imshow(values, aspect="auto", origin="lower", interpolation="nearest")
        axis.set_xlabel("column / last-axis index")
        axis.set_ylabel("row / flattened leading axes")
        plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_title(label, fontsize=9)


def _manifest_summary(path: Path) -> str:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "status", "sample_id", "global_index", "frequency_hz", "noise_std",
        "parameter_dofs", "state_dofs", "observation_count", "jacobian_shape",
    )
    lines = [f"{key}: {manifest[key]}" for key in keys if key in manifest]
    if manifest.get("map"):
        lines.append(f"MAP: {manifest['map']}")
    if manifest.get("outputs"):
        enabled = [name for name, enabled in manifest["outputs"].items() if enabled]
        lines.append("saved outputs: " + ", ".join(enabled))
    return "\n".join(lines) or json.dumps(manifest, indent=2)


def _load_optional(sample_dir: Path, name: str) -> np.ndarray | None:
    path = sample_dir / name
    return np.load(path, allow_pickle=False) if path.exists() else None


def _save_finite_element_views(
    sample_dir: Path,
    output: Path,
    dpi: int,
    config_path: Path,
) -> None:
    """Rebuild the configured FE spaces and plot fields on the actual mesh."""
    try:
        import dolfin as dl
        import hippylib as hp
    except ImportError as error:
        raise RuntimeError(
            "Spatial finite-element plots require the same DOLFIN/hIPPYlib environment used to generate the sample."
        ) from error

    here = Path(__file__).resolve().parent
    msh_dir = here.parent
    if str(msh_dir) not in sys.path:
        sys.path.insert(0, str(msh_dir))
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from helmholtz_model import MultiFrequencyHelmholtzModel, helmholtz_settings
    from inverse_analysis import create_smooth_prior_mean

    config = json.loads(config_path.read_text(encoding="utf-8"))
    sample_manifest = sample_dir / "manifest.json"
    manifest = json.loads(sample_manifest.read_text(encoding="utf-8")) if sample_manifest.exists() else {}
    settings = helmholtz_settings()
    settings.update(config.get("model_settings", {}))
    settings.update(manifest.get("model_settings", {}))
    model = MultiFrequencyHelmholtzModel(settings)
    smoothing_settings = manifest.get(
        "prior_mean_smoothing_settings", config.get("prior_mean_smoothing", {})
    )

    def vector_function(space: Any, values: np.ndarray) -> Any:
        function = dl.Function(space)
        local_values = np.asarray(values).reshape(-1)
        if function.vector().size() != local_values.size:
            raise ValueError(
                f"Cannot map saved vector of length {local_values.size} onto FE space "
                f"of dimension {function.vector().size()}; pass the generation config with --config."
            )
        function.vector().set_local(local_values)
        function.vector().apply("insert")
        return function

    def plot_field(axis: Any, space: Any, values: np.ndarray, title: str) -> None:
        function = vector_function(space, values)
        plt.sca(axis)
        color_plot = dl.plot(function, title=title)
        plt.colorbar(color_plot, ax=axis, fraction=0.046, pad=0.04)

    def load(name: str) -> np.ndarray | None:
        return _load_optional(sample_dir, name)

    velocity = load("velocity_field.npy")
    true_parameter = load("true_parameter.npy")
    map_parameter = load("map_parameter.npy")
    prior_mean = load("prior_mean_parameter.npy")
    if prior_mean is None:
        if "prior_mean_smoothing_settings" in manifest and true_parameter is not None:
            prior_mean = create_smooth_prior_mean(
                model, true_parameter, smoothing_settings
            ).vector().get_local()
        else:
            prior_mean = model.prior.mean.get_local()
    initial_parameter = load("map_initial_parameter.npy")
    if initial_parameter is None:
        initial_guess = manifest.get("map_solver_settings", config.get("map_solver", {})).get(
            "initial_guess", "prior_mean"
        )
        if initial_guess == "prior_mean":
            initial_parameter = prior_mean
        elif initial_guess == "true_parameter":
            initial_parameter = true_parameter

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    field_axis = axes[0, 0]
    if velocity is not None and np.squeeze(velocity).ndim == 2:
        image = field_axis.imshow(np.squeeze(velocity), origin="lower", aspect="auto", cmap="viridis")
        field_axis.set_title("Input velocity field")
        field_axis.set_xlabel("x pixel")
        field_axis.set_ylabel("y pixel")
        figure.colorbar(image, ax=field_axis, fraction=0.046, pad=0.04)
    else:
        field_axis.text(0.5, 0.5, "velocity_field.npy unavailable", ha="center")
        field_axis.set_axis_off()

    parameter_space = model.Vh[hp.PARAMETER]
    parameter_fields = (
        (true_parameter, "True parameter"),
        (prior_mean, "Prior mean parameter"),
        (initial_parameter, "Initial parameter for MAP solve"),
        (map_parameter, "MAP parameter"),
    )
    parameter_axes = (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1])
    for axis, (values, title) in zip(parameter_axes, parameter_fields):
        if values is None:
            axis.text(0.5, 0.5, f"{title} unavailable", ha="center")
            axis.set_axis_off()
        else:
            plot_field(axis, parameter_space, values, title)

    difference_axis = axes[1, 2]
    if (map_parameter is not None and true_parameter is not None
            and map_parameter.size == true_parameter.size):
        difference = np.ravel(map_parameter) - np.ravel(true_parameter)
        plot_field(difference_axis, parameter_space, difference, "MAP − true parameter")
    else:
        difference_axis.text(0.5, 0.5, "MAP − true parameter unavailable", ha="center")
        difference_axis.set_axis_off()
    figure.suptitle("Parameter fields on the finite-element mesh", fontsize=14)
    figure.tight_layout()
    fields_path = output
    figure.savefig(fields_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved spatial parameter plots to {fields_path}")

    state_arrays = (
        ("true_state.npy", "True state"),
        ("map_state.npy", "MAP state"),
        ("map_adjoint.npy", "MAP adjoint"),
    )
    available_states = [(load(name), title) for name, title in state_arrays if (sample_dir / name).exists()]
    if available_states:
        state_space = model.Vh[hp.STATE]
        figure, axes = plt.subplots(2 * len(available_states), model.n_sources,
                                    figsize=(4.2 * model.n_sources, 3.6 * 2 * len(available_states)),
                                    squeeze=False)
        for state_index, (values, title) in enumerate(available_states):
            state_function = vector_function(state_space, values)
            for source in range(model.n_sources):
                for component, component_name in enumerate(("Real", "Imaginary")):
                    axis = axes[2 * state_index + component, source]
                    scalar_space = state_space.sub(source).sub(component).collapse()
                    scalar_function = dl.interpolate(
                        state_function.sub(source).sub(component), scalar_space
                    )
                    plt.sca(axis)
                    color_plot = dl.plot(scalar_function, title=f"{title} — source {source + 1} ({component_name})")
                    plt.colorbar(color_plot, ax=axis, fraction=0.046, pad=0.04)
        figure.suptitle("State fields by source and real/imaginary component", fontsize=14)
        figure.tight_layout()
        state_path = output.with_name(f"{output.stem}_states{output.suffix or '.png'}")
        figure.savefig(state_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved spatial state plots to {state_path}")

    clean = load("clean_observations.npy")
    noisy = load("noisy_observations.npy")
    if clean is not None or noisy is not None:
        for data, label in ((clean, "clean"), (noisy, "noisy")):
            if data is None:
                continue
            if data.size != model.targets.shape[0] * model.n_sources * 2:
                raise ValueError(
                    "Observation vector size does not match the reconstructed model's sensors/sources; "
                    "pass the correct generation config with --config."
                )
            data_view = data.reshape(model.targets.shape[0], model.n_sources, 2)
            figure, axes = plt.subplots(3, model.n_sources, figsize=(4.2 * model.n_sources, 9), squeeze=False)
            for source in range(model.n_sources):
                real = data_view[:, source, 0]
                imaginary = data_view[:, source, 1]
                for row, (values, component) in enumerate((
                    (real, "Real"), (imaginary, "Imaginary"),
                    (np.hypot(real, imaginary), "Magnitude"),
                )):
                    axis = axes[row, source]
                    scatter = axis.scatter(
                        model.targets[:, 0], model.targets[:, 1], c=values,
                        cmap="RdBu_r" if component != "Magnitude" else "viridis",
                        s=36, edgecolors="black", linewidths=0.35,
                    )
                    axis.set_title(f"{label.capitalize()} — source {source + 1} ({component})")
                    axis.set_aspect("equal")
                    axis.set_xlabel("x")
                    axis.set_ylabel("y")
                    figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
            figure.suptitle(f"{label.capitalize()} observations at sensor locations", fontsize=14)
            figure.tight_layout()
            observation_path = output.with_name(f"{output.stem}_{label}_observations{output.suffix or '.png'}")
            figure.savefig(observation_path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved spatial observation plots to {observation_path}")

    sensitivities = load("map_data_sensitivities.npy")
    gradient = load("misfit_gradient_at_true.npy")
    fisher = load("fisher_information_diagonal.npy")
    spatial_parameter_arrays: list[tuple[np.ndarray, str]] = []
    if sensitivities is not None:
        spatial_parameter_arrays.extend(
            (row, f"MAP sensitivity direction {index + 1}")
            for index, row in enumerate(np.atleast_2d(sensitivities))
        )
    if gradient is not None:
        spatial_parameter_arrays.append((gradient, "Misfit gradient at true parameter"))
    if fisher is not None:
        spatial_parameter_arrays.append((fisher, "Fisher information diagonal"))
    if spatial_parameter_arrays:
        columns = min(3, len(spatial_parameter_arrays))
        rows = math.ceil(len(spatial_parameter_arrays) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.2 * rows), squeeze=False)
        for axis, (values, title) in zip(axes.ravel(), spatial_parameter_arrays):
            plot_field(axis, parameter_space, values, title)
        for axis in axes.ravel()[len(spatial_parameter_arrays):]:
            axis.set_visible(False)
        figure.suptitle("Sensitivity and information fields", fontsize=14)
        figure.tight_layout()
        sensitivity_path = output.with_name(f"{output.stem}_sensitivity_fields{output.suffix or '.png'}")
        figure.savefig(sensitivity_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved spatial sensitivity plots to {sensitivity_path}")


def visualize_sample(
    sample_dir: Path,
    output: Path | None = None,
    dpi: int = 140,
    show: bool = False,
    config_path: Path | None = None,
) -> Path:
    """Render all NPY arrays + manifest into a single figure and save it.

    Returns the path of the saved figure.
    """
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory does not exist: {sample_dir}")

    arrays = sorted(sample_dir.glob("*.npy"))
    manifest_path = sample_dir / "manifest.json"
    if not arrays and not manifest_path.exists():
        raise FileNotFoundError(f"No .npy files or manifest.json found in {sample_dir}")

    panel_count = len(arrays) + int(manifest_path.exists())
    columns = min(3, panel_count)
    rows = math.ceil(panel_count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.8 * rows), squeeze=False)
    flat_axes = axes.ravel()

    for axis, path in zip(flat_axes, arrays):
        _plot_array(axis, path)
    next_axis = len(arrays)
    if manifest_path.exists():
        axis = flat_axes[next_axis]
        axis.text(
            0.02, 0.98, _manifest_summary(manifest_path),
            va="top", family="monospace", fontsize=8, wrap=True,
        )
        axis.set_title("manifest.json")
        axis.set_axis_off()
        next_axis += 1
    for axis in flat_axes[next_axis:]:
        axis.set_visible(False)

    figure.suptitle(f"Generated sample: {sample_dir}", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    if output is None:
        output = sample_dir / "visualization.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    all_arrays_output = output.with_name(f"{output.stem}_all_arrays{output.suffix or '.png'}")
    figure.savefig(all_arrays_output, dpi=dpi, bbox_inches="tight")
    print(f"Saved all-arrays overview to {all_arrays_output}")
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config" / "default.json"
    _save_finite_element_views(sample_dir, output, dpi, config_path)

    if show:
        plt.show()
    plt.close(figure)
    return output


def _discover_samples(root: Path) -> list[Path]:
    """Find sample directories under root (dirs containing *.npy or manifest.json)."""
    if (root / "manifest.json").exists() or any(root.glob("*.npy")):
        return [root]
    return sorted(
        child for child in root.iterdir()
        if child.is_dir() and ((child / "manifest.json").exists() or any(child.glob("*.npy")))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sample_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_data" / "sample_0000000",
        help="sample directory, or a parent directory containing multiple samples "
             "(default: generated_data/sample_0000000)",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        help="output image path; for multiple samples this is treated as a directory",
    )
    parser.add_argument("--dpi", type=int, default=140, help="resolution for the saved image")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parent / "config" / "default.json",
        help="generation config used to create this sample (default: config/default.json)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="also open the figures interactively (default: only save)",
    )
    args = parser.parse_args()

    samples = _discover_samples(args.sample_dir)
    if not samples:
        raise FileNotFoundError(f"No samples found under {args.sample_dir}")

    multi = len(samples) > 1
    for sample in samples:
        if args.output is None:
            out = sample / "visualization.png"
        elif multi or args.output.suffix == "":
            out = args.output / f"{sample.name}.png"
        else:
            out = args.output
        visualize_sample(sample, out, args.dpi, show=args.show, config_path=args.config)


if __name__ == "__main__":
    main()