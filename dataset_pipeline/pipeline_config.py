"""Configuration and model-file indexing for the msh dataset pipeline."""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load JSON configuration and resolve input/output paths relative to it."""
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    config["_config_path"] = str(path)
    for key in ("model_dir", "output_dir"):
        value = Path(config[key]).expanduser()
        config[key] = str((path.parent / value).resolve() if not value.is_absolute() else value)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Reject invalid counts, missing paths, and unsupported frequency layouts."""
    if int(config.get("sample_count", 0)) < 1:
        raise ValueError("sample_count must be a positive integer")
    if float(config.get("velocity_scale", 0.0)) <= 0:
        raise ValueError("velocity_scale must be positive")
    if float(config.get("noise_relative_std", -1.0)) < 0:
        raise ValueError("noise_relative_std must be non-negative")
    if not Path(config["model_dir"]).is_dir():
        raise FileNotFoundError(f"Model input directory does not exist: {config['model_dir']}")
    settings = config.get("model_settings", {})
    frequencies = settings.get("source_frequencies")
    if frequencies:
        flattened = [float(freq) for source_freqs in frequencies for freq in source_freqs]
        if not flattened or len(set(flattened)) != 1:
            raise ValueError(
                "This pipeline currently requires every source to use the same single "
                "frequency. The msh combined model uses one shared state/observation "
                "layout and does not safely separate different frequencies by source."
            )


def _natural_key(path: Path) -> tuple[Any, ...]:
    """Sort names such as model2.npy before model10.npy."""
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r"(\d+)", path.name))


def discover_model_files(config: dict[str, Any]) -> list[Path]:
    """Find matching input NPY files in deterministic natural order."""
    model_dir = Path(config["model_dir"])
    pattern = config.get("model_glob", "model*.npy")
    files = sorted((p for p in model_dir.iterdir()
                    if p.is_file() and fnmatch.fnmatch(p.name, pattern)), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {model_dir}")
    return files


class SampleIndex:
    """Map one global sample index onto model file and within-file index."""

    def __init__(self, files: list[Path]):
        self.files = files
        self.lengths: list[int] = []
        self._arrays: dict[Path, Any] = {}
        for path in files:
            import numpy as np
            array = np.load(path, mmap_mode="r")
            if array.ndim not in (3, 4):
                raise ValueError(f"Expected [samples,H,W] or [samples,C,H,W], got {array.shape} in {path}")
            self.lengths.append(int(array.shape[0]))
        self.total_samples = sum(self.lengths)

    def get(self, global_index: int):
        """Return the input field and provenance for a zero-based sample index."""
        if global_index < 0 or global_index >= self.total_samples:
            raise IndexError(f"Sample index {global_index} outside 0..{self.total_samples - 1}")
        offset = global_index
        for path, length in zip(self.files, self.lengths):
            if offset < length:
                if path not in self._arrays:
                    import numpy as np
                    self._arrays[path] = np.load(path, mmap_mode="r")
                field = self._arrays[path][offset]
                return field, {
                    "global_index": global_index,
                    "source_file": str(path),
                    "file_sample_index": offset,
                }
            offset -= length
        raise RuntimeError("Sample index mapping failed")
