# Modular MSH dataset pipeline

This pipeline generates per-input-model parameter/data samples and their forward sensitivities without importing or executing `msh/main.py` or the legacy generation scripts. Configuration is in `config/default.json`; model/PDE code remains in the parent `msh` directory.

## Choose the total sample count

Set `sample_count` in the JSON config, or override it on the runner command line with `--sample-count N`. Input files matching `model_glob` are discovered and naturally sorted (for example, `model2.npy` before `model10.npy`). Each file is treated as a sample block. The loader reads as many blocks as needed to reach the requested count and fails early if their combined first dimensions are too small. It accepts `[samples,H,W]` and `[samples,channels,H,W]` arrays; the shipped config selects channel 0 for the latter, matching the existing `vel[j, :, :, :][0]` convention.

Examples, from this directory:

```text
python3 run_sample_job.py --config config/default.json --sample-count 750
bash run_local.sh config/default.json 8 2
bash submit_slurm.sh config/default.json 8 750
```

The local launcher runs eight deterministic shards with at most two concurrent tasks. The Slurm launcher submits an array of eight tasks, each receiving its share of the requested 750 samples. Set `PYTHON_BIN` to the Python executable for the environment containing DOLFIN, hIPPYlib, hippyflow, SciPy, and NumPy. Slurm resource settings can be overridden with `CPUS_PER_TASK`, `SLURM_MEMORY`, `SLURM_WALLTIME`, and `SLURM_LOG_DIR`. The submit script is scheduler-specific; it does not activate a conda/virtual environment or load site modules because those vary by cluster.

For reliable and non-overlapping task assignment, each task processes global sample IDs `task_id, task_id + num_tasks, ...`. IDs remain stable across reruns. Samples are stored independently in `generated_data/sample_XXXXXXX/`; a completed manifest makes retries skip completed work. To regenerate, pass `--overwrite` to the Python runner.

## Per-sample outputs

Output flags in the JSON config control which arrays are written:

- `true_parameter.npy`: interpolated velocity field multiplied by `velocity_scale` (default $10^{-4}$, matching the existing inverse scripts).
- `prior_mean_parameter.npy`, `map_initial_parameter.npy`: the heat-diffused true parameter used as prior mean, and the initial parameter used for the MAP solve, respectively. Smoothing uses the `prior_mean_smoothing` settings (defaults match `main.py`: 20 diffusion steps, `dt=0.01`, `kappa=0.05`, natural/Neumann boundary conditions).
- `velocity_field.npy`: source image selected from the downloaded NPY.
- `true_state.npy`: forward PDE state degrees of freedom.
- `clean_observations.npy`, `noisy_observations.npy`: pointwise data vectors; noise is Gaussian with configured relative standard deviation and a floor.
- `forward_jacobian.npy`: the data-with-respect-to-parameter Jacobian, stored in `[observation, parameter]` order.
- `misfit_gradient_at_true.npy`: $J^T(clean-noisy)/noise_variance$, the data-misfit gradient at the true parameter.
- `fisher_information_diagonal.npy`: diagonal of $J^T J/noise_variance$.
- `map_parameter.npy`, `map_state.npy`, `map_adjoint.npy`: the converged (or best-returned) reduced-space MAP point for the generated noisy observations.
- `map_data_perturbations.npy`, `map_data_sensitivities.npy`: random data perturbations and notebook-style actions $H^{-1}J^T\Gamma^{-1}\delta d$, with axes `[direction, observation]` and `[direction, parameter]`, respectively.
- `manifest.json`: provenance, dimensions, noise settings, frequency, and output/Jacobian schema.

The dense forward Jacobian is optional via `outputs.forward_jacobian`. Enabling data sensitivity or Fisher diagonal also computes it, then only the requested result arrays are written. MAP estimation and MAP-data sensitivity are separately controlled by `outputs.map_point` and `outputs.map_sensitivity`; MAP sensitivities require a MAP solve and therefore also save the MAP point. The inverse sensitivity uses a configured Gaussian perturbation scale, the configured observation noise variance, the exact reduced Hessian by default, and a prior-preconditioned hIPPYlib CG solve. At 32 sensors, 4 sources, and real/imaginary observations, the current configured data dimension is 256.

Adjust `prior_mean_smoothing`, `map_solver`, and `map_sensitivity` in the config JSON to control prior diffusion, optimizer tolerances, iteration limits, initial guess, number and scale of data perturbations, exact/Gauss-Newton Hessian selection, and sensitivity CG tolerance.

## Visualize a generated sample

Use `visualize_sample.py` to create notebook-style spatial figures: input velocity, true/prior-mean/initial/MAP parameter fields, true/MAP/adjoint state real and imaginary parts by source, observations at sensor locations, and parameter sensitivity fields. The requested spatial overview is saved as `visualization.png`; the complete generic array collage is saved as `visualization_all_arrays.png`. It rebuilds the finite-element spaces from the generation config (or the settings stored in the manifest). Run it in the DOLFIN/hIPPYlib environment used to generate the samples; pass `--config path/to/generation.json` if a non-default config was used.

```text
python3 visualize_sample.py generated_data/sample_0000000
python3 visualize_sample.py generated_data/sample_0000000 --output sample_0000000.png
```

If no sample directory is given, the script uses `generated_data/sample_0000000` relative to `dataset_pipeline`.

## Model limitations

The current `msh` combined multifrequency implementation uses one shared state layout across sources. This pipeline therefore validates that all configured sources use the same single frequency; distinct frequencies per source must not be silently mixed into this dataset. Tune mesh resolution and the frequency in `model_settings` before launching large arrays. Array files are memory-mapped and individual fields are loaded on demand, so the full source corpus is not read into RAM.
