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
- `map_Data_Sensitiviites.npy`: full derivative of the MAP parameter with respect to observations, computed one unit observation perturbation at a time. Its shape is `[parameter, observation]`, exactly the transpose shape of `forward_jacobian.npy` (`[observation, parameter]`). Column $i$ is $H^{-1}J^T\Gamma^{-1}e_i$, matching the supplied `MAPJacobian.mult` implementation. Correctly spelled `map_data_sensitivities.npy` and legacy `map_data_jacobian.npy` are hard-link aliases when supported.
- `manifest.json`: provenance, dimensions, noise settings, frequency, and output/Jacobian schema.

The dense forward Jacobian is optional via `outputs.forward_jacobian`. Enabling data sensitivity or Fisher diagonal also computes it, then only the requested result arrays are written. MAP estimation is controlled by `outputs.map_point`; set `map_solver.disable_prior_preconditioner` to `true` to run its Newton-CG solves without the prior Hessian preconditioner (the default is `false`). `model_settings.verbose: true` enables hIPPYlib's CG termination output for each Newton iteration; `map_solver.print_level` can override the verbosity level directly (1 shows CG termination reasons, 2 also prints each CG iteration). Verbosity does not change the objective or its gradient. The full MAP data Jacobian is enabled by `outputs.map_data_sensitivities` (or the compatibility flags `outputs.map_data_jacobian` / `outputs.map_sensitivity`). It uses the exact reduced Hessian, observation-noise variance, and prior-preconditioned hIPPYlib CG by default, as in `MAPJacobian.mult`; independently, set `map_sensitivity.disable_prior_preconditioner` to `true` to disable prior preconditioning for those sensitivity solves. It requires one Hessian solve per observation; with 256 observations, that means 256 CG solves per sample, substantially more expensive than a single random-direction sensitivity action. `map_sensitivity.cg_relative_tolerance` and `cg_max_iter` tune each solve. The manifest records the preconditioner settings, iterations, columns that hit the iteration limit, and the matrix shape. At 8,321 parameter DOFs and 256 observations, a float64 matrix is about 16.3 MiB per sample (about 159 GiB for 10,000 samples). At 32 sensors, 4 sources, and real/imaginary observations, the current configured data dimension is 256.

Adjust `prior_mean_smoothing`, `map_solver`, and the settings in `map_sensitivity` to control prior diffusion, optimizer tolerances, iteration limits, initial guess, preconditioning, and accuracy/cost of the full MAP-data Jacobian. Changes to the config apply only to jobs started after the change; already-running workers have loaded their config.

## Visualize a generated sample

Use `visualize_sample.py` to create notebook-style spatial figures: input velocity, true/prior-mean/initial/MAP parameter fields, true/MAP/adjoint state real and imaginary parts by source, observations at sensor locations, and representative columns of the MAP-data Jacobian as parameter fields. The requested spatial overview is saved as `visualization.png`; the complete generic array collage is saved as `visualization_all_arrays.png`. It rebuilds the finite-element spaces from the generation config (or the settings stored in the manifest). Run it in the DOLFIN/hIPPYlib environment used to generate the samples; pass `--config path/to/generation.json` if a non-default config was used.

```text
python3 visualize_sample.py generated_data/sample_0000000
python3 visualize_sample.py generated_data/sample_0000000 --output sample_0000000.png
```

If no sample directory is given, the script uses `generated_data/sample_0000000` relative to `dataset_pipeline`.

## Submit a large sample run to Slurm

`submit_slurm_samples.sh` submits 10,000 samples as one-core jobs and keeps at most 20 of its own jobs queued/running, submitting another task as soon as a slot opens. Each job processes a small deterministic share of the sample IDs; completed samples are skipped if a failed task is retried. Defaults are partition `vv`, one CPU per job, 16 GB, 48 hours, and 10 samples per job. You must set `SLURM_ACCOUNT` to your valid project/allocation charge code; the script passes it to Slurm as `--account` and refuses to submit if it is missing. Set `CONDA_ENV` if the worker needs a conda environment. From the login node, run:

```text
SLURM_ACCOUNT=YOUR_CHARGE_CODE bash submit_slurm_samples.sh config/default.json
```

Override settings through environment variables, for example `TOTAL_SAMPLES=10000 SAMPLES_PER_JOB=1 MAX_ACTIVE_JOBS=20 PARTITION=vv-dev SLURM_WALLTIME=02:00:00 bash submit_slurm_samples.sh config/default.json`. Use `vv`/`vv-large` with an appropriate walltime for production; the `*-dev` queues have a two-hour limit. The controller uses `squeue` only for job IDs it submitted, retries failed tasks twice, and writes logs under `logs_slurm_samples/`.

## Model limitations

The current `msh` combined multifrequency implementation uses one shared state layout across sources. This pipeline therefore validates that all configured sources use the same single frequency; distinct frequencies per source must not be silently mixed into this dataset. Tune mesh resolution and the frequency in `model_settings` before launching large arrays. Array files are memory-mapped and individual fields are loaded on demand, so the full source corpus is not read into RAM.


'''
m) arsadam@ccgo3:/storage/arushi/posa_dino/dino_tutorial/applications/multisourcehelmholtz/msh/dataset_pipeline$ python3 run_sample_job.py --config config/default.json --sample-count 1
Task 0/1: 1 sample(s), requested total=1
Discovered 4 model files / 2000 fields
Multi-frequency model with 4 (source, frequency) pairs
  Pair 1: Source 1, Freq 1000Hz
  Pair 2: Source 2, Freq 1000Hz
  Pair 3: Source 3, Freq 1000Hz
  Pair 4: Source 4, Freq 1000Hz
Number of observation points: 32
Base mesh: 64x64, cells=16384, vertices=8321
Function spaces: state=264200 (4 sources x 2 components), parameter=8321, adjoint=264200
  Using Gaussian regularization (BiLaplacian)
  Frequency 1000Hz: wave number=1.519683e+01, state=264200, parameter=8321, observations=256
Created 1 frequency models from 4 (source, frequency) pairs
Generating sample 0 from /storage/arushi/posa_dino/dino_tutorial/applications/multisourcehelmholtz/downloads/model/model1.npy
  Jacobian adjoints: 64/256
  Jacobian adjoints: 128/256
  Jacobian adjoints: 192/256
  Solving MAP point for sample 0

It  cg_it cost            misfit          reg             (g,dm)          ||g||L2        alpha          tolcg         
  1   6    2.893125e+02    2.892743e+02    3.823011e-02   -2.530760e+03   6.534629e+05   1.000000e+00   1.000000e-01
  2  35    5.430800e+01    5.337376e+01    9.342336e-01   -4.946693e+02   1.623025e+05   1.000000e+00   1.000000e-01
  3  69    4.160854e+01    3.948369e+01    2.124845e+00   -1.234640e+02   3.923908e+04   2.500000e-01   1.000000e-01
  4  19    2.701902e+01    2.492030e+01    2.098722e+00   -2.958831e+01   4.702132e+04   1.000000e+00   1.000000e-01
  5  92    2.288065e+01    2.006866e+01    2.811985e+00   -9.096462e+01   1.530638e+04   6.250000e-02   1.000000e-01
  6  30    1.917818e+01    1.617888e+01    2.999302e+00   -7.414951e+00   1.777242e+04   1.000000e+00   1.000000e-01
  7 199    1.841315e+01    1.508218e+01    3.330974e+00   -1.415614e+02   6.290465e+03   7.812500e-03   9.811399e-02
  8  74    1.609429e+01    1.117970e+01    4.914584e+00   -4.738205e+00   7.305154e+03   1.000000e+00   1.000000e-01
  9  51    1.568034e+01    1.070653e+01    4.973815e+00   -8.280664e-01   4.607948e+03   1.000000e+00   8.397371e-02
 10 500    1.538636e+01    9.405418e+00    5.980937e+00   -1.643869e+00   1.949673e+03   5.000000e-01   5.462235e-02
 11  23    1.516807e+01    9.186930e+00    5.981138e+00   -4.380330e-01   5.784285e+03   1.000000e+00   9.408370e-02
 12 500    1.515006e+01    9.143380e+00    6.006682e+00   -8.903908e-01   1.372736e+03   3.125000e-02   4.583350e-02
 13 264    1.495368e+01    8.181206e+00    6.772477e+00   -3.992181e-01   1.240181e+03   1.000000e+00   4.356444e-02
 14  32    1.494966e+01    8.177873e+00    6.771786e+00   -8.046469e-03   8.018682e+02   1.000000e+00   3.503007e-02
 15 500    1.494227e+01    8.186980e+00    6.755295e+00   -2.404383e-02   1.383055e+02   5.000000e-01   1.454820e-02
 16 232    1.493868e+01    8.220590e+00    6.718087e+00   -6.994894e-03   4.517422e+02   1.000000e+00   2.629268e-02
 17 500    1.493732e+01    8.198987e+00    6.738331e+00   -8.688061e-03   5.083077e+01   5.000000e-01   8.819681e-03
 18  50    1.493550e+01    8.199282e+00    6.736222e+00   -3.627866e-03   5.057249e+02   1.000000e+00   2.781933e-02
 19 500    1.493518e+01    8.190680e+00    6.744501e+00   -2.808649e-03   5.054993e+01   1.000000e+00   8.795283e-03
 20  51    1.493390e+01    8.191447e+00    6.742451e+00   -2.565156e-03   4.490474e+02   1.000000e+00   2.621414e-02
 21 500    1.493365e+01    8.196030e+00    6.737619e+00   -4.872979e-04   4.009713e+01   1.000000e+00   7.833326e-03
 22 500    1.493360e+01    8.193279e+00    6.740325e+00   -9.061562e-05   2.622008e+01   1.000000e+00   6.334416e-03
 23 500    1.493359e+01    8.193898e+00    6.739696e+00   -2.024545e-05   9.739693e+00   1.000000e+00   3.860665e-03
 24 500    1.493359e+01    8.193294e+00    6.740298e+00   -5.219639e-06   2.663059e+00   1.000000e+00   2.018738e-03
 25 500    1.493359e+01    8.193401e+00    6.740189e+00   -1.297045e-06   1.880641e+00   1.000000e+00   1.696456e-03
  MAP result: converged=True, iterations=25, cost=1.493359e+01
  Computing MAP-data sensitivities for sample 0
Maximum Number of Iterations Reached
Not Converged. Final residual norm  26.866751619554833
  MAP sensitivity 1/1: CG iterations=1001, norm=6.6755e+01
Completed sample 0: obs=256, params=8321, elapsed=4189.6s
'''