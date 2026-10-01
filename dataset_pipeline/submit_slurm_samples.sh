#!/usr/bin/env bash
# Submit a fixed-size pool of Slurm jobs and refill slots as jobs finish.
# Defaults: 10,000 samples, 10 samples/job, at most 20 queued/running jobs.
# Run this script from a login node, not from inside an sbatch allocation.

set -uo pipefail

# Resolve the directory containing THIS script. When Slurm runs the script as
# a batch job, BASH_SOURCE points into /var/spool/slurmd/... , so callers must
# pass REAL_HERE explicitly in worker mode.
ORIG_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$ORIG_HERE/$(basename "${BASH_SOURCE[0]}")"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-10000}"
SAMPLES_PER_JOB="${SAMPLES_PER_JOB:-10}"
MAX_ACTIVE_JOBS="${MAX_ACTIVE_JOBS:-20}"
PARTITION="${PARTITION:-debug}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${SLURM_MEMORY:-16G}"
WALLTIME="${SLURM_WALLTIME:-48:00:00}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_RETRIES="${MAX_RETRIES:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV="${CONDA_ENV:-}"
ACCOUNT="${SLURM_ACCOUNT:-FTA-SUB-Ghattas}"
LOG_DIR_DEFAULT="${SLURM_LOG_DIR:-$ORIG_HERE/logs_slurm_samples}"
STATUS_DIR_DEFAULT="${SLURM_STATUS_DIR:-$ORIG_HERE/.slurm_sample_status}"

log_time() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
    log_time "ERROR: $*" >&2
    exit 2
}

# ---------------------------------------------------------------------------
# Worker mode: each Slurm task runs inside the allocation.
# Args: TASK_ID NUM_TASKS TOTAL CONFIG STATUS_DIR REAL_HERE PYTHON_BIN CONDA_ENV
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--worker" ]]; then
    shift
    if [[ "$#" -ne 8 ]]; then
        echo "Internal error: worker expects TASK_ID NUM_TASKS TOTAL CONFIG STATUS_DIR REAL_HERE PYTHON_BIN CONDA_ENV" >&2
        exit 2
    fi
    TASK_ID="$1"
    NUM_TASKS="$2"
    WORKER_TOTAL="$3"
    WORKER_CONFIG="$4"
    WORKER_STATUS_DIR="$5"
    WORKER_REAL_HERE="$6"
    WORKER_PYTHON="$7"
    WORKER_CONDA_ENV="${8:-}"

    mkdir -p "$WORKER_STATUS_DIR"
    STATUS_FILE="$WORKER_STATUS_DIR/task_${TASK_ID}.exit"
    write_status() {
        rc=$?
        printf '%s\n' "$rc" > "${STATUS_FILE}.tmp.$$"
        mv "${STATUS_FILE}.tmp.$$" "$STATUS_FILE"
        exit "$rc"
    }
    trap write_status EXIT

    export OMP_NUM_THREADS="${CPUS_PER_TASK:-1}"
    export OPENBLAS_NUM_THREADS="${CPUS_PER_TASK:-1}"
    export MKL_NUM_THREADS="${CPUS_PER_TASK:-1}"
    export NUMEXPR_NUM_THREADS="${CPUS_PER_TASK:-1}"
    cd "$WORKER_REAL_HERE"

    if [[ -n "$WORKER_CONDA_ENV" ]]; then
        [[ -f "$HOME/.bashrc" ]] && source "$HOME/.bashrc"
        if ! command -v conda >/dev/null 2>&1; then
            echo "CONDA_ENV was specified but conda is unavailable; initialize conda in ~/.bashrc or leave CONDA_ENV empty." >&2
            exit 127
        fi
        conda activate "$WORKER_CONDA_ENV"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $TASK_ID/$NUM_TASKS starting on $(hostname)"
    echo "Processing $WORKER_TOTAL samples using $WORKER_PYTHON; up to $SAMPLES_PER_JOB samples assigned to this task."
    "$WORKER_PYTHON" "$WORKER_REAL_HERE/run_sample_job.py" \
        --config "$WORKER_CONFIG" \
        --sample-count "$WORKER_TOTAL" \
        --task-id "$TASK_ID" \
        --num-tasks "$NUM_TASKS"
    exit $?
fi

# ---------------------------------------------------------------------------
# Submitter mode
# ---------------------------------------------------------------------------
CONFIG_ARG="${1:-$ORIG_HERE/config/default.json}"
LOG_DIR="${LOG_DIR_DEFAULT}"
STATUS_DIR="${STATUS_DIR_DEFAULT}"
mkdir -p "$LOG_DIR" "$STATUS_DIR"

CONFIG="$(realpath "$CONFIG_ARG")"

[[ "$TOTAL_SAMPLES" =~ ^[1-9][0-9]*$ ]] || fail "TOTAL_SAMPLES must be a positive integer"
[[ "$SAMPLES_PER_JOB" =~ ^[1-9][0-9]*$ ]] || fail "SAMPLES_PER_JOB must be a positive integer"
[[ "$MAX_ACTIVE_JOBS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_ACTIVE_JOBS must be a positive integer"
[[ "$CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]] || fail "CPUS_PER_TASK must be a positive integer"
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "POLL_SECONDS must be a positive integer"
[[ "$MAX_RETRIES" =~ ^[0-9]+$ ]] || fail "MAX_RETRIES must be zero or a positive integer"
[[ -f "$CONFIG" ]] || fail "Config file not found: $CONFIG"
[[ -n "$ACCOUNT" && "$ACCOUNT" != "XXXX" ]] || fail "Set SLURM_ACCOUNT to your valid project/allocation charge code before submitting."
command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available; run this from the cluster login node"
command -v squeue >/dev/null 2>&1 || fail "squeue is not available; run this from the cluster login node"

NUM_JOBS=$(((TOTAL_SAMPLES + SAMPLES_PER_JOB - 1) / SAMPLES_PER_JOB))
if (( NUM_JOBS < MAX_ACTIVE_JOBS )); then
    POOL_SIZE="$NUM_JOBS"
else
    POOL_SIZE="$MAX_ACTIVE_JOBS"
fi

declare -A TASK_FOR_JOB=()
declare -A ATTEMPT_FOR_JOB=()
FAILED_TASKS=()
NEXT_TASK=0

submit_task() {
    local task_id="$1"
    local attempt="$2"
    local status_file="$STATUS_DIR/task_${task_id}.exit"
    local job_name="msh_${task_id}"
    local sbatch_args=(
        --parsable
        --job-name="$job_name"
        --partition="$PARTITION"
        --nodes=1
        --ntasks=1
        --cpus-per-task="$CPUS_PER_TASK"
        --mem="$MEMORY"
        --time="$WALLTIME"
        --account="$ACCOUNT"
        --output="$LOG_DIR/%x_%j.out"
        --error="$LOG_DIR/%x_%j.err"
        --export="ALL,CPUS_PER_TASK=$CPUS_PER_TASK,SAMPLES_PER_JOB=$SAMPLES_PER_JOB"
    )

    rm -f "$status_file"
    local job_id
    if ! job_id="$(sbatch "${sbatch_args[@]}" "$SCRIPT" \
        --worker "$task_id" "$NUM_JOBS" "$TOTAL_SAMPLES" "$CONFIG" \
        "$STATUS_DIR" "$ORIG_HERE" "$PYTHON_BIN" "$CONDA_ENV")"; then
        log_time "sbatch failed for task $task_id (attempt $((attempt + 1)))" >&2
        return 1
    fi
    job_id="${job_id%%;*}"
    TASK_FOR_JOB["$job_id"]="$task_id"
    ATTEMPT_FOR_JOB["$job_id"]="$attempt"
    log_time "Submitted task $task_id/$NUM_JOBS as Slurm job $job_id (attempt $((attempt + 1)); $SAMPLES_PER_JOB samples/job)"
}

fill_pool() {
    while (( NEXT_TASK < NUM_JOBS && ${#TASK_FOR_JOB[@]} < POOL_SIZE )); do
        if ! submit_task "$NEXT_TASK" 0; then
            FAILED_TASKS+=("$NEXT_TASK (sbatch submission failed)")
        fi
        ((NEXT_TASK += 1))
    done
}

log_time "Submitting $TOTAL_SAMPLES samples as $NUM_JOBS jobs ($SAMPLES_PER_JOB samples/job), with at most $POOL_SIZE active jobs."
log_time "Account=$ACCOUNT Partition=$PARTITION CPUs/job=$CPUS_PER_TASK Memory/job=$MEMORY Time/job=$WALLTIME"
log_time "Config=$CONFIG Logs=$LOG_DIR Status=$STATUS_DIR"
fill_pool
while (( ${#TASK_FOR_JOB[@]} > 0 || NEXT_TASK < NUM_JOBS )); do
    for job_id in "${!TASK_FOR_JOB[@]}"; do
        task_id="${TASK_FOR_JOB[$job_id]}"
        attempt="${ATTEMPT_FOR_JOB[$job_id]}"

        if squeue -h -j "$job_id" -o '%A' 2>/dev/null | grep -q .; then
            continue
        fi

        status_file="$STATUS_DIR/task_${task_id}.exit"
        if [[ -f "$status_file" ]]; then
            job_rc="$(cat "$status_file")"
        else
            job_rc=1
        fi
        unset 'TASK_FOR_JOB[$job_id]' 'ATTEMPT_FOR_JOB[$job_id]'

        if [[ "$job_rc" == "0" ]]; then
            log_time "Task $task_id completed successfully."
        elif (( attempt < MAX_RETRIES )); then
            next_attempt=$((attempt + 1))
            log_time "Task $task_id failed (exit $job_rc); retrying ($next_attempt/$MAX_RETRIES)."
            if ! submit_task "$task_id" "$next_attempt"; then
                FAILED_TASKS+=("$task_id (retry submission failed)")
            fi
        else
            log_time "Task $task_id failed (exit $job_rc); retry limit reached."
            FAILED_TASKS+=("$task_id (exit $job_rc)")
        fi
    done

    fill_pool
    if (( ${#TASK_FOR_JOB[@]} > 0 )); then
        sleep "$POLL_SECONDS"
    fi
done

if (( ${#FAILED_TASKS[@]} > 0 )); then
    log_time "Finished with ${#FAILED_TASKS[@]} failed task(s): ${FAILED_TASKS[*]}" >&2
    exit 1
fi
log_time "All $TOTAL_SAMPLES samples completed successfully."