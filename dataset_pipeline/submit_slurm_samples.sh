#!/usr/bin/env bash
# Submit a fixed-size pool of Slurm jobs, refilling slots as jobs finish.
# Run from a login node, not from inside an sbatch allocation.
#
# Each task is submitted exactly once. If a task fails, it is recorded and
# reported at the end; it is not retried.

set -uo pipefail

# ---------------------------------------------------------------------------
# Locate this script. When Slurm runs it as a batch job, BASH_SOURCE points
# into /var/spool/slurmd/..., so worker mode receives REAL_HERE explicitly.
# ---------------------------------------------------------------------------
ORIG_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$ORIG_HERE/$(basename "${BASH_SOURCE[0]}")"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-20}"
SAMPLES_PER_JOB="${SAMPLES_PER_JOB:-10}"
MAX_ACTIVE_JOBS="${MAX_ACTIVE_JOBS:-10}"
PARTITION="${PARTITION:-debug}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${SLURM_MEMORY:-16G}"
WALLTIME="${SLURM_WALLTIME:-02:00:00}"
POLL_SECONDS="${POLL_SECONDS:-30}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV="${CONDA_ENV:-}"
ACCOUNT="${SLURM_ACCOUNT:-FTA-SUB-Ghattas}"
LOG_DIR_DEFAULT="${SLURM_LOG_DIR:-$ORIG_HERE/logs_slurm_samples}"
STATUS_DIR_DEFAULT="${SLURM_STATUS_DIR:-$ORIG_HERE/.slurm_sample_status}"

# Seconds to wait for a freshly submitted job to appear in squeue.
REGISTRATION_TIMEOUT="${REGISTRATION_TIMEOUT:-30}"

SKIP_MAP="${SKIP_MAP:-FALSE}"
export SKIP_MAP

log_time() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
    log_time "ERROR: $*" >&2
    exit 2
}

# ---------------------------------------------------------------------------
# Worker mode
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

    extra_args=()
    if [[ "${SKIP_MAP^^}" == "TRUE" ]]; then
        extra_args+=(--skip_map)
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $TASK_ID/$NUM_TASKS starting on $(hostname)"
    echo "Processing $WORKER_TOTAL samples using $WORKER_PYTHON; up to $SAMPLES_PER_JOB samples assigned to this task."
    echo "SKIP_MAP=$SKIP_MAP (extra_args: ${extra_args[*]:-none})"
    export PYTHONUNBUFFERED=1
    "$WORKER_PYTHON" "$WORKER_REAL_HERE/run_sample_job.py" \
        --config "$WORKER_CONFIG" \
        --sample-count "$WORKER_TOTAL" \
        --task-id "$TASK_ID" \
        --num-tasks "$NUM_TASKS" \
        "${extra_args[@]}"
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

[[ "$TOTAL_SAMPLES"   =~ ^[1-9][0-9]*$ ]] || fail "TOTAL_SAMPLES must be a positive integer"
[[ "$SAMPLES_PER_JOB" =~ ^[1-9][0-9]*$ ]] || fail "SAMPLES_PER_JOB must be a positive integer"
[[ "$MAX_ACTIVE_JOBS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_ACTIVE_JOBS must be a positive integer"
[[ "$CPUS_PER_TASK"   =~ ^[1-9][0-9]*$ ]] || fail "CPUS_PER_TASK must be a positive integer"
[[ "$POLL_SECONDS"    =~ ^[1-9][0-9]*$ ]] || fail "POLL_SECONDS must be a positive integer"
[[ "$REGISTRATION_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail "REGISTRATION_TIMEOUT must be a positive integer"
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

POOL_TAG="mshpool_${USER:-user}_$$"

declare -A TASK_FOR_JOB=()
FAILED_TASKS=()
NEXT_TASK=0

# Count jobs currently in Slurm for our tag.
count_active_jobs() {
    local n
    n="$(squeue -h -u "${USER:-$(id -un)}" -o '%j' 2>/dev/null \
        | grep -c "^${POOL_TAG}_" || true)"
    printf '%s\n' "${n:-0}"
}

# Block until the given job ID is visible to squeue (or has a sacct record),
# or timeout. Returns 0 if visible/known, 1 on timeout.
wait_for_job_registration() {
    local job_id="$1"
    local waited=0
    while (( waited < REGISTRATION_TIMEOUT )); do
        if squeue -h -j "$job_id" -o '%A' 2>/dev/null | grep -q .; then
            return 0
        fi
        if sacct -j "$job_id" --format=JobID --noheader --parsable2 2>/dev/null \
                | grep -q .; then
            return 0
        fi
        sleep 1
        ((waited += 1))
    done
    return 1
}

# Parse a Slurm job ID out of sbatch output that may contain a welcome banner.
parse_job_id() {
    grep -oE '^[0-9]+(;[^ ]*)?$' | head -1 | cut -d';' -f1
}

submit_task() {
    local task_id="$1"
    local status_file="$STATUS_DIR/task_${task_id}.exit"
    local job_name="${POOL_TAG}_${task_id}"
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
        --export="ALL,CPUS_PER_TASK=$CPUS_PER_TASK,SAMPLES_PER_JOB=$SAMPLES_PER_JOB,SKIP_MAP=$SKIP_MAP"
    )

    rm -f "$status_file"

    local raw job_id
    if ! raw="$(sbatch "${sbatch_args[@]}" "$SCRIPT" \
        --worker "$task_id" "$NUM_JOBS" "$TOTAL_SAMPLES" "$CONFIG" \
        "$STATUS_DIR" "$ORIG_HERE" "$PYTHON_BIN" "$CONDA_ENV" 2>&1)"; then
        log_time "sbatch failed for task $task_id" >&2
        return 1
    fi

    job_id="$(printf '%s\n' "$raw" | parse_job_id)"
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        log_time "Could not parse a job ID for task $task_id from sbatch output:" >&2
        printf '%s\n' "$raw" >&2
        return 1
    fi

    TASK_FOR_JOB["$job_id"]="$task_id"
    log_time "Submitted task $task_id/$NUM_JOBS as Slurm job $job_id ($SAMPLES_PER_JOB samples/job)"

    if ! wait_for_job_registration "$job_id"; then
        log_time "WARNING: job $job_id (task $task_id) not visible to squeue after ${REGISTRATION_TIMEOUT}s" >&2
    fi
    return 0
}

fill_pool() {
    while (( NEXT_TASK < NUM_JOBS )); do
        local active
        active="$(count_active_jobs)"
        if (( active >= POOL_SIZE )); then
            break
        fi
        if ! submit_task "$NEXT_TASK"; then
            FAILED_TASKS+=("$NEXT_TASK (sbatch submission failed)")
        fi
        ((NEXT_TASK += 1))
        sleep 1
    done
}

# Ask Slurm what happened to a finished job. Populates two globals:
#   RESOLVED_RC      exit code (0 = success)
#   RESOLVED_REASON  human-readable explanation
# Prefers the per-task status file; falls back to sacct.
RESOLVED_RC=""
RESOLVED_REASON=""
resolve_job_result() {
    local job_id="$1"
    local task_id="$2"
    local status_file="$STATUS_DIR/task_${task_id}.exit"

    if [[ -f "$status_file" ]]; then
        RESOLVED_RC="$(cat "$status_file")"
        RESOLVED_REASON="status file exit $RESOLVED_RC"
        return
    fi

    # No status file; give sacct a moment, then query.
    sleep 2
    local slurm_state slurm_exit slurm_code
    slurm_state="$(sacct -j "$job_id" --format=State --noheader --parsable2 2>/dev/null \
                    | head -1 | tr -d ' ')"
    slurm_exit="$(sacct -j "$job_id" --format=ExitCode --noheader --parsable2 2>/dev/null \
                    | head -1 | tr -d ' ')"
    slurm_code="${slurm_exit%%:*}"

    if [[ "$slurm_state" == "COMPLETED" && "$slurm_code" == "0" ]]; then
        RESOLVED_RC=0
        RESOLVED_REASON="Slurm COMPLETED 0:0 (no status file)"
    else
        RESOLVED_RC="${slurm_code:-1}"
        RESOLVED_REASON="no status file; Slurm State=${slurm_state:-UNKNOWN} ExitCode=${slurm_exit:-?}"
    fi
}

log_time "Submitting $TOTAL_SAMPLES samples as $NUM_JOBS jobs ($SAMPLES_PER_JOB samples/job), with at most $POOL_SIZE active jobs."
log_time "Account=$ACCOUNT Partition=$PARTITION CPUs/job=$CPUS_PER_TASK Memory/job=$MEMORY Time/job=$WALLTIME"
log_time "Config=$CONFIG Logs=$LOG_DIR Status=$STATUS_DIR"
log_time "Pool tag: $POOL_TAG  SKIP_MAP=$SKIP_MAP"

fill_pool

while (( ${#TASK_FOR_JOB[@]} > 0 || NEXT_TASK < NUM_JOBS )); do
    for job_id in "${!TASK_FOR_JOB[@]}"; do
        task_id="${TASK_FOR_JOB[$job_id]}"

        if squeue -h -j "$job_id" -o '%A' 2>/dev/null | grep -q .; then
            continue
        fi

        resolve_job_result "$job_id" "$task_id"
        job_rc="$RESOLVED_RC"
        reason="$RESOLVED_REASON"
        unset 'TASK_FOR_JOB[$job_id]'

        if [[ "$job_rc" == "0" ]]; then
            log_time "Task $task_id completed successfully ($reason)."
        else
            log_time "Task $task_id failed (exit $job_rc; $reason)."
            FAILED_TASKS+=("$task_id (exit $job_rc)")
        fi
    done

    fill_pool

    if (( ${#TASK_FOR_JOB[@]} > 0 || NEXT_TASK < NUM_JOBS )); then
        sleep "$POLL_SECONDS"
    fi
done

if (( ${#FAILED_TASKS[@]} > 0 )); then
    log_time "Finished with ${#FAILED_TASKS[@]} failed task(s): ${FAILED_TASKS[*]}" >&2
    exit 1
fi
log_time "All $TOTAL_SAMPLES samples completed successfully."