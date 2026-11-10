#!/usr/bin/env bash
# Usage: SLURM_ACCOUNT=XXXX ./submit_all.sh [CONFIG]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-10000}"
SAMPLES_PER_JOB="${SAMPLES_PER_JOB:-10}"
MAX_ACTIVE_JOBS="${MAX_ACTIVE_JOBS:-10}"
PARTITION="${PARTITION:-debug}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${SLURM_MEMORY:-16G}"
WALLTIME="${SLURM_WALLTIME:-48:00:00}"
ACCOUNT="${SLURM_ACCOUNT:-FTA-SUB-Ghattas}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_RETRIES="${MAX_RETRIES:-2}"

CONFIG="$(realpath "${1:-$HERE/config/default.json}")"
LOG_DIR="$HERE/logs_slurm_samples"
STATUS_DIR="$HERE/.slurm_sample_status"
mkdir -p "$LOG_DIR" "$STATUS_DIR"

[[ "$ACCOUNT" == "XXXX" ]] && { echo "Set SLURM_ACCOUNT"; exit 2; }
[[ -f "$CONFIG" ]]         || { echo "No config: $CONFIG"; exit 2; }

NUM_JOBS=$(( (TOTAL_SAMPLES + SAMPLES_PER_JOB - 1) / SAMPLES_PER_JOB ))
POOL_SIZE=$(( NUM_JOBS < MAX_ACTIVE_JOBS ? NUM_JOBS : MAX_ACTIVE_JOBS ))

declare -A JOB_OF_TASK=()     # task_id -> job_id
declare -A ATTEMPT_OF_TASK=() # task_id -> attempt
FAILED=()
NEXT_TASK=0
SUBMITTED=0

submit_task() {
    local task_id="$1" attempt="$2"
    local status_file="$STATUS_DIR/task_${task_id}.exit"
    rm -f "$status_file"

    local job_id
    job_id=$(sbatch --parsable \
        --job-name="msh_${task_id}" \
        --partition="$PARTITION" \
        --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEMORY" --time="$WALLTIME" --account="$ACCOUNT" \
        --output="$LOG_DIR/msh_${task_id}_%j.out" \
        --error="$LOG_DIR/msh_${task_id}_%j.err" \
        --wrap "exec bash '$HERE/run_one_task.sh' '$task_id' '$NUM_JOBS' '$TOTAL_SAMPLES' '$CONFIG' '$STATUS_DIR'") || return 1

    job_id="${job_id%%;*}"
    JOB_OF_TASK["$task_id"]="$job_id"
    ATTEMPT_OF_TASK["$task_id"]="$attempt"
    SUBMITTED=$((SUBMITTED+1))
    echo "[$(date '+%F %T')] Submitted task $task_id/$NUM_JOBS as job $job_id (attempt $((attempt+1)))"
}

fill_pool() {
    while (( NEXT_TASK < NUM_JOBS && ${#JOB_OF_TASK[@]} < POOL_SIZE )); do
        submit_task "$NEXT_TASK" 0 || FAILED+=("$NEXT_TASK (sbatch failed)")
        NEXT_TASK=$((NEXT_TASK+1))
    done
}

echo "[$(date '+%F %T')] $TOTAL_SAMPLES samples, $NUM_JOBS jobs, pool=$POOL_SIZE"
fill_pool

while (( ${#JOB_OF_TASK[@]} > 0 || NEXT_TASK < NUM_JOBS )); do
    for task_id in "${!JOB_OF_TASK[@]}"; do
        job_id="${JOB_OF_TASK[$task_id]}"
        attempt="${ATTEMPT_OF_TASK[$task_id]}"

        # still queued/running?
        if squeue -h -j "$job_id" -o '%A' 2>/dev/null | grep -q .; then
            continue
        fi

        status_file="$STATUS_DIR/task_${task_id}.exit"
        rc="$(cat "$status_file" 2>/dev/null || echo 1)"
        unset 'JOB_OF_TASK[$task_id]' 'ATTEMPT_OF_TASK[$task_id]'

        if [[ "$rc" == "0" ]]; then
            echo "[$(date '+%F %T')] Task $task_id OK"
        elif (( attempt < MAX_RETRIES )); then
            echo "[$(date '+%F %T')] Task $task_id failed (rc=$rc), retry $((attempt+1))/$MAX_RETRIES"
            submit_task "$task_id" "$((attempt+1))" || FAILED+=("$task_id (retry sbatch failed)")
        else
            echo "[$(date '+%F %T')] Task $task_id failed (rc=$rc), giving up"
            FAILED+=("$task_id (rc=$rc)")
        fi
    done

    fill_pool
    (( ${#JOB_OF_TASK[@]} > 0 )) && sleep "$POLL_SECONDS"
done

if (( ${#FAILED[@]} > 0 )); then
    echo "Finished with ${#FAILED[@]} failures: ${FAILED[*]}" >&2
    exit 1
fi
echo "[$(date '+%F %T')] All $TOTAL_SAMPLES samples completed."