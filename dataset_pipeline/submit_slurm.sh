#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$HERE/config/default.json}"
TASKS="${2:-1}"
SAMPLE_COUNT="${3:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPUS="${CPUS_PER_TASK:-1}"
MEMORY="${SLURM_MEMORY:-16G}"
WALLTIME="${SLURM_WALLTIME:-24:00:00}"
LOG_DIR="${SLURM_LOG_DIR:-$HERE/logs}"
mkdir -p "$LOG_DIR"

if [[ "$TASKS" -lt 1 ]]; then
    echo "TASKS must be a positive integer" >&2
    exit 2
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    EXTRA_EXPORT=""
    if [[ -n "$SAMPLE_COUNT" ]]; then
        EXTRA_EXPORT=",PIPELINE_SAMPLE_COUNT=$SAMPLE_COUNT"
    fi
    echo "Submitting $TASKS Slurm array tasks using $CONFIG"
    sbatch \
        --array="0-$((TASKS - 1))" \
        --cpus-per-task="$CPUS" \
        --mem="$MEMORY" \
        --time="$WALLTIME" \
        --output="$LOG_DIR/%A_%a.out" \
        --export="ALL,PIPELINE_CONFIG=$CONFIG,PIPELINE_TASKS=$TASKS,PIPELINE_PYTHON=$PYTHON_BIN$EXTRA_EXPORT" \
        "$0"
    exit 0
fi

TASK_ID="$SLURM_ARRAY_TASK_ID"
CONFIG="${PIPELINE_CONFIG:-$CONFIG}"
TASKS="${PIPELINE_TASKS:-$TASKS}"
PYTHON_BIN="${PIPELINE_PYTHON:-$PYTHON_BIN}"
ARGS=(--config "$CONFIG" --task-id "$TASK_ID" --num-tasks "$TASKS")
if [[ -n "${PIPELINE_SAMPLE_COUNT:-}" ]]; then
    ARGS+=(--sample-count "$PIPELINE_SAMPLE_COUNT")
fi
"$PYTHON_BIN" "$HERE/run_sample_job.py" "${ARGS[@]}"
