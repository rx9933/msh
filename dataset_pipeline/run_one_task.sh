#!/usr/bin/env bash
# Usage: run_one_task.sh TASK_ID NUM_TASKS TOTAL CONFIG STATUS_DIR
set -uo pipefail

TASK_ID="$1"
NUM_TASKS="$2"
TOTAL="$3"
CONFIG="$4"
STATUS_DIR="$5"

HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$STATUS_DIR"
STATUS_FILE="$STATUS_DIR/task_${TASK_ID}.exit"
trap 'rc=$?; echo "$rc" > "$STATUS_FILE"; exit $rc' EXIT

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# ---- conda setup --------------------------------------------------------
CONDA_BASE="/scratch/09579/a2097855"

if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "conda.sh not found under $CONDA_BASE; check CONDA_BASE" >&2
    exit 127
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate torchfem

echo "[$(date '+%F %T')] using python: $(command -v python3)"
python3 -c "import sys; print('  ->', sys.executable)"
# -----------------------------------------------------------------------------
# ------------------------------------------------------------------------
export PYTHONPATH="/scratch/09579/a2097855/hippylib:${PYTHONPATH:-}"
export PYTHONPATH="/scratch/09579/a2097855/hippyflow:${PYTHONPATH:-}"
cd "$HERE"

echo "[$(date '+%F %T')] Task $TASK_ID/$NUM_TASKS starting on $(hostname)"
python3 "$HERE/run_sample_job.py" \
    --config "$CONFIG" \
    --sample-count "$TOTAL" \
    --task-id "$TASK_ID" \
    --num-tasks "$NUM_TASKS"