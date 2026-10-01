#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$HERE/config/default.json}"
TASKS="${2:-1}"
PARALLEL="${3:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$TASKS" -lt 1 || "$PARALLEL" -lt 1 ]]; then
    echo "TASKS and PARALLEL must be positive integers" >&2
    exit 2
fi

seq 0 "$((TASKS - 1))" | xargs -r -n 1 -P "$PARALLEL" -I '{}' \
    "$PYTHON_BIN" "$HERE/run_sample_job.py" \
    --config "$CONFIG" --task-id '{}' --num-tasks "$TASKS"
