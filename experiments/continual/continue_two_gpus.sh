#!/usr/bin/env bash
# Two independent continual-learning branches; each task remains sequential within its branch.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
PY="$REPO/.venv/bin/python"
RUNTIME=${OPENPI_CL_STORAGE_ROOT:-/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cl_runtime}
export OPENPI_CL_STORAGE_ROOT="$RUNTIME"
export SEQUENTIAL_GPU=${SEQUENTIAL_GPU:-0} REPLAY_GPU=${REPLAY_GPU:-2}
export SEQUENTIAL_PORT=${SEQUENTIAL_PORT:-8000} REPLAY_PORT=${REPLAY_PORT:-8002}
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

if [[ "${1:-start}" == "--supervisor" ]]; then
    : "${PAIR_LOG:?Launch using start}"
    seq_log="$RUNTIME/logs/continual_sequential_seed42_${RUN_TAG}.log"
    replay_log="$RUNTIME/logs/continual_replay20_seed42_${RUN_TAG}.log"
    ln -sfn -- "$seq_log" "$RUNTIME/logs/continual_sequential_seed42.latest.log"
    ln -sfn -- "$replay_log" "$RUNTIME/logs/continual_replay20_seed42.latest.log"
    CUDA_VISIBLE_DEVICES="$SEQUENTIAL_GPU" JAX_PLATFORMS=cuda \
        "$PY" -B -m experiments.continual.continue_branch --method sequential \
        --storage-root "$RUNTIME" --port "$SEQUENTIAL_PORT" > "$seq_log" 2>&1 &
    seq_pid=$!
    printf '%s\n' "$seq_pid" > "$seq_log.pid"
    CUDA_VISIBLE_DEVICES="$REPLAY_GPU" JAX_PLATFORMS=cuda \
        "$PY" -B -m experiments.continual.continue_branch --method replay \
        --storage-root "$RUNTIME" --port "$REPLAY_PORT" > "$replay_log" 2>&1 &
    replay_pid=$!
    printf '%s\n' "$replay_pid" > "$replay_log.pid"
    echo "Sequential: GPU=$SEQUENTIAL_GPU PID=$seq_pid LOG=$seq_log"
    echo "Replay20: GPU=$REPLAY_GPU PID=$replay_pid LOG=$replay_log"
    # A failure in one branch does not discard useful progress in the other.
    seq_code=0; wait "$seq_pid" || seq_code=$?
    replay_code=0; wait "$replay_pid" || replay_code=$?
    printf '%s\n' "$seq_code" > "$seq_log.exit"
    printf '%s\n' "$replay_code" > "$replay_log.exit"
    echo "Exit status: sequential=$seq_code replay20=$replay_code"
    if (( seq_code != 0 || replay_code != 0 )); then
        printf '1\n' > "$PAIR_LOG.exit"
        exit 1
    fi
    comparison="$RUNTIME/reports/paired_seed42_4tasks_${RUN_TAG}.json"
    if "$PY" -B -m experiments.continual.compare \
        --storage-root "$RUNTIME" \
        --sequential "$RUNTIME/reports/v2_sequential_seed42_metrics.json" \
        --replay "$RUNTIME/reports/v2_replay20_seed42_metrics.json" \
        --output "$comparison"; then
        printf '0\n' > "$PAIR_LOG.exit"
        echo "ALL COMPLETE: $comparison"
    else
        printf '1\n' > "$PAIR_LOG.exit"
        exit 1
    fi
    exit 0
fi
if [[ "${1:-start}" != "start" && "${1:-start}" != "check" ]] || (( $# > 1 )); then
    echo "Usage: bash experiments/continual/continue_two_gpus.sh [start|check]" >&2
    exit 2
fi
[[ "$SEQUENTIAL_GPU" != "$REPLAY_GPU" && "$SEQUENTIAL_PORT" != "$REPLAY_PORT" ]] || {
    echo "The two branches require different GPUs and ports." >&2; exit 1;
}
"$PY" -B -m experiments.continual.storage --storage-root "$RUNTIME" > /dev/null
# Locks stay held by the detached supervisor until both branches exit.
exec 8>"$RUNTIME/logs/continual_pair_seed42.lock"
flock -n 8 || { echo "This two-GPU experiment is already running." >&2; exit 1; }
exec 9>"$RUNTIME/logs/t1_sequential_seed42.lock"
flock -n 9 || { echo "The original T1 job is still running." >&2; exit 1; }
for method in sequential replay; do
    JAX_PLATFORMS=cpu "$PY" -B -m experiments.continual.storage --storage-root "$RUNTIME" -- \
        "$PY" -B -m experiments.continual.continue_branch --method "$method" \
        --storage-root "$RUNTIME" --port 8000 --check
done
# Check GPU visibility and port availability without allocating GPU memory.
for gpu in "$SEQUENTIAL_GPU" "$REPLAY_GPU"; do
    nvidia-smi -i "$gpu" --query-gpu=index,memory.free,utilization.gpu --format=csv
done
"$PY" -B - <<'PY'
import os
import socket
for name in ("SEQUENTIAL_PORT", "REPLAY_PORT"):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", int(os.environ[name])))
print("Both evaluation ports are currently available.")
PY
[[ "${1:-start}" == "check" ]] && exit 0
export RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
export PAIR_LOG="$RUNTIME/logs/continual_pair_seed42_${RUN_TAG}.log"
nohup "$PY" -B -m experiments.continual.storage --storage-root "$RUNTIME" -- \
    bash "$REPO/experiments/continual/continue_two_gpus.sh" --supervisor \
    > "$PAIR_LOG" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$PAIR_LOG.pid"
ln -sfn -- "$PAIR_LOG" "$RUNTIME/logs/continual_pair_seed42.latest.log"
echo "Started supervisor PID $(cat "$PAIR_LOG.pid"). SSH can now be disconnected."
echo "Sequential log: $RUNTIME/logs/continual_sequential_seed42.latest.log"
echo "Replay20 log: $RUNTIME/logs/continual_replay20_seed42.latest.log"
echo "Pair status log: $RUNTIME/logs/continual_pair_seed42.latest.log"
