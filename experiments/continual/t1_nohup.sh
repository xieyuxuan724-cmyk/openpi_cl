#!/usr/bin/env bash
# Launch formal T1 training followed by LIBERO evaluation on the data disk.
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
PY="$REPO/.venv/bin/python"
RUNTIME=${OPENPI_CL_STORAGE_ROOT:-/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cl_runtime}
DATA=/mnt/disk20t/xieyuxuan/denghaoxindatadisk/libero_data
BASE=/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cache_test/pi05_base_params/params
NORM="$RUNTIME/reports/per_task_norm_v1"
OUT="$RUNTIME/checkpoints/v2_sequential_seed42"
SIM="$RUNTIME/envs/libero-py310/bin/python"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OPENPI_CL_STORAGE_ROOT="$RUNTIME"

if [[ "${1:-start}" == "--worker" ]]; then
    : "${RUN_LOG:?Launch through the start command}"
    phase=train
    finish() {
        code=$?
        printf '%s phase=%s exit_code=%s\n' "$(date -Is)" "$phase" "$code"
        printf '%s\n' "$code" > "$RUN_LOG.exit"
    }
    trap finish EXIT
    echo "[$(date -Is)] TRAIN: GPU=$CUDA_VISIBLE_DEVICES, output=$OUT"
    "$PY" -B -m experiments.continual.run train \
        --stage 1 --method sequential --seed 42 \
        --dataset-root "$DATA" --base-params "$BASE" --norm-assets "$NORM" \
        --storage-root "$RUNTIME" --output "$OUT" \
        --batch-size 8 --steps 10000 --warmup 500 \
        --unfreeze-vision --normalization quantile
    phase=evaluate
    echo "[$(date -Is)] EVALUATE: T1, 40 episodes, initial states 10..49"
    "$PY" -B -m experiments.continual.evaluate \
        --checkpoint "$OUT/cl_pi05_continual_v2/T1/9999" \
        --norm-assets "$NORM" --storage-root "$RUNTIME" \
        --sim-python "$SIM" --port "${EVAL_PORT:-8000}" \
        --trials 40 --initial-state-offset 10 --eval-seed 7 --videos-per-task 3
    phase=complete
    exit 0
fi

if [[ "${1:-start}" != "start" || $# -gt 1 ]]; then
    echo "Usage: GPU=0 bash experiments/continual/t1_nohup.sh [start]" >&2
    exit 2
fi
# Validate storage before creating logs, locks or checkpoints.
"$PY" -B -m experiments.continual.storage --storage-root "$RUNTIME" > /dev/null
for required in "$DATA" "$BASE" "$NORM/T1/norm_stats.json" "$SIM"; do
    [[ -e "$required" ]] || { echo "Missing: $required" >&2; exit 1; }
done
exec 9>"$RUNTIME/logs/t1_sequential_seed42.lock"
flock -n 9 || { echo "T1 job already running; inspect its log." >&2; exit 1; }
if [[ -e "$OUT/cl_pi05_continual_v2/T1" ]]; then
    echo "T1 output already exists: $OUT. Inspect it before resuming or evaluating." >&2
    exit 1
fi
export RUN_LOG="$RUNTIME/logs/t1_sequential_seed42_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup "$PY" -B -m experiments.continual.storage --storage-root "$RUNTIME" -- \
    bash "$REPO/experiments/continual/t1_nohup.sh" --worker \
    > "$RUN_LOG" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$RUN_LOG.pid"
ln -sfn -- "$RUN_LOG" "$RUNTIME/logs/t1_sequential_seed42.latest.log"
echo "Started supervisor PID: $pid"
echo "Log: $RUN_LOG"
echo "PID file: $RUN_LOG.pid"
echo "Exit status after completion: $RUN_LOG.exit (0 means train + evaluation succeeded)"
echo "Follow: tail -n 50 -f $RUNTIME/logs/t1_sequential_seed42.latest.log"
