#!/usr/bin/env bash
# Engineering smoke test only: three tuning initial states, not benchmark results.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
PY="$REPO/.venv/bin/python"
RUNTIME=${OPENPI_CL_STORAGE_ROOT:-/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cl_runtime}
CKPT="$RUNTIME/checkpoints/smoke_t1_gpu0_20261029_094234/cl_pi05_continual_v2/T1/99"
SIM="$RUNTIME/envs/libero-py310/bin/python"
NORM="$RUNTIME/reports/per_task_norm_v1"
TAG="smoke_$(date +%Y%m%d_%H%M%S)_$$"
LABEL="smoke_t1_gpu0_20261029_094234_T1_offset0_n3_$TAG"
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYNPUT_BACKEND=dummy

"$PY" -B -m experiments.continual.storage --storage-root "$RUNTIME" > /dev/null
for required in "$CKPT/_CHECKPOINT_METADATA" "$SIM" "$NORM/T1/norm_stats.json"; do
    [[ -e "$required" ]] || { echo "Missing: $required" >&2; exit 1; }
done
exec 9>"$RUNTIME/logs/eval_smoke_gpu2.lock"
flock -n 9 || { echo "Evaluation smoke test already running." >&2; exit 1; }
LOG="$RUNTIME/logs/eval_gpu2_$TAG.log"
ln -sfn -- "$LOG" "$RUNTIME/logs/eval_smoke_gpu2.latest.log"
echo "GPU: $CUDA_VISIBLE_DEVICES; checkpoint: $CKPT"
echo "Main log: $LOG"
echo "Policy server log: $RUNTIME/logs/${LABEL}_T1_server.log"
echo "Results: $RUNTIME/reports/$LABEL/T1.jsonl"
echo "Videos: $RUNTIME/videos/$LABEL/T1"
echo "Using tuning initial states 0..2. This is not a formal success-rate evaluation."
"$PY" -B -m experiments.continual.storage --storage-root "$RUNTIME" -- \
    "$PY" -B -m experiments.continual.evaluate \
    --checkpoint "$CKPT" --norm-assets "$NORM" --storage-root "$RUNTIME" \
    --sim-python "$SIM" --port "${EVAL_PORT:-8002}" \
    --trials 3 --initial-state-offset 0 --eval-seed 7 --videos-per-task 3 \
    --run-tag "$TAG" 2>&1 | tee "$LOG"
echo "Smoke test completed. Inspect the JSONL and videos listed above."
