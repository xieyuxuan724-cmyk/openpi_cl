#!/usr/bin/env bash
# Commit only the reviewed 2026-10-29 continual-learning changes, then push.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
BRANCH=codex/continual-pi05-protocol
REMOTE=https://github.com/xieyuxuan724-cmyk/openpi_cl.git
[[ "$(git branch --show-current)" == "$BRANCH" ]] || {
    echo "Expected branch: $BRANCH; stop to inspect the current branch." >&2; exit 1;
}
[[ "$(git remote get-url origin)" == "$REMOTE" ]] || {
    echo "origin differs from $REMOTE; stop to inspect the remote." >&2; exit 1;
}
git diff --cached --quiet || {
    echo "Other changes are already staged. Inspect them before running this script." >&2; exit 1;
}
FILES=(
    experiments/continual/analyze.py
    experiments/continual/evaluate.py
    experiments/continual/evaluate_test.py
    experiments/continual/protocol.py
    experiments/continual/run.py
    experiments/continual/continue_branch.py
    experiments/continual/continue_branch_test.py
    experiments/continual/shared_baseline.py
    experiments/continual/t1_nohup.sh
    experiments/continual/eval_smoke_gpu2.sh
    experiments/continual/continue_two_gpus.sh
    experiments/continual/CONTINUE_TWO_GPUS.md
    experiments/continual/CHANGES_2026-10-29.md
    experiments/continual/push_20261029.sh
)
for path in "${FILES[@]}"; do
    [[ -f "$path" ]] || { echo "Missing expected file: $path" >&2; exit 1; }
done
git diff --check -- "${FILES[@]}"
git add -- "${FILES[@]}"
git diff --cached --check
git diff --cached --stat
if ! git diff --cached --quiet; then
    git commit -m "Fix LIBERO evaluation and continue paired pi05 experiments on two GPUs"
else
    echo "Today's selected files are already committed; continuing to push."
fi
git push -u origin "$BRANCH"
echo "Push complete: $BRANCH"
git status --short
