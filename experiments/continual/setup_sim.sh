#!/usr/bin/env bash
# Run from the repository root. All mutable installation output stays on the data disk.
set -euo pipefail
ROOT=${OPENPI_CL_STORAGE_ROOT:-/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cl_runtime}
SIM="$ROOT/envs/libero-py310"
INDEX_URL=${SIM_INDEX_URL:-https://pypi.org/simple}
export TMPDIR="$ROOT/tmp" TMP="$ROOT/tmp" TEMP="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/cache/pip"
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -B -c 'import os; from pathlib import Path; from experiments.continual.storage import configure; configure(Path(os.environ["TMPDIR"]).parent, minimum_gib=10)'
mkdir -p "$ROOT/envs" "$PIP_CACHE_DIR"
if [ ! -x "$SIM/bin/python" ]; then
    /usr/bin/python3.10 -m venv "$SIM"
fi
# The inherited proxy on this server rejects package downloads; use direct HTTPS.
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy
"$SIM/bin/python" -m pip install --index-url "$INDEX_URL" --timeout 30 --retries 2 'pip==25.0.1' wheel
"$SIM/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu --timeout 30 --retries 2 'torch==1.11.0+cpu'
"$SIM/bin/python" -m pip install --index-url "$INDEX_URL" --timeout 30 --retries 2 -r experiments/continual/sim_requirements.txt
"$SIM/bin/python" -m pip install --index-url "$INDEX_URL" --no-deps -e third_party/libero -e packages/openpi-client
"$SIM/bin/python" -m pip check
"$SIM/bin/python" -m pip freeze > "$ROOT/reports/libero_sim_freeze.txt"
"$SIM/bin/python" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['TMPDIR']).parent
benchmark = Path('third_party/libero/libero/libero').resolve()
config_dir = root / 'cache/libero'
config_dir.mkdir(parents=True, exist_ok=True)
config = {'benchmark_root': str(benchmark), 'bddl_files': str(benchmark / 'bddl_files'),
          'init_states': str(benchmark / 'init_files'), 'assets': str(benchmark / 'assets'),
          'datasets': '/mnt/disk20t/xieyuxuan/denghaoxindatadisk/libero_data'}
path = config_dir / 'config.yaml'
if path.exists() and json.loads(path.read_text()) != config:
    raise RuntimeError(f'Existing LIBERO config differs: {path}')
path.write_text(json.dumps(config, indent=2) + '\n')
link = Path('examples/libero/.venv')
target = root / 'envs/libero-py310'
if not link.exists() and not link.is_symlink():
    link.symlink_to(target, target_is_directory=True)
elif link.resolve() != target.resolve():
    raise RuntimeError('Existing simulation environment points to another location')
print(f'Simulation Python: {target / "bin/python"}')
PY
