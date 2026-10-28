"""Data-disk storage bootstrap; uses only the Python standard library."""

import argparse
from datetime import datetime
from datetime import timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

UTC = timezone.utc  # noqa: UP017 -- shared with the Python 3.10 simulator.

DEFAULT_ROOT = Path("/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cl_runtime")


def check_space(path: Path, minimum_gib: float) -> float:
    if minimum_gib < 0:
        raise ValueError("Minimum free space must be nonnegative")
    free = shutil.disk_usage(path).free / 2**30
    if free < minimum_gib:
        raise OSError(f"Insufficient disk space at {path}: {free:.1f} GiB available; need {minimum_gib:.1f} GiB")
    return free


def configure(
    root: Path, output: Path | None = None, *, minimum_gib: float = 100, require_data_disk: bool = True
) -> dict:
    """Validate resolved paths before creating directories or importing ML libraries."""
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("Storage root must be absolute")
    root = root.resolve()
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    if require_data_disk and ancestor.stat().st_dev == Path("/").stat().st_dev:
        raise ValueError(f"Storage root is on the system filesystem (or data disk is not mounted): {root}")
    free = check_space(ancestor, minimum_gib)
    output = output.expanduser().resolve() if output is not None else root / "checkpoints"
    if not output.is_relative_to(root / "checkpoints"):
        raise ValueError(f"Checkpoint output must be under {root / 'checkpoints'}")
    directories = {
        "tmp": root / "tmp",
        "cache": root / "cache",
        "logs": root / "logs",
        "checkpoints": root / "checkpoints",
        "reports": root / "reports",
        "videos": root / "videos",
        "hf": root / "cache/huggingface",
        "datasets": root / "cache/huggingface/datasets",
        "hub": root / "cache/huggingface/hub",
        "xet": root / "cache/huggingface/xet",
        "hf_assets": root / "cache/huggingface/assets",
        "hf_modules": root / "cache/huggingface/modules",
        "inductor": root / "cache/torch/inductor",
        "wandb_data": root / "logs/wandb/data",
        "openpi": root / "cache/openpi",
        "jax": root / "cache/jax",
        "torch": root / "cache/torch",
        "triton": root / "cache/triton",
        "cuda": root / "cache/cuda",
        "numba": root / "cache/numba",
        "matplotlib": root / "cache/matplotlib",
        "libero": root / "cache/libero",
        "wandb": root / "logs/wandb",
        "wandb_cache": root / "cache/wandb",
        "wandb_config": root / "cache/wandb-config",
    }
    # Reject symlinks that escape the managed tree or lead onto a different disk.
    for path in [*directories.values(), output]:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Storage path escapes managed root: {path}")
        existing = resolved
        while not existing.exists():
            existing = existing.parent
        if existing.stat().st_dev != ancestor.stat().st_dev:
            raise ValueError(f"Storage path uses a different filesystem: {path}")
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    # Keep the experiment output absent: the trainer uses its existence to detect reruns.
    for key in ("tmp", "cache", "logs", "checkpoints"):
        with tempfile.TemporaryFile(dir=directories[key]):
            pass
    old_openpi = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    env = {
        "OPENPI_CL_STORAGE_ROOT": str(root),
        "TMPDIR": str(directories["tmp"]),
        "TMP": str(directories["tmp"]),
        "TEMP": str(directories["tmp"]),
        "XDG_CACHE_HOME": str(directories["cache"]),
        "HF_HOME": str(directories["hf"]),
        "HF_DATASETS_CACHE": str(directories["datasets"]),
        "HF_HUB_CACHE": str(directories["hub"]),
        "HUGGINGFACE_HUB_CACHE": str(directories["hub"]),
        "HF_XET_CACHE": str(directories["xet"]),
        "HF_ASSETS_CACHE": str(directories["hf_assets"]),
        "HF_MODULES_CACHE": str(directories["hf_modules"]),
        "OPENPI_DATA_HOME": str(directories["openpi"]),
        "JAX_COMPILATION_CACHE_DIR": str(directories["jax"]),
        "TORCH_HOME": str(directories["torch"]),
        "TORCHINDUCTOR_CACHE_DIR": str(directories["inductor"]),
        "TRITON_CACHE_DIR": str(directories["triton"]),
        "CUDA_CACHE_PATH": str(directories["cuda"]),
        "NUMBA_CACHE_DIR": str(directories["numba"]),
        "MPLCONFIGDIR": str(directories["matplotlib"]),
        "LIBERO_CONFIG_PATH": str(directories["libero"]),
        "WANDB_DIR": str(directories["wandb"]),
        "WANDB_CACHE_DIR": str(directories["wandb_cache"]),
        "WANDB_CONFIG_DIR": str(directories["wandb_config"]),
        "WANDB_DATA_DIR": str(directories["wandb_data"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OPENPI_CHECKPOINT_MIN_FREE_GIB": "20",
    }
    # Seed only the small public tokenizer asset; never move/delete old caches or copy credentials.
    relative = Path("big_vision/paligemma_tokenizer.model")
    source, target = old_openpi / relative, directories["openpi"] / relative
    if not target.resolve().is_relative_to(root):
        raise ValueError("Tokenizer cache path escapes managed root")
    if source.is_file() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as dst, source.open("rb") as src:
                shutil.copyfileobj(src, dst)
        except FileExistsError:
            pass
    os.environ.update(env)
    tempfile.tempdir = str(directories["tmp"])
    sys.dont_write_bytecode = True
    return {"root": str(root), "output": str(output), "free_gib_at_start": free, "environment": env}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root", type=Path, default=Path(os.environ.get("OPENPI_CL_STORAGE_ROOT", DEFAULT_ROOT))
    )
    parser.add_argument("--minimum-free-gib", type=float, default=100)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    storage = configure(args.storage_root, minimum_gib=args.minimum_free_gib)
    if not command:
        print(json.dumps(storage, indent=2))
        return
    name = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    log_path = Path(storage["root"]) / "logs" / f"{name}.log"
    record = {"command": command, "cwd": str(Path.cwd()), "storage": storage}
    record_path = log_path.with_suffix(".json")
    record_path.write_text(json.dumps(record, indent=2))
    print(f"Combined stdout/stderr log: {log_path}", flush=True)
    with log_path.open("xb", buffering=0) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for line in iter(process.stdout.readline, b""):
                log.write(line)
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            code = 130
        finally:
            process.stdout.close()
    record["exit_code"] = code
    record_path.write_text(json.dumps(record, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
