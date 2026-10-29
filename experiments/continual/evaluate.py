"""Evaluate all seen tasks with one checkpoint and each task's saved normalization."""

import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from experiments.continual.protocol import TASKS
from experiments.continual.storage import DEFAULT_ROOT
from experiments.continual.storage import configure


def server_command(manifest, checkpoint, norm_assets, storage_root, task, port):
    identity = manifest["identity"]
    if identity.get("protocol_version") != 2 or task not in TASKS[: manifest["stage"]]:
        raise ValueError("Expected a version 2 checkpoint and an already learned task")
    command = [
        sys.executable,
        "-B",
        "-m",
        "experiments.continual.run",
        "serve",
        "--stage",
        str(manifest["stage"]),
        "--checkpoint",
        str(checkpoint),
        "--eval-task",
        task,
        "--norm-assets",
        str(norm_assets),
        "--storage-root",
        str(storage_root),
        "--port",
        str(port),
    ]
    for field in ("method", "dataset_root", "base_params", "seed", "batch_size", "steps", "warmup", "normalization"):
        command += ["--" + field.replace("_", "-"), str(identity[field])]
    for field in ("replay_budget", "replay_fraction"):
        if identity.get(field) is not None:
            command += ["--" + field.replace("_", "-"), str(identity[field])]
    command.append("--unfreeze-vision" if identity["unfreeze_vision"] else "--freeze-vision")
    return command


def wait_for_server(process, port, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited with code {process.returncode}; inspect its log")
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        try:
            connection.request("GET", "/healthz")
            response = connection.getresponse()
            if response.status == 200 and response.read().strip() == b"OK":
                return
        except (OSError, http.client.HTTPException):
            pass
        finally:
            connection.close()
        time.sleep(1)
    raise TimeoutError("Policy server did not become ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-assets", type=Path, required=True)
    parser.add_argument("--sim-python", type=Path, default=Path("examples/libero/.venv/bin/python"))
    parser.add_argument("--splits", type=Path, default=Path("cl/splits.json"))
    parser.add_argument(
        "--storage-root", type=Path, default=Path(os.environ.get("OPENPI_CL_STORAGE_ROOT", DEFAULT_ROOT))
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--run-tag", default="", help="Unique suffix for reruns after a failed evaluation")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--initial-state-offset", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=7)
    parser.add_argument("--videos-per-task", type=int, default=3)
    args = parser.parse_args()
    if args.run_tag and not all(char.isalnum() or char in "_-" for char in args.run_tag):
        parser.error("run-tag must contain only letters, digits, underscores or hyphens")
    storage = configure(args.storage_root, minimum_gib=1)
    if not args.sim_python.is_file():
        parser.error("Simulator Python does not exist; install the separate LIBERO environment first")
    checkpoint = args.checkpoint.resolve()
    manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
    if checkpoint.name != str(manifest["identity"]["steps"] - 1):
        parser.error("Report only the final checkpoint of a stage")
    if not 1 <= manifest["stage"] <= 4 or args.trials <= 0 or args.initial_state_offset < 0:
        parser.error("Invalid stage or evaluation episode range")
    from experiments.continual.protocol import digest

    if digest(args.splits) != manifest["identity"]["split_sha256"]:
        parser.error("Evaluation split differs from training")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    root = Path(storage["root"])
    group = checkpoint.parents[2].name
    label = f"{group}_T{manifest['stage']}_offset{args.initial_state_offset}_n{args.trials}"
    if args.run_tag:
        label += "_" + args.run_tag
    report_dir = root / "reports" / label
    report_dir.mkdir(exist_ok=False)
    for task in TASKS[: manifest["stage"]]:
        server = server_command(manifest, checkpoint, args.norm_assets.resolve(), root, task, args.port)
        server += ["--splits", str(args.splits.resolve())]
        command = [
            # Preserve the venv executable symlink so Python finds pyvenv.cfg.
            str(args.sim_python.absolute()),
            "-B",
            "-m",
            "examples.libero.main",
            "--args.host",
            "127.0.0.1",
            "--args.port",
            str(args.port),
            "--args.task-suite-name",
            "libero_10",
            "--args.task-keys",
            task,
            "--args.splits-path",
            str(args.splits.resolve()),
            "--args.num-trials-per-task",
            str(args.trials),
            "--args.initial-state-offset",
            str(args.initial_state_offset),
            "--args.stage",
            str(manifest["stage"]),
            "--args.checkpoint-label",
            str(checkpoint),
            "--args.seed",
            str(args.eval_seed),
            "--args.videos-per-task",
            str(args.videos_per_task),
            "--args.result-path",
            str(report_dir / f"{task}.jsonl"),
            "--args.video-out-path",
            str(root / "videos" / label / task),
        ]
        log_path = root / "logs" / f"{label}_{task}_server.log"
        with log_path.open("x") as log:
            process = subprocess.Popen(server, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_for_server(process, args.port)
                sim_env = os.environ.copy()
                sim_env.setdefault("MUJOCO_GL", "osmesa")
                sim_env.setdefault("PYOPENGL_PLATFORM", sim_env["MUJOCO_GL"])
                sim_env.setdefault("PYNPUT_BACKEND", "dummy")
                subprocess.run(command, check=True, env=sim_env)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    print(f"Evaluation complete: {report_dir}")


if __name__ == "__main__":
    main()
