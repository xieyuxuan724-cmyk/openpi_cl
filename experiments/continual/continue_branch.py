"""Continue one experiment branch from the completed common T1 baseline."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.continual.normalization import validate_assets
from experiments.continual.protocol import TASKS
from experiments.continual.protocol import digest
from experiments.continual.protocol import read_splits
from experiments.continual.storage import DEFAULT_ROOT
from experiments.continual.storage import configure


def committed(path):
    metadata = path / "_CHECKPOINT_METADATA"
    return metadata.is_file() and bool(json.loads(metadata.read_text()).get("commit_timestamp_nsecs"))


def verify_stage(path, expected):
    actual = json.loads((path / "manifest.json").read_text())
    for field in ("identity", "stage", "norm_sha256", "previous"):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"Existing stage has a different {field}: {path}")
    if bool(actual.get("shared_t1")) != bool(expected.get("shared_t1")):
        raise ValueError(f"Shared-baseline provenance differs: {path}")


def evaluation_files(directory, checkpoint):
    """Require complete, unique, matching trials before skipping any evaluation."""
    manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
    paths = []
    for task in TASKS[: manifest["stage"]]:
        path = directory / f"{task}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 40 or sorted(row["episode"] for row in rows) != list(range(10, 50)):
            raise ValueError(f"Incomplete or duplicate evaluation trials: {path}")
        expected = {
            "protocol": manifest["identity"],
            "stage": manifest["stage"],
            "task_key": task,
            "checkpoint": str(checkpoint.resolve()),
            "seed": 7,
            "suite": "libero_10",
            "replan_steps": 5,
            "max_steps": 520,
            "num_steps_wait": 10,
            "resize_size": 224,
            "normalization_task": task,
            "norm_sha256": manifest["norm_sha256"][task],
        }
        for row in rows:
            if any(row.get(k) != v for k, v in expected.items()) or type(row.get("success")) is not bool:
                raise ValueError(f"Evaluation metadata differs: {path}")
            if row.get("error"):
                raise ValueError(f"Evaluation contains infrastructure errors: {path}")
        paths.append(path)
    return paths


def run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    try:
        return subprocess.run(list(map(str, command)), check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, file=sys.stderr)
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("sequential", "replay"), required=True)
    parser.add_argument(
        "--storage-root", type=Path, default=Path(os.environ.get("OPENPI_CL_STORAGE_ROOT", DEFAULT_ROOT))
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--check", action="store_true", help="Validate inputs and T2 plan without starting training")
    args = parser.parse_args()
    root = Path(configure(args.storage_root, minimum_gib=100)["root"])
    norm = root / "reports/per_task_norm_v1"
    splits_path = Path("cl/splits.json")
    validate_assets(norm, read_splits(splits_path), splits_path, 4)
    shared = root / "checkpoints/v2_sequential_seed42/cl_pi05_continual_v2/T1/9999"
    if not committed(shared):
        raise ValueError(f"Completed common T1 checkpoint is missing: {shared}")
    source = json.loads((shared.parent / "manifest.json").read_text())["identity"]
    group = "v2_sequential_seed42" if args.method == "sequential" else "v2_replay20_seed42"
    output = root / "checkpoints" / group
    sim = root / "envs/libero-py310/bin/python"
    if not sim.is_file():
        raise ValueError(f"Simulator environment missing: {sim}")
    baseline_dir = root / "reports/v2_sequential_seed42_T1_offset10_n40"
    results = evaluation_files(baseline_dir, shared)
    common = [
        "--method",
        args.method,
        "--dataset-root",
        source["dataset_root"],
        "--base-params",
        source["base_params"],
        "--norm-assets",
        norm,
        "--storage-root",
        root,
        "--output",
        output,
        "--seed",
        "42",
        "--batch-size",
        "8",
        "--steps",
        "10000",
        "--warmup",
        "500",
        "--unfreeze-vision",
        "--normalization",
        "quantile",
    ]
    if args.method == "replay":
        common += ["--replay-fraction", "0.2"]
    previous = shared
    for stage in range(2, 5):
        transition = ["--stage", str(stage), "--previous", previous, *common]
        if args.method == "replay" and stage == 2:
            transition += ["--shared-t1"]
        plan_command = [sys.executable, "-B", "-m", "experiments.continual.run", "plan", *transition]
        planned = json.loads(run(plan_command, capture_output=True, text=True).stdout)
        print(
            f"T{stage}: current starts={planned['current_frames']}; "
            f"replay buffers={ {k: len(v) for k, v in planned['replay_start_offsets'].items()} }",
            flush=True,
        )
        if args.check:
            print(f"Preflight passed for {args.method}. T1 baseline: 40 trials; T2 plan validated.", flush=True)
            return
        stage_dir = output / "cl_pi05_continual_v2" / f"T{stage}"
        checkpoint = stage_dir / "9999"
        resume = []
        if stage_dir.exists():
            verify_stage(stage_dir, planned)
            if not committed(checkpoint):
                saved = [p for p in stage_dir.iterdir() if p.name.isdigit() and committed(p)]
                if not saved:
                    raise ValueError(
                        f"No committed checkpoint to resume in {stage_dir}; inspect this failed stage first"
                    )
                resume = ["--resume"]
                print(f"Resume T{stage} from its latest committed checkpoint; data iterator is restarted.", flush=True)
        if not committed(checkpoint):
            run([sys.executable, "-B", "-m", "experiments.continual.run", "train", *transition, *resume])
        verify_stage(stage_dir, planned)
        if not committed(checkpoint):
            raise ValueError(f"Training returned without a committed final checkpoint: {checkpoint}")
        # Verify all saved normalization assets before evaluating or reusing results.
        for task, expected_hash in planned["norm_sha256"].items():
            if digest(checkpoint / "assets" / task / "norm_stats.json") != expected_hash:
                raise ValueError(f"Saved normalization differs for {task}")
        label = f"{group}_T{stage}_offset10_n40"
        candidates = [root / "reports" / label, *sorted((root / "reports").glob(label + "_retry_*"))]
        complete = None
        for directory in candidates:
            if not directory.exists():
                continue
            try:
                complete = evaluation_files(directory, checkpoint)
                break
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"Leaving incomplete evaluation intact: {directory}: {exc}", flush=True)
        if complete is None:
            tag = f"retry_{time.time_ns()}" if any(p.exists() for p in candidates) else ""
            command = [
                sys.executable,
                "-B",
                "-m",
                "experiments.continual.evaluate",
                "--checkpoint",
                checkpoint,
                "--norm-assets",
                norm,
                "--storage-root",
                root,
                "--sim-python",
                sim,
                "--port",
                str(args.port),
                "--trials",
                "40",
                "--initial-state-offset",
                "10",
                "--eval-seed",
                "7",
                "--videos-per-task",
                "3",
            ]
            if tag:
                command += ["--run-tag", tag]
            run(command)
            directory = root / "reports" / (label + ("_" + tag if tag else ""))
            complete = evaluation_files(directory, checkpoint)
        for path in complete:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            print(f"T{stage} / {path.stem}: {sum(row['success'] for row in rows)}/40 succeeded", flush=True)
        results.extend(complete)
        previous = checkpoint
    metrics = root / "reports" / f"{group}_metrics.json"
    if metrics.exists():
        saved = json.loads(metrics.read_text())
        if saved.get("inputs") != {str(p): digest(p) for p in results}:
            raise ValueError(f"Existing metrics have different input records: {metrics}")
    else:
        command = [
            sys.executable,
            "-B",
            "-m",
            "experiments.continual.analyze",
            "metrics",
            "--storage-root",
            root,
            "--output",
            metrics,
            "--results",
            *results,
        ]
        if args.method == "replay":
            command += ["--shared-t1"]
        run(command)
    print(f"COMPLETE {args.method}: {metrics}", flush=True)


if __name__ == "__main__":
    main()
