"""Audit training data or aggregate one seed/method's evaluation matrix."""

import argparse
import json
import os
from pathlib import Path

from experiments.continual.normalization import validate_assets
from experiments.continual.protocol import TASKS
from experiments.continual.protocol import digest
from experiments.continual.protocol import read_splits
from experiments.continual.protocol import summarize
from experiments.continual.storage import DEFAULT_ROOT
from experiments.continual.storage import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "metrics"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--storage-root", type=Path, default=Path(os.environ.get("OPENPI_CL_STORAGE_ROOT", DEFAULT_ROOT))
    )
    parser.add_argument("--splits", type=Path, default=Path("cl/splits.json"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--norm-file", type=Path)
    parser.add_argument("--norm-assets", type=Path)
    parser.add_argument("--normalization", choices=["quantile", "zscore"], default="quantile")
    parser.add_argument("--results", type=Path, nargs="+")
    args = parser.parse_args()
    storage = configure(args.storage_root, minimum_gib=1)
    if not args.output.resolve().is_relative_to(Path(storage["root"]) / "reports"):
        parser.error("Analysis output must be under storage-root/reports")
    if args.mode == "metrics":
        if not args.results:
            parser.error("metrics requires --results")
        rows = [json.loads(line) for p in args.results for line in p.read_text().splitlines() if line.strip()]
        signatures = {
            (r["seed"], r["suite"], r["replan_steps"], r["max_steps"], r.get("num_steps_wait"), r.get("resize_size"))
            for r in rows
        }
        if len(signatures) != 1:
            raise ValueError("Do not combine different evaluation protocols/seeds")
        protocols = {json.dumps(r.get("protocol"), sort_keys=True) for r in rows}
        if len(protocols) != 1 or None in [r.get("protocol") for r in rows]:
            raise ValueError("Evaluation files have missing or inconsistent training protocol identity")
        for task in TASKS:
            task_rows = [row for row in rows if row["task_key"] == task]
            hashes = {row.get("norm_sha256") for row in task_rows}
            if len(hashes) != 1 or None in hashes or any(row.get("normalization_task") != task for row in task_rows):
                raise ValueError(f"Missing, changed or misrouted normalization for {task}")
        for stage in range(1, 5):
            if len({row["checkpoint"] for row in rows if row["stage"] == stage}) != 1:
                raise ValueError("Each stage must use a single checkpoint across tasks")
        report = summarize(rows)
        report["protocol"] = rows[0]["protocol"]
        report["evaluation_episode_ids"] = sorted({row["episode"] for row in rows})
        report["norm_sha256"] = {
            task: next(row["norm_sha256"] for row in rows if row["task_key"] == task) for task in TASKS
        }
        report["evaluation_signature"] = list(next(iter(signatures)))
        report["inputs"] = {str(p): digest(p) for p in args.results}
    else:
        if args.dataset_root is None or (args.norm_file is None) == (args.norm_assets is None):
            parser.error("audit requires --dataset-root and exactly one of --norm-file / --norm-assets")
        import numpy as np
        import pyarrow.parquet as pq

        splits = read_splits(args.splits)
        root = args.dataset_root
        info = json.loads((root / "meta/info.json").read_text())
        episodes = {
            e["episode_index"]: e for e in map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines())
        }
        norm_hashes = (
            validate_assets(args.norm_assets, splits, args.splits, 4)
            if args.norm_assets
            else {"shared": digest(args.norm_file)}
        )
        report = {
            "split_sha256": digest(args.splits),
            "norm_sha256": norm_hashes,
            "normalization": args.normalization,
            "tasks": {},
            "standard": "Training episodes only. Outside threshold: abs(z)>1.01 for quantile or abs(z)>3 for zscore; no clipping.",
        }
        for task in TASKS:
            norm_file = args.norm_assets / task / "norm_stats.json" if args.norm_assets else args.norm_file
            stats = json.loads(norm_file.read_text())["norm_stats"]["actions"]
            low, high = np.array(stats["q01"]), np.array(stats["q99"])
            item = splits["tasks"][task]
            actions = []
            for i in item["train_episode_ids"]:
                if item["task_text"] not in episodes[i]["tasks"]:
                    raise ValueError(f"Wrong task in episode {i}")
                path = info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)
                actions.append(np.asarray(pq.read_table(root / path, columns=["actions"])["actions"].to_pylist()))
            a = np.concatenate(actions)
            if not np.isfinite(a).all():
                raise ValueError(f"Non-finite actions in {task}")
            z = (
                2 * (a - low) / (high - low + 1e-6) - 1
                if args.normalization == "quantile"
                else (a - np.array(stats["mean"])) / (np.array(stats["std"]) + 1e-6)
            )
            report["tasks"][task] = {
                "train_episodes": len(actions),
                "val_episodes": len(item["val_episode_ids"]),
                "train_frames": len(a),
                "raw_action_min": a.min(0).tolist(),
                "raw_action_max": a.max(0).tolist(),
                "normalized_abs_max": np.abs(z).max(0).tolist(),
                "outside_fraction_per_dimension": (np.abs(z) > (1.01 if args.normalization == "quantile" else 3))
                .mean(0)
                .tolist(),
                "effective_epochs_at_batch8_steps10000": 80000 / len(a),
                "replay1000_fraction": 1000 / len(a),
                "effective_epochs_at_batch2_steps10000": 20000 / len(a),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)


if __name__ == "__main__":
    main()
