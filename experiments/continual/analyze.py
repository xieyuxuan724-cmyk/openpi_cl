"""Audit training data or aggregate one seed/method's evaluation matrix."""

import argparse
import json
from pathlib import Path

from experiments.continual.protocol import TASKS
from experiments.continual.protocol import digest
from experiments.continual.protocol import read_splits
from experiments.continual.protocol import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "metrics"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("cl/splits.json"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--norm-file", type=Path)
    parser.add_argument("--results", type=Path, nargs="+")
    args = parser.parse_args()
    if args.mode == "metrics":
        if not args.results:
            parser.error("metrics requires --results")
        rows = [json.loads(line) for p in args.results for line in p.read_text().splitlines() if line.strip()]
        signatures = {(r["seed"], r["suite"], r["replan_steps"], r["max_steps"]) for r in rows}
        if len(signatures) != 1:
            raise ValueError("Do not combine different evaluation protocols/seeds")
        protocols = {json.dumps(r.get("protocol"), sort_keys=True) for r in rows}
        if len(protocols) != 1 or None in [r.get("protocol") for r in rows]:
            raise ValueError("Evaluation files have missing or inconsistent training protocol identity")
        report = summarize(rows)
        report["inputs"] = {str(p): digest(p) for p in args.results}
    else:
        if args.dataset_root is None or args.norm_file is None:
            parser.error("audit requires --dataset-root and --norm-file")
        import numpy as np
        import pyarrow.parquet as pq

        splits = read_splits(args.splits)
        root = args.dataset_root
        info = json.loads((root / "meta/info.json").read_text())
        episodes = {
            e["episode_index"]: e for e in map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines())
        }
        stats = json.loads(args.norm_file.read_text())["norm_stats"]["actions"]
        low, high = np.array(stats["q01"]), np.array(stats["q99"])
        if np.any(high <= low):
            raise ValueError("Degenerate action normalization interval")
        report = {
            "split_sha256": digest(args.splits),
            "norm_sha256": digest(args.norm_file),
            "tasks": {},
            "standard": "Audit train episodes only. Outside fraction uses abs(normalized)>1.01 to avoid quantile interpolation noise; no clipping.",
        }
        for task in TASKS:
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
            z = 2 * (a - low) / (high - low + 1e-6) - 1
            report["tasks"][task] = {
                "train_episodes": len(actions),
                "val_episodes": len(item["val_episode_ids"]),
                "train_frames": len(a),
                "raw_action_min": a.min(0).tolist(),
                "raw_action_max": a.max(0).tolist(),
                "normalized_abs_max": np.abs(z).max(0).tolist(),
                "outside_fraction_per_dimension": (np.abs(z) > 1.01).mean(0).tolist(),
                "effective_epochs_at_batch2_steps10000": 20000 / len(a),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)


if __name__ == "__main__":
    main()
