"""Compute immutable per-task statistics from training episodes on the data disk."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.continual.protocol import TASKS
from experiments.continual.protocol import digest
from experiments.continual.protocol import read_splits
from experiments.continual.storage import DEFAULT_ROOT
from experiments.continual.storage import configure

RECIPE = "libero_delta_actions_chunk10_train_only_v1"


def action_chunks(actions, horizon):
    """Match LeRobot's within-episode action window, including last-frame padding."""
    import numpy as np

    indices = np.minimum(np.arange(len(actions))[:, None] + np.arange(horizon), len(actions) - 1)
    return actions[indices]


def validate_assets(root, splits, split_path, stage, horizon=10):
    """Read only seen-task metadata; future-task statistics are never required."""
    hashes = {}
    for task in TASKS[:stage]:
        directory = Path(root) / task
        metadata = json.loads((directory / "metadata.json").read_text())
        expected = {
            "recipe": RECIPE,
            "split_sha256": digest(split_path),
            "task": task,
            "task_text": splits["tasks"][task]["task_text"],
            "episode_ids": splits["tasks"][task]["train_episode_ids"],
            "action_horizon": horizon,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Normalization provenance mismatch for {task}")
        hashes[task] = digest(directory / "norm_stats.json")
        if hashes[task] != metadata["norm_sha256"]:
            raise ValueError(f"Normalization file changed for {task}")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("cl/splits.json"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--storage-root", type=Path, default=Path(os.environ.get("OPENPI_CL_STORAGE_ROOT", DEFAULT_ROOT))
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    storage = configure(args.storage_root, minimum_gib=1)
    root = Path(storage["root"])
    output = args.output or root / "reports" / "per_task_norm_v1"
    if not output.resolve().is_relative_to(root / "reports"):
        parser.error("Statistics must be under storage-root/reports")

    import numpy as np
    import pyarrow.parquet as pq

    from openpi import transforms
    from openpi.shared import normalize
    from openpi.training import config

    splits = read_splits(args.splits)
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    episodes = {
        e["episode_index"]: e
        for e in map(json.loads, (args.dataset_root / "meta/episodes.jsonl").read_text().splitlines())
    }
    cfg = config.get_config("cl_pi05_t1_lora_smoke")
    if cfg.model.action_horizon != 10 or cfg.data.extra_delta_transform:
        raise ValueError("This statistics recipe expects LIBERO delta actions and horizon 10")
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    transform = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
    dummy_image = np.zeros((2, 2, 3), dtype=np.uint8)
    for task in args.tasks:
        directory = output / task
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite statistics: {directory}")
        item = splits["tasks"][task]
        running = {key: normalize.RunningStats() for key in ("state", "actions")}
        source_hash = hashlib.sha256()
        frame_count = 0
        for episode_id in item["train_episode_ids"]:
            if item["task_text"] not in episodes[episode_id]["tasks"]:
                raise ValueError(f"Wrong task label for episode {episode_id}")
            path = info["data_path"].format(episode_chunk=episode_id // info["chunks_size"], episode_index=episode_id)
            table = pq.read_table(args.dataset_root / path, columns=["state", "actions"])
            states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
            if (
                len(actions) != episodes[episode_id]["length"]
                or not np.isfinite(states).all()
                or not np.isfinite(actions).all()
            ):
                raise ValueError(f"Invalid numeric data in episode {episode_id}")
            source_hash.update(str(episode_id).encode())
            source_hash.update(states.tobytes())
            source_hash.update(actions.tobytes())
            # These LIBERO transforms do not mix numeric values with image pixels.
            # Use the actual training transforms without decoding unused images.
            transformed = transform(
                {
                    "state": states,
                    "actions": action_chunks(actions, 10),
                    "image": dummy_image,
                    "wrist_image": dummy_image,
                    "prompt": item["task_text"],
                }
            )
            for key, accumulator in running.items():
                accumulator.update(np.asarray(transformed[key], dtype=np.float64))
            frame_count += len(actions)
        stats = {key: accumulator.get_statistics() for key, accumulator in running.items()}
        for key, stat in stats.items():
            if not all(np.isfinite(value).all() for value in (stat.mean, stat.std, stat.q01, stat.q99)):
                raise ValueError(f"Non-finite {key} statistics for {task}")
        normalize.save(directory, stats)
        metadata = {
            "recipe": RECIPE,
            "split_sha256": digest(args.splits),
            "task": task,
            "task_text": item["task_text"],
            "episode_ids": item["train_episode_ids"],
            "action_horizon": 10,
            "frames": frame_count,
            "action_label_vectors": frame_count * 10,
            "numeric_source_sha256": source_hash.hexdigest(),
            "norm_sha256": digest(directory / "norm_stats.json"),
            "dataset_root": str(args.dataset_root.resolve()),
            "quantile_estimator": "OpenPI RunningStats histogram, 5000 bins",
            "constant_state_dimensions": np.flatnonzero(stats["state"].std == 0).tolist(),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"task": task, "frames": frame_count, "output": str(directory)}), flush=True)


if __name__ == "__main__":
    main()
