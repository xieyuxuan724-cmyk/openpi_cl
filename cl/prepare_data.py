import argparse
import json
import random
from pathlib import Path


TASKS = {
    "T1": "pick up the book and place it in the back compartment of the caddy",
    "T2": "turn on the stove and put the moka pot on it",
    "T3": "put the yellow and white mug in the microwave and close it",
    "T4": "put the black bowl in the bottom drawer of the cabinet and close it",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        default="/mnt/e/pi05/episodes.jsonl",
        help="Path to official LIBERO episodes.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="cl",
        help="Directory for experiment metadata",
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/mccxin/.cache/huggingface/lerobot/physical-intelligence/libero",
        help="Local LeRobot dataset root",
    )
    parser.add_argument(
        "--dataset-revision",
        default="main",
        help="LIBERO dataset revision",
    )
    args = parser.parse_args()

    episodes_path = Path(args.episodes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    splits = {
        "seed": 42,
        "train_ratio": 0.8,
        "tasks": {},
    }

    rng = random.Random(42)

    for task_id, task_text in TASKS.items():
        matched = [
            episode["episode_index"]
            for episode in episodes
            if task_text in episode.get("tasks", [])
        ]

        if not matched:
            raise RuntimeError(f"No episodes found for {task_id}: {task_text}")

        rng.shuffle(matched)

        split_index = int(len(matched) * 0.8)
        train_ids = matched[:split_index]
        val_ids = matched[split_index:]

        splits["tasks"][task_id] = {
            "task_text": task_text,
            "all_episode_ids": matched,
            "train_episode_ids": train_ids,
            "val_episode_ids": val_ids,
        }

        print(
            f"{task_id}: total={len(matched)}, "
            f"train={len(train_ids)}, val={len(val_ids)}"
        )

    splits_path = output_dir / "splits.json"
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    (output_dir / "dataset_root.txt").write_text(
        args.dataset_root + "\n",
        encoding="utf-8",
    )

    (output_dir / "dataset_revision.txt").write_text(
        args.dataset_revision + "\n",
        encoding="utf-8",
    )

    print(f"\nWrote: {splits_path}")
    print(f"Wrote: {output_dir / 'dataset_root.txt'}")
    print(f"Wrote: {output_dir / 'dataset_revision.txt'}")


if __name__ == "__main__":
    main()
