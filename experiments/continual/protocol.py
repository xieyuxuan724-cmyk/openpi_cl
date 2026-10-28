"""Pure, testable definitions for the four-task continual learning protocol."""

import hashlib
import json
from pathlib import Path
import random

TASKS = ("T1", "T2", "T3", "T4")


def read_splits(path):
    splits = json.loads(Path(path).read_text())
    seen = set()
    for task in TASKS:
        item = splits["tasks"][task]
        train, val = item["train_episode_ids"], item["val_episode_ids"]
        ids = train + val
        if not train or not val or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"Invalid or overlapping episode split: {task}")
        if set(ids) != set(item["all_episode_ids"]):
            raise ValueError(f"Incomplete split: {task}")
        seen.update(ids)
    return splits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_plan(splits, episodes, stage, budget, seed):
    """Return episode order, selected frame offsets, and current-frame count.

    Replay is sampled once per previous task, without replacement. The seed for
    each task is stage-independent, so later stages retain exactly the same buffer.
    A stored transition is a start frame plus its within-episode action chunk.
    """
    if stage not in range(1, 5) or budget < 0:
        raise ValueError("stage must be 1..4 and budget must be nonnegative")
    current = splits["tasks"][TASKS[stage - 1]]["train_episode_ids"]
    order = list(current)
    current_count = sum(episodes[i]["length"] for i in current)
    indices = list(range(current_count))
    buffers = {}
    offset = current_count
    if budget:
        for task_index, task in enumerate(TASKS[: stage - 1]):
            ids = splits["tasks"][task]["train_episode_ids"]
            count = sum(episodes[i]["length"] for i in ids)
            if budget > count:
                raise ValueError(f"Replay budget exceeds available frames for {task}")
            selected = sorted(random.Random(seed + 1009 * task_index).sample(range(count), budget))
            buffers[task] = selected
            indices.extend(offset + i for i in selected)
            order.extend(ids)
            offset += count
    return order, indices, current_count, buffers


def summarize(records):
    """Compute final average SR and signed diagonal-to-final forgetting."""
    cells = {}
    identities = set()
    for row in records:
        stage, task = row["stage"], row["task_key"]
        if stage not in range(5) or task not in TASKS:
            raise ValueError("Invalid evaluation stage/task")
        identity = (stage, task, row["episode"])
        if identity in identities:
            raise ValueError(f"Duplicate episode: {identity}")
        identities.add(identity)
        if row.get("error"):
            raise ValueError("Evaluation contains infrastructure errors; rerun it")
        cells.setdefault((stage, task), []).append(int(row["success"]))
    rates = {k: sum(v) / len(v) for k, v in cells.items()}
    required = [(stage, task) for stage in range(1, 5) for task in TASKS[:stage]]
    if any(k not in rates for k in required):
        raise ValueError("Incomplete lower-triangular evaluation matrix")
    episode_sets = [{r["episode"] for r in records if (r["stage"], r["task_key"]) == k} for k in required]
    if any(ids != episode_sets[0] for ids in episode_sets):
        raise ValueError("Unequal evaluation episode identities")
    counts = {len(cells[k]) for k in required}
    if len(counts) != 1:
        raise ValueError("Unequal evaluation episode counts")
    drops = {task: rates[(i + 1, task)] - rates[(4, task)] for i, task in enumerate(TASKS[:3])}
    return {
        "success_matrix": {str(s): {t: rates[(s, t)] for t in TASKS if (s, t) in rates} for s in range(5)},
        "episodes_per_cell": counts.pop(),
        "final_average_success": sum(rates[(4, t)] for t in TASKS) / 4,
        "final_forgetting": sum(drops.values()) / 3,
        "per_task_final_drop": drops,
        "diagonal_average_success": sum(rates[(i + 1, t)] for i, t in enumerate(TASKS)) / 4,
        "definition": "Final forgetting = mean_i<4 (SR_after_i_on_i - SR_after_4_on_i); negative is improvement.",
    }
