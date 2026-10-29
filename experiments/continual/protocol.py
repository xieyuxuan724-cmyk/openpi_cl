"""Pure, testable definitions for the four-task continual learning protocol."""

import bisect
import hashlib
import json
import math
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


def replay_plan(splits, episodes, stage, budget, seed, *, fraction=None):
    """Return episode order, selected frame offsets, and current-frame count.

    Replay is sampled once per previous task, without replacement. The seed for
    each task is stage-independent, so later stages retain exactly the same buffer.
    A stored transition is a start frame plus its within-episode action chunk.
    """
    if stage not in range(1, 5) or budget < 0:
        raise ValueError("stage must be 1..4 and budget must be nonnegative")
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("Replay fraction must be in (0, 1]")
    current = splits["tasks"][TASKS[stage - 1]]["train_episode_ids"]
    order = list(current)
    current_count = sum(episodes[i]["length"] for i in current)
    indices = list(range(current_count))
    buffers = {}
    offset = current_count
    if budget or fraction is not None:
        for task_index, task in enumerate(TASKS[: stage - 1]):
            ids = splits["tasks"][task]["train_episode_ids"]
            count = sum(episodes[i]["length"] for i in ids)
            task_budget = math.floor(count * fraction) if fraction is not None else budget
            if task_budget <= 0:
                raise ValueError(f"Replay fraction selects zero frames for {task}")
            if task_budget > count:
                raise ValueError(f"Replay budget exceeds available frames for {task}")
            selected = sorted(random.Random(seed + 1009 * task_index).sample(range(count), task_budget))
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
        if row["success"] not in (True, False, 0, 1):
            raise ValueError("Success must be boolean")
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
    nbt = {
        task: sum(rates[(i + 1, task)] - rates[(stage, task)] for stage in range(i + 2, 5)) / (3 - i)
        for i, task in enumerate(TASKS[:3])
    }
    normalized = {
        task: nbt[task] / rates[(i + 1, task)] if rates[(i + 1, task)] > 0 else None for i, task in enumerate(TASKS[:3])
    }
    return {
        "per_task_nbt": nbt,
        "nbt_mean_over_past_tasks": sum(nbt.values()) / 3,
        "nbt_mean_over_all_tasks_last_zero": sum(nbt.values()) / 4,
        "per_task_normalized_nbt": normalized,
        "normalized_nbt_undefined_tasks": [task for task, value in normalized.items() if value is None],
        "cell_wilson_95": {
            str(stage): {
                task: wilson_interval(sum(cells[(stage, task)]), len(cells[(stage, task)])) for task in TASKS[:stage]
            }
            for stage in range(1, 5)
        },
        "nbt_definition": "For each task, mean diagonal-minus-later SR over all later stages; report both K-1 and K denominators (last task set to zero for K).",
        "success_matrix": {str(s): {t: rates[(s, t)] for t in TASKS if (s, t) in rates} for s in range(5)},
        "episodes_per_cell": counts.pop(),
        "final_average_success": sum(rates[(4, t)] for t in TASKS) / 4,
        "final_forgetting": sum(drops.values()) / 3,
        "per_task_final_drop": drops,
        "diagonal_average_success": sum(rates[(i + 1, t)] for i, t in enumerate(TASKS)) / 4,
        "definition": "Final forgetting = mean_i<4 (SR_after_i_on_i - SR_after_4_on_i); negative is improvement.",
    }


def validate_eval_metadata(metadata, stage, checkpoint, task_keys):
    """Prevent a task from being evaluated using another task's action scale."""
    if metadata.get("stage") != stage or metadata.get("checkpoint") != str(Path(checkpoint).resolve()):
        raise ValueError("Connected policy server does not match requested stage/checkpoint")
    if metadata.get("identity", {}).get("protocol_version") != 2:
        raise ValueError("Evaluation requires a version 2 per-task normalization server")
    if len(task_keys) != 1 or metadata.get("eval_task") != task_keys[0]:
        raise ValueError("Evaluate exactly the task selected by the server's --eval-task")
    if not metadata.get("norm_sha256"):
        raise ValueError("Server has no normalization fingerprint")


def wilson_interval(successes, count):
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def replay_audit(splits, episodes, buffers, horizon=10):
    """Report both sampled observations and the distinct action-label coverage."""
    report = {}
    for task, starts in buffers.items():
        ends = []
        for episode_id in splits["tasks"][task]["train_episode_ids"]:
            ends.append((ends[-1] if ends else 0) + episodes[episode_id]["length"])
        labels = set()
        for start in starts:
            episode_end = ends[bisect.bisect_right(ends, start)]
            labels.update(range(start, min(start + horizon, episode_end)))
        report[task] = {
            "train_frames": ends[-1],
            "stored_start_frames": len(starts),
            "stored_start_fraction": len(starts) / ends[-1],
            "unique_action_label_frames": len(labels),
            "unique_action_label_fraction": len(labels) / ends[-1],
            "action_label_vectors_with_padding": len(starts) * horizon,
        }
    return report


def validate_previous_identity(previous, identity, stage, *, shared_t1=False):
    """Allow an explicit common T1 baseline; all later transitions stay strict."""
    if previous["stage"] != stage - 1:
        raise ValueError("Previous checkpoint belongs to a different stage")
    source = previous["identity"]
    if not shared_t1:
        if source != identity:
            raise ValueError("Previous checkpoint belongs to a different protocol")
        return
    if not (
        stage == 2
        and source["method"] == "sequential"
        and identity["method"] == "replay"
        and source.get("replay_budget") is None
        and source.get("replay_fraction") is None
        and identity.get("replay_budget") is None
        and identity.get("replay_fraction") == 0.2
    ):
        raise ValueError("Shared T1 is only supported for sequential T1 -> replay20 T2")
    ignored = {"method", "replay_budget", "replay_fraction"}
    if {k: v for k, v in source.items() if k not in ignored} != {k: v for k, v in identity.items() if k not in ignored}:
        raise ValueError("Shared T1 must have identical training settings")
