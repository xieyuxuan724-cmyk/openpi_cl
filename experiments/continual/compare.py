"""Compare paired sequential / 20% replay metrics without mixing protocols."""

import argparse
import json
import os
from pathlib import Path
import statistics

from experiments.continual.storage import DEFAULT_ROOT
from experiments.continual.storage import configure

METRICS = ("final_average_success", "diagonal_average_success", "final_forgetting", "nbt_mean_over_past_tasks")


def compare(sequential, replay):
    def index(reports, method):
        result = {}
        for report in reports:
            identity = report["protocol"]
            if identity["method"] != method or identity["seed"] in result:
                raise ValueError("Incorrect method or duplicate training seed")
            if method == "replay" and (identity["replay_fraction"] != 0.2 or identity["replay_budget"] is not None):
                raise ValueError("Main comparison requires replay_fraction=0.2")
            result[identity["seed"]] = report
        return result

    left, right = index(sequential, "sequential"), index(replay, "replay")
    if not left or left.keys() != right.keys():
        raise ValueError("Both methods require exactly the same training seeds")
    paired = []
    comparable_protocols = set()
    for seed in sorted(left):
        a, b = left[seed], right[seed]

        def common(report):
            return {
                key: value
                for key, value in report["protocol"].items()
                if key not in ("method", "replay_fraction", "replay_budget", "seed")
            }

        if common(a) != common(b) or a["evaluation_signature"] != b["evaluation_signature"]:
            raise ValueError("Paired methods use different training or evaluation settings")
        comparable_protocols.add(json.dumps([common(a), a["evaluation_signature"]], sort_keys=True))
        if a["evaluation_episode_ids"] != b["evaluation_episode_ids"] or a["norm_sha256"] != b["norm_sha256"]:
            raise ValueError("Paired methods use different initial states or normalization files")
        if a["episodes_per_cell"] != b["episodes_per_cell"]:
            raise ValueError("Different evaluation sample counts")
        paired.append({"seed": seed, "replay_minus_sequential": {key: b[key] - a[key] for key in METRICS}})
    if len(comparable_protocols) != 1:
        raise ValueError("Do not pool seeds with different experimental settings")

    def aggregate(values):
        return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None}

    return {
        "seed_count": len(left),
        "paired_results": paired,
        "sequential": {key: aggregate([row[key] for row in left.values()]) for key in METRICS},
        "replay20": {key: aggregate([row[key] for row in right.values()]) for key in METRICS},
        "paired_delta": {key: aggregate([row["replay_minus_sequential"][key] for row in paired]) for key in METRICS},
        "interpretation": "Positive success delta is better; negative forgetting/NBT delta is better. One seed is a pilot; no significance claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential", nargs="+", type=Path, required=True)
    parser.add_argument("--replay", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--storage-root", type=Path, default=Path(os.environ.get("OPENPI_CL_STORAGE_ROOT", DEFAULT_ROOT))
    )
    args = parser.parse_args()
    storage = configure(args.storage_root, minimum_gib=1)
    if not args.output.resolve().is_relative_to(Path(storage["root"]) / "reports"):
        parser.error("Output must be under storage-root/reports")
    report = compare(
        [json.loads(path.read_text()) for path in args.sequential],
        [json.loads(path.read_text()) for path in args.replay],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)


if __name__ == "__main__":
    main()
