import pytest

from experiments.continual.compare import METRICS
from experiments.continual.compare import compare


def report(method, seed, success):
    return {
        "protocol": {
            "method": method,
            "seed": seed,
            "replay_fraction": 0.2 if method == "replay" else None,
            "replay_budget": None,
            "batch_size": 8,
        },
        "evaluation_signature": [7, "libero_10", 5, 520, 10],
        "episodes_per_cell": 40,
        "evaluation_episode_ids": list(range(10, 50)),
        "norm_sha256": {"T1": "abc"},
        **dict.fromkeys(METRICS, success),
    }


def test_paired_comparison_rejects_mixed_protocols():
    result = compare([report("sequential", 42, 0.5)], [report("replay", 42, 0.75)])
    assert result["paired_delta"]["final_average_success"] == {"mean": 0.25, "sample_std": None}
    with pytest.raises(ValueError, match="same training seeds"):
        compare([report("sequential", 42, 0.5)], [report("replay", 43, 0.75)])
    mixed = report("replay", 42, 0.75)
    mixed["protocol"]["batch_size"] = 2
    with pytest.raises(ValueError, match="different training"):
        compare([report("sequential", 42, 0.5)], [mixed])
