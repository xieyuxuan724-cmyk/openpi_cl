import pytest

from experiments.continual.protocol import TASKS
from experiments.continual.protocol import replay_plan
from experiments.continual.protocol import summarize


def fixture_data():
    splits = {"tasks": {t: {"train_episode_ids": [i]} for i, t in enumerate(TASKS)}}
    episodes = {i: {"length": 20} for i in range(4)}
    return splits, episodes


def test_fixed_buffer_and_partition():
    splits, episodes = fixture_data()
    order, indices, count, buffers = replay_plan(splits, episodes, 2, 5, 42)
    assert order == [1, 0]
    assert count == 20
    assert indices[:20] == list(range(20))
    assert all(20 <= i < 40 for i in indices[20:])
    assert len(set(indices[20:])) == 5
    assert buffers["T1"] == replay_plan(splits, episodes, 4, 5, 42)[3]["T1"]


def test_sequential_has_no_past_data():
    splits, episodes = fixture_data()
    assert replay_plan(splits, episodes, 4, 0, 42) == ([3], list(range(20)), 20, {})


def test_budget_cannot_silently_shrink():
    with pytest.raises(ValueError, match="Replay budget"):
        replay_plan(*fixture_data(), 2, 21, 42)


def test_metrics_signed_forgetting_and_invalid_runs():
    records = [
        {"stage": stage, "task_key": task, "episode": ep, "success": stage < 4}
        for stage in range(1, 5)
        for task in TASKS[:stage]
        for ep in range(2)
    ]
    result = summarize(records)
    assert result["final_forgetting"] == 1
    assert result["final_average_success"] == 0
    with pytest.raises(ValueError, match="Duplicate episode"):
        summarize([*records, records[0]])
    with pytest.raises(ValueError, match="Unequal evaluation"):
        summarize(records[:-1])
    with pytest.raises(ValueError, match="infrastructure errors"):
        summarize([dict(r, error="environment") for r in records])
