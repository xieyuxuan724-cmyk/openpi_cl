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


def test_nbt_uses_all_later_stages_and_handles_zero_diagonal():
    records = [
        {"stage": stage, "task_key": task, "episode": ep, "success": (task == "T1" and stage in {1, 4})}
        for stage in range(1, 5)
        for task in TASKS[:stage]
        for ep in range(2)
    ]
    result = summarize(records)
    assert result["final_forgetting"] == 0
    assert result["per_task_nbt"]["T1"] == pytest.approx(2 / 3)
    assert result["nbt_mean_over_past_tasks"] == pytest.approx(2 / 9)
    assert result["nbt_mean_over_all_tasks_last_zero"] == pytest.approx(1 / 6)
    assert result["per_task_normalized_nbt"]["T2"] is None
    assert result["cell_wilson_95"]["1"]["T1"][1] == pytest.approx(1)


def test_fraction_buffer_is_per_task_and_fixed_across_stages():
    from experiments.continual.protocol import replay_audit

    splits, episodes = fixture_data()
    episodes[0]["length"] = 23
    episodes[1]["length"] = 47
    plan = replay_plan(splits, episodes, 3, 0, 42, fraction=0.2)
    assert {task: len(indices) for task, indices in plan[3].items()} == {"T1": 4, "T2": 9}
    assert plan[3]["T1"] == replay_plan(splits, episodes, 4, 0, 42, fraction=0.2)[3]["T1"]
    audit = replay_audit(splits, episodes, plan[3])
    assert audit["T1"]["stored_start_fraction"] == 4 / 23
    assert 4 <= audit["T1"]["unique_action_label_frames"] <= 23
