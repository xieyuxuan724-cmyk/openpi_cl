import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.continual.normalization import RECIPE
from experiments.continual.normalization import action_chunks
from experiments.continual.normalization import validate_assets
from experiments.continual.protocol import digest
from experiments.continual.protocol import validate_eval_metadata
from openpi import transforms
from openpi.shared import normalize
from openpi.training import checkpoints
from openpi.training import config
from openpi.training import data_loader


def stats(center):
    return {
        "actions": normalize.NormStats(
            mean=np.array([center]), std=np.array([2.0]), q01=np.array([center - 2]), q99=np.array([center + 2])
        )
    }


@pytest.mark.parametrize("quantile", [True, False])
def test_replay_normalization_routes_before_repack(quantile):
    samples = [{"episode_index": 19, "raw": np.array([[12.0]])}, {"episode_index": 3, "raw": np.array([[12.0]])}]
    cfg = config.DataConfig(
        task_norm_stats={"T1": stats(10), "T2": stats(20)},
        episode_norm_keys={19: "T1", 3: "T2"},
        repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({"actions": "raw"})]),
        use_quantile_norm=quantile,
    )
    dataset = data_loader.transform_dataset(samples, cfg)
    np.testing.assert_allclose(dataset[0]["actions"], [[1]], atol=2e-6)
    np.testing.assert_allclose(dataset[1]["actions"], [[-4]], atol=3e-6)
    for index, task in enumerate(("T1", "T2")):
        restored = transforms.Unnormalize(cfg.task_norm_stats[task], use_quantiles=quantile)(dataset[index])
        np.testing.assert_allclose(restored["actions"], samples[index]["raw"], atol=1e-5)
    with pytest.raises(ValueError, match="no normalization"):
        data_loader.TaskNormalizedDataset([{"episode_index": 100}], cfg)[0]


def test_chunk_padding_does_not_cross_episode():
    np.testing.assert_array_equal(
        action_chunks(np.array([[1], [2], [3]]), 3), [[[1], [2], [3]], [[2], [3], [3]], [[3], [3], [3]]]
    )


def test_balanced_replay_batches_and_seed():
    sampler = data_loader.BalancedReplaySampler(23, 39, 8, 42)
    indices = list(sampler)
    assert indices == list(data_loader.BalancedReplaySampler(23, 39, 8, 42))
    assert len(indices) == 32
    for offset in range(0, len(indices), 8):
        batch = indices[offset : offset + 8]
        assert sum(index < 23 for index in batch) == 4
        assert all(0 <= index < 39 for index in batch)
    assert list(sampler) != indices
    with pytest.raises(ValueError, match="even batch"):
        data_loader.BalancedReplaySampler(10, 20, 3, 0)


def test_checkpoint_carries_all_seen_task_statistics(tmp_path, monkeypatch):
    cfg = config.DataConfig(asset_id="T2", norm_stats=stats(20), task_norm_stats={"T1": stats(10), "T2": stats(20)})
    loader = SimpleNamespace(data_config=lambda: cfg)
    monkeypatch.setattr(checkpoints, "_split_params", lambda state: ({}, {}))

    class Manager:
        def save(self, step, items):
            items["assets"](tmp_path)

    checkpoints.save_state(Manager(), None, loader, 0)
    for task, center in (("T1", 10), ("T2", 20)):
        np.testing.assert_equal(normalize.load(tmp_path / task)["actions"].mean, [center])


def test_metadata_rejects_wrong_task():
    metadata = {
        "stage": 3,
        "checkpoint": str(Path("checkpoint").resolve()),
        "identity": {"protocol_version": 2},
        "eval_task": "T1",
        "norm_sha256": "abc",
    }
    validate_eval_metadata(metadata, 3, "checkpoint", ("T1",))
    for keys in (("T2",), ("T1", "T2")):
        with pytest.raises(ValueError, match="exactly the task"):
            validate_eval_metadata(metadata, 3, "checkpoint", keys)


def test_assets_need_only_seen_tasks_and_reject_tampering(tmp_path):
    split_path = tmp_path / "splits.json"
    split_path.write_text("{}")
    splits = {"tasks": {"T1": {"train_episode_ids": [5], "task_text": "task"}}}
    directory = tmp_path / "T1"
    normalize.save(directory, stats(10))
    meta = {
        "recipe": RECIPE,
        "split_sha256": digest(split_path),
        "task": "T1",
        "task_text": "task",
        "episode_ids": [5],
        "action_horizon": 10,
        "norm_sha256": digest(directory / "norm_stats.json"),
    }
    (directory / "metadata.json").write_text(json.dumps(meta))
    assert list(validate_assets(tmp_path, splits, split_path, 1)) == ["T1"]
    normalize.save(directory, stats(20))
    with pytest.raises(ValueError, match="changed"):
        validate_assets(tmp_path, splits, split_path, 1)


def test_replay_old_tasks_are_balanced_despite_different_sizes():
    sampler = data_loader.BalancedReplaySampler(10, 110, 8, 42, (10, 90))
    past = [index for _ in range(100) for index in sampler if index >= 10]
    assert 0.45 < sum(index < 20 for index in past) / len(past) < 0.55
    with pytest.raises(ValueError, match="partition"):
        data_loader.BalancedReplaySampler(10, 110, 8, 42, (10, 80))
