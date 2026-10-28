"""CPU integration check against real LeRobot samples, including episode boundaries."""

import argparse
import dataclasses
import json
import os
from pathlib import Path

from experiments.continual.normalization import action_chunks
from experiments.continual.normalization import validate_assets
from experiments.continual.protocol import TASKS
from experiments.continual.protocol import read_splits
from experiments.continual.storage import DEFAULT_ROOT
from experiments.continual.storage import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("cl/splits.json"))
    parser.add_argument(
        "--storage-root", type=Path, default=Path(os.environ.get("OPENPI_CL_STORAGE_ROOT", DEFAULT_ROOT))
    )
    args = parser.parse_args()
    configure(args.storage_root, minimum_gib=1)
    os.environ["JAX_PLATFORMS"] = "cpu"

    import numpy as np
    import pyarrow.parquet as pq

    from experiments.continual.data import ContinualLiberoDataConfig
    from openpi import transforms
    from openpi.shared import normalize
    from openpi.training import config
    from openpi.training import data_loader

    splits = read_splits(args.splits)
    validate_assets(args.norm_assets, splits, args.splits, 4)
    order = [splits["tasks"][task]["train_episode_ids"][0] for task in ("T4", "T1", "T3", "T2")]
    routing = {
        episode: task for task in TASKS for episode in splits["tasks"][task]["train_episode_ids"] if episode in order
    }
    cfg = config.get_config("cl_pi05_t1_lora_smoke")
    factory = ContinualLiberoDataConfig(
        **{
            field.name: getattr(cfg.data, field.name)
            for field in dataclasses.fields(cfg.data)
            if field.name not in ("base_config", "assets")
        },
        base_config=dataclasses.replace(
            cfg.data.base_config,
            dataset_root=str(args.dataset_root),
            episode_ids=tuple(order),
            task_norm_stats={task: normalize.load(args.norm_assets / task) for task in TASKS},
            episode_norm_keys=routing,
        ),
        assets=config.AssetsConfig(assets_dir=str(args.norm_assets), asset_id="T4"),
    )
    data_cfg = factory.create(cfg.assets_dirs, cfg.model)
    data_cfg = dataclasses.replace(data_cfg, model_transforms=transforms.Group())
    raw_dataset = data_loader.create_torch_dataset(data_cfg, 10, cfg.model)
    processed = data_loader.transform_dataset(raw_dataset, data_cfg)
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    offset = 0
    checked = 0
    for episode in order:
        path = info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode)
        table = pq.read_table(args.dataset_root / path, columns=["state", "actions"])
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        chunks = action_chunks(actions, 10)
        for index in (0, len(actions) // 2, len(actions) - 1):
            sample = raw_dataset[offset + index]
            assert int(np.asarray(sample["episode_index"]).item()) == episode, "LeRobot changed episode order"
            np.testing.assert_allclose(np.asarray(sample["actions"]), chunks[index], atol=1e-6)
            transformed = processed[offset + index]
            task = routing[episode]
            expected = transforms.Normalize(data_cfg.task_norm_stats[task], use_quantiles=True)(
                {"actions": chunks[index]}
            )
            np.testing.assert_allclose(transformed["actions"], expected["actions"], atol=1e-6)
            restored = transforms.Unnormalize(data_cfg.task_norm_stats[task], use_quantiles=True)(
                {"actions": transformed["actions"], "state": transformed["state"]}
            )
            np.testing.assert_allclose(restored["actions"], chunks[index], atol=1e-5)
            checked += 1
        offset += len(actions)
    assert len(raw_dataset) == offset
    # Replay Subset indices must preserve the original episode ID used for routing.
    subset_cfg = dataclasses.replace(data_cfg, frame_indices=(0, offset - 1))
    subset = data_loader.transform_dataset(data_loader.create_torch_dataset(subset_cfg, 10, cfg.model), subset_cfg)
    np.testing.assert_allclose(subset[0]["actions"], processed[0]["actions"])
    np.testing.assert_allclose(subset[1]["actions"], processed[offset - 1]["actions"])
    print(
        json.dumps(
            {
                "checked_samples": checked,
                "task_count": 4,
                "episode_order": order,
                "chunk_boundary_and_normalization_roundtrip": "passed",
                "replay_subset_routing": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
