"""Train or serve one explicitly identified continual-learning stage."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

from experiments.continual.normalization import RECIPE
from experiments.continual.normalization import validate_assets
from experiments.continual.protocol import TASKS
from experiments.continual.protocol import digest
from experiments.continual.protocol import read_splits
from experiments.continual.protocol import replay_audit
from experiments.continual.protocol import replay_plan
from experiments.continual.protocol import validate_previous_identity
from experiments.continual.storage import DEFAULT_ROOT
from experiments.continual.storage import configure


def validate_lora_metadata(expected, available):
    """Match the NNX /value wrapper removal used by model.restore_params."""
    if available and all(key.endswith("/value") for key in available):
        available = {key.removesuffix("/value"): value for key, value in available.items()}
    lora = {key: value for key, value in expected.items() if "lora" in key}
    if not lora or not lora.keys() <= available.keys():
        raise ValueError("Previous checkpoint is missing LoRA parameters")
    for key, value in lora.items():
        if tuple(value.shape) != tuple(available[key].shape):
            raise ValueError(f"Previous checkpoint has incompatible LoRA shape: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "train", "serve"])
    parser.add_argument("--stage", type=int, choices=range(1, 5), required=True)
    parser.add_argument("--method", choices=["sequential", "replay"], default="sequential")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--norm-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--storage-root", type=Path, default=Path(os.environ.get("OPENPI_CL_STORAGE_ROOT", DEFAULT_ROOT))
    )
    parser.add_argument("--minimum-free-gib", type=float, default=100)
    parser.add_argument("--splits", type=Path, default=Path("cl/splits.json"))
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--shared-t1", action="store_true", help="Explicitly branch replay20 T2 from sequential T1")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=500)
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument("--replay-fraction", type=float, default=0.2)
    replay.add_argument(
        "--replay-budget", type=int, help="Optional fixed sample count per past task instead of a fraction"
    )
    vision = parser.add_mutually_exclusive_group()
    vision.add_argument("--freeze-vision", dest="unfreeze_vision", action="store_false")
    vision.add_argument("--unfreeze-vision", dest="unfreeze_vision", action="store_true")
    parser.set_defaults(unfreeze_vision=True)
    parser.add_argument("--normalization", choices=["quantile", "zscore"], default="quantile")
    parser.add_argument("--eval-task", choices=TASKS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.warmup <= 0:
        parser.error("steps, batch-size and warmup must be positive")
    if args.replay_budget is not None and args.replay_budget <= 0:
        parser.error("replay-budget must be positive")
    if not 0 < args.replay_fraction <= 1:
        parser.error("replay-fraction must be in (0, 1]")
    if args.method == "replay" and args.batch_size % 2:
        parser.error("replay requires an even batch-size for exact 1:1 sampling")
    if args.shared_t1 and (args.stage != 2 or args.method != "replay" or args.previous is None):
        parser.error("--shared-t1 requires replay T2 and --previous")
    if args.stage == 1 and args.previous:
        parser.error("T1 must start from base parameters")
    if args.mode != "serve" and args.stage > 1 and args.previous is None:
        parser.error("T2--T4 require --previous pointing to the previous stage checkpoint")
    if args.mode == "serve" and (args.eval_task is None or args.eval_task not in TASKS[: args.stage]):
        parser.error("serve requires --eval-task identifying an already learned task")
    if args.mode != "serve" and args.eval_task is not None:
        parser.error("--eval-task is only valid for serve")
    if args.output is None:
        variant = "vision_ft" if args.unfreeze_vision else "frozen_vision"
        replay_label = (
            (f"m{args.replay_budget}" if args.replay_budget else f"frac{args.replay_fraction:g}")
            if args.method == "replay"
            else "none"
        )
        args.output = (
            args.storage_root
            / "checkpoints"
            / f"v2_{args.method}_{replay_label}_seed{args.seed}_{variant}_{args.normalization}_b{args.batch_size}"
        )
    storage = configure(args.storage_root, args.output, minimum_gib=args.minimum_free_gib)
    splits = read_splits(args.splits)
    episodes = {
        e["episode_index"]: e
        for e in map(json.loads, (args.dataset_root / "meta/episodes.jsonl").read_text().splitlines())
    }
    for item in splits["tasks"].values():
        for i in item["all_episode_ids"]:
            if item["task_text"] not in episodes[i]["tasks"]:
                raise ValueError(f"Episode {i} has the wrong task")
    order, indices, current_count, buffers = replay_plan(
        splits,
        episodes,
        args.stage,
        (args.replay_budget or 0) if args.method == "replay" else 0,
        args.seed,
        fraction=args.replay_fraction if args.method == "replay" and args.replay_budget is None else None,
    )
    norm_hashes = validate_assets(args.norm_assets, splits, args.splits, args.stage)
    identity = {
        "method": args.method,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "replay_budget": args.replay_budget if args.method == "replay" else None,
        "replay_fraction": args.replay_fraction if args.method == "replay" and args.replay_budget is None else None,
        "unfreeze_vision": args.unfreeze_vision,
        "split_sha256": digest(args.splits),
        "protocol_version": 2,
        "normalization_recipe": RECIPE,
        "normalization": args.normalization,
        "action_horizon": 10,
        "replay_sampling": "current_past_equal_per_batch_past_tasks_uniform",
        "optimizer": "adamw_b1_0.9_b2_0.95_eps_1e-8_wd_1e-10_clip_1",
        "peak_lr": 2.5e-5,
        "end_lr": 2.5e-6,
        "base_params": str(args.base_params.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
    }
    if args.previous:
        previous_manifest = json.loads((args.previous.parent / "manifest.json").read_text())
        validate_previous_identity(previous_manifest, identity, args.stage, shared_t1=args.shared_t1)
        previous_hashes = previous_manifest["norm_sha256"]
        if previous_hashes != {task: norm_hashes[task] for task in TASKS[: args.stage - 1]}:
            raise ValueError("Previously learned tasks must retain exactly the same normalization")
        for task, expected_hash in previous_hashes.items():
            if digest(args.previous / "assets" / task / "norm_stats.json") != expected_hash:
                raise ValueError(f"Checkpoint normalization mismatch: {task}")
        if args.previous.name != str(args.steps - 1):
            raise ValueError("Previous checkpoint must be the stage-end checkpoint")
    manifest = {
        "shared_t1": args.shared_t1,
        "stage": args.stage,
        "norm_sha256": norm_hashes,
        "storage": storage,
        "identity": identity,
        "episode_order": order,
        "current_frames": current_count,
        "replay_start_offsets": buffers,
        "replay_audit": replay_audit(splits, episodes, buffers),
        "replay_label_frames_upper_bound": sum(map(len, buffers.values())) * 10,
        "previous": str(args.previous.resolve()) if args.previous else None,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": os.sys.executable,
        "working_tree_patch_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "flax", "torch", "orbax-checkpoint", "lerobot")
        },
    }
    if args.mode == "plan":
        print(json.dumps(manifest, indent=2))
        return

    import flax.nnx as nnx

    from experiments.continual.data import ContinualLiberoDataConfig
    from openpi.shared import nnx_utils
    from openpi.shared import normalize
    from openpi.training import config
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    c = config.get_config("cl_pi05_t1_lora_smoke")
    task_stats = {task: normalize.load(args.norm_assets / task) for task in TASKS[: args.stage]}
    episode_norm_keys = {
        episode_id: task
        for task in TASKS[: args.stage]
        for episode_id in splits["tasks"][task]["train_episode_ids"]
        if episode_id in order
    }
    freeze = c.model.get_freeze_filter()
    if not args.unfreeze_vision:
        freeze = nnx.Any(freeze, nnx_utils.PathRegex("PaliGemma/img/.*"))
    base = dataclasses.replace(
        c.data.base_config,
        dataset_root=str(args.dataset_root.resolve()),
        episode_ids=tuple(order),
        frame_indices=tuple(indices) if buffers else None,
        current_frame_count=current_count if buffers else None,
        replay_group_sizes=tuple(map(len, buffers.values())) if buffers else None,
        task_norm_stats=task_stats,
        episode_norm_keys=episode_norm_keys,
    )
    data = ContinualLiberoDataConfig(
        **{
            field.name: getattr(c.data, field.name)
            for field in dataclasses.fields(c.data)
            if field.name not in ("base_config", "assets")
        },
        base_config=base,
        assets=config.AssetsConfig(
            assets_dir=str(args.norm_assets.resolve()), asset_id=args.eval_task or TASKS[args.stage - 1]
        ),
        normalization=args.normalization,
    )
    c = dataclasses.replace(
        c,
        name="cl_pi05_continual_v2",
        exp_name=f"T{args.stage}",
        checkpoint_base_dir=str(args.output.resolve()),
        data=data,
        freeze_filter=freeze,
        seed=args.seed,
        batch_size=args.batch_size,
        num_train_steps=args.steps,
        save_interval=1000,
        keep_period=None,
        log_interval=10,
        optimizer=optimizer.AdamW(),
        ema_decay=None,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup, peak_lr=2.5e-5, decay_steps=max(args.steps, args.warmup + 1), decay_lr=2.5e-6
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(args.previous / "params" if args.previous else args.base_params)
        ),
        resume=args.resume,
        overwrite=False,
    )

    if args.mode == "serve":
        if args.checkpoint is None:
            parser.error("serve requires --checkpoint")
        saved = json.loads((args.checkpoint.parent / "manifest.json").read_text())
        if saved["identity"] != identity or saved["stage"] != args.stage or saved["norm_sha256"] != norm_hashes:
            raise ValueError("Evaluation configuration differs from checkpoint manifest")
        from openpi.policies.policy_config import create_trained_policy
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer

        for task, expected_hash in norm_hashes.items():
            if digest(args.checkpoint / "assets" / task / "norm_stats.json") != expected_hash:
                raise ValueError(f"Checkpoint normalization mismatch: {task}")
        policy = create_trained_policy(c, args.checkpoint, sample_kwargs={"num_steps": 10})
        WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata={
                "checkpoint": str(args.checkpoint.resolve()),
                "stage": args.stage,
                "identity": identity,
                "eval_task": args.eval_task,
                "norm_sha256": norm_hashes[args.eval_task],
            },
        ).serve_forever()
        return

    # Validate LoRA presence before allowing a cross-stage load.
    if args.previous:
        import orbax.checkpoint as ocp

        metadata = ocp.PyTreeCheckpointer().metadata(args.previous / "params")
        from flax.traverse_util import flatten_dict
        import jax

        expected = jax.eval_shape(lambda: nnx.state(c.model.create(jax.random.key(args.seed)))).to_pure_dict()
        validate_lora_metadata(flatten_dict(expected, sep="/"), flatten_dict(metadata["params"], sep="/"))
    manifest_path = c.checkpoint_dir / "manifest.json"
    if args.resume:
        saved = json.loads(manifest_path.read_text())
        if saved["identity"] != identity or saved["stage"] != args.stage or saved["norm_sha256"] != norm_hashes:
            raise ValueError("Resume configuration mismatch")
    elif c.checkpoint_dir.exists():
        raise FileExistsError(c.checkpoint_dir)

    from scripts import train

    train.init_logging()
    # The trainer creates its checkpoint directory before emitting the manifest.
    train.main(c, run_manifest=manifest)


if __name__ == "__main__":
    main()
