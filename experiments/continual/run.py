"""Train or serve one explicitly identified continual-learning stage."""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess

from experiments.continual.protocol import digest
from experiments.continual.protocol import read_splits
from experiments.continual.protocol import replay_plan
from experiments.continual.storage import DEFAULT_ROOT
from experiments.continual.storage import configure


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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--replay-budget", type=int, default=100)
    parser.add_argument("--unfreeze-vision", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.warmup <= 0:
        parser.error("steps, batch-size and warmup must be positive")
    if args.method == "replay" and args.replay_budget <= 0:
        parser.error("replay requires a positive budget")
    if args.stage == 1 and args.previous:
        parser.error("T1 must start from base parameters")
    if args.stage > 1 and args.previous is None:
        parser.error("T2--T4 require --previous pointing to the previous stage checkpoint")
    if args.output is None:
        variant = "vision_ft" if args.unfreeze_vision else "frozen_vision"
        args.output = args.storage_root / "checkpoints" / f"{args.method}_seed{args.seed}_{variant}"
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
        splits, episodes, args.stage, args.replay_budget if args.method == "replay" else 0, args.seed
    )
    norm_file = args.norm_assets / "physical-intelligence/libero/norm_stats.json"
    identity = {
        "method": args.method,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "replay_budget": args.replay_budget,
        "unfreeze_vision": args.unfreeze_vision,
        "split_sha256": digest(args.splits),
        "norm_sha256": digest(norm_file),
        "base_params": str(args.base_params.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
    }
    if args.previous:
        previous_manifest = json.loads((args.previous.parent / "manifest.json").read_text())
        if previous_manifest["stage"] != args.stage - 1 or previous_manifest["identity"] != identity:
            raise ValueError("Previous checkpoint belongs to a different stage or protocol")
        if args.previous.name != str(args.steps - 1):
            raise ValueError("Previous checkpoint must be the stage-end checkpoint")
    manifest = {
        "stage": args.stage,
        "storage": storage,
        "identity": identity,
        "episode_order": order,
        "current_frames": current_count,
        "replay_start_offsets": buffers,
        "replay_label_frames_upper_bound": sum(map(len, buffers.values())) * 10,
        "previous": str(args.previous.resolve()) if args.previous else None,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": os.sys.executable,
    }
    if args.mode == "plan":
        print(json.dumps(manifest, indent=2))
        return

    import flax.nnx as nnx

    from openpi.shared import nnx_utils
    from openpi.training import config
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    c = config.get_config("cl_pi05_t1_lora_smoke")
    freeze = c.model.get_freeze_filter()
    if not args.unfreeze_vision:
        freeze = nnx.Any(freeze, nnx_utils.PathRegex("PaliGemma/img/.*"))
    base = dataclasses.replace(
        c.data.base_config,
        dataset_root=str(args.dataset_root.resolve()),
        episode_ids=tuple(order),
        frame_indices=tuple(indices) if buffers else None,
        current_frame_count=current_count if buffers else None,
    )
    data = dataclasses.replace(
        c.data,
        base_config=base,
        assets=config.AssetsConfig(assets_dir=str(args.norm_assets.resolve()), asset_id="physical-intelligence/libero"),
    )
    c = dataclasses.replace(
        c,
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
        if saved["identity"] != identity or saved["stage"] != args.stage:
            raise ValueError("Evaluation configuration differs from checkpoint manifest")
        from openpi.policies.policy_config import create_trained_policy
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer

        policy = create_trained_policy(c, args.checkpoint, sample_kwargs={"num_steps": 10})
        WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata={"checkpoint": str(args.checkpoint.resolve()), "stage": args.stage, "identity": identity},
        ).serve_forever()
        return

    # Validate LoRA presence before allowing a cross-stage load.
    if args.previous:
        import orbax.checkpoint as ocp

        metadata = ocp.PyTreeCheckpointer().metadata(args.previous / "params")
        from flax.traverse_util import flatten_dict
        import jax

        expected = jax.eval_shape(lambda: nnx.state(c.model.create(jax.random.key(args.seed)))).to_pure_dict()
        expected_lora = {k for k in flatten_dict(expected, sep="/") if "lora" in k}
        available = flatten_dict(metadata["params"], sep="/")
        if not expected_lora.issubset(available):
            raise ValueError("Previous checkpoint is missing LoRA parameters")
    manifest_path = c.checkpoint_dir / "manifest.json"
    if args.resume:
        saved = json.loads(manifest_path.read_text())
        if saved["identity"] != identity or saved["stage"] != args.stage:
            raise ValueError("Resume configuration mismatch")
    elif c.checkpoint_dir.exists():
        raise FileExistsError(c.checkpoint_dir)

    from scripts import train

    train.init_logging()
    # The trainer creates its checkpoint directory before emitting the manifest.
    train.main(c, run_manifest=manifest)


if __name__ == "__main__":
    main()
