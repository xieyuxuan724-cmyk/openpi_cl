"""Validate all four LIBERO scenes and the evaluation client without loading a policy."""

import argparse
import json
import os
from pathlib import Path

from experiments.continual.protocol import read_splits
from experiments.continual.storage import DEFAULT_ROOT
from experiments.continual.storage import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root", type=Path, default=Path(os.environ.get("OPENPI_CL_STORAGE_ROOT", DEFAULT_ROOT))
    )
    parser.add_argument("--splits", type=Path, default=Path("cl/splits.json"))
    parser.add_argument("--renderer", choices=["osmesa", "egl"], default="osmesa")
    parser.add_argument("--output-name", default="libero_sim_smoke")
    args = parser.parse_args()
    storage = configure(args.storage_root, minimum_gib=1)
    root = Path(storage["root"])
    os.environ["LIBERO_CONFIG_PATH"] = str(root / "cache/libero")
    os.environ["MUJOCO_GL"] = args.renderer
    os.environ["PYOPENGL_PLATFORM"] = args.renderer
    os.environ.setdefault("PYNPUT_BACKEND", "dummy")
    if args.renderer == "osmesa":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output = root / "reports" / args.output_name
    if not output.resolve().is_relative_to(root / "reports"):
        parser.error("Output must stay inside the reports directory")
    output.mkdir(exist_ok=False)

    import imageio.v2 as imageio
    from libero.libero import benchmark
    import numpy as np

    # Import the real evaluation entrypoint, including its WebSocket client.
    from examples.libero.main import LIBERO_DUMMY_ACTION
    from examples.libero.main import _get_libero_env

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    by_text = {suite.get_task(index).language: index for index in range(suite.n_tasks)}
    rows = []
    for key, item in read_splits(args.splits)["tasks"].items():
        index = by_text[item["task_text"]]
        task = suite.get_task(index)
        initial_states = suite.get_task_init_states(index)
        if len(initial_states) < 50:
            raise ValueError(f"Expected at least 50 evaluation initial states for {key}")
        env, _ = _get_libero_env(task, 256, 7)
        try:
            env.reset()
            obs = env.set_init_state(initial_states[10])
            for _ in range(12):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            for image_key in ("agentview_image", "robot0_eye_in_hand_image"):
                image = np.asarray(obs[image_key])
                if image.shape != (256, 256, 3) or not np.isfinite(image).all() or image.std() < 1:
                    raise ValueError(f"Invalid rendered image for {key}: {image_key}")
            imageio.imwrite(output / f"{key}.png", np.asarray(obs["agentview_image"])[::-1, ::-1])
            rows.append(
                {
                    "task": key,
                    "suite_index": index,
                    "initial_states": len(initial_states),
                    "steps": 12,
                    "image_shape": list(obs["agentview_image"].shape),
                    "status": "passed",
                }
            )
            print(json.dumps(rows[-1]), flush=True)
        finally:
            env.close()
    (output / "report.json").write_text(
        json.dumps(
            {
                "renderer": args.renderer,
                "tasks": rows,
                "scope": "Scene loading, reset, initial states, image rendering, stepping and evaluation imports; no policy success measurement.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
