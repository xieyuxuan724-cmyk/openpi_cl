from pathlib import Path

import pytest

from experiments.continual.evaluate import server_command


def test_server_command_reconstructs_protocol_and_selects_task():
    identity = {
        "protocol_version": 2,
        "method": "replay",
        "dataset_root": "dataset",
        "base_params": "base",
        "seed": 43,
        "batch_size": 8,
        "steps": 10000,
        "warmup": 500,
        "normalization": "quantile",
        "replay_budget": None,
        "replay_fraction": 0.2,
        "unfreeze_vision": True,
    }

    command = server_command(
        {"identity": identity, "stage": 3}, Path("checkpoint"), Path("norm"), Path("disk"), "T1", 8001
    )
    assert command[command.index("--eval-task") + 1] == "T1"
    assert command[command.index("--stage") + 1] == "3"
    assert command[command.index("--replay-fraction") + 1] == "0.2"
    assert "--previous" not in command
    assert "--unfreeze-vision" in command
    with pytest.raises(ValueError, match="already learned"):
        server_command({"identity": identity, "stage": 3}, Path("checkpoint"), Path("norm"), Path("disk"), "T4", 8001)
