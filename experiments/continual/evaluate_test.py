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


def test_evaluation_preserves_venv_path_and_libero_cli(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    from experiments.continual import evaluate
    from experiments.continual import protocol

    checkpoint = tmp_path / "checkpoints/group/config/T1/99"
    checkpoint.mkdir(parents=True)
    manifest = {"stage": 1, "identity": {"steps": 100, "split_sha256": "split"}}
    (checkpoint.parent / "manifest.json").write_text(json.dumps(manifest))
    for directory in ("logs", "reports"):
        (tmp_path / directory).mkdir()
    simulator = tmp_path / "venv/bin/python"
    simulator.parent.mkdir(parents=True)
    simulator.symlink_to(sys.executable)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--sim-python",
            str(simulator),
            "--norm-assets",
            str(tmp_path / "norm"),
            "--storage-root",
            str(tmp_path),
            "--port",
            "0",
        ],
    )
    monkeypatch.setattr(evaluate, "configure", lambda *a, **kw: {"root": str(tmp_path)})
    monkeypatch.setattr(protocol, "digest", lambda path: "split")
    monkeypatch.setattr(evaluate, "server_command", lambda *a: ["policy-server"])
    monkeypatch.setattr(evaluate, "wait_for_server", lambda *a: None)
    process = SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr(evaluate.subprocess, "Popen", lambda *a, **kw: process)
    commands = []
    monkeypatch.setattr(evaluate.subprocess, "run", lambda command, **kw: commands.append(command))
    evaluate.main()
    (command,) = commands
    assert command[0] == str(simulator)
    assert command[0] != str(simulator.resolve())
    assert command[1:4] == ["-B", "-m", "examples.libero.main"]
    assert command[command.index("--args.task-keys") + 1] == "T1"
    assert command[command.index("--args.num-trials-per-task") + 1] == "40"
    assert all(value.startswith("--args.") for value in command if value.startswith("--"))


def test_readiness_uses_http_health_endpoint(monkeypatch):
    from types import SimpleNamespace

    from experiments.continual import evaluate

    calls = []
    response = SimpleNamespace(status=200, read=lambda: b"OK\n")
    connection = SimpleNamespace(
        request=lambda *args: calls.append(args),
        getresponse=lambda: response,
        close=lambda: calls.append(("closed",)),
    )
    monkeypatch.setattr(evaluate.http.client, "HTTPConnection", lambda *a, **kw: connection)
    evaluate.wait_for_server(SimpleNamespace(poll=lambda: None), 8002)
    assert calls == [("GET", "/healthz"), ("closed",)]
