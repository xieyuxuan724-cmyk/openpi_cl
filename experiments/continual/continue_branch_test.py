import json

import pytest

from experiments.continual.continue_branch import committed
from experiments.continual.continue_branch import evaluation_files
from experiments.continual.continue_branch import verify_stage
from experiments.continual.protocol import validate_previous_identity
from experiments.continual.shared_baseline import align_shared_t1


def identities():
    source = {
        "method": "sequential",
        "replay_budget": None,
        "replay_fraction": None,
        "seed": 42,
        "steps": 10000,
        "batch_size": 8,
        "normalization": "quantile",
    }
    return source, dict(source, method="replay", replay_fraction=0.2)


def test_common_t1_requires_explicit_matching_transition():
    source, target = identities()
    previous = {"stage": 1, "identity": source}
    validate_previous_identity(previous, target, 2, shared_t1=True)
    with pytest.raises(ValueError, match="protocol"):
        validate_previous_identity(previous, target, 2)
    for change in ({"seed": 43}, {"batch_size": 4}, {"normalization": "zscore"}):
        with pytest.raises(ValueError, match="identical training"):
            validate_previous_identity(previous, dict(target, **change), 2, shared_t1=True)
    with pytest.raises(ValueError, match="only supported"):
        validate_previous_identity({"stage": 2, "identity": source}, target, 3, shared_t1=True)
    with pytest.raises(ValueError, match="only supported"):
        validate_previous_identity(previous, dict(target, replay_fraction=0.1), 2, shared_t1=True)
    validate_previous_identity({"stage": 2, "identity": target}, target, 3)


def test_shared_analysis_keeps_raw_baseline_and_requires_actual_lineage(tmp_path):
    source, target = identities()
    first = tmp_path / "sequential/T1/9999"
    second = tmp_path / "replay/T2/9999"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first.parent / "manifest.json").write_text(json.dumps({"stage": 1, "identity": source}))
    child = {"stage": 2, "identity": target, "shared_t1": True, "previous": str(first)}
    child_file = second.parent / "manifest.json"
    child_file.write_text(json.dumps(child))
    rows = [
        {"stage": 1, "task_key": "T1", "protocol": source, "checkpoint": str(first)},
        {"stage": 2, "task_key": "T1", "protocol": target, "checkpoint": str(second)},
    ]
    aligned, provenance = align_shared_t1(rows)
    assert rows[0]["protocol"]["method"] == "sequential"
    assert aligned[0]["protocol"]["method"] == "replay"
    assert provenance["checkpoint"] == str(first)
    child_file.write_text(json.dumps(dict(child, previous=str(tmp_path / "other"))))
    with pytest.raises(ValueError, match="not initialized"):
        align_shared_t1(rows)


def test_partial_checkpoint_and_mismatched_resume_rejected(tmp_path):
    checkpoint = tmp_path / "9999"
    checkpoint.mkdir()
    assert not committed(checkpoint)
    meta = checkpoint / "_CHECKPOINT_METADATA"
    meta.write_text(json.dumps({"commit_timestamp_nsecs": None}))
    assert not committed(checkpoint)
    meta.write_text(json.dumps({"commit_timestamp_nsecs": 123}))
    assert committed(checkpoint)
    manifest = {"identity": {"seed": 42}, "stage": 2, "previous": "T1", "norm_sha256": {"T1": "hash"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    verify_stage(tmp_path, manifest)
    with pytest.raises(ValueError, match="previous"):
        verify_stage(tmp_path, dict(manifest, previous="other"))


def test_completed_evaluation_requires_exact_trials_and_metadata(tmp_path):
    checkpoint = tmp_path / "T2/9999"
    checkpoint.mkdir(parents=True)
    _, identity = identities()
    (checkpoint.parent / "manifest.json").write_text(
        json.dumps({"identity": identity, "stage": 2, "norm_sha256": {"T1": "h1", "T2": "h2"}})
    )
    directory = tmp_path / "results"
    directory.mkdir()
    for task, norm_hash in [("T1", "h1"), ("T2", "h2")]:
        rows = [
            {
                "protocol": identity,
                "stage": 2,
                "task_key": task,
                "checkpoint": str(checkpoint),
                "seed": 7,
                "suite": "libero_10",
                "replan_steps": 5,
                "max_steps": 520,
                "num_steps_wait": 10,
                "resize_size": 224,
                "normalization_task": task,
                "norm_sha256": norm_hash,
                "episode": i,
                "success": True,
            }
            for i in range(10, 50)
        ]
        (directory / f"{task}.jsonl").write_text("\n".join(map(json.dumps, rows)))
    assert len(evaluation_files(directory, checkpoint)) == 2
    path = directory / "T2.jsonl"
    original = path.read_text()
    path.write_text("\n".join(original.splitlines()[:-1]))
    with pytest.raises(ValueError, match="Incomplete"):
        evaluation_files(directory, checkpoint)
    path.write_text(original.replace('"h2"', '"wrong_hash"'))
    with pytest.raises(ValueError, match="metadata"):
        evaluation_files(directory, checkpoint)


def test_lora_metadata_accepts_nnx_wrappers_and_rejects_missing_or_wrong_shapes():
    from types import SimpleNamespace

    from experiments.continual.run import validate_lora_metadata

    leaf = SimpleNamespace(shape=(18, 32, 1024))
    expected = {"model/lora_a": leaf}
    validate_lora_metadata(expected, {"model/lora_a/value": leaf})
    validate_lora_metadata(expected, {"model/lora_a": leaf})
    with pytest.raises(ValueError, match="missing LoRA"):
        validate_lora_metadata(expected, {"model/kernel/value": leaf})
    with pytest.raises(ValueError, match="incompatible LoRA shape"):
        validate_lora_metadata(expected, {"model/lora_a/value": SimpleNamespace(shape=(18, 16, 1024))})


def test_shared_baseline_full_matrix_aggregation_and_paired_comparison(tmp_path, monkeypatch):
    import sys

    from experiments.continual import analyze
    from experiments.continual.compare import compare

    source, target = identities()
    first = tmp_path / "sequential/T1/9999"
    first.mkdir(parents=True)
    (first.parent / "manifest.json").write_text(json.dumps({"stage": 1, "identity": source}))
    second = tmp_path / "replay/T2/9999"
    second.mkdir(parents=True)
    (second.parent / "manifest.json").write_text(
        json.dumps({"stage": 2, "identity": target, "shared_t1": True, "previous": str(first)})
    )
    rows = [
        {
            "stage": stage,
            "task_key": task,
            "episode": episode,
            "success": True,
            "protocol": source if stage == 1 else target,
            "checkpoint": str(first if stage == 1 else tmp_path / f"replay/T{stage}/9999"),
            "seed": 7,
            "suite": "libero_10",
            "replan_steps": 5,
            "max_steps": 520,
            "num_steps_wait": 10,
            "resize_size": 224,
            "norm_sha256": task + "_hash",
            "normalization_task": task,
        }
        for stage in range(1, 5)
        for task in ("T1", "T2", "T3", "T4")[:stage]
        for episode in range(10, 50)
    ]
    inputs = tmp_path / "raw.jsonl"
    raw = "\n".join(map(json.dumps, rows))
    inputs.write_text(raw)
    output = tmp_path / "reports/replay_metrics.json"
    monkeypatch.setattr(analyze, "configure", lambda *a, **kw: {"root": str(tmp_path)})
    monkeypatch.setattr(
        sys, "argv", ["analyze", "metrics", "--output", str(output), "--results", str(inputs), "--shared-t1"]
    )
    analyze.main()
    report = json.loads(output.read_text())
    assert report["final_average_success"] == 1
    assert report["final_forgetting"] == 0
    assert report["shared_t1_baseline"]["checkpoint"] == str(first)
    assert inputs.read_text() == raw
    sequential_report = dict(report, protocol=source)
    paired = compare([sequential_report], [report])
    assert paired["paired_delta"]["final_average_success"]["mean"] == 0
