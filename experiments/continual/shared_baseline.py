"""Reuse a measured common T1 baseline without rewriting its source records."""

import json
from pathlib import Path

from experiments.continual.protocol import validate_previous_identity


def align_shared_t1(rows):
    baseline = [row for row in rows if row["stage"] == 1]
    later = [row for row in rows if row["stage"] > 1]
    if not baseline or not later:
        raise ValueError("Shared baseline requires T1 and later-stage records")
    source_paths = {row["checkpoint"] for row in baseline}
    identities = {json.dumps(row["protocol"], sort_keys=True) for row in later}
    if len(source_paths) != 1 or len(identities) != 1:
        raise ValueError("Ambiguous shared baseline or later training protocols")
    source = Path(next(iter(source_paths))).resolve()
    manifest = json.loads((source.parent / "manifest.json").read_text())
    identity = later[0]["protocol"]
    validate_previous_identity(manifest, identity, 2, shared_t1=True)
    for row in baseline:
        if row["task_key"] != "T1" or row["protocol"] != manifest["identity"]:
            raise ValueError("Shared baseline records do not match their checkpoint")
    stage2 = {row["checkpoint"] for row in later if row["stage"] == 2}
    if len(stage2) != 1:
        raise ValueError("Expected exactly one T2 checkpoint")
    child = json.loads((Path(next(iter(stage2))).parent / "manifest.json").read_text())
    if not child.get("shared_t1") or Path(child["previous"]).resolve() != source:
        raise ValueError("Replay T2 was not initialized from this shared T1")
    if child["identity"] != identity:
        raise ValueError("Replay T2 records have an inconsistent protocol")
    # Only the in-memory analysis view is aligned. Raw JSONL files remain intact.
    aligned = [dict(row, protocol=identity) if row["stage"] == 1 else row for row in rows]
    provenance = {
        "checkpoint": str(source),
        "source_protocol": manifest["identity"],
        "note": "Identical measured T1 baseline shared by both branches; not a new replay rollout.",
    }
    return aligned, provenance
