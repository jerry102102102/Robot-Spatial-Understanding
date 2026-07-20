"""Deterministic negative controls for evidence-integrity and abstention tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .errors import IntegrityError, SchemaError
from .simulation import SimulationRun, evaluate_completeness
from .util import load_json, sha256_file, sha256_json, write_deterministic_npz, write_json


CORRUPTION_KINDS = frozenset(
    {
        "dropped-frame",
        "out-of-order",
        "wrong-id",
        "stale-tail",
        "terminal-status-tamper",
        "digest-tamper",
    }
)


def _manifest_digest(manifest: dict[str, Any]) -> str:
    return sha256_json({key: value for key, value in manifest.items() if key != "manifest_sha256"})


def corrupt_run(
    source: str | Path,
    output: str | Path,
    *,
    kind: str,
    channel: str = "pose",
) -> Path:
    if kind not in CORRUPTION_KINDS:
        raise SchemaError(f"unknown corruption kind {kind!r}; expected one of {sorted(CORRUPTION_KINDS)}")
    run = SimulationRun.load(source)
    target = Path(output)
    if target.exists():
        raise IntegrityError(f"output path already exists: {target}")
    shutil.copytree(run.root, target)
    manifest = load_json(target / "run.json")
    manifest["derived_from"] = {
        "run_id": run.manifest["run_id"],
        "run_manifest_sha256": run.digest,
        "corruption_kind": kind,
    }
    manifest["run_id"] = f"{run.manifest['run_id']}/corrupt/{kind}"
    try:
        if kind == "terminal-status-tamper":
            events_path = target / manifest["events"]["path"]
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "time_s": manifest["interval"]["end_s"],
                            "type": "action_status",
                            "status": "succeeded",
                            "producer": "corruption-control",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            manifest["events"]["sha256"] = sha256_file(events_path)
            manifest["events"]["count"] += 1
        else:
            channel_record = manifest["channels"].get(channel)
            if not isinstance(channel_record, dict) or channel_record.get("status") != "available":
                raise SchemaError(f"corruption kind {kind!r} requires available channel {channel!r}")
            stream_path = target / channel_record["path"]
            with np.load(stream_path, allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name]) for name in archive.files}
            sample_count = len(arrays["time_s"])
            if sample_count < 2:
                raise SchemaError(f"corruption kind {kind!r} requires at least two samples")
            sample_arrays = {
                name
                for name, values in arrays.items()
                if values.ndim > 0 and values.shape[0] == sample_count and name not in {"joint_ids", "entity_ids", "sensor_ids"}
            }
            if kind == "dropped-frame":
                remove = sample_count // 2
                for name in sample_arrays:
                    arrays[name] = np.delete(arrays[name], remove, axis=0)
            elif kind == "stale-tail":
                keep = max(1, sample_count * 2 // 3)
                for name in sample_arrays:
                    arrays[name] = arrays[name][:keep]
            elif kind == "out-of-order":
                arrays["time_s"] = arrays["time_s"].copy()
                arrays["time_s"][0], arrays["time_s"][1] = arrays["time_s"][1], arrays["time_s"][0]
            elif kind == "wrong-id":
                identifier_array = next(
                    (name for name in ("entity_ids", "joint_ids", "sensor_ids", "body_a") if name in arrays),
                    None,
                )
                if identifier_array is None or arrays[identifier_array].size == 0:
                    raise SchemaError(f"channel {channel!r} has no identity array to corrupt")
                arrays[identifier_array] = arrays[identifier_array].copy()
                arrays[identifier_array].flat[0] = f"wrong/{arrays[identifier_array].flat[0]}"
            elif kind == "digest-tamper":
                with stream_path.open("ab") as handle:
                    handle.write(b"tamper")
                return target
            write_deterministic_npz(stream_path, arrays)
            channel_record["sha256"] = sha256_file(stream_path)
        manifest["manifest_sha256"] = _manifest_digest(manifest)
        write_json(target / "run.json", manifest)
        completeness = evaluate_completeness(target, manifest)
        write_json(target / "completeness.json", completeness)
        SimulationRun.load(target)
        return target
    except Exception:
        if kind != "digest-tamper":
            shutil.rmtree(target, ignore_errors=True)
        raise
