#!/usr/bin/env python3
"""Cross-check measured mesh bounds and metrics against independent trimesh."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any


class CrosscheckError(ValueError):
    """A cross-engine geometry setup or comparison error."""


def json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def read_model(path: Path) -> dict[str, Any]:
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CrosscheckError(f"cannot read model artifact {path}: {error}") from error
    if not isinstance(model, dict) or model.get("schema_version") != "robot-spatial.v2":
        raise CrosscheckError("crosscheck requires robot-spatial.v2 model.json")
    return model


def crosscheck(model_path: Path, absolute_tolerance: float, relative_tolerance: float) -> dict[str, Any]:
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise CrosscheckError("tolerances must be non-negative")
    try:
        import numpy as np
        import trimesh
    except ImportError as error:
        raise CrosscheckError("trimesh and NumPy are required in the active Python environment") from error
    model = read_model(model_path)
    results: list[dict[str, Any]] = []
    discrepancies: list[dict[str, Any]] = []
    maximum_bound_error = 0.0
    maximum_area_error = 0.0
    maximum_volume_error = 0.0
    for frame_name, record in sorted(model["geometry_analysis"].items()):
        if record.get("geometry_type") != "mesh":
            continue
        if record.get("status") != "measured":
            raise CrosscheckError(f"mesh {frame_name!r} was not measured in the candidate artifact")
        source = record["source"]
        source_path = Path(source["path"])
        raw = source_path.read_bytes()
        source_sha = hashlib.sha256(raw).hexdigest()
        if source_sha != source["sha256"]:
            raise CrosscheckError(f"mesh digest changed for {source_path}")
        loaded = trimesh.load(str(source_path), force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise CrosscheckError(f"trimesh did not produce one mesh for {source_path}")
        vertices = np.asarray(loaded.vertices, dtype=float) * np.asarray(source["declared_scale_xyz"], dtype=float)
        oracle = trimesh.Trimesh(vertices=vertices, faces=np.asarray(loaded.faces), process=False)
        topology_oracle = oracle.copy()
        topology_oracle.process(validate=True)
        expected_bounds = record["bounds_in_geometry_frame"]
        oracle_minimum = oracle.bounds[0].tolist()
        oracle_maximum = oracle.bounds[1].tolist()
        bound_error = max(
            abs(float(expected_bounds[key][axis]) - float(oracle_value[axis]))
            for key, oracle_value in (("min_xyz_m", oracle_minimum), ("max_xyz_m", oracle_maximum))
            for axis in range(3)
        )
        area_error = abs(float(record["surface_area_m2"]) - float(oracle.area))
        volume_error: float | None = None
        if record.get("volume_trust") == "exact_for_closed_consistently_oriented_triangle_surface":
            volume_error = abs(float(record["volume_m3"]) - abs(float(oracle.volume)))
        bound_limit = absolute_tolerance + relative_tolerance * max(1.0, *(abs(value) for value in [*oracle_minimum, *oracle_maximum]))
        area_limit = absolute_tolerance + relative_tolerance * max(1.0, abs(float(oracle.area)))
        volume_limit = absolute_tolerance + relative_tolerance * max(1.0, abs(float(oracle.volume)))
        reasons: list[str] = []
        if bound_error > bound_limit:
            reasons.append("bounds")
        if area_error > area_limit:
            reasons.append("surface_area")
        if volume_error is not None and volume_error > volume_limit:
            reasons.append("volume")
        if bool(record["topology"]["watertight"]) != bool(topology_oracle.is_watertight):
            reasons.append("watertight_classification")
        result = {
            "geometry_frame": frame_name,
            "mesh_path": str(source_path.resolve()),
            "mesh_sha256": source_sha,
            "candidate_format": source["format"],
            "oracle_vertex_count_unprocessed": int(len(oracle.vertices)),
            "oracle_vertex_count_processed": int(len(topology_oracle.vertices)),
            "oracle_face_count": int(len(oracle.faces)),
            "maximum_bound_coordinate_error_m": bound_error,
            "surface_area_error_m2": area_error,
            "volume_error_m3": volume_error,
            "candidate_watertight": bool(record["topology"]["watertight"]),
            "oracle_watertight": bool(topology_oracle.is_watertight),
            "status": "passed" if not reasons else "failed",
            "failed_metrics": reasons,
        }
        results.append(result)
        maximum_bound_error = max(maximum_bound_error, bound_error)
        maximum_area_error = max(maximum_area_error, area_error)
        if volume_error is not None:
            maximum_volume_error = max(maximum_volume_error, volume_error)
        if reasons:
            discrepancies.append(result)
    return {
        "schema_version": "robot-spatial-cross-engine-mesh.v1",
        "status": "passed" if not discrepancies else "failed",
        "robot": model["robot"]["name"],
        "source_model": {
            "path": str(model_path.resolve()),
            "spatial_truth_sha256": hashlib.sha256(json.dumps({key: value for key, value in model.items() if key != "artifacts"}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest(),
            "urdf_sha256": model["source"]["sha256"],
        },
        "engines": {
            "candidate": {"name": "robot-spatial", "schema_version": "robot-spatial.v2"},
            "oracle": {
                "name": "trimesh",
                "version": importlib.metadata.version("trimesh"),
                "numpy_version": np.__version__,
            },
            "python": {"version": platform.python_version(), "platform": platform.platform()},
        },
        "coverage": {
            "measured_mesh_count": len(results),
            "compared_metrics": ["local vertex bounds", "surface area", "watertight classification", "volume when candidate trust is trustworthy"],
            "not_compared": ["root-frame placement", "principal axes", "shape heuristic", "triangle collision"],
        },
        "tolerances": {"absolute": absolute_tolerance, "relative": relative_tolerance},
        "maximum_bound_coordinate_error_m": maximum_bound_error,
        "maximum_surface_area_error_m2": maximum_area_error,
        "maximum_volume_error_m3": maximum_volume_error,
        "discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies,
        "mesh_results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-9)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = crosscheck(args.model, args.absolute_tolerance, args.relative_tolerance)
    except (OSError, CrosscheckError, ValueError) as error:
        print(json_dump({"status": "error", "error": str(error)}), file=sys.stderr, end="")
        return 2
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json_dump(result), encoding="utf-8")
    print(json_dump(result), end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
