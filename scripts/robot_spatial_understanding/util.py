"""Canonical serialization, hashing, paths, and small math helpers."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .errors import IntegrityError, SchemaError


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SchemaError(f"cannot read JSON {path}: {error}") from error


def load_structured(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json(path)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - packaging installs PyYAML
            raise SchemaError("YAML input requires PyYAML; install the package dependencies") from error
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise SchemaError(f"cannot read YAML {path}: {error}") from error
    raise SchemaError(f"unsupported structured file {path}; expected .json, .yaml, or .yml")


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaError(f"{label} must be an array")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{label} must be a non-empty string")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SchemaError(f"{label} must be a finite number")
    return float(value)


def safe_relative_path(root: Path, relative: str, label: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise IntegrityError(f"{label} escapes artifact root: {relative!r}") from error
    return candidate


def ensure_new_directory(path: Path) -> None:
    if path.exists():
        raise IntegrityError(f"output path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()


def is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.resolve().relative_to(other.resolve())
        return True
    except ValueError:
        return False


FORBIDDEN_ORACLE_KEYS = frozenset(
    {
        "reward",
        "rewards",
        "success",
        "is_success",
        "official_success",
        "oracle",
        "oracle_result",
        "evaluator",
        "evaluator_result",
        "task_success",
    }
)


def find_forbidden_keys(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN_ORACLE_KEYS:
                found.append(child_path)
            found.extend(find_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_forbidden_keys(child, f"{path}[{index}]"))
    return found


def pair_matches(left: str, right: str, expected: Iterable[str]) -> bool:
    pair = list(expected)
    if len(pair) != 2:
        raise SchemaError("body pair must contain exactly two entity IDs")
    return {left, right} == {str(pair[0]), str(pair[1])}


def euclidean(left: Iterable[float], right: Iterable[float]) -> float:
    a, b = list(left), list(right)
    if len(a) != len(b):
        raise SchemaError("cannot measure vectors with different dimensions")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def quaternion_normalized(value: Iterable[float]) -> list[float]:
    quaternion = [float(component) for component in value]
    if len(quaternion) != 4:
        raise SchemaError("quaternion_xyzw must contain four values")
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm <= 1e-12 or not math.isfinite(norm):
        raise SchemaError("quaternion_xyzw has zero or non-finite norm")
    return [component / norm for component in quaternion]


def quaternion_angle(left: Iterable[float], right: Iterable[float]) -> float:
    a = quaternion_normalized(left)
    b = quaternion_normalized(right)
    dot = min(1.0, abs(sum(x * y for x, y in zip(a, b))))
    return 2.0 * math.acos(dot)


def quaternion_conjugate(value: Iterable[float]) -> list[float]:
    x, y, z, w = quaternion_normalized(value)
    return [-x, -y, -z, w]


def quaternion_multiply(left: Iterable[float], right: Iterable[float]) -> list[float]:
    ax, ay, az, aw = list(left)
    bx, by, bz, bw = list(right)
    return quaternion_normalized(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def rotate_vector(quaternion: Iterable[float], vector: Iterable[float]) -> list[float]:
    qx, qy, qz, qw = quaternion_normalized(quaternion)
    vx, vy, vz = [float(component) for component in vector]
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    ]


def relative_pose(
    reference_position: Iterable[float],
    reference_quaternion: Iterable[float],
    target_position: Iterable[float],
    target_quaternion: Iterable[float],
) -> tuple[list[float], list[float]]:
    inverse = quaternion_conjugate(reference_quaternion)
    delta = [float(target) - float(reference) for target, reference in zip(target_position, reference_position)]
    return rotate_vector(inverse, delta), quaternion_multiply(inverse, target_quaternion)


def runtime_identity() -> dict[str, str]:
    return {
        "pid_namespace": str(os.getpid()),
        "implementation": "robot-spatial-understanding",
    }


def write_deterministic_npz(path: Path, arrays: dict[str, Any]) -> None:
    """Write a NumPy-compatible NPZ without wall-clock timestamps."""
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - package dependency
        raise SchemaError("NPZ streams require NumPy; install the package dependencies") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
