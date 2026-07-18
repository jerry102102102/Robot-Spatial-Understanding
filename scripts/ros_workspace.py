#!/usr/bin/env python3
"""Deterministic ROS source-workspace discovery and Xacro package lookup support."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse


PACKAGE_URI = re.compile(r"package://([A-Za-z0-9_.+-]+)/")
IGNORED_DIRECTORY_NAMES = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".pytest_cache",
    "__pycache__",
    "build",
    "install",
    "log",
})


class WorkspaceError(ValueError):
    """A ROS workspace is missing, ambiguous, or cannot be provenance-bound."""


def sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise WorkspaceError(f"cannot hash {path}: {error}") from error


def _package_name(package_xml: Path) -> str:
    try:
        root = ET.fromstring(package_xml.read_bytes())
    except (OSError, ET.ParseError) as error:
        raise WorkspaceError(f"cannot parse ROS package manifest {package_xml}: {error}") from error
    if root.tag != "package":
        raise WorkspaceError(f"ROS package manifest root must be <package>: {package_xml}")
    names = [element.text.strip() for element in root.findall("name") if element.text and element.text.strip()]
    if len(names) != 1:
        raise WorkspaceError(f"ROS package manifest must contain exactly one non-empty <name>: {package_xml}")
    return names[0]


def nearest_package_root(path: Path) -> Path | None:
    """Return the closest ancestor containing package.xml, if any."""
    current = path.resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        if (directory / "package.xml").is_file():
            return directory
    return None


def discover_packages(roots: Iterable[Path]) -> dict[str, Path]:
    """Scan explicit source roots and reject ambiguous package names."""
    manifests: set[Path] = set()
    normalized_roots: list[Path] = []
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        if not root.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {root}")
        if root not in normalized_roots:
            normalized_roots.append(root)
    if not normalized_roots:
        raise WorkspaceError("at least one workspace root is required")
    for root in normalized_roots:
        for directory, child_names, file_names in os.walk(root, followlinks=False):
            child_names[:] = sorted(
                name for name in child_names
                if name not in IGNORED_DIRECTORY_NAMES and not (Path(directory) / name).is_symlink()
            )
            if "package.xml" in file_names:
                manifests.add((Path(directory) / "package.xml").resolve())
    packages: dict[str, Path] = {}
    for manifest in sorted(manifests):
        name = _package_name(manifest)
        directory = manifest.parent.resolve()
        previous = packages.get(name)
        if previous is not None and previous != directory:
            raise WorkspaceError(
                f"ambiguous ROS package {name!r}: both {previous} and {directory}; narrow --workspace-root"
            )
        packages[name] = directory
    return dict(sorted(packages.items()))


def _tree_file_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for directory, child_names, file_names in os.walk(root, followlinks=False):
        child_names[:] = sorted(
            name for name in child_names
            if name not in IGNORED_DIRECTORY_NAMES and not (Path(directory) / name).is_symlink()
        )
        for name in sorted(file_names):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                resolved = path.resolve()
                if not resolved.is_file():
                    raise WorkspaceError(f"source symlink does not resolve to a regular file: {path}")
                records.append({
                    "path": relative,
                    "kind": "symlink",
                    "target": os.readlink(path),
                    "resolved_path": str(resolved),
                    "sha256": sha256_path(resolved),
                })
            elif path.is_file():
                records.append({"path": relative, "kind": "file", "sha256": sha256_path(path)})
    return records


def tree_manifest(root: Path, *, package_name: str | None = None) -> dict[str, Any]:
    """Hash a conservative source tree; identity excludes its machine-local root path."""
    resolved = root.resolve()
    if not resolved.is_dir():
        raise WorkspaceError(f"source tree root is not a directory: {resolved}")
    files = _tree_file_records(resolved)
    identity = {
        "package_name": package_name,
        "files": files,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "package_name": package_name,
        "root": str(resolved),
        "file_count": len(files),
        "tree_sha256": digest,
        "files": files,
    }


def package_references_from_urdf(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise WorkspaceError(f"cannot inspect package URIs in {path}: {error}") from error
    return sorted(set(PACKAGE_URI.findall(text)))


def normalize_expanded_urdf(path: Path, package_map: dict[str, Path]) -> dict[str, Any]:
    """Remove non-semantic comments and rewrite package-owned absolute mesh paths."""
    try:
        raw = path.read_bytes()
        root = ET.fromstring(raw)
    except (OSError, ET.ParseError) as error:
        raise WorkspaceError(f"cannot normalize expanded URDF {path}: {error}") from error
    roots = sorted(
        ((name, directory.resolve()) for name, directory in package_map.items()),
        key=lambda item: len(str(item[1])),
        reverse=True,
    )
    rewrites: list[dict[str, str]] = []
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        for attribute, value in list(element.attrib.items()):
            if attribute not in {"filename", "url"}:
                continue
            candidate: Path | None = None
            if value.startswith("file://"):
                parsed = urlparse(value)
                if parsed.netloc in {"", "localhost"}:
                    candidate = Path(unquote(parsed.path)).resolve()
            elif Path(value).is_absolute():
                candidate = Path(value).resolve()
            if candidate is None:
                continue
            for package_name, package_root in roots:
                try:
                    relative = candidate.relative_to(package_root)
                except ValueError:
                    continue
                replacement = f"package://{package_name}/{relative.as_posix()}"
                element.set(attribute, replacement)
                rewrites.append({"attribute": attribute, "from": value, "to": replacement})
                break
    ET.indent(root, space="  ")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="robot-spatial-normalized-",
            suffix=".urdf",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        ET.ElementTree(root).write(temporary_path, encoding="utf-8", xml_declaration=True)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {
        "schema_version": "robot-spatial-urdf-normalization.v1",
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "output_sha256": sha256_path(path),
        "removed_xml_comment_count": raw.count(b"<!--"),
        "rewritten_package_path_count": len(rewrites),
        "rewritten_packages": sorted({record["to"].removeprefix("package://").split("/", 1)[0] for record in rewrites}),
        "rewrites": rewrites,
        "meaning": "XML comments are non-semantic; package-owned absolute filename/url attributes are rewritten to equivalent package:// URIs",
    }


def write_ament_index_shim(directory: Path) -> Path:
    """Create the minimal ament_index_python API needed by Xacro $(find ...)."""
    package = directory / "ament_index_python"
    package.mkdir(parents=True, exist_ok=False)
    (package / "__init__.py").write_text(
        "from .packages import PackageNotFoundError\n",
        encoding="utf-8",
    )
    (package / "packages.py").write_text(
        '''from __future__ import annotations
import json
import os
from pathlib import Path

class PackageNotFoundError(LookupError):
    pass

def _mapping():
    return json.loads(os.environ["ROBOT_SPATIAL_PACKAGE_MAP_JSON"])

def _record(name):
    path = os.environ.get("ROBOT_SPATIAL_PACKAGE_LOOKUP_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(name + "\\n")

def get_package_share_directory(name):
    mapping = _mapping()
    if name not in mapping:
        raise PackageNotFoundError(name)
    _record(name)
    return mapping[name]

def get_package_share_path(name):
    return Path(get_package_share_directory(name))

def get_package_prefix(name):
    return str(Path(get_package_share_directory(name)).parent)

def get_packages_with_prefixes():
    return {name: str(Path(path).parent) for name, path in _mapping().items()}
''',
        encoding="utf-8",
    )
    return directory


def xacro_environment(package_map: dict[str, Path], shim_root: Path, lookup_log: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["ROBOT_SPATIAL_PACKAGE_MAP_JSON"] = json.dumps(
        {name: str(path.resolve()) for name, path in sorted(package_map.items())},
        sort_keys=True,
    )
    environment["ROBOT_SPATIAL_PACKAGE_LOOKUP_LOG"] = str(lookup_log.resolve())
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(shim_root.resolve()) + (os.pathsep + previous if previous else "")
    return environment


def read_package_lookups(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return sorted(set(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
    except OSError as error:
        raise WorkspaceError(f"cannot read Xacro package lookup log {path}: {error}") from error
