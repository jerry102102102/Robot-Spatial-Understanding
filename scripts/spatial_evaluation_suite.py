#!/usr/bin/env python3
"""Grade a public multi-robot spatial-question suite with isolated private keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from spatial_evaluation import EvaluationError, json_dump, jsonl_dump, read_jsonl, verify_answers


PUBLIC_SCHEMA = "robot-spatial-evaluation-suite.v1"
PRIVATE_SCHEMA = "robot-spatial-evaluation-suite-key.v1"
REPORT_SCHEMA = "robot-spatial-evaluation-suite-report.v1"
BUILD_SCHEMA = "robot-spatial-evaluation-suite-build.v1"
EVALUATION_MANIFEST_SCHEMA = "robot-spatial-evaluation-manifest.v1"
CANDIDATE_INPUT_CONTEXT = "generated_context"
CANDIDATE_INPUT_RAW = "raw_sources"
RAW_TASK_SCHEMA = "robot-spatial-raw-source-task.v1"
REQUIRED_CONTEXT_FILES = (
    "agent-context.json",
    "agent-guide.md",
    "entity-cards.jsonl",
    "entity-index.json",
    "fact-index.json",
    "facts.jsonl",
    "context.md",
    "model.json",
)
OPTIONAL_CONTEXT_FILES = (
    "scene.svg",
    "articulation-grammar.json",
    "constraint-graph.json",
    "constraint-evaluation.json",
    "configuration-atlas.json",
    "concept-graph.json",
    "concept-language.rsl",
    "functional-model.json",
    "action-assurance.json",
)
OPTIONAL_CONTEXT_DIRECTORIES = ("render-atlas", "motion-atlas")
RAW_FORBIDDEN_FILES = set(REQUIRED_CONTEXT_FILES) | set(OPTIONAL_CONTEXT_FILES) | {
    "articulation-comparison.json",
    "constraint-graph.json",
    "constraint-evaluation.json",
    "constraint-solution.json",
    "configuration-atlas.json",
    "concept-graph.json",
    "concept-language.rsl",
    "functional-model.json",
}
RAW_PATH_OPTIONS = {"configuration_atlas_spec", "constraint_spec", "functional_spec", "semantics", "invariants", "observation_query", "observations", "package_map", "pose", "scene", "srdf"}
RAW_BOOLEAN_OPTIONS = {
    "include_workspace_samples",
    "inspect_meshes",
    "motion_atlas",
    "render",
    "surface_collisions",
}
RAW_INTEGER_OPTIONS = {"workspace_samples"}
RAW_NUMBER_OPTIONS = {"contact_tolerance_m", "motion_angular_step_rad", "motion_linear_step_m"}
RAW_STRING_OPTIONS = {"pose_name"}
PRIVATE_LEAK_MARKERS = (
    '"schema_version": "robot-spatial-answer-key.v1"',
    '"private_artifacts"',
)


class EvaluationSuiteError(ValueError):
    """An invalid suite, private key manifest, or public artifact boundary."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationSuiteError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationSuiteError(f"{label} must contain a JSON object")
    return value


def _string(value: dict[str, Any], field: str, label: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise EvaluationSuiteError(f"{label}.{field} must be a non-empty string")
    return result


def _resolve_under(base: Path, relative: str, label: str, must_exist: bool = True) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise EvaluationSuiteError(f"{label} must be relative to its declared root")
    root = base.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise EvaluationSuiteError(f"{label} escapes its declared root: {relative!r}") from error
    if must_exist and not resolved.is_file():
        raise EvaluationSuiteError(f"{label} does not exist: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_address_locations(value: Any) -> Any:
    """Replace machine-local locations when the same record carries a content digest."""
    if isinstance(value, list):
        return [_content_address_locations(item) for item in value]
    if not isinstance(value, dict):
        return value
    digest = value.get("sha256")
    content_bound = isinstance(digest, str) and len(digest) == 64
    result: dict[str, Any] = {}
    for key, item in value.items():
        if content_bound and key in {"path", "urdf"}:
            result[key] = f"sha256:{digest}"
        else:
            result[key] = _content_address_locations(item)
    return result


def _empty_output_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise EvaluationSuiteError(f"{label} must not exist or must be an empty directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _copy_public_file(source: Path, destination: Path, sanitize_json: bool = False) -> None:
    if not source.is_file():
        raise EvaluationSuiteError(f"public source artifact does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if sanitize_json:
        destination.write_text(json_dump(_content_address_locations(_read_json(source, str(source)))), encoding="utf-8")
    else:
        shutil.copy2(source, destination)


def _copy_raw_source_tree(source_root: Path, destination_root: Path) -> list[Path]:
    if not source_root.is_dir() or source_root.is_symlink():
        raise EvaluationSuiteError(f"raw source root must be a real directory: {source_root}")
    copied: list[Path] = []
    forbidden = RAW_FORBIDDEN_FILES
    for source in sorted(source_root.rglob("*")):
        if source.is_symlink():
            raise EvaluationSuiteError(f"raw source tree must not contain symlinks: {source}")
        if not source.is_file():
            continue
        relative_source = source.relative_to(source_root)
        generated_directories = sorted(set(relative_source.parts) & set(OPTIONAL_CONTEXT_DIRECTORIES))
        if generated_directories:
            raise EvaluationSuiteError(
                f"source-only task contains generated context directory {generated_directories[0]!r}: {source}"
            )
        if source.name in forbidden:
            raise EvaluationSuiteError(
                f"source-only task contains generated context artifact {source.name!r}: {source}"
            )
        destination = destination_root / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    if not copied:
        raise EvaluationSuiteError(f"raw source root contains no files: {source_root}")
    return copied


def _relative_directory_under(base: Path, relative: str, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise EvaluationSuiteError(f"{label} must be relative to its declared root")
    root = base.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise EvaluationSuiteError(f"{label} escapes its declared root: {relative!r}") from error
    if not resolved.is_dir():
        raise EvaluationSuiteError(f"{label} does not exist: {resolved}")
    return resolved


def _contains_xacro_elements(path: Path) -> bool:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise EvaluationSuiteError(f"cannot parse raw task XML {path}: {error}") from error
    for element in root.iter():
        if isinstance(element.tag, str) and element.tag.startswith("{"):
            namespace = element.tag[1:].split("}", 1)[0].lower()
            if "xacro" in namespace:
                return True
    return False


def _validate_raw_task_spec(task_spec: dict[str, Any], task_root: Path, label: str) -> None:
    allowed_top_level = {
        "schema_version",
        "workflow",
        "input_format",
        "entrypoint",
        "workspace_roots",
        "expansion",
        "export_options",
        "ros_observation_adapter",
        "ros_action_adapters",
        "articulation_sources",
        "articulation_comparisons",
        "constraint_graphs",
        "configuration_atlases",
        "action_assurances",
    }
    if set(task_spec) - allowed_top_level:
        raise EvaluationSuiteError(
            f"{label} contains unsupported fields: {sorted(set(task_spec) - allowed_top_level)}"
        )
    if task_spec.get("schema_version") != RAW_TASK_SCHEMA:
        raise EvaluationSuiteError(f"{label} must use schema_version {RAW_TASK_SCHEMA}")
    workflow = task_spec.get("workflow", "direct")
    if workflow not in {"direct", "prepare"}:
        raise EvaluationSuiteError(f"{label}.workflow must be 'direct' or 'prepare'")
    input_format = task_spec.get("input_format")
    if input_format not in {"urdf", "xacro"}:
        raise EvaluationSuiteError(f"{label}.input_format must be 'urdf' or 'xacro'")
    entrypoint = _string(task_spec, "entrypoint", label)
    entrypoint_path = _resolve_under(task_root, entrypoint, f"{label}.entrypoint")
    try:
        entrypoint_path.relative_to((task_root / "source").resolve())
    except ValueError as error:
        raise EvaluationSuiteError(f"{label}.entrypoint must be inside source/") from error
    contains_xacro = _contains_xacro_elements(entrypoint_path)
    if input_format == "urdf":
        if entrypoint_path.suffix.lower() != ".urdf":
            raise EvaluationSuiteError(f"{label}.entrypoint must be an expanded .urdf file")
        if contains_xacro:
            raise EvaluationSuiteError(f"{label}.entrypoint contains unexpanded Xacro elements but input_format is 'urdf'")
        if "expansion" in task_spec:
            raise EvaluationSuiteError(f"{label}.expansion is permitted only for input_format 'xacro'")
    else:
        if not contains_xacro:
            raise EvaluationSuiteError(f"{label}.entrypoint has no executable Xacro elements")
        expansion = task_spec.get("expansion")
        if not isinstance(expansion, dict):
            raise EvaluationSuiteError(f"{label}.expansion must be an object for input_format 'xacro'")
        allowed_expansion = {"executable", "mappings", "output"}
        unknown_expansion = set(expansion) - allowed_expansion
        if unknown_expansion:
            raise EvaluationSuiteError(
                f"{label}.expansion contains unsupported fields: {sorted(unknown_expansion)}"
            )
        executable = expansion.get("executable")
        if not isinstance(executable, str) or not executable or "/" in executable or "\\" in executable:
            raise EvaluationSuiteError(
                f"{label}.expansion.executable must be a portable command name supplied by the evaluator"
            )
        output = expansion.get("output")
        if (
            not isinstance(output, str)
            or not output
            or Path(output).name != output
            or not output.lower().endswith(".urdf")
        ):
            raise EvaluationSuiteError(
                f"{label}.expansion.output must be a basename ending in .urdf"
            )
        mappings = expansion.get("mappings", [])
        if (
            not isinstance(mappings, list)
            or any(not isinstance(mapping, str) or ":=" not in mapping or not mapping.split(":=", 1)[0] for mapping in mappings)
            or len(set(mappings)) != len(mappings)
        ):
            raise EvaluationSuiteError(
                f"{label}.expansion.mappings must be a unique array of name:=value strings"
            )

    articulation_sources = task_spec.get("articulation_sources")
    articulation_comparisons = task_spec.get("articulation_comparisons")
    constraint_graphs = task_spec.get("constraint_graphs")
    configuration_atlases = task_spec.get("configuration_atlases")
    action_assurances = task_spec.get("action_assurances")
    ros_action_adapters = task_spec.get("ros_action_adapters")
    source_ids: list[str] = []
    graph_ids: set[str] = set()
    if articulation_sources is None:
        if articulation_comparisons is not None:
            raise EvaluationSuiteError(
                f"{label}.articulation_comparisons requires articulation_sources"
            )
        if constraint_graphs is not None:
            raise EvaluationSuiteError(
                f"{label}.constraint_graphs requires articulation_sources"
            )
        if configuration_atlases is not None:
            raise EvaluationSuiteError(
                f"{label}.configuration_atlases requires constraint_graphs"
            )
    else:
        if not isinstance(articulation_sources, list) or not articulation_sources:
            raise EvaluationSuiteError(
                f"{label}.articulation_sources must contain at least one source record"
            )
        for index, source in enumerate(articulation_sources):
            source_label = f"{label}.articulation_sources[{index}]"
            if not isinstance(source, dict) or set(source) != {"source_id", "format", "path"}:
                raise EvaluationSuiteError(
                    f"{source_label} must contain exactly source_id, format, and path"
                )
            source_id = _task_identifier(source.get("source_id"), f"{source_label}.source_id")
            if source_id in source_ids:
                raise EvaluationSuiteError(f"{label}.articulation_sources has duplicate source_id {source_id!r}")
            source_ids.append(source_id)
            source_format = source.get("format")
            if source_format not in {"urdf", "sdf", "mjcf"}:
                raise EvaluationSuiteError(
                    f"{source_label}.format must be 'urdf', 'sdf', or 'mjcf'"
                )
            relative = source.get("path")
            if not isinstance(relative, str) or not relative:
                raise EvaluationSuiteError(f"{source_label}.path must be a relative file path")
            source_path = _resolve_under(task_root, relative, f"{source_label}.path")
            try:
                source_path.relative_to((task_root / "source").resolve())
            except ValueError as error:
                raise EvaluationSuiteError(f"{source_label}.path must be inside source/") from error

        if articulation_comparisons is not None:
            if not isinstance(articulation_comparisons, list) or not articulation_comparisons:
                raise EvaluationSuiteError(
                    f"{label}.articulation_comparisons must be a non-empty array when provided"
                )
            pairs: set[tuple[str, str]] = set()
            for index, comparison in enumerate(articulation_comparisons):
                comparison_label = f"{label}.articulation_comparisons[{index}]"
                if not isinstance(comparison, dict) or set(comparison) != {
                    "reference", "candidate", "correspondence"
                }:
                    raise EvaluationSuiteError(
                        f"{comparison_label} must contain exactly reference, candidate, and correspondence"
                    )
                reference = comparison.get("reference")
                candidate = comparison.get("candidate")
                if reference not in source_ids or candidate not in source_ids or reference == candidate:
                    raise EvaluationSuiteError(
                        f"{comparison_label} must name two distinct declared articulation source IDs"
                    )
                pair = (reference, candidate)
                if pair in pairs:
                    raise EvaluationSuiteError(f"{label}.articulation_comparisons has duplicate pair {pair!r}")
                pairs.add(pair)
                correspondence = comparison.get("correspondence")
                if not isinstance(correspondence, str) or not correspondence:
                    raise EvaluationSuiteError(
                        f"{comparison_label}.correspondence must be a relative file path"
                    )
                correspondence_path = _resolve_under(
                    task_root, correspondence, f"{comparison_label}.correspondence"
                )
                try:
                    correspondence_path.relative_to((task_root / "source").resolve())
                except ValueError as error:
                    raise EvaluationSuiteError(
                        f"{comparison_label}.correspondence must be inside source/"
                    ) from error

        if constraint_graphs is not None:
            if not isinstance(constraint_graphs, list) or not constraint_graphs:
                raise EvaluationSuiteError(
                    f"{label}.constraint_graphs must be a non-empty array when provided"
                )
            for index, graph in enumerate(constraint_graphs):
                graph_label = f"{label}.constraint_graphs[{index}]"
                if not isinstance(graph, dict) or set(graph) != {
                    "graph_id", "articulation_source", "spec"
                }:
                    raise EvaluationSuiteError(
                        f"{graph_label} must contain exactly graph_id, articulation_source, and spec"
                    )
                graph_id = _task_identifier(graph.get("graph_id"), f"{graph_label}.graph_id")
                if graph_id in graph_ids:
                    raise EvaluationSuiteError(f"{label}.constraint_graphs has duplicate graph_id {graph_id!r}")
                graph_ids.add(graph_id)
                articulation_source = graph.get("articulation_source")
                if articulation_source not in source_ids:
                    raise EvaluationSuiteError(
                        f"{graph_label}.articulation_source must name a declared articulation source ID"
                    )
                spec = graph.get("spec")
                if not isinstance(spec, str) or not spec:
                    raise EvaluationSuiteError(f"{graph_label}.spec must be a relative file path")
                spec_path = _resolve_under(task_root, spec, f"{graph_label}.spec")
                try:
                    spec_path.relative_to((task_root / "source").resolve())
                except ValueError as error:
                    raise EvaluationSuiteError(
                        f"{graph_label}.spec must be inside source/"
                    ) from error

        if configuration_atlases is not None:
            if constraint_graphs is None:
                raise EvaluationSuiteError(
                    f"{label}.configuration_atlases requires constraint_graphs"
                )
            if not isinstance(configuration_atlases, list) or not configuration_atlases:
                raise EvaluationSuiteError(
                    f"{label}.configuration_atlases must be a non-empty array when provided"
                )
            atlas_ids: set[str] = set()
            atlas_graphs: set[str] = set()
            for index, atlas in enumerate(configuration_atlases):
                atlas_label = f"{label}.configuration_atlases[{index}]"
                if not isinstance(atlas, dict) or set(atlas) != {
                    "atlas_id", "constraint_graph", "spec"
                }:
                    raise EvaluationSuiteError(
                        f"{atlas_label} must contain exactly atlas_id, constraint_graph, and spec"
                    )
                atlas_id = _task_identifier(atlas.get("atlas_id"), f"{atlas_label}.atlas_id")
                if atlas_id in atlas_ids:
                    raise EvaluationSuiteError(
                        f"{label}.configuration_atlases has duplicate atlas_id {atlas_id!r}"
                    )
                atlas_ids.add(atlas_id)
                graph_id = atlas.get("constraint_graph")
                if graph_id not in graph_ids:
                    raise EvaluationSuiteError(
                        f"{atlas_label}.constraint_graph must name a declared constraint graph ID"
                    )
                if graph_id in atlas_graphs:
                    raise EvaluationSuiteError(
                        f"{label}.configuration_atlases has more than one atlas for constraint graph {graph_id!r}"
                    )
                atlas_graphs.add(graph_id)
                spec = atlas.get("spec")
                if not isinstance(spec, str) or not spec:
                    raise EvaluationSuiteError(f"{atlas_label}.spec must be a relative file path")
                spec_path = _resolve_under(task_root, spec, f"{atlas_label}.spec")
                try:
                    spec_path.relative_to((task_root / "source").resolve())
                except ValueError as error:
                    raise EvaluationSuiteError(
                        f"{atlas_label}.spec must be inside source/"
                    ) from error

    workspace_roots = task_spec.get("workspace_roots")
    if workflow == "direct":
        if workspace_roots is not None:
            raise EvaluationSuiteError(f"{label}.workspace_roots is permitted only for workflow 'prepare'")
    else:
        if (
            not isinstance(workspace_roots, list)
            or not workspace_roots
            or any(not isinstance(root, str) or not root for root in workspace_roots)
            or len(set(workspace_roots)) != len(workspace_roots)
        ):
            raise EvaluationSuiteError(
                f"{label}.workspace_roots must be a unique non-empty array for workflow 'prepare'"
            )
        source_root = (task_root / "source").resolve()
        for index, root in enumerate(workspace_roots):
            root_path = _relative_directory_under(task_root, root, f"{label}.workspace_roots[{index}]")
            try:
                root_path.relative_to(source_root)
            except ValueError as error:
                raise EvaluationSuiteError(
                    f"{label}.workspace_roots[{index}] must be inside source/"
                ) from error
        if input_format == "xacro" and task_spec["expansion"]["output"] != "resolved.urdf":
            raise EvaluationSuiteError(
                f"{label}.expansion.output must be 'resolved.urdf' for workflow 'prepare'"
            )

    options = task_spec.get("export_options", {})
    if not isinstance(options, dict):
        raise EvaluationSuiteError(f"{label}.export_options must be an object")
    allowed_options = (
        RAW_PATH_OPTIONS
        | RAW_BOOLEAN_OPTIONS
        | RAW_INTEGER_OPTIONS
        | RAW_NUMBER_OPTIONS
        | RAW_STRING_OPTIONS
        | {"inspect_mesh_kind"}
    )
    unknown = set(options) - allowed_options
    if unknown:
        raise EvaluationSuiteError(f"{label}.export_options contains unsupported fields: {sorted(unknown)}")
    if workflow == "prepare" and "package_map" in options:
        raise EvaluationSuiteError(
            f"{label}.export_options.package_map is not allowed for workflow 'prepare'; prepare generates it"
        )
    if ("observations" in options) != ("observation_query" in options):
        raise EvaluationSuiteError(
            f"{label}.export_options.observations and observation_query must be provided together"
        )
    if "observations" in options and "scene" not in options:
        raise EvaluationSuiteError(f"{label}.export_options.observations requires scene")
    if "configuration_atlas_spec" in options and "constraint_spec" not in options:
        raise EvaluationSuiteError(
            f"{label}.export_options.configuration_atlas_spec requires constraint_spec"
        )

    ros_adapter = task_spec.get("ros_observation_adapter")
    if ros_adapter is not None:
        if not isinstance(ros_adapter, dict):
            raise EvaluationSuiteError(f"{label}.ros_observation_adapter must be an object")
        allowed_adapter = {
            "config",
            "capture",
            "observation_query",
            "output_filename",
            "report_filename",
        }
        unknown_adapter = set(ros_adapter) - allowed_adapter
        if unknown_adapter:
            raise EvaluationSuiteError(
                f"{label}.ros_observation_adapter contains unsupported fields: {sorted(unknown_adapter)}"
            )
        if "observations" in options or "observation_query" in options:
            raise EvaluationSuiteError(
                f"{label} must not combine ros_observation_adapter with pre-normalized observation export options"
            )
        if "scene" not in options:
            raise EvaluationSuiteError(f"{label}.ros_observation_adapter requires export_options.scene")
        for field in ("config", "capture", "observation_query"):
            relative = ros_adapter.get(field)
            if not isinstance(relative, str) or not relative:
                raise EvaluationSuiteError(f"{label}.ros_observation_adapter.{field} must be a relative file path")
            adapter_path = _resolve_under(task_root, relative, f"{label}.ros_observation_adapter.{field}")
            try:
                adapter_path.relative_to((task_root / "source").resolve())
            except ValueError as error:
                raise EvaluationSuiteError(
                    f"{label}.ros_observation_adapter.{field} must be inside source/"
                ) from error
        output_names: list[str] = []
        for field in ("output_filename", "report_filename"):
            filename = ros_adapter.get(field)
            if (
                not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
                or not filename.endswith(".json")
            ):
                raise EvaluationSuiteError(
                    f"{label}.ros_observation_adapter.{field} must be a JSON basename created under candidate work"
                )
            output_names.append(filename)
        if len(set(output_names)) != len(output_names):
            raise EvaluationSuiteError(f"{label}.ros_observation_adapter output filenames must differ")
    for option in RAW_PATH_OPTIONS:
        if option in options:
            relative = options[option]
            if not isinstance(relative, str) or not relative:
                raise EvaluationSuiteError(f"{label}.export_options.{option} must be a relative file path")
            option_path = _resolve_under(task_root, relative, f"{label}.export_options.{option}")
            try:
                option_path.relative_to((task_root / "source").resolve())
            except ValueError as error:
                raise EvaluationSuiteError(
                    f"{label}.export_options.{option} must be inside source/"
                ) from error
    for option in RAW_BOOLEAN_OPTIONS:
        if option in options and not isinstance(options[option], bool):
            raise EvaluationSuiteError(f"{label}.export_options.{option} must be boolean")
    for option in RAW_INTEGER_OPTIONS:
        value = options.get(option)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise EvaluationSuiteError(f"{label}.export_options.{option} must be a non-negative integer")
    for option in RAW_NUMBER_OPTIONS:
        value = options.get(option)
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise EvaluationSuiteError(f"{label}.export_options.{option} must be a finite non-negative number")
    for option in RAW_STRING_OPTIONS:
        value = options.get(option)
        if value is not None and (not isinstance(value, str) or not value):
            raise EvaluationSuiteError(f"{label}.export_options.{option} must be a non-empty string")
    if "inspect_mesh_kind" in options:
        kinds = options["inspect_mesh_kind"]
        if (
            not isinstance(kinds, list)
            or not kinds
            or any(kind not in {"visual", "collision"} for kind in kinds)
            or len(set(kinds)) != len(kinds)
        ):
            raise EvaluationSuiteError(
                f"{label}.export_options.inspect_mesh_kind must be a unique non-empty array of visual/collision"
            )
    if action_assurances is not None:
        if "functional_spec" not in options:
            raise EvaluationSuiteError(f"{label}.action_assurances requires export_options.functional_spec")
        if not isinstance(action_assurances, list) or not action_assurances:
            raise EvaluationSuiteError(f"{label}.action_assurances must be a non-empty array when provided")
        assurance_ids: set[str] = set()
        output_names: set[str] = set()
        for index, assurance in enumerate(action_assurances):
            assurance_label = f"{label}.action_assurances[{index}]"
            if not isinstance(assurance, dict) or set(assurance) != {
                "assurance_id",
                "functional_model_source",
                "evidence_bundle",
                "output",
            }:
                raise EvaluationSuiteError(
                    f"{assurance_label} must contain exactly assurance_id, functional_model_source, evidence_bundle, and output"
                )
            assurance_id = _task_identifier(assurance.get("assurance_id"), f"{assurance_label}.assurance_id")
            if assurance_id in assurance_ids:
                raise EvaluationSuiteError(f"{label}.action_assurances has duplicate assurance_id {assurance_id!r}")
            assurance_ids.add(assurance_id)
            if assurance.get("functional_model_source") != "exported_functional_model":
                raise EvaluationSuiteError(
                    f"{assurance_label}.functional_model_source must be 'exported_functional_model'"
                )
            evidence_bundle = assurance.get("evidence_bundle")
            if not isinstance(evidence_bundle, str) or not evidence_bundle:
                raise EvaluationSuiteError(f"{assurance_label}.evidence_bundle must be a relative file path")
            bundle_path = _resolve_under(task_root, evidence_bundle, f"{assurance_label}.evidence_bundle")
            try:
                bundle_path.relative_to((task_root / "source").resolve())
            except ValueError as error:
                raise EvaluationSuiteError(f"{assurance_label}.evidence_bundle must be inside source/") from error
            output = assurance.get("output")
            if (
                not isinstance(output, str)
                or not output
                or Path(output).name != output
                or not output.endswith(".json")
            ):
                raise EvaluationSuiteError(
                    f"{assurance_label}.output must be a JSON basename created under candidate work"
                )
            if output in output_names:
                raise EvaluationSuiteError(f"{label}.action_assurances has duplicate output {output!r}")
            output_names.add(output)

    if ros_action_adapters is not None:
        if "functional_spec" not in options:
            raise EvaluationSuiteError(f"{label}.ros_action_adapters requires export_options.functional_spec")
        if not isinstance(ros_action_adapters, list) or not ros_action_adapters:
            raise EvaluationSuiteError(
                f"{label}.ros_action_adapters must be a non-empty array when provided"
            )
        adapter_ids: set[str] = set()
        adapter_output_names: set[str] = set()
        existing_outputs = {
            assurance["output"]
            for assurance in (action_assurances or [])
            if isinstance(assurance, dict) and isinstance(assurance.get("output"), str)
        }
        if ros_adapter is not None:
            existing_outputs.update(
                ros_adapter[field]
                for field in ("output_filename", "report_filename")
                if isinstance(ros_adapter.get(field), str)
            )
        for index, adapter in enumerate(ros_action_adapters):
            adapter_label = f"{label}.ros_action_adapters[{index}]"
            expected_fields = {
                "adapter_id",
                "functional_model_source",
                "config",
                "capture",
                "supplemental_sources",
                "evidence_source_output",
                "bundle_output",
                "report_output",
                "assurance_output",
            }
            if not isinstance(adapter, dict) or set(adapter) != expected_fields:
                raise EvaluationSuiteError(
                    f"{adapter_label} must contain exactly {sorted(expected_fields)}"
                )
            adapter_id = _task_identifier(adapter.get("adapter_id"), f"{adapter_label}.adapter_id")
            if adapter_id in adapter_ids:
                raise EvaluationSuiteError(
                    f"{label}.ros_action_adapters has duplicate adapter_id {adapter_id!r}"
                )
            adapter_ids.add(adapter_id)
            if adapter.get("functional_model_source") != "exported_functional_model":
                raise EvaluationSuiteError(
                    f"{adapter_label}.functional_model_source must be 'exported_functional_model'"
                )
            for field in ("config", "capture"):
                relative = adapter.get(field)
                if not isinstance(relative, str) or not relative:
                    raise EvaluationSuiteError(f"{adapter_label}.{field} must be a relative file path")
                source_path = _resolve_under(task_root, relative, f"{adapter_label}.{field}")
                try:
                    source_path.relative_to((task_root / "source").resolve())
                except ValueError as error:
                    raise EvaluationSuiteError(f"{adapter_label}.{field} must be inside source/") from error
            supplemental_sources = adapter.get("supplemental_sources")
            if not isinstance(supplemental_sources, list):
                raise EvaluationSuiteError(f"{adapter_label}.supplemental_sources must be an array")
            supplemental_outputs: set[str] = set()
            supplemental_paths: set[str] = set()
            for source_index, source in enumerate(supplemental_sources):
                source_label = f"{adapter_label}.supplemental_sources[{source_index}]"
                if not isinstance(source, dict) or set(source) != {"source", "output"}:
                    raise EvaluationSuiteError(
                        f"{source_label} must contain exactly source and output"
                    )
                relative = source.get("source")
                if not isinstance(relative, str) or not relative:
                    raise EvaluationSuiteError(f"{source_label}.source must be a relative file path")
                source_path = _resolve_under(task_root, relative, f"{source_label}.source")
                try:
                    source_path.relative_to((task_root / "source").resolve())
                except ValueError as error:
                    raise EvaluationSuiteError(f"{source_label}.source must be inside source/") from error
                output = source.get("output")
                if (
                    not isinstance(output, str)
                    or not output
                    or Path(output).name != output
                    or not output.endswith(".json")
                ):
                    raise EvaluationSuiteError(
                        f"{source_label}.output must be a JSON basename copied under candidate work"
                    )
                if relative in supplemental_paths or output in supplemental_outputs:
                    raise EvaluationSuiteError(
                        f"{adapter_label}.supplemental_sources repeats a source path or output"
                    )
                supplemental_paths.add(relative)
                supplemental_outputs.add(output)
            generated_names: list[str] = []
            for field in (
                "evidence_source_output",
                "bundle_output",
                "report_output",
                "assurance_output",
            ):
                filename = adapter.get(field)
                if (
                    not isinstance(filename, str)
                    or not filename
                    or Path(filename).name != filename
                    or not filename.endswith(".json")
                ):
                    raise EvaluationSuiteError(
                        f"{adapter_label}.{field} must be a JSON basename created under candidate work"
                    )
                generated_names.append(filename)
            all_names = [*supplemental_outputs, *generated_names]
            if len(all_names) != len(set(all_names)):
                raise EvaluationSuiteError(
                    f"{adapter_label} supplemental and generated output filenames must differ"
                )
            collisions = set(all_names) & (adapter_output_names | existing_outputs)
            if collisions:
                raise EvaluationSuiteError(
                    f"{label}.ros_action_adapters output filenames collide: {sorted(collisions)}"
                )
            adapter_output_names.update(all_names)


def _task_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value) is None:
        raise EvaluationSuiteError(f"{label} must match [a-z0-9][a-z0-9_-]*")
    return value


def _scan_public_tree(public_root: Path) -> None:
    forbidden_names: list[str] = []
    leaks: list[dict[str, str]] = []
    for path in sorted(item for item in public_root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(public_root))
        lowered = path.name.lower()
        if "answer-key" in lowered or "private" in lowered:
            forbidden_names.append(relative)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in PRIVATE_LEAK_MARKERS:
            if marker in text:
                leaks.append({"path": relative, "marker": marker})
    if forbidden_names or leaks:
        raise EvaluationSuiteError(
            f"public tree contains private-key indicators: filenames={forbidden_names}, content={leaks}"
        )


def build_suite(config_path: Path, public_out: Path, private_out: Path) -> dict[str, Any]:
    config = _read_json(config_path, "evaluation-suite build config")
    if config.get("schema_version") != BUILD_SCHEMA:
        raise EvaluationSuiteError(f"build config must use schema_version {BUILD_SCHEMA}")
    suite_id = _task_identifier(config.get("suite_id"), "build.suite_id")
    candidate_input = config.get("candidate_input", CANDIDATE_INPUT_CONTEXT)
    if candidate_input not in {CANDIDATE_INPUT_CONTEXT, CANDIDATE_INPUT_RAW}:
        raise EvaluationSuiteError(
            f"build.candidate_input must be {CANDIDATE_INPUT_CONTEXT!r} or {CANDIDATE_INPUT_RAW!r}"
        )
    runtime_requirements = config.get("runtime_requirements", {})
    if not isinstance(runtime_requirements, dict):
        raise EvaluationSuiteError("build.runtime_requirements must be an object")
    for runtime_name, requirement in runtime_requirements.items():
        if not isinstance(runtime_name, str) or not runtime_name or not isinstance(requirement, dict):
            raise EvaluationSuiteError("build.runtime_requirements must map names to objects")
        unknown_runtime_fields = set(requirement) - {"executable", "provision", "version"}
        if unknown_runtime_fields:
            raise EvaluationSuiteError(
                f"build.runtime_requirements[{runtime_name!r}] has unsupported fields: {sorted(unknown_runtime_fields)}"
            )
        if any(not isinstance(value, str) or not value for value in requirement.values()):
            raise EvaluationSuiteError(
                f"build.runtime_requirements[{runtime_name!r}] values must be non-empty strings"
            )
        executable = requirement.get("executable")
        if executable is not None and ("/" in executable or "\\" in executable):
            raise EvaluationSuiteError(
                f"build.runtime_requirements[{runtime_name!r}].executable must be a portable command name"
            )
    task_configs = config.get("tasks")
    if not isinstance(task_configs, list) or not task_configs:
        raise EvaluationSuiteError("build.tasks must be a non-empty array")
    public_root = public_out.resolve()
    private_root = private_out.resolve()
    if public_root == private_root or public_root in private_root.parents or private_root in public_root.parents:
        raise EvaluationSuiteError("public and private output roots must be disjoint and neither may contain the other")
    public_root = _empty_output_directory(public_root, "public output")
    private_root = _empty_output_directory(private_root, "private output")

    public_tasks: list[dict[str, Any]] = []
    private_keys: dict[str, dict[str, str]] = {}
    seen: set[str] = set()
    for index, task in enumerate(task_configs):
        label = f"build.tasks[{index}]"
        if not isinstance(task, dict):
            raise EvaluationSuiteError(f"{label} must be an object")
        task_id = _task_identifier(task.get("task_id"), f"{label}.task_id")
        if task_id in seen:
            raise EvaluationSuiteError(f"duplicate build task_id {task_id!r}")
        seen.add(task_id)
        robot_family = _string(task, "robot_family", label)
        evaluation_dir = Path(_string(task, "evaluation_dir", label)).resolve()
        answer_key_source = Path(_string(task, "answer_key", label)).resolve()
        if not evaluation_dir.is_dir() or not answer_key_source.is_file():
            raise EvaluationSuiteError(f"{label} references a missing evaluation or answer-key path")

        task_root = public_root / "tasks" / task_id
        evaluation_root = task_root / "evaluation"
        copied: list[Path] = []
        task_surface: dict[str, Any] = {}
        if candidate_input == CANDIDATE_INPUT_CONTEXT:
            context_dir = Path(_string(task, "context_dir", label)).resolve()
            if not context_dir.is_dir():
                raise EvaluationSuiteError(f"{label} references a missing context directory")
            context_root = task_root / "context"
            for filename in REQUIRED_CONTEXT_FILES:
                source = context_dir / filename
                destination = context_root / filename
                _copy_public_file(
                    source,
                    destination,
                    sanitize_json=filename in {"agent-context.json", "model.json"},
                )
                copied.append(destination)
            for filename in OPTIONAL_CONTEXT_FILES:
                source = context_dir / filename
                if source.is_file():
                    destination = context_root / filename
                    _copy_public_file(source, destination)
                    copied.append(destination)
            for directory_name in OPTIONAL_CONTEXT_DIRECTORIES:
                source_directory = context_dir / directory_name
                if not source_directory.exists():
                    continue
                if not source_directory.is_dir() or source_directory.is_symlink():
                    raise EvaluationSuiteError(
                        f"{label} optional context artifact {directory_name!r} must be a real directory"
                    )
                for source in sorted(source_directory.rglob("*")):
                    if source.is_symlink():
                        raise EvaluationSuiteError(
                            f"{label} optional context artifact contains symlink {source}"
                        )
                    if not source.is_file():
                        continue
                    destination = context_root / directory_name / source.relative_to(source_directory)
                    _copy_public_file(source, destination)
                    copied.append(destination)
            task_surface["context_entrypoint"] = str(
                (context_root / "agent-context.json").relative_to(public_root)
            )
        else:
            source_dir = Path(_string(task, "source_dir", label)).resolve()
            copied.extend(_copy_raw_source_tree(source_dir, task_root / "source"))
            candidate_task = task.get("candidate_task")
            if not isinstance(candidate_task, dict):
                raise EvaluationSuiteError(f"{label}.candidate_task must be an object")
            task_spec_path = task_root / "task.json"
            task_spec_path.parent.mkdir(parents=True, exist_ok=True)
            task_spec_path.write_text(json_dump(candidate_task), encoding="utf-8")
            _validate_raw_task_spec(candidate_task, task_root, f"{label}.candidate_task")
            if candidate_task.get("input_format") == "xacro":
                xacro_requirement = runtime_requirements.get("xacro")
                if not isinstance(xacro_requirement, dict):
                    raise EvaluationSuiteError(
                        f"{label} uses Xacro but build.runtime_requirements.xacro is missing"
                    )
                if xacro_requirement.get("executable") != candidate_task["expansion"]["executable"]:
                    raise EvaluationSuiteError(
                        f"{label}.candidate_task expansion executable does not match runtime_requirements.xacro"
                    )
            copied.append(task_spec_path)
            task_surface.update({
                "task_spec": str(task_spec_path.relative_to(public_root)),
                "source_root": str((task_root / "source").relative_to(public_root)),
            })
        for filename in ("manifest.json", "questions.jsonl", "answer-template.jsonl"):
            source = evaluation_dir / filename
            destination = evaluation_root / filename
            _copy_public_file(source, destination)
            copied.append(destination)

        key_destination = private_root / "keys" / f"{task_id}.answer-key.jsonl"
        key_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(answer_key_source, key_destination)
        relative = lambda path: str(path.relative_to(public_root))
        public_tasks.append({
            "task_id": task_id,
            "robot_family": robot_family,
            "source": task.get("source", {}),
            "evaluation_manifest": relative(evaluation_root / "manifest.json"),
            "questions": relative(evaluation_root / "questions.jsonl"),
            "answer_template": relative(evaluation_root / "answer-template.jsonl"),
            "submission": f"{task_id}/answers.jsonl",
            "artifacts": {relative(path): _sha256(path) for path in sorted(copied)},
            **task_surface,
        })
        private_keys[task_id] = {
            "answer_key": str(key_destination.relative_to(private_root)),
            "sha256": _sha256(key_destination),
        }

    instructions_path = public_root / "INSTRUCTIONS.md"
    if candidate_input == CANDIDATE_INPUT_CONTEXT:
        candidate_workflow = (
            "For each task, read `context/agent-context.json` first, use `query-concepts` for compositional structure and its minimal proof closure, "
            "use `query-functions` for explicit component function, capability grounding, relational affordances, and project-inventory boundaries, "
            "use digest-verifying `retrieve` for typed entities and facts, "
            "read a bound `render_atlas/` and exact `render_view/` card before any supplied semantic SVG when present, and consult "
            "`context/model.json` only when the targeted context is insufficient."
        )
        meaning = "instruction-isolated multi-robot evaluation of the AI-readable spatial representation; expected answers are absent"
    else:
        candidate_workflow = (
            "For each task, read `task.json` and only its raw `source/` inputs. If `ros_observation_adapter` is present, first establish one "
            "concrete URDF: for `workflow: prepare`, run the skill's `prepare` once without observations into a new task-local bootstrap "
            "directory; for direct Xacro, expand and validate first; for direct URDF, validate it. Then run "
            "`ros_observation_adapter.py normalize` with that concrete URDF plus the declared config, capture, and scene; write its declared "
            "observation/report filenames under candidate work. Add the generated observation and declared query paths to a final export or "
            "second preparation; never write generated files into public source. If `workflow` is `prepare`, run the skill's final `prepare` "
            "command with the declared entrypoint, every `workspace_roots` directory, the evaluator-provided executable for Xacro, "
            "every declared mapping, and the declared export options; create the new preparation root under that task's submissions "
            "directory and answer from its `context/`. If `workflow` is `direct`, then for `input_format: xacro`, resolve the portable "
            "command named by `expansion.executable`, expand every mapping into the declared output basename, validate it, and export a "
            "fresh context; for `input_format: urdf`, validate the entrypoint and export directly. Translate each `export_options` key "
            "to the corresponding CLI option, using paths exactly as declared. "
            "When `articulation_sources` is present, compile and verify every declared source in task-local candidate work, then run each "
            "declared `articulation_comparisons` pair with its digest-bound correspondence. Use the exact mapped-law comparison and unseen-pose "
            "all-frame execution crosscheck; a name-sensitive law hash by itself neither proves nor disproves equivalence. "
            "When `constraint_graphs` is present, compile each graph from its named articulation source and digest-bound spec in task-local "
            "candidate work, verify regeneration, and use `evaluate-constraints` at explicit poses. Use `solve-constraints` only with an "
            "explicit seed and selected solved variables. Treat the articulation tree as a parameterization rather than the complete "
            "mechanism, and report numerical local mobility as pose-conditioned rather than global DOF. "
            "When `configuration_atlases` is present, generate each atlas from its named freshly compiled constraint graph and digest-bound "
            "atlas spec, verify exact regeneration and every stored node, and inspect exact chart/node/component records. Treat multi-seed "
            "solutions, proximity components, and observed rank drops as finite declared-sampling evidence only—not exhaustive branches, "
            "certified singularities, global topology, or physical truth. "
            "If `render` is true, preserve the generated `render-atlas/`, use its typed view records for visual-grounding questions, and "
            "run `verify-render` when view/numeric consistency is material. If `motion_atlas` is true, preserve `motion-atlas/`, use its "
            "driver/endpoint records for causal questions, and run `verify-motion-atlas`; translate motion step values to their exact CLI "
            "options. Do not use `--generate-evaluation`. Load the generated "
            "`agent-context.json`; every export also produces `concept-graph.json`, `concept-language.rsl`, and `articulation-grammar.json`. "
            "Use `query-concepts` for structural summaries, unique tree paths, driver effects, frame laws, constraint dependencies, and finite-node comparisons; "
            "preserve exact/ asserted/ finite modalities and return a negative only inside its declared complete closed-world projection. "
            "When `functional_spec` is present, preserve the generated `functional-model.json` and use `query-functions`; keep project function assertions, "
            "deterministic structural requirement grounding, unevaluated preconditions, intended effects, inventory scope, and physical executability distinct. "
            "When `ros_action_adapters` is present, first copy every declared supplemental evidence source byte-for-byte from its public `source` path to its declared "
            "candidate-work `output` basename. Then run `ros_action_adapter.py normalize` for each adapter with the exact freshly exported `functional-model.json`, "
            "declared config/capture, declared lifecycle-source/bundle/report output basenames, and every copied supplemental source. Compile the generated bundle into "
            "the declared assurance output with `action-assurance`, verify it, and use `query-action-assurance`. Never run `execute-capture` in an evaluation: the supplied "
            "capture is immutable input. Preserve exact goal UUID/payload/config/capture/source digests, client-versus-server timestamps, ignored other goals, unknown and "
            "duplicate statuses, publisher-identity visibility, and feedback/result non-promotion. Status-only traffic is not goal acceptance, and a `SUCCEEDED` server "
            "report is not physical success, effect observation, causation, authorization, or safety. "
            "When `action_assurances` is present, compile each declared evidence bundle against that exact generated `functional-model.json` into its declared "
            "candidate-work output, verify exact regeneration and every evidence-source digest, and use `query-action-assurance`. Select condition evidence at the "
            "declared decision time and lifecycle/effect evidence at evaluation time. Keep declared readiness, dispatch authorization, action-server goal/status/result, "
            "post-execution effect observation, causal success, physical-world truth, and safety distinct; a goal acceptance or succeeded result proves none of the latter boundaries. "
            "Never infer component purpose from URDF names or geometry, and never turn inventory absence into physical impossibility. Use the articulation grammar's typed variables, mimic equations, "
            "joint operators, and ordered frame derivations for pose-independent laws, and use `evaluate-articulation` for an unseen "
            "binding or `verify-articulation-grammar` when grammar/FK agreement is material. Do not treat an FK snapshot, Jacobian, or "
            "finite motion atlas as the general articulation law. Perform digest-verifying retrieval or fresh "
            "deterministic queries as needed, and answer from that newly generated representation."
        )
        meaning = "instruction-isolated multi-robot evaluation from raw URDF/Xacro project sources and optional declared URDF/SDF/MJCF articulation, supplemental mechanism-constraint, and finite configuration-atlas bundles; no generated canonical context or expected answers are public"
    instructions_path.write_text(
        "# Multi-robot blind spatial evaluation\n\n"
        "Use only this public directory and the supplied `understand-robot-spatial` skill. "
        "Do not search for private keys, evaluator reports, generated evaluator context, or prior submissions outside the allowed roots.\n\n"
        f"{candidate_workflow} Answer every record in `evaluation/questions.jsonl` using the exact two-field JSONL shape from "
        "`answer-template.jsonl`. Write each result to the task's `submission` path relative to the submissions root supplied by the evaluator.\n\n"
        "State no explanations inside `answer`; preserve arrays, objects, booleans, units, frame direction, and ordering exactly as requested. "
        "Angle-bracket alternatives in a submission contract are controlled enums: choose exactly one listed token. The common status tokens are "
        "`exact` (deterministically established), `asserted` (explicitly declared), `sampled` (finite-sample evidence), `indeterminate` "
        "(meaningful but evidence incomplete or analysis not run), `not_provided` (optional evidence layer absent), `not_established` "
        "(semantic role or intent not explicitly declared), and `unsupported` (engine or representation cannot evaluate the operation/content).\n",
        encoding="utf-8",
    )
    public_manifest = {
        "schema_version": PUBLIC_SCHEMA,
        "suite_id": suite_id,
        "candidate_input": candidate_input,
        "meaning": meaning,
        "runtime_requirements": runtime_requirements,
        "instructions": "INSTRUCTIONS.md",
        "artifacts": {"INSTRUCTIONS.md": _sha256(instructions_path)},
        "tasks": public_tasks,
        "isolation_requirement": "The candidate must not have filesystem, context, tool, log, cache, or conversation access to the private output or grader report.",
    }
    public_manifest_path = public_root / "manifest.json"
    public_manifest_path.write_text(json_dump(public_manifest), encoding="utf-8")
    _scan_public_tree(public_root)
    private_manifest = {
        "schema_version": PRIVATE_SCHEMA,
        "suite_id": suite_id,
        "candidate_input": candidate_input,
        "public_manifest_sha256": _sha256(public_manifest_path),
        "keys": private_keys,
        "privacy": "candidate-readable access to this directory invalidates the blind evaluation",
    }
    private_manifest_path = private_root / "manifest.json"
    private_manifest_path.write_text(json_dump(private_manifest), encoding="utf-8")
    return {
        "status": "built",
        "suite_id": suite_id,
        "candidate_input": candidate_input,
        "task_count": len(public_tasks),
        "public_manifest": str(public_manifest_path),
        "public_manifest_sha256": _sha256(public_manifest_path),
        "private_manifest": str(private_manifest_path),
        "public_private_roots_disjoint": True,
        "public_leak_scan": "passed",
    }


def _unique_ids(records: list[dict[str, Any]], label: str) -> list[str]:
    ids: list[str] = []
    for index, record in enumerate(records, 1):
        question_id = record.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise EvaluationSuiteError(f"{label} record {index} has no non-empty question_id")
        if question_id in ids:
            raise EvaluationSuiteError(f"{label} has duplicate question_id {question_id!r}")
        ids.append(question_id)
    return ids


def _validate_public_task(
    suite_root: Path,
    task: dict[str, Any],
    index: int,
    candidate_input: str = CANDIDATE_INPUT_CONTEXT,
) -> tuple[dict[str, str], list[dict[str, Any]], list[str], dict[str, Any]]:
    label = f"public.tasks[{index}]"
    task_id = _string(task, "task_id", label)
    normalized = {
        "task_id": task_id,
        "robot_family": _string(task, "robot_family", label),
        "evaluation_manifest": _string(task, "evaluation_manifest", label),
        "questions": _string(task, "questions", label),
        "answer_template": _string(task, "answer_template", label),
        "submission": _string(task, "submission", label),
    }
    artifacts = task.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise EvaluationSuiteError(f"{label}.artifacts must map every public relative path to SHA-256")
    if not all(
        isinstance(path, str)
        and path
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for path, digest in artifacts.items()
    ):
        raise EvaluationSuiteError(f"{label}.artifacts contains an invalid path or SHA-256")
    required_paths = {normalized["evaluation_manifest"], normalized["questions"], normalized["answer_template"]}
    if candidate_input == CANDIDATE_INPUT_RAW:
        normalized["task_spec"] = _string(task, "task_spec", label)
        normalized["source_root"] = _string(task, "source_root", label)
        required_paths.add(normalized["task_spec"])
    if not required_paths.issubset(artifacts):
        raise EvaluationSuiteError(f"{label}.artifacts must bind its evaluation manifest, questions, and answer template")
    for relative, expected_digest in artifacts.items():
        path = _resolve_under(suite_root, relative, f"{label}.artifacts[{relative!r}]")
        actual_digest = _sha256(path)
        if actual_digest != expected_digest:
            raise EvaluationSuiteError(
                f"public artifact digest mismatch for task {task_id!r}: {relative!r}; "
                f"expected {expected_digest}, got {actual_digest}"
            )

    if candidate_input == CANDIDATE_INPUT_RAW:
        task_spec_path = _resolve_under(suite_root, normalized["task_spec"], f"{label}.task_spec")
        source_root = _relative_directory_under(suite_root, normalized["source_root"], f"{label}.source_root")
        if source_root != (task_spec_path.parent / "source").resolve():
            raise EvaluationSuiteError(f"{label}.source_root must be the source/ directory beside task_spec")
        source_files = sorted(path for path in source_root.rglob("*") if path.is_file())
        if not source_files:
            raise EvaluationSuiteError(f"{label}.source_root contains no files")
        forbidden = RAW_FORBIDDEN_FILES
        generated = [str(path.relative_to(suite_root)) for path in source_files if path.name in forbidden]
        if generated:
            raise EvaluationSuiteError(
                f"source-only public task contains generated context artifacts: {generated}"
            )
        bound_source_files = {
            relative for relative in artifacts if relative.startswith(normalized["source_root"].rstrip("/") + "/")
        }
        actual_source_files = {str(path.relative_to(suite_root.resolve())) for path in source_files}
        if bound_source_files != actual_source_files:
            raise EvaluationSuiteError(f"{label}.artifacts must bind every and only every file under source_root")
        _validate_raw_task_spec(_read_json(task_spec_path, f"raw task spec for {task_id}"), task_spec_path.parent, label)

    evaluation_manifest_path = _resolve_under(
        suite_root, normalized["evaluation_manifest"], f"{label}.evaluation_manifest"
    )
    evaluation_manifest = _read_json(evaluation_manifest_path, f"evaluation manifest for {task_id}")
    if evaluation_manifest.get("schema_version") != EVALUATION_MANIFEST_SCHEMA:
        raise EvaluationSuiteError(
            f"task {task_id!r} evaluation manifest must use {EVALUATION_MANIFEST_SCHEMA}"
        )
    questions_path = _resolve_under(suite_root, normalized["questions"], f"{label}.questions")
    template_path = _resolve_under(suite_root, normalized["answer_template"], f"{label}.answer_template")
    questions = read_jsonl(questions_path)
    templates = read_jsonl(template_path)
    question_ids = _unique_ids(questions, f"task {task_id!r} questions")
    template_ids = _unique_ids(templates, f"task {task_id!r} answer template")
    if set(template_ids) != set(question_ids) or len(template_ids) != len(question_ids):
        raise EvaluationSuiteError(f"task {task_id!r} answer template IDs do not exactly match question IDs")
    for index_value, template in enumerate(templates, 1):
        if set(template) != {"question_id", "answer"} or template["answer"] is not None:
            raise EvaluationSuiteError(
                f"task {task_id!r} answer-template record {index_value} must contain only question_id and null answer"
            )
    if evaluation_manifest.get("question_count") != len(questions):
        raise EvaluationSuiteError(f"task {task_id!r} evaluation manifest question_count is stale")
    actual_capabilities: dict[str, int] = {}
    for question in questions:
        if question.get("schema_version") != "robot-spatial-question.v1":
            raise EvaluationSuiteError(f"task {task_id!r} contains a question with an unsupported schema")
        capability = question.get("capability")
        if not isinstance(capability, str):
            raise EvaluationSuiteError(f"task {task_id!r} question capability must be a string")
        actual_capabilities[capability] = actual_capabilities.get(capability, 0) + 1
    if evaluation_manifest.get("capability_counts") != dict(sorted(actual_capabilities.items())):
        raise EvaluationSuiteError(f"task {task_id!r} evaluation manifest capability_counts are stale")
    return normalized, questions, question_ids, evaluation_manifest


def grade_suite(
    public_manifest_path: Path,
    private_manifest_path: Path,
    submissions_root: Path,
    per_task_accuracy: float = 1.0,
    minimum_task_pass_rate: float = 1.0,
    minimum_overall_accuracy: float = 1.0,
) -> dict[str, Any]:
    for name, value in (
        ("per_task_accuracy", per_task_accuracy),
        ("minimum_task_pass_rate", minimum_task_pass_rate),
        ("minimum_overall_accuracy", minimum_overall_accuracy),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
            raise EvaluationSuiteError(f"{name} must be between 0 and 1")

    public = _read_json(public_manifest_path, "public evaluation-suite manifest")
    if public.get("schema_version") != PUBLIC_SCHEMA:
        raise EvaluationSuiteError(f"public manifest must use schema_version {PUBLIC_SCHEMA}")
    suite_id = _string(public, "suite_id", "public")
    candidate_input = public.get("candidate_input", CANDIDATE_INPUT_CONTEXT)
    if candidate_input not in {CANDIDATE_INPUT_CONTEXT, CANDIDATE_INPUT_RAW}:
        raise EvaluationSuiteError("public.candidate_input is unsupported")
    runtime_requirements = public.get("runtime_requirements", {})
    if not isinstance(runtime_requirements, dict):
        raise EvaluationSuiteError("public.runtime_requirements must be an object")
    for runtime_name, requirement in runtime_requirements.items():
        if not isinstance(runtime_name, str) or not runtime_name or not isinstance(requirement, dict):
            raise EvaluationSuiteError("public.runtime_requirements must map names to objects")
        if set(requirement) - {"executable", "provision", "version"}:
            raise EvaluationSuiteError(f"public runtime requirement {runtime_name!r} is malformed")
        if any(not isinstance(value, str) or not value for value in requirement.values()):
            raise EvaluationSuiteError(f"public runtime requirement {runtime_name!r} has an invalid value")
        executable = requirement.get("executable")
        if executable is not None and ("/" in executable or "\\" in executable):
            raise EvaluationSuiteError(f"public runtime requirement {runtime_name!r} is not portable")
    tasks = public.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise EvaluationSuiteError("public.tasks must be a non-empty array")
    root_artifacts = public.get("artifacts")
    if not isinstance(root_artifacts, dict) or not root_artifacts:
        raise EvaluationSuiteError("public.artifacts must bind suite-level instructions and metadata")
    for relative, expected_digest in root_artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected_digest, str):
            raise EvaluationSuiteError("public.artifacts must map relative paths to SHA-256 strings")
        path = _resolve_under(public_manifest_path.parent, relative, f"public.artifacts[{relative!r}]")
        if _sha256(path) != expected_digest:
            raise EvaluationSuiteError(f"suite-level public artifact digest mismatch: {relative!r}")

    private = _read_json(private_manifest_path, "private evaluation-suite key manifest")
    if private.get("schema_version") != PRIVATE_SCHEMA:
        raise EvaluationSuiteError(f"private manifest must use schema_version {PRIVATE_SCHEMA}")
    if private.get("suite_id") != suite_id:
        raise EvaluationSuiteError("private suite_id does not match public suite_id")
    if private.get("candidate_input", CANDIDATE_INPUT_CONTEXT) != candidate_input:
        raise EvaluationSuiteError("private candidate_input does not match public candidate_input")
    if private.get("public_manifest_sha256") != _sha256(public_manifest_path):
        raise EvaluationSuiteError("private manifest does not bind this exact public manifest")
    private_keys = private.get("keys")
    if not isinstance(private_keys, dict):
        raise EvaluationSuiteError("private.keys must map task IDs to isolated key records")

    normalized_tasks: list[tuple[dict[str, str], list[str], dict[str, Any]]] = []
    task_ids: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise EvaluationSuiteError(f"public.tasks[{index}] must be an object")
        normalized, _questions, question_ids, evaluation_manifest = _validate_public_task(
            public_manifest_path.parent, task, index, candidate_input
        )
        if candidate_input == CANDIDATE_INPUT_RAW:
            task_spec_path = _resolve_under(
                public_manifest_path.parent,
                normalized["task_spec"],
                f"public.tasks[{index}].task_spec",
            )
            task_spec = _read_json(task_spec_path, f"raw task spec for {normalized['task_id']}")
            if task_spec.get("input_format") == "xacro":
                requirement = runtime_requirements.get("xacro")
                if not isinstance(requirement, dict):
                    raise EvaluationSuiteError("public raw Xacro task requires runtime_requirements.xacro")
                if requirement.get("executable") != task_spec["expansion"]["executable"]:
                    raise EvaluationSuiteError(
                        f"raw Xacro task {normalized['task_id']!r} executable does not match public runtime requirement"
                    )
        task_id = normalized["task_id"]
        if task_id in task_ids:
            raise EvaluationSuiteError(f"duplicate public task_id {task_id!r}")
        task_ids.append(task_id)
        normalized_tasks.append((normalized, question_ids, evaluation_manifest))
    if set(private_keys) != set(task_ids):
        raise EvaluationSuiteError("private key IDs must exactly match public task IDs")

    task_reports: list[dict[str, Any]] = []
    total_questions = 0
    total_correct = 0
    for normalized, question_ids, evaluation_manifest in normalized_tasks:
        task_id = normalized["task_id"]
        private_record = private_keys[task_id]
        if not isinstance(private_record, dict):
            raise EvaluationSuiteError(f"private key record for {task_id!r} must be an object")
        answer_key_relative = _string(private_record, "answer_key", f"private.keys[{task_id!r}]")
        expected_key_digest = _string(private_record, "sha256", f"private.keys[{task_id!r}]")
        if len(expected_key_digest) != 64 or any(character not in "0123456789abcdef" for character in expected_key_digest):
            raise EvaluationSuiteError(f"private key SHA-256 for {task_id!r} is invalid")
        answer_key_path = _resolve_under(
            private_manifest_path.parent, answer_key_relative, f"private answer key {task_id!r}"
        )
        actual_key_digest = _sha256(answer_key_path)
        if actual_key_digest != expected_key_digest:
            raise EvaluationSuiteError(
                f"private answer-key digest mismatch for {task_id!r}; expected {expected_key_digest}, got {actual_key_digest}"
            )
        keys = read_jsonl(answer_key_path)
        key_ids = _unique_ids(keys, f"private answer key {task_id!r}")
        if set(key_ids) != set(question_ids) or len(key_ids) != len(question_ids):
            raise EvaluationSuiteError(f"private answer-key IDs do not exactly match public questions for {task_id!r}")
        expected_truth = evaluation_manifest.get("spatial_truth_sha256")
        for key in keys:
            if key.get("schema_version") != "robot-spatial-answer-key.v1":
                raise EvaluationSuiteError(f"private answer key {task_id!r} contains an unsupported schema")
            if key.get("evidence", {}).get("spatial_truth_sha256") != expected_truth:
                raise EvaluationSuiteError(f"private answer key {task_id!r} is bound to a different spatial truth")

        submission_path = _resolve_under(
            submissions_root,
            normalized["submission"],
            f"candidate submission {task_id!r}",
            must_exist=False,
        )
        submission_present = submission_path.is_file()
        answers = read_jsonl(submission_path) if submission_present else []
        report = verify_answers(keys, answers, per_task_accuracy)
        total_questions += report["total_questions"]
        total_correct += report["correct_answers"]
        task_reports.append({
            "task_id": task_id,
            "robot_family": normalized["robot_family"],
            "status": report["status"],
            "submission_present": submission_present,
            "submission_sha256": _sha256(submission_path) if submission_present else None,
            "question_count": len(question_ids),
            "report": report,
        })

    task_count = len(task_reports)
    passed_tasks = sum(report["status"] == "passed" for report in task_reports)
    task_pass_rate = passed_tasks / task_count
    overall_accuracy = total_correct / total_questions if total_questions else 0.0
    family_summary: dict[str, dict[str, Any]] = {}
    for task in task_reports:
        bucket = family_summary.setdefault(task["robot_family"], {"tasks": 0, "passed": 0, "questions": 0, "correct": 0})
        bucket["tasks"] += 1
        bucket["passed"] += task["status"] == "passed"
        bucket["questions"] += task["report"]["total_questions"]
        bucket["correct"] += task["report"]["correct_answers"]
    for bucket in family_summary.values():
        bucket["task_pass_rate"] = bucket["passed"] / bucket["tasks"]
        bucket["accuracy"] = bucket["correct"] / bucket["questions"] if bucket["questions"] else 0.0
    passed = (
        task_pass_rate + 1e-15 >= minimum_task_pass_rate
        and overall_accuracy + 1e-15 >= minimum_overall_accuracy
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "meaning": "all public artifacts and private bindings are verified before per-task grading; aggregate success requires both configured gates",
        "suite_id": suite_id,
        "candidate_input": candidate_input,
        "task_count": task_count,
        "passed_task_count": passed_tasks,
        "failed_task_count": task_count - passed_tasks,
        "task_pass_rate": task_pass_rate,
        "minimum_task_pass_rate": minimum_task_pass_rate,
        "total_questions": total_questions,
        "correct_answers": total_correct,
        "overall_accuracy": round(overall_accuracy, 12),
        "minimum_overall_accuracy": minimum_overall_accuracy,
        "per_task_accuracy": per_task_accuracy,
        "family_summary": dict(sorted(family_summary.items())),
        "tasks": task_reports,
        "privacy_warning": "this report may contain expected answers inside task failures; keep it outside candidate-readable surfaces",
    }


def self_check_suite(public_manifest_path: Path, private_manifest_path: Path) -> dict[str, Any]:
    public = _read_json(public_manifest_path, "public evaluation-suite manifest")
    private = _read_json(private_manifest_path, "private evaluation-suite key manifest")
    tasks = public.get("tasks")
    keys = private.get("keys")
    if not isinstance(tasks, list) or not isinstance(keys, dict):
        raise EvaluationSuiteError("suite manifests are malformed")
    with tempfile.TemporaryDirectory(prefix="robot-spatial-suite-control-") as temporary:
        submissions_root = Path(temporary)
        last_submission: Path | None = None
        last_answers: list[dict[str, Any]] = []
        for index, task in enumerate(tasks):
            if not isinstance(task, dict):
                raise EvaluationSuiteError(f"public.tasks[{index}] must be an object")
            task_id = _string(task, "task_id", f"public.tasks[{index}]")
            private_record = keys.get(task_id)
            if not isinstance(private_record, dict):
                raise EvaluationSuiteError(f"missing private key record for {task_id!r}")
            key_path = _resolve_under(
                private_manifest_path.parent,
                _string(private_record, "answer_key", f"private.keys[{task_id!r}]"),
                f"private answer key {task_id!r}",
            )
            answer_records = [
                {"question_id": key["question_id"], "answer": key["answer"]}
                for key in read_jsonl(key_path)
            ]
            submission = _resolve_under(
                submissions_root,
                _string(task, "submission", f"public.tasks[{index}]"),
                f"control submission {task_id!r}",
                must_exist=False,
            )
            submission.parent.mkdir(parents=True, exist_ok=True)
            submission.write_text(jsonl_dump(answer_records), encoding="utf-8")
            last_submission, last_answers = submission, answer_records
        perfect = grade_suite(public_manifest_path, private_manifest_path, submissions_root)
        assert last_submission is not None and last_answers
        last_submission.write_text(jsonl_dump(last_answers[:-1]), encoding="utf-8")
        missing = grade_suite(public_manifest_path, private_manifest_path, submissions_root)
    passed = perfect["status"] == "passed" and missing["status"] == "failed"
    return {
        "schema_version": "robot-spatial-evaluation-suite-self-check.v1",
        "status": "passed" if passed else "failed",
        "suite_id": perfect["suite_id"],
        "candidate_input": perfect.get("candidate_input", CANDIDATE_INPUT_CONTEXT),
        "task_count": perfect["task_count"],
        "question_count": perfect["total_questions"],
        "perfect_submission_control": {
            "status": perfect["status"],
            "task_pass_rate": perfect["task_pass_rate"],
            "overall_accuracy": perfect["overall_accuracy"],
        },
        "one_missing_answer_control": {
            "status": missing["status"],
            "task_pass_rate": missing["task_pass_rate"],
            "overall_accuracy": missing["overall_accuracy"],
        },
        "temporary_control_submissions_removed": True,
    }


def public_result_summary(private_report: dict[str, Any]) -> dict[str, Any]:
    if private_report.get("schema_version") != REPORT_SCHEMA:
        raise EvaluationSuiteError(f"private report must use schema_version {REPORT_SCHEMA}")
    task_summaries: list[dict[str, Any]] = []
    for index, task in enumerate(private_report.get("tasks", [])):
        if not isinstance(task, dict) or not isinstance(task.get("report"), dict):
            raise EvaluationSuiteError(f"private report task {index} is malformed")
        report = task["report"]
        failures = report.get("failures", [])
        if not isinstance(failures, list):
            raise EvaluationSuiteError(f"private report task {index} failures must be an array")
        missing = sum(isinstance(failure, dict) and failure.get("reason") == "missing answer" for failure in failures)
        mismatched = len(failures) - missing
        task_summaries.append({
            "task_id": task.get("task_id"),
            "robot_family": task.get("robot_family"),
            "status": task.get("status"),
            "submission_present": task.get("submission_present"),
            "submission_sha256": task.get("submission_sha256"),
            "question_count": report.get("total_questions"),
            "correct_answers": report.get("correct_answers"),
            "accuracy": report.get("accuracy"),
            "per_capability": report.get("per_capability"),
            "failure_counts": {
                "missing": missing,
                "mismatched_or_wrong_type": mismatched,
                "duplicate_question_ids": len(report.get("duplicate_question_ids", [])),
                "unexpected_question_ids": len(report.get("unexpected_question_ids", [])),
                "malformed_records": len(report.get("malformed_records", [])),
            },
        })
    return {
        "schema_version": "robot-spatial-evaluation-suite-public-result.v1",
        "status": private_report.get("status"),
        "suite_id": private_report.get("suite_id"),
        "candidate_input": private_report.get("candidate_input", CANDIDATE_INPUT_CONTEXT),
        "task_count": private_report.get("task_count"),
        "passed_task_count": private_report.get("passed_task_count"),
        "failed_task_count": private_report.get("failed_task_count"),
        "task_pass_rate": private_report.get("task_pass_rate"),
        "minimum_task_pass_rate": private_report.get("minimum_task_pass_rate"),
        "total_questions": private_report.get("total_questions"),
        "correct_answers": private_report.get("correct_answers"),
        "overall_accuracy": private_report.get("overall_accuracy"),
        "minimum_overall_accuracy": private_report.get("minimum_overall_accuracy"),
        "per_task_accuracy": private_report.get("per_task_accuracy"),
        "family_summary": private_report.get("family_summary"),
        "tasks": task_summaries,
        "privacy": {
            "expected_answers_omitted": True,
            "actual_answers_omitted": True,
            "question_level_failure_reasons_omitted": True,
            "safe_to_publish_after_manual_isolation_metadata_review": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="assemble disjoint public and private suite roots from exported contexts")
    build.add_argument("config", type=Path)
    build.add_argument("--public-out", type=Path, required=True)
    build.add_argument("--private-out", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="validate all bindings and grade candidate submissions")
    verify.add_argument("public_manifest", type=Path)
    verify.add_argument("private_manifest", type=Path)
    verify.add_argument("--submissions-root", type=Path, required=True)
    verify.add_argument("--per-task-accuracy", type=float, default=1.0)
    verify.add_argument("--minimum-task-pass-rate", type=float, default=1.0)
    verify.add_argument("--minimum-overall-accuracy", type=float, default=1.0)
    verify.add_argument("--report", type=Path)
    self_check = subparsers.add_parser("self-check", help="prove a perfect synthetic suite passes and one missing answer fails")
    self_check.add_argument("public_manifest", type=Path)
    self_check.add_argument("private_manifest", type=Path)
    summarize = subparsers.add_parser("summarize", help="remove expected/actual answer material from a private grader report")
    summarize.add_argument("private_report", type=Path)
    summarize.add_argument("--out", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == "build":
        print(json_dump(build_suite(args.config, args.public_out, args.private_out)), end="")
        return 0
    if args.command == "self-check":
        report = self_check_suite(args.public_manifest, args.private_manifest)
        print(json_dump(report), end="")
        return 0 if report["status"] == "passed" else 1
    if args.command == "summarize":
        report = public_result_summary(_read_json(args.private_report, "private suite report"))
        serialized = json_dump(report)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0
    report = grade_suite(
        args.public_manifest,
        args.private_manifest,
        args.submissions_root,
        args.per_task_accuracy,
        args.minimum_task_pass_rate,
        args.minimum_overall_accuracy,
    )
    serialized = json_dump(report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, EvaluationError, EvaluationSuiteError) as error:
        print(json_dump({"status": "error", "error": str(error)}), file=sys.stderr, end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
