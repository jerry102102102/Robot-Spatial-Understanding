#!/usr/bin/env python3
"""Build and query a progressive-disclosure context pack for robot spatial facts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA = "robot-spatial-agent-context.v1"
CARD_SCHEMA = "robot-spatial-entity-card.v1"
ENTITY_INDEX_SCHEMA = "robot-spatial-entity-index.v1"
FACT_INDEX_SCHEMA = "robot-spatial-fact-index.v1"
RETRIEVAL_SCHEMA = "robot-spatial-context-retrieval.v1"


class ContextError(ValueError):
    """An invalid, inconsistent, or ambiguous spatial context pack query."""


def _json_bytes(data: Any) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _jsonl_line(data: Any) -> bytes:
    return (json.dumps(data, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ContextError(f"cannot read context artifact {path}: {error}") from error


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _roles(canonical: dict[str, Any], frame_name: str) -> tuple[list[str], str | None]:
    annotation = canonical.get("semantics", {}).get("frames", {}).get(frame_name, {})
    return list(annotation.get("roles", [])), annotation.get("meaning")


def _related_fact_ids(entity_id: str, facts: list[dict[str, Any]]) -> list[str]:
    entity_type, _, name = entity_id.partition("/")
    related: list[str] = []
    for fact in facts:
        include = fact["subject"] == entity_id
        object_value = fact.get("object")
        if entity_type == "link":
            include = include or fact["subject"] == f"frame/{name}"
            include = include or object_value == name
            include = include or (
                isinstance(object_value, dict)
                and (object_value.get("parent") == name or object_value.get("child") == name)
            )
        if entity_type == "render_view":
            include = include or fact.get("qualifiers", {}).get("view_entity") == entity_id
            include = include or (
                isinstance(object_value, dict) and object_value.get("view_entity") == entity_id
            )
        if entity_type in {"motion_driver", "motion_view"}:
            qualifier = "motion_driver_entity" if entity_type == "motion_driver" else "motion_view_entity"
            include = include or fact.get("qualifiers", {}).get(qualifier) == entity_id
            include = include or (
                isinstance(object_value, dict) and object_value.get(qualifier) == entity_id
            )
        if entity_type in {"articulation_variable", "articulation_operator", "articulation_derivation"}:
            include = include or fact["subject"] == entity_id
        if include:
            related.append(fact["fact_id"])
    return sorted(set(related))


def _trust_record(fact_ids: list[str], facts_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected = [facts_by_id[fact_id] for fact_id in fact_ids]
    exact_count = sum(bool(fact["evidence"]["exact"]) for fact in selected)
    sources: dict[str, int] = {}
    for fact in selected:
        source = fact["evidence"]["source_type"]
        sources[source] = sources.get(source, 0) + 1
    if selected and exact_count == len(selected):
        classification = "verified_exact"
    elif exact_count:
        classification = "mixed_exact_and_asserted_or_approximate"
    elif selected:
        classification = "asserted_or_approximate"
    else:
        classification = "canonical_summary_without_bound_fact"
    return {
        "classification": classification,
        "bound_fact_count": len(selected),
        "exact_fact_count": exact_count,
        "nonexact_fact_count": len(selected) - exact_count,
        "source_type_counts": dict(sorted(sources.items())),
    }


def _link_card(name: str, canonical: dict[str, Any]) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    incoming = next((joint for joint_name, joint in canonical["joints"].items() if joint["child_link"] == name), None)
    incoming_name = next((joint_name for joint_name, joint in canonical["joints"].items() if joint["child_link"] == name), None)
    outgoing = [
        {"joint": joint_name, "type": joint["type"], "child_link": joint["child_link"]}
        for joint_name, joint in sorted(canonical["joints"].items())
        if joint["parent_link"] == name
    ]
    roles, meaning = _roles(canonical, name)
    link = canonical["links"][name]
    geometry_frames = [record["frame"] for key in ("visuals", "collisions") for record in link[key]]
    frame = canonical["frames"][name]
    inertial = link["inertial"]
    data = {
        "root_link": name == canonical["robot"]["root_link"],
        "parent_link": incoming["parent_link"] if incoming else None,
        "incoming_joint": incoming_name,
        "incoming_joint_type": incoming["type"] if incoming else None,
        "outgoing_joints": outgoing,
        "geometry_frames": geometry_frames,
        "has_inertial": inertial is not None,
        "declared_inertial": inertial,
        "asserted_roles": roles,
        "asserted_meaning": meaning,
        "pose_in_root_at_exported_pose": frame["world_from_frame"],
    }
    summary = (
        f"LINK {name}; root={_compact(data['root_link'])}; "
        f"parent={_compact(data['parent_link'])}; incoming_joint={_compact(incoming_name)}; "
        f"incoming_type={_compact(data['incoming_joint_type'])}; children={_compact(outgoing)}; "
        f"geometry_frames={_compact(geometry_frames)}; roles_asserted={_compact(roles)}; "
        f"declared_inertial={_compact(inertial)}; "
        f"pose[{canonical['pose']['name']}]={canonical['robot']['root_link']}_from_{name} "
        f"xyz_m={_compact(frame['world_from_frame']['translation_xyz_m'])} "
        f"quat_xyzw={_compact(frame['world_from_frame']['quaternion_xyzw'])}."
    )
    queries = [
        {"command": "chain", "arguments": {"from": "<link>", "to": name}, "use_when": "ordered ancestry or path is required"},
        {"command": "transform", "arguments": {"from": "<frame>", "to": name, "pose": "<pose.json>"}, "use_when": "a transform at another pose or reference frame is required"},
        {"command": "mass-properties", "arguments": {"subtree_root": name, "frame": "<frame>", "pose": "<pose.json>"}, "use_when": "declared subtree mass, center of mass, or aggregate inertia is required"},
    ]
    return data, summary, queries


def _joint_card(name: str, canonical: dict[str, Any], facts: list[dict[str, Any]]) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    joint = canonical["joints"][name]
    affected = sorted(
        fact["object"] for fact in facts
        if fact["subject"] == f"joint/{name}" and fact["predicate"] == "can_change_pose_of_link"
    )
    data = {
        "type": joint["type"],
        "parent_link": joint["parent_link"],
        "child_link": joint["child_link"],
        "pre_motion_frame": joint["pre_motion_frame"],
        "parent_from_pre_motion_origin": {
            "xyz_m": joint["origin_xyz_m"],
            "rpy_rad": joint["origin_rpy_rad"],
        },
        "axis_in_pre_motion_frame": joint["axis_in_pre_motion_frame"],
        "axis_in_root_at_exported_pose": joint["axis_in_root_frame_at_pose"],
        "position_at_exported_pose": joint["position_at_pose"],
        "position_unit": joint["position_unit"],
        "limits": joint["limits"],
        "mimic": joint["mimic"],
        "dynamics_declaration": joint.get("dynamics"),
        "actuation_declarations": joint.get("actuation_declarations"),
        "static_gravity_load_under_export_convention": (
            canonical.get("physical_analysis", {})
            .get("declared_static_gravity_loads_under_root_frame_convention", {})
            .get("independent_driver_loads")
            or {}
        ).get(name),
        "static_gravity_load_under_scene_convention": (
            (
                canonical.get("physical_analysis", {})
                .get("declared_static_gravity_loads_under_scene_gravity", {})
                .get("loads")
                or {}
            ).get("independent_driver_loads")
            or {}
        ).get(name),
        "affected_links": affected,
    }
    summary = (
        f"JOINT {name}; type={joint['type']}; edge={joint['parent_link']}->{joint['child_link']}; "
        f"pre_motion_frame={joint['pre_motion_frame']}; parent_from_joint xyz_m={_compact(joint['origin_xyz_m'])} "
        f"rpy_rad={_compact(joint['origin_rpy_rad'])}; axis_in_pre_motion={_compact(joint['axis_in_pre_motion_frame'])}; "
        f"axis_in_{canonical['robot']['root_link']}[{canonical['pose']['name']}]={_compact(joint['axis_in_root_frame_at_pose'])}; "
        f"position={_compact(joint['position_at_pose'])} {joint['position_unit'] or 'none'}; "
        f"limits={_compact(joint['limits'])}; mimic={_compact(joint['mimic'])}; "
        f"dynamics_declared={_compact(joint.get('dynamics'))}; "
        f"actuation_declared={_compact(joint.get('actuation_declarations'))}; "
        f"gravity_load_export_convention={_compact(data['static_gravity_load_under_export_convention'])}; "
        f"gravity_load_scene_convention={_compact(data['static_gravity_load_under_scene_convention'])}; "
        f"affected_links={_compact(affected)}."
    )
    queries = [
        {"command": "affects", "arguments": {"joint": name}, "use_when": "complete causal descendants are required"},
        {"command": "axis", "arguments": {"joint": name, "frame": "<frame>", "pose": "<pose.json>"}, "use_when": "the signed axis in another frame or pose is required"},
        {"command": "gravity-loads", "arguments": {"gravity_frame": "<frame>", "gravity": "<gx gy gz>", "pose": "<pose.json>"}, "use_when": "gravity generalized force or ideal static holding effort is required"},
        {"command": "actuation", "arguments": {"joint": name}, "use_when": "embedded ros2_control, transmission, or dynamics declarations are required"},
    ]
    if canonical.get("world_scene", {}).get("status") == "parsed_validated_and_bound":
        queries.append({"command": "scene-gravity-loads", "arguments": {"scene": "<scene.json>", "pose": "<pose.json>"}, "use_when": "world-scene mounting-aware gravity load for this joint is required"})
    return data, summary, queries


def _frame_card(name: str, canonical: dict[str, Any]) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    frame = canonical["frames"][name]
    roles, meaning = _roles(canonical, name)
    geometry = canonical.get("geometry_analysis", {}).get(name)
    inertial = canonical["links"][frame["owner"]]["inertial"] if frame["type"] == "inertial" else None
    data = {
        "semantic_type": frame["type"],
        "parent_frame": frame["parent_frame"],
        "owner": frame["owner"],
        "pose_in_root_at_exported_pose": frame["world_from_frame"],
        "asserted_roles": roles,
        "asserted_meaning": meaning,
        "geometry": ({
            "kind": geometry["kind"],
            "link": geometry["link"],
            "geometry_type": geometry["geometry_type"],
            "measurement_status": geometry["status"],
            "bounds_in_root_frame_at_pose": geometry.get("bounds_in_root_frame_at_pose"),
        } if geometry else None),
        "declared_inertial": inertial,
    }
    summary = (
        f"FRAME {name}; semantic_type={frame['type']}; parent_frame={_compact(frame['parent_frame'])}; "
        f"owner={frame['owner']}; roles_asserted={_compact(roles)}; "
        f"pose[{canonical['pose']['name']}]={canonical['robot']['root_link']}_from_{name} "
        f"xyz_m={_compact(frame['world_from_frame']['translation_xyz_m'])} "
        f"quat_xyzw={_compact(frame['world_from_frame']['quaternion_xyzw'])}; "
        f"geometry={_compact(data['geometry'])}; declared_inertial={_compact(inertial)}."
    )
    queries = [{
        "command": "transform",
        "arguments": {"from": "<frame>", "to": name, "pose": "<pose.json>"},
        "use_when": "a transform at another pose or reference frame is required",
    }]
    if geometry:
        queries.append({
            "command": "bounds",
            "arguments": {
                "geometry_frame": name,
                "pose": "<pose.json>",
                "inspect_mesh_kind": geometry["kind"] if geometry["geometry_type"] == "mesh" else None,
            },
            "use_when": "geometry bounds at another pose are required",
        })
    if inertial:
        queries.append({
            "command": "mass-properties",
            "arguments": {"subtree_root": frame["owner"], "frame": "<frame>", "pose": "<pose.json>"},
            "use_when": "pose-conditioned declared mass center or inertia for this link/subtree is required",
        })
    return data, summary, queries


def _simple_card(entity_type: str, name: str, data: dict[str, Any]) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    label = entity_type.upper()
    return data, f"{label} {name}; data={_compact(data)}.", []


def build_entity_cards(
    canonical: dict[str, Any],
    facts: list[dict[str, Any]],
    functional_model: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    facts_by_id = {fact["fact_id"]: fact for fact in facts}
    raw: list[tuple[str, str, str, dict[str, Any], str, list[dict[str, Any]]]] = []

    robot_name = canonical["robot"]["name"]
    robot_data = {
        **canonical["robot"],
        "pose": canonical["pose"],
        "validation": canonical["validation"],
        "capabilities": canonical["capabilities"],
        "declared_mass_properties": {
            key: value
            for key, value in canonical.get("physical_analysis", {}).get("declared_mass_properties", {}).items()
            if key != "per_link_declared_inertials"
        },
        "declared_static_gravity_loads_under_root_frame_convention": {
            key: value
            for key, value in canonical.get("physical_analysis", {})
            .get("declared_static_gravity_loads_under_root_frame_convention", {})
            .items()
            if key != "per_link_modeled_contributions"
        },
        "actuation_declaration_coverage": canonical.get("actuation", {}).get("coverage"),
        "world_scene": {
            key: canonical.get("world_scene", {}).get(key)
            for key in ("status", "scene_id", "snapshot", "world_frame", "robot_mount", "gravity")
            if key in canonical.get("world_scene", {})
        },
        "observed_world": {
            "status": canonical.get("observed_world", {}).get("observation", {}).get("status", "not_provided"),
            "query": canonical.get("observed_world", {}).get("observation", {}).get("query"),
            "readiness": canonical.get("observed_world", {}).get("observation", {}).get("readiness"),
            "analysis_status": canonical.get("observed_world", {}).get("analysis", {}).get("status"),
        },
    }
    data, summary, queries = _simple_card("robot", robot_name, robot_data)
    queries.append({
        "command": "mass-properties",
        "arguments": {"frame": canonical["robot"]["root_link"], "pose": "<pose.json>"},
        "use_when": "whole-model declared mass, center of mass, or aggregate inertia is required",
    })
    queries.extend([
        {
            "command": "gravity-loads",
            "arguments": {"gravity_frame": canonical["robot"]["root_link"], "gravity": "0 0 -9.80665", "pose": "<pose.json>"},
            "use_when": "whole-model static gravity generalized forces are required",
        },
        {
            "command": "actuation",
            "arguments": {},
            "use_when": "all embedded actuation/control declarations are required",
        },
    ])
    if canonical.get("world_scene", {}).get("status") == "parsed_validated_and_bound":
        queries.extend([
            {
                "command": "scene-summary",
                "arguments": {"scene": "<scene.json>", "pose": "<pose.json>"},
                "use_when": "root mounting, world objects, snapshot gravity, and robot/environment collision are required",
            },
            {
                "command": "scene-gravity-loads",
                "arguments": {"scene": "<scene.json>", "pose": "<pose.json>"},
                "use_when": "gravity loads must follow the world-scene root mounting rather than a root-local convention",
            },
        ])
    raw.append((f"robot/{robot_name}", "robot", robot_name, data, summary, queries))
    for name in sorted(canonical["links"]):
        data, summary, queries = _link_card(name, canonical)
        raw.append((f"link/{name}", "link", name, data, summary, queries))
    for name in sorted(canonical["joints"]):
        data, summary, queries = _joint_card(name, canonical, facts)
        raw.append((f"joint/{name}", "joint", name, data, summary, queries))
    for name in sorted(canonical["frames"]):
        data, summary, queries = _frame_card(name, canonical)
        raw.append((f"frame/{name}", "frame", name, data, summary, queries))

    render_atlas = canonical.get("artifacts", {}).get("semantic_render_atlas")
    if isinstance(render_atlas, dict):
        render_id = render_atlas["render_id"]
        atlas_data = {
            key: render_atlas[key]
            for key in (
                "path",
                "manifest_sha256",
                "schema_version",
                "render_id",
                "render_input_sha256",
                "pose_binding",
                "coordinate_contract",
                "coverage",
                "epistemic_scope",
            )
        }
        atlas_data["view_entities"] = [
            f"render_view/{render_id}/{view_id}" for view_id in sorted(render_atlas["views"])
        ]
        data, summary, queries = _simple_card("render_atlas", render_id, atlas_data)
        queries.append({
            "command": "verify-render",
            "arguments": {"atlas": render_atlas["path"], "pose": canonical["pose"]["name"]},
            "use_when": "the projection, pixel mapping, entity IDs, and SVG digests must be regenerated and verified",
        })
        raw.append((f"render_atlas/{render_id}", "render_atlas", render_id, data, summary, queries))
        for view_id, view in sorted(render_atlas["views"].items()):
            view_name = f"{render_id}/{view_id}"
            view_data = {
                "render_atlas": f"render_atlas/{render_id}",
                "view_id": view_id,
                "title": view["title"],
                "projection": view["projection"],
                "screen": view["screen"],
                "scene_projection_bounds_uv_m": view["scene_projection_bounds_uv_m"],
                "geometry": view["geometry"],
                "kinematic_edges": view["kinematic_edges"],
                "link_frames": view["link_frames"],
                "highlight_frame_axes": view["highlight_frame_axes"],
                "artifact": view["artifact"],
                "interpretation": "semantic convex projection derived from the canonical numeric model; not photorealistic visibility or an independent spatial oracle",
            }
            data, summary, queries = _simple_card("render_view", view_name, view_data)
            queries.append({
                "command": "verify-render",
                "arguments": {"atlas": render_atlas["path"], "pose": canonical["pose"]["name"]},
                "use_when": "numeric projection coordinates or the SVG must be trusted",
            })
            raw.append((f"render_view/{view_name}", "render_view", view_name, data, summary, queries))

    articulation = canonical.get("artifacts", {}).get("articulation_grammar")
    if isinstance(articulation, dict):
        grammar_id = articulation["grammar_id"]
        grammar_data = {
            key: articulation[key]
            for key in (
                "path",
                "sha256",
                "schema_version",
                "grammar_id",
                "grammar_input_sha256",
                "law_identity",
                "source_binding",
                "coordinate_contract",
                "language_contract",
                "evaluation_contract",
                "layer_contract",
                "coverage",
                "epistemic_scope",
            )
        }
        grammar_data["variable_entities"] = [
            f"articulation_variable/{grammar_id}/{driver}"
            for driver in sorted(articulation["independent_variables"])
        ]
        grammar_data["operator_entities"] = [
            f"articulation_operator/{grammar_id}/{joint}"
            for joint in sorted(articulation["joint_operators"])
        ]
        grammar_data["derivation_entities"] = [
            f"articulation_derivation/{grammar_id}/{frame}"
            for frame in sorted(articulation["frame_derivations"])
        ]
        data, summary, queries = _simple_card("articulation_grammar", grammar_id, grammar_data)
        queries.extend([
            {
                "command": "evaluate-articulation",
                "arguments": {"grammar": articulation["path"], "pose": "<pose.json>"},
                "use_when": "the general law must be bound at a new pose without reparsing its source format",
            },
            {
                "command": "verify-articulation-grammar",
                "arguments": {"grammar": articulation["path"]},
                "use_when": "the grammar must be regenerated and cross-checked over fresh all-frame FK probes",
            },
        ])
        raw.append((f"articulation_grammar/{grammar_id}", "articulation_grammar", grammar_id, data, summary, queries))
        for driver, variable in sorted(articulation["independent_variables"].items()):
            variable_name = f"{grammar_id}/{driver}"
            variable_data = {
                "articulation_grammar": f"articulation_grammar/{grammar_id}",
                "driver_joint": driver,
                **variable,
            }
            data, summary, queries = _simple_card("articulation_variable", variable_name, variable_data)
            queries.append({
                "command": "evaluate-articulation",
                "arguments": {"grammar": articulation["path"], "pose": "<pose.json>"},
                "use_when": "evaluate this driver within its complete mimic-constrained domain",
            })
            raw.append((f"articulation_variable/{variable_name}", "articulation_variable", variable_name, data, summary, queries))
        for joint, operator in sorted(articulation["joint_operators"].items()):
            operator_name = f"{grammar_id}/{joint}"
            operator_data = {
                "articulation_grammar": f"articulation_grammar/{grammar_id}",
                "joint_position_rule": articulation["joint_position_rules"][joint],
                **operator,
            }
            data, summary, queries = _simple_card("articulation_operator", operator_name, operator_data)
            queries.append({
                "command": "evaluate-articulation",
                "arguments": {"grammar": articulation["path"], "pose": "<pose.json>"},
                "use_when": "apply this typed operator inside an exact frame derivation",
            })
            raw.append((f"articulation_operator/{operator_name}", "articulation_operator", operator_name, data, summary, queries))
        for frame, derivation in sorted(articulation["frame_derivations"].items()):
            derivation_name = f"{grammar_id}/{frame}"
            derivation_data = {
                "articulation_grammar": f"articulation_grammar/{grammar_id}",
                **derivation,
            }
            data, summary, queries = _simple_card("articulation_derivation", derivation_name, derivation_data)
            queries.append({
                "command": "evaluate-articulation",
                "arguments": {"grammar": articulation["path"], "target": frame, "pose": "<pose.json>"},
                "use_when": "evaluate this exact root-to-frame operator composition at a new binding",
            })
            raw.append((f"articulation_derivation/{derivation_name}", "articulation_derivation", derivation_name, data, summary, queries))

    constraint_graph = canonical.get("artifacts", {}).get("constraint_graph")
    if isinstance(constraint_graph, dict):
        graph_id = constraint_graph["constraint_graph_id"]
        graph_data = {
            key: constraint_graph[key]
            for key in (
                "path",
                "sha256",
                "schema_version",
                "constraint_graph_id",
                "constraint_graph_sha256",
                "source_binding",
                "structural_graph",
                "executable_contract",
                "coverage",
                "evaluation",
                "epistemic_scope",
            )
        }
        graph_data["attachment_entities"] = [
            f"attachment/{graph_id}/{record['attachment_id']}"
            for record in constraint_graph["attachments"]
        ]
        graph_data["constraint_entities"] = [
            f"constraint/{graph_id}/{record['constraint_id']}"
            for record in constraint_graph["constraints"]
        ]
        data, summary, queries = _simple_card("constraint_graph", graph_id, graph_data)
        queries.extend([
            {
                "command": "evaluate-constraints",
                "arguments": {"graph": constraint_graph["path"], "pose": "<pose.json>"},
                "use_when": "loop closures, cross-branch relations, coordinate couplings, or local mobility must be evaluated at a new binding",
            },
            {
                "command": "verify-constraint-graph",
                "arguments": {"graph": constraint_graph["path"]},
                "use_when": "the graph must be regenerated from the bound grammar and supplemental spec and executed again",
            },
        ])
        raw.append((f"constraint_graph/{graph_id}", "constraint_graph", graph_id, data, summary, queries))
        attachment_evaluations = constraint_graph["evaluation"].get("attachments", {})
        for attachment in constraint_graph["attachments"]:
            attachment_id = attachment["attachment_id"]
            attachment_name = f"{graph_id}/{attachment_id}"
            attachment_data = {
                "constraint_graph": f"constraint_graph/{graph_id}",
                **attachment,
                "evaluation_at_export_pose": attachment_evaluations.get(attachment["frame_id"]),
            }
            data, summary, queries = _simple_card("attachment", attachment_name, attachment_data)
            queries.append({
                "command": "evaluate-constraints",
                "arguments": {"graph": constraint_graph["path"], "pose": "<pose.json>"},
                "use_when": "the root pose of this rigid attachment or its constraint residuals are required at another binding",
            })
            raw.append((f"attachment/{attachment_name}", "attachment", attachment_name, data, summary, queries))
        constraint_evaluations = {
            record["constraint_id"]: record
            for record in constraint_graph["evaluation"].get("constraints", [])
        }
        for constraint in constraint_graph["constraints"]:
            constraint_id = constraint["constraint_id"]
            constraint_name = f"{graph_id}/{constraint_id}"
            constraint_data = {
                "constraint_graph": f"constraint_graph/{graph_id}",
                **constraint,
                "evaluation_at_export_pose": constraint_evaluations.get(constraint_id),
                "assertion_boundary": "the constraint is supplied mechanism semantics, not inferred from the spanning tree and not a physical observation",
            }
            data, summary, queries = _simple_card("constraint", constraint_name, constraint_data)
            queries.extend([
                {
                    "command": "evaluate-constraints",
                    "arguments": {"graph": constraint_graph["path"], "pose": "<pose.json>"},
                    "use_when": "typed residual components and satisfaction at another binding are required",
                },
                {
                    "command": "solve-constraints",
                    "arguments": {"graph": constraint_graph["path"], "pose": "<seed-pose.json>", "solve_for": ["<independent-driver>"]},
                    "use_when": "a local branch solution is required with explicitly chosen solved variables and all others fixed",
                },
            ])
            raw.append((f"constraint/{constraint_name}", "constraint", constraint_name, data, summary, queries))

    configuration_atlas = canonical.get("artifacts", {}).get("configuration_atlas")
    if isinstance(configuration_atlas, dict):
        atlas_id = configuration_atlas["configuration_atlas_id"]
        atlas_data = {
            key: configuration_atlas[key]
            for key in (
                "path",
                "sha256",
                "schema_version",
                "configuration_atlas_id",
                "configuration_atlas_sha256",
                "status",
                "source_binding",
                "coverage",
                "epistemic_scope",
            )
        }
        atlas_data["chart_entities"] = [
            f"configuration_chart/{atlas_id}/{chart['chart_id']}"
            for chart in configuration_atlas["charts"]
        ]
        data, summary, queries = _simple_card("configuration_atlas", atlas_id, atlas_data)
        queries.extend([
            {
                "command": "configuration-atlas",
                "arguments": {"graph": "constraint-graph.json", "spec": "<configuration-atlas-spec.json>"},
                "use_when": "the explicit chart sampling, seeds, or graph artifact changes and finite witnesses must be regenerated",
            },
            {
                "command": "verify-configuration-atlas",
                "arguments": {"graph": "constraint-graph.json", "spec": "<configuration-atlas-spec.json>", "atlas": configuration_atlas["path"]},
                "use_when": "exact regeneration and execution of every stored configuration node must be checked",
            },
        ])
        raw.append((f"configuration_atlas/{atlas_id}", "configuration_atlas", atlas_id, data, summary, queries))
        for chart in configuration_atlas["charts"]:
            chart_id = chart["chart_id"]
            chart_name = f"{atlas_id}/{chart_id}"
            sample_summaries = []
            node_entities = []
            for sample in chart["samples"]:
                sample_nodes = []
                for node in sample["solutions"]:
                    node_name = f"{atlas_id}/{chart_id}/{node['sample_index']:04d}/{node['node_id'].split('/')[-1]}"
                    node_entity = f"configuration_node/{node_name}"
                    sample_nodes.append(node_entity)
                    node_entities.append(node_entity)
                sample_summaries.append({
                    "sample_index": sample["sample_index"],
                    "parameter_value": sample["parameter_value"],
                    "attempt_count": sample["attempt_count"],
                    "converged_attempt_count": sample["converged_attempt_count"],
                    "unique_solution_count": sample["unique_solution_count"],
                    "minimum_solutions_required": sample["minimum_solutions_required"],
                    "coverage_status": sample["coverage_status"],
                    "configuration_node_entities": sample_nodes,
                })
            chart_data = {
                "configuration_atlas": f"configuration_atlas/{atlas_id}",
                **{
                    key: chart[key]
                    for key in (
                        "chart_id",
                        "parameter_driver",
                        "parameter_values",
                        "solve_for",
                        "driver_scales",
                        "seed_contracts",
                        "solution_merge_tolerance_normalized",
                        "continuation_edge_max_distance_normalized",
                        "minimum_solutions_per_sample",
                        "periodic_driver_metric",
                        "observed_rank_reference",
                        "coverage",
                    )
                },
                "samples": sample_summaries,
                "configuration_node_entities": node_entities,
                "component_entities": [
                    f"configuration_component/{atlas_id}/{chart_id}/{component['component_id'].split('/')[-1]}"
                    for component in chart["witness_components"]
                ],
                "interpretation": "finite multi-seed one-parameter exploration; sample minima, proximity components, and observed rank drops are evidence only, not exhaustive topology",
            }
            data, summary, queries = _simple_card("configuration_chart", chart_name, chart_data)
            queries.append({
                "command": "configuration-atlas",
                "arguments": {"graph": "constraint-graph.json", "spec": "<configuration-atlas-spec.json>"},
                "use_when": "this chart contract, seed set, parameter samples, merge tolerance, or proximity metric changes",
            })
            raw.append((f"configuration_chart/{chart_name}", "configuration_chart", chart_name, data, summary, queries))
            node_lookup = {
                node["node_id"]: node
                for sample in chart["samples"]
                for node in sample["solutions"]
            }
            for node in node_lookup.values():
                node_name = f"{atlas_id}/{chart_id}/{node['sample_index']:04d}/{node['node_id'].split('/')[-1]}"
                node_data = {
                    "configuration_atlas": f"configuration_atlas/{atlas_id}",
                    "configuration_chart": f"configuration_chart/{chart_name}",
                    **node,
                    "interpretation": "constraint-satisfying stored witness; rank-drop labels are relative to maxima observed in this finite chart and are not certified singularities",
                }
                data, summary, queries = _simple_card("configuration_node", node_name, node_data)
                queries.append({
                    "command": "evaluate-constraints",
                    "arguments": {"graph": "constraint-graph.json", "pose": node["independent_driver_positions"]},
                    "use_when": "the typed residuals and local Jacobian at this exact stored witness must be re-executed",
                })
                raw.append((f"configuration_node/{node_name}", "configuration_node", node_name, data, summary, queries))
            for component in chart["witness_components"]:
                component_suffix = component["component_id"].split("/")[-1]
                component_name = f"{atlas_id}/{chart_id}/{component_suffix}"
                component_data = {
                    "configuration_atlas": f"configuration_atlas/{atlas_id}",
                    "configuration_chart": f"configuration_chart/{chart_name}",
                    "node_entities": [
                        f"configuration_node/{atlas_id}/{chart_id}/{node_lookup[node_id]['sample_index']:04d}/{node_id.split('/')[-1]}"
                        for node_id in component["node_ids"]
                    ],
                    "source_node_ids": component["node_ids"],
                    "interpretation": "connected component of finite declared proximity edges; not a topological branch certificate",
                }
                data, summary, queries = _simple_card("configuration_component", component_name, component_data)
                raw.append((f"configuration_component/{component_name}", "configuration_component", component_name, data, summary, queries))

    concept_graph = canonical.get("artifacts", {}).get("concept_graph")
    if isinstance(concept_graph, dict):
        concept_id = concept_graph["concept_graph_id"]
        concept_data = {
            key: concept_graph[key]
            for key in (
                "path",
                "sha256",
                "schema_version",
                "concept_graph_id",
                "concept_graph_sha256",
                "language_path",
                "language_sha256",
                "coverage",
                "epistemic_scope",
            )
        }
        concept_name = concept_id.removeprefix("concept_graph/")
        data, summary, queries = _simple_card("concept_graph", concept_name, concept_data)
        queries.extend([
            {
                "command": "query-concepts",
                "arguments": {"graph": concept_graph["path"], "query": "<robot-spatial-concept-query.v1.json>"},
                "use_when": "a structural summary, exact tree path, driver effect, frame law, asserted constraint, finite-node comparison, or entity proof closure is required",
            },
            {
                "command": "verify-concept-graph",
                "arguments": {
                    "concept": concept_graph["path"],
                    "language": concept_graph["language_path"],
                },
                "use_when": "the concept clauses and controlled-language rendering must be regenerated from their exact bound artifacts",
            },
        ])
        raw.append((concept_id, "concept_graph", concept_name, data, summary, queries))

    functional_artifact = canonical.get("artifacts", {}).get("functional_model")
    if isinstance(functional_artifact, dict) and isinstance(functional_model, dict):
        functional_id = functional_model["functional_model_id"]
        model_name = functional_id.removeprefix("functional_model/")
        model_data = {
            **functional_artifact,
            "function_set_id": functional_model["function_set_id"],
            "ontology_contract": functional_model["ontology_contract"],
            "query_contract": functional_model["query_contract"],
            "structural_evidence_clause_count": len(functional_model["structural_evidence_clauses"]),
        }
        data, summary, queries = _simple_card("functional_model", model_name, model_data)
        queries.extend([
            {
                "command": "query-functions",
                "arguments": {
                    "model": functional_artifact["path"],
                    "query": "<robot-spatial-functional-query.v1.json>",
                },
                "use_when": "project-declared component function, capability grounding, relational affordance, conditional action, or scoped inventory absence is required",
            },
            {
                "command": "verify-functional-model",
                "arguments": {
                    "model": functional_artifact["path"],
                    "spec": "<robot-spatial-function-affordance-spec.v1.json>",
                },
                "use_when": "the functional assertions and their structural grounding must be exactly regenerated",
            },
        ])
        raw.append((functional_id, "functional_model", model_name, data, summary, queries))

        projection_contract = (
            ("object_types", "object_type_id", "functional_object_type", None),
            ("components", "component_id", "functional_component", "describe_component"),
            ("functions", "function_id", "functional_function", "explain_function"),
            ("conditions", "condition_id", "functional_condition", None),
            ("effects", "effect_id", "functional_effect", None),
            ("capabilities", "capability_id", "functional_capability", "explain_capability"),
            ("affordances", "affordance_id", "functional_affordance", "explain_affordance"),
        )
        for collection, id_key, entity_type, query_intent in projection_contract:
            for record in functional_model["projections"][collection]:
                entity_id = record[id_key]
                entity_name = entity_id.split("/", 1)[1]
                entity_data = {
                    **record,
                    "functional_model": functional_id,
                    "epistemic_class": "project_declared_function_knowledge",
                }
                data, summary, queries = _simple_card(entity_type, entity_name, entity_data)
                if query_intent is not None:
                    parameter_name = {
                        "describe_component": "component",
                        "explain_function": "function",
                        "explain_capability": "capability",
                        "explain_affordance": "affordance",
                    }[query_intent]
                    queries.append({
                        "command": "query-functions",
                        "arguments": {
                            "model": functional_artifact["path"],
                            "query": {
                                "intent": query_intent,
                                "parameters": {parameter_name: entity_id},
                            },
                        },
                        "use_when": "the declaration plus its complete functional and structural proof closure is required",
                    })
                raw.append((entity_id, entity_type, entity_name, data, summary, queries))

    motion_atlas = canonical.get("artifacts", {}).get("counterfactual_motion_atlas")
    if isinstance(motion_atlas, dict):
        motion_id = motion_atlas["motion_id"]
        atlas_data = {
            key: motion_atlas[key]
            for key in (
                "path",
                "manifest_sha256",
                "schema_version",
                "motion_id",
                "motion_input_sha256",
                "baseline_pose_binding",
                "perturbation_policy",
                "coordinate_contract",
                "coverage",
                "epistemic_scope",
            )
        }
        atlas_data["driver_entities"] = [
            f"motion_driver/{motion_id}/{driver}" for driver in sorted(motion_atlas["drivers"])
        ]
        data, summary, queries = _simple_card("motion_atlas", motion_id, atlas_data)
        queries.append({
            "command": "verify-motion-atlas",
            "arguments": {"atlas": motion_atlas["path"], "pose": canonical["pose"]["name"]},
            "use_when": "signed endpoint FK, causality deltas, shared view mappings, typed entities, and SVG digests must be regenerated",
        })
        raw.append((f"motion_atlas/{motion_id}", "motion_atlas", motion_id, data, summary, queries))
        for driver, driver_record in sorted(motion_atlas["drivers"].items()):
            driver_name = f"{motion_id}/{driver}"
            driver_data = {
                key: driver_record[key]
                for key in (
                    "driver_joint",
                    "joint_type",
                    "joint_position_unit",
                    "baseline_position",
                    "nominal_step",
                    "feasible_interval",
                    "physical_joints_driven",
                    "baseline_physical_joint_positions",
                    "structural_causality",
                    "endpoints",
                )
            }
            driver_data["motion_atlas"] = f"motion_atlas/{motion_id}"
            driver_data["view_entities"] = [
                f"motion_view/{motion_id}/{driver}/{view_id}" for view_id in sorted(driver_record["views"])
            ]
            data, summary, queries = _simple_card("motion_driver", driver_name, driver_data)
            queries.extend([
                {
                    "command": "motion-atlas",
                    "arguments": {"pose": canonical["pose"]["name"]},
                    "use_when": "a fresh finite signed perturbation record is required",
                },
                {
                    "command": "jacobian",
                    "arguments": {"target": "<frame>", "frame": canonical["robot"]["root_link"], "pose": canonical["pose"]["name"]},
                    "use_when": "the infinitesimal derivative must be distinguished from a finite endpoint delta",
                },
            ])
            raw.append((f"motion_driver/{driver_name}", "motion_driver", driver_name, data, summary, queries))
            for view_id, view in sorted(driver_record["views"].items()):
                view_name = f"{motion_id}/{driver}/{view_id}"
                view_data = {
                    "motion_atlas": f"motion_atlas/{motion_id}",
                    "motion_driver": f"motion_driver/{motion_id}/{driver}",
                    "view_id": view_id,
                    "title": view["title"],
                    "projection": view["projection"],
                    "screen": view["screen"],
                    "combined_projection_bounds_uv_m": view["combined_projection_bounds_uv_m"],
                    "samples": view["samples"],
                    "motion_vectors": view["motion_vectors"],
                    "artifact": view["artifact"],
                    "interpretation": "finite exact-FK signed endpoints on one shared screen; not interpolation, continuous motion, dynamics, or physical evidence",
                }
                data, summary, queries = _simple_card("motion_view", view_name, view_data)
                queries.append({
                    "command": "verify-motion-atlas",
                    "arguments": {"atlas": motion_atlas["path"], "pose": canonical["pose"]["name"]},
                    "use_when": "endpoint pixel displacement or the overlay SVG must be trusted",
                })
                raw.append((f"motion_view/{view_name}", "motion_view", view_name, data, summary, queries))

    world_scene = canonical.get("world_scene", {})
    if world_scene.get("status") == "parsed_validated_and_bound":
        mount = world_scene["robot_mount"]
        instance_name = mount["instance_id"]
        instance_data = {
            **mount,
            "scene_id": world_scene["scene_id"],
            "snapshot": world_scene["snapshot"],
            "scene_gravity": world_scene["gravity"],
            "robot_environment_collision_summary": {
                "status": world_scene["robot_environment_collision"]["status"],
                "minimum_separation": world_scene["robot_environment_collision"]["minimum_separation"],
                "coverage": world_scene["robot_environment_collision"]["coverage"],
            },
        }
        observed = canonical.get("observed_world", {}).get("observation")
        if isinstance(observed, dict):
            instance_data["effective_root_at_observation_query"] = {
                "query": observed["query"],
                "pose": observed["effective_state"]["world_from_robot_root"],
                "source": observed["effective_state"]["sources"]["robot_root"],
                "selection": observed["selections"]["robot_root_pose"],
            }
        data, summary, queries = _simple_card("robot_instance", instance_name, instance_data)
        queries.extend([
            {"command": "scene-transform", "arguments": {"from": f"scene_frame/{world_scene['world_frame']}", "to": mount["root_entity"], "pose": "<pose.json>"}, "use_when": "the mounted world pose of the robot root is required"},
            {"command": "scene-collisions", "arguments": {"scene": "<scene.json>", "pose": "<pose.json>"}, "use_when": "robot versus declared environment collision or clearance is required"},
            {"command": "scene-gravity-loads", "arguments": {"scene": "<scene.json>", "pose": "<pose.json>"}, "use_when": "world-scene gravity loads are required"},
        ])
        raw.append((f"robot_instance/{instance_name}", "robot_instance", instance_name, data, summary, queries))

        typed_poses = world_scene["typed_frame_poses_in_world"]
        scene_frame_names = [world_scene["world_frame"], *sorted(world_scene.get("scene_frames", {}))]
        for frame_name in scene_frame_names:
            entity_id = f"scene_frame/{frame_name}"
            if frame_name == world_scene["world_frame"]:
                frame_data = {
                    "name": frame_name,
                    "parent": None,
                    "world_frame": True,
                    "world_from_frame": typed_poses[entity_id],
                    "semantics": {"category": "world", "roles": ["world_frame"], "meaning": "root of this scene snapshot"},
                    "source": world_scene["source"]["provenance"],
                }
            else:
                frame_data = world_scene["scene_frames"][frame_name]
            data, summary, queries = _simple_card("scene_frame", frame_name, frame_data)
            queries.append({"command": "scene-transform", "arguments": {"from": entity_id, "to": "<typed-entity>", "pose": "<pose.json>"}, "use_when": "a snapshot-bound transform to another scene or robot entity is required"})
            raw.append((entity_id, "scene_frame", frame_name, data, summary, queries))

        collision_geometry = world_scene["robot_environment_collision"]["geometry_analysis"]
        for geometry_id, geometry_data in sorted(collision_geometry.items()):
            if not geometry_id.startswith("robot_geometry/"):
                continue
            short_name = geometry_id.removeprefix("robot_geometry/")
            data, summary, queries = _simple_card("robot_geometry", short_name, geometry_data)
            queries.extend([
                {"command": "scene-transform", "arguments": {"from": f"scene_frame/{world_scene['world_frame']}", "to": geometry_id, "pose": "<pose.json>"}, "use_when": "the mounted world pose of this robot geometry frame is required"},
                {"command": "scene-collisions", "arguments": {"scene": "<scene.json>", "pose": "<pose.json>"}, "use_when": "environment pair status, witness points, or clearance is required"},
            ])
            raw.append((geometry_id, "robot_geometry", short_name, data, summary, queries))
        for object_id, object_record in sorted(world_scene.get("objects", {}).items()):
            entity_id = f"scene_object/{object_id}"
            geometry_ids = [record["entity_id"] for record in object_record["collision_geometries"]]
            object_data = {**object_record, "collision_geometry_entities": geometry_ids}
            if isinstance(observed, dict):
                object_data["effective_pose_at_observation_query"] = {
                    "query": observed["query"],
                    "pose": observed["effective_state"]["world_from_objects"].get(object_id),
                    "source": observed["effective_state"]["sources"]["objects"].get(object_id),
                    "selection": observed["selections"]["object_poses"].get(object_id),
                }
            data, summary, queries = _simple_card("scene_object", object_id, object_data)
            queries.extend([
                {"command": "scene-transform", "arguments": {"from": f"scene_frame/{world_scene['world_frame']}", "to": entity_id}, "use_when": "the object pose in world is required"},
                {"command": "scene-collisions", "arguments": {"scene": "<scene.json>", "pose": "<pose.json>"}, "use_when": "robot collision or clearance against this object is required"},
            ])
            raw.append((entity_id, "scene_object", object_id, data, summary, queries))
            for geometry_id in geometry_ids:
                short_name = geometry_id.removeprefix("scene_geometry/")
                geometry_data = collision_geometry[geometry_id]
                data, summary, queries = _simple_card("scene_geometry", short_name, geometry_data)
                queries.extend([
                    {"command": "scene-transform", "arguments": {"from": f"scene_frame/{world_scene['world_frame']}", "to": geometry_id}, "use_when": "the collision-geometry frame pose is required"},
                    {"command": "scene-collisions", "arguments": {"scene": "<scene.json>", "pose": "<pose.json>"}, "use_when": "pair status, witness points, or minimum separation is required"},
                ])
                raw.append((geometry_id, "scene_geometry", short_name, data, summary, queries))

    observed = canonical.get("observed_world", {}).get("observation")
    if isinstance(observed, dict):
        log = observed["observation_log"]
        log_data = {
            "observation_log": log,
            "query_source": observed.get("query_source"),
            "query": observed["query"],
            "selection_method": observed["selection_method"],
            "selections": observed["selections"],
            "effective_state": observed["effective_state"],
            "readiness": observed["readiness"],
            "epistemic_layers": observed["epistemic_layers"],
            "analysis": canonical["observed_world"].get("analysis"),
        }
        data, summary, queries = _simple_card("observation_log", log["id"], log_data)
        queries.extend([
            {"command": "observe-summary", "arguments": {"scene": "<scene.json>", "observations": "<observations.json>", "observation_query": "<query.json>"}, "use_when": "the complete time-selected nominal world state is required"},
            {"command": "observe-transform", "arguments": {"from": "<typed-entity>", "to": "<typed-entity>"}, "use_when": "an observation-conditioned typed transform is required"},
            {"command": "observe-collisions", "arguments": {}, "use_when": "nominal declared geometry collision under selected observations is required"},
        ])
        raw.append((f"observation_log/{log['id']}", "observation_log", log["id"], data, summary, queries))
        normalization = log.get("normalization")
        if isinstance(normalization, dict):
            capture_id = normalization["capture_id"]
            capture_data = {
                "capture_id": capture_id,
                "capture_sha256": normalization["capture_sha256"],
                "adapter_id": normalization["adapter_id"],
                "config_sha256": normalization["config_sha256"],
                "normalization_method": normalization["method"],
                "clock_policy": normalization["clock_policy"],
                "authority_policy": normalization["authority_policy"],
                "tf_policy": normalization["tf_policy"],
                "normalized_observation_log": f"observation_log/{log['id']}",
            }
            data, summary, queries = _simple_card("ros_capture", capture_id, capture_data)
            queries.append({
                "command": "ros_observation_adapter.py normalize",
                "arguments": {"config": "<adapter-config.json>", "capture": "<ros-capture.json>", "scene": "<scene.json>"},
                "use_when": "the exact ROS capture must be deterministically regenerated into its observation log",
            })
            raw.append((f"ros_capture/{capture_id}", "ros_capture", capture_id, data, summary, queries))

    actuation = canonical.get("actuation", {})
    for name, system in sorted(actuation.get("ros2_control_systems", {}).items()):
        data, summary, queries = _simple_card("ros2_control_system", name, system)
        queries.append({"command": "actuation", "arguments": {"system": name}, "use_when": "the complete embedded declaration is required"})
        raw.append((f"ros2_control_system/{name}", "ros2_control_system", name, data, summary, queries))
        for sensor_name, sensor in sorted(system.get("sensors", {}).items()):
            sensor_id = f"{name}/{sensor_name}"
            data, summary, queries = _simple_card("control_sensor", sensor_id, {"system": name, **sensor})
            raw.append((f"control_sensor/{sensor_id}", "control_sensor", sensor_id, data, summary, queries))
        for gpio_name, gpio in sorted(system.get("gpios", {}).items()):
            gpio_id = f"{name}/{gpio_name}"
            data, summary, queries = _simple_card("control_gpio", gpio_id, {"system": name, **gpio})
            raw.append((f"control_gpio/{gpio_id}", "control_gpio", gpio_id, data, summary, queries))
    for name, transmission in sorted(actuation.get("legacy_transmissions", {}).items()):
        data, summary, queries = _simple_card("transmission", name, transmission)
        queries.append({"command": "actuation", "arguments": {"transmission": name}, "use_when": "the complete embedded declaration is required"})
        raw.append((f"transmission/{name}", "transmission", name, data, summary, queries))
        for actuator in transmission.get("actuators", []):
            actuator_id = f"{name}/{actuator['name']}"
            data, summary, queries = _simple_card("actuator", actuator_id, {"transmission": name, **actuator})
            raw.append((f"actuator/{actuator_id}", "actuator", actuator_id, data, summary, queries))

    semantics = canonical.get("semantics", {})
    for name, group in sorted(semantics.get("groups", {}).items()):
        data, summary, queries = _simple_card("semantic_group", name, group)
        raw.append((f"group/{name}", "semantic_group", name, data, summary, queries))
    for name, end_effector in sorted(semantics.get("end_effectors", {}).items()):
        data, summary, queries = _simple_card("semantic_end_effector", name, end_effector)
        raw.append((f"end_effector/{name}", "semantic_end_effector", name, data, summary, queries))

    srdf = canonical.get("srdf", {})
    for name, group in sorted(srdf.get("groups", {}).items()):
        data, summary, queries = _simple_card("srdf_group", name, group)
        raw.append((f"srdf_group/{name}", "srdf_group", name, data, summary, queries))
    for name, end_effector in sorted(srdf.get("end_effectors", {}).items()):
        data, summary, queries = _simple_card("srdf_end_effector", name, end_effector)
        raw.append((f"srdf_end_effector/{name}", "srdf_end_effector", name, data, summary, queries))

    invariant_validation = canonical.get("invariant_validation", {})
    for result in invariant_validation.get("results", []):
        name = result["id"]
        data, summary, queries = _simple_card("invariant", name, result)
        raw.append((f"invariant/{name}", "invariant", name, data, summary, queries))

    cards: list[dict[str, Any]] = []
    functional_types = {
        "functional_model",
        "functional_object_type",
        "functional_component",
        "functional_function",
        "functional_condition",
        "functional_effect",
        "functional_capability",
        "functional_affordance",
    }
    for entity_id, entity_type, name, data, summary, queries in sorted(raw):
        fact_ids = _related_fact_ids(entity_id, facts)
        trust = _trust_record(fact_ids, facts_by_id)
        if entity_type in functional_types:
            trust = {
                "classification": "project_asserted_with_digest_bound_proof_model",
                "bound_fact_count": 0,
                "exact_fact_count": 0,
                "nonexact_fact_count": 0,
                "source_type_counts": {"project_function_affordance_spec": 1},
            }
        cards.append({
            "schema_version": CARD_SCHEMA,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "name": name,
            "summary_cnl": summary,
            "data": data,
            "tool_queries": queries,
            "fact_ids": fact_ids,
            "trust": trust,
        })
    return cards


def _unresolved_claims(canonical: dict[str, Any]) -> list[dict[str, str]]:
    unresolved: list[dict[str, str]] = []
    if canonical.get("semantics", {}).get("status") == "not_provided":
        unresolved.append({"topic": "semantic_roles", "status": "not_established", "instruction": "do not infer base, flange, TCP, group, or end-effector roles from names"})
    if canonical.get("srdf", {}).get("status") == "not_provided":
        unresolved.append({"topic": "planning_semantics", "status": "not_provided", "instruction": "do not claim SRDF groups, named poses, end effectors, or disabled-collision policy"})
    unmeasured_by_kind = {
        kind: sorted(
            name
            for name, record in canonical.get("geometry_analysis", {}).items()
            if record.get("kind") == kind and record.get("status") != "measured"
        )
        for kind in ("visual", "collision")
    }
    for kind, unmeasured in unmeasured_by_kind.items():
        if unmeasured:
            unresolved.append({
                "topic": f"{kind}_mesh_shape",
                "status": "not_measured",
                "instruction": (
                    f"do not make {kind} mesh shape claims for {unmeasured}; "
                    f"rerun with --inspect-mesh-kind {kind} and a package map when needed"
                ),
            })
    surface = canonical.get("collision_surface", {})
    if surface.get("status") == "not_requested":
        unresolved.append({"topic": "self_collision", "status": "not_requested", "instruction": "AABB candidates are not triangle collision results; run surface-collisions"})
    elif surface.get("self_collision_status") == "indeterminate":
        unresolved.append({"topic": "self_collision", "status": "indeterminate", "instruction": "do not claim collision freedom"})
    if canonical.get("kinematic_analysis", {}).get("targets"):
        unresolved.append({"topic": "reachable_workspace", "status": "finite_sample_only", "instruction": "sampled AABBs do not prove the complete reachable set or reachability of interior points"})
    mass_properties = canonical.get("physical_analysis", {}).get("declared_mass_properties", {})
    mass_coverage = mass_properties.get("coverage", {})
    if mass_properties.get("status") == "indeterminate":
        unresolved.append({"topic": "declared_mass_properties", "status": "indeterminate", "instruction": "fix invalid or incomplete inertial declarations before reporting aggregate mass, center of mass, or inertia"})
    elif mass_properties.get("status") == "not_provided":
        unresolved.append({"topic": "declared_mass_properties", "status": "not_provided", "instruction": "the URDF declares no inertial properties; do not infer mass or center of mass from geometry"})
    elif mass_coverage.get("missing_inertial_links"):
        unresolved.append({"topic": "physical_mass_completeness", "status": "not_established", "instruction": "aggregate mass properties cover declared inertials only; links without inertial elements are not proof of zero physical mass"})
    gravity = canonical.get("physical_analysis", {}).get("declared_static_gravity_loads_under_root_frame_convention", {})
    if gravity.get("status") == "indeterminate":
        unresolved.append({"topic": "static_gravity_loads", "status": "indeterminate", "instruction": "fix invalid or incomplete selected inertials before reporting gravity generalized forces"})
    elif gravity.get("status") == "not_provided":
        unresolved.append({"topic": "static_gravity_loads", "status": "not_provided", "instruction": "do not infer gravity loads without valid inertial declarations"})
    else:
        unresolved.append({
            "topic": "physical_gravity_and_static_effort",
            "status": "model_convention_only",
            "instruction": "the export assumes gravity [0,0,-9.80665] in the URDF root; actual mounting orientation, payload, contacts, friction, transmissions, and hardware feasibility are not established",
        })
    actuation = canonical.get("actuation", {})
    has_actuation = bool(actuation.get("ros2_control_systems") or actuation.get("legacy_transmissions"))
    unresolved.append({
        "topic": "actuation_runtime_and_hardware",
        "status": "declarations_only" if has_actuation else "not_declared_in_expanded_urdf",
        "instruction": (
            "embedded control declarations do not prove plugin availability, external controller configuration, interface claiming, hardware connectivity, calibration, or command execution"
            if has_actuation
            else "no embedded actuation declaration is not proof that the physical robot is unactuated or uncontrolled"
        ),
    })
    world_scene = canonical.get("world_scene", {})
    if world_scene.get("status") != "parsed_validated_and_bound":
        unresolved.append({
            "topic": "world_root_mount_environment_and_gravity",
            "status": "not_provided",
            "instruction": "robot-local URDF coordinates do not establish a world pose, real gravity direction, or external obstacles; provide a robot-spatial-world-scene.v1 snapshot",
        })
    else:
        unresolved.append({
            "topic": "physical_world_snapshot_truth",
            "status": "declared_snapshot_only",
            "instruction": "scene transforms and collisions are exact consequences of the named static snapshot, but the parser does not prove that its calibration, timestamp, object set, or geometry matches the current physical world",
        })
        scene_collision = world_scene["robot_environment_collision"]
        if scene_collision["status"] == "indeterminate" or scene_collision["coverage"]["indeterminate_pair_count"]:
            unresolved.append({
                "topic": "robot_environment_collision_coverage",
                "status": "partial_or_indeterminate",
                "instruction": "inspect the per-pair coverage; a collision result may coexist with other indeterminate pairs, and collision-free claims require all relevant pairs to be resolved",
            })
        scene_gravity = canonical.get("physical_analysis", {}).get("declared_static_gravity_loads_under_scene_gravity", {})
        unresolved.append({
            "topic": "scene_gravity_physical_truth",
            "status": "snapshot_convention_only" if scene_gravity.get("status") != "not_provided" else "not_provided",
            "instruction": "world-scene gravity loads use the declared root mount and gravity vector; their physical truth is limited by supplied provenance and still excludes payload, contact, motion, transmission loss, controller, and hardware behavior",
        })
    observation = canonical.get("observed_world", {}).get("observation")
    if not isinstance(observation, dict):
        unresolved.append({
            "topic": "timestamped_observed_state",
            "status": "not_provided",
            "instruction": "model and static-scene declarations do not establish what was observed at a particular time; provide a bound observation log and explicit query-time age/fallback policy",
        })
    else:
        unresolved.append({
            "topic": "observation_source_truth_and_physical_completeness",
            "status": "not_established",
            "instruction": "a current time selection establishes sample age only; calibration, covariance-bounded geometry, source truth, omitted-object absence, continuous-time collision, and physical safety remain unverified",
        })
        normalization = observation.get("observation_log", {}).get("normalization")
        if isinstance(normalization, dict):
            unresolved.append({
                "topic": "ros_capture_clock_authority_and_transport_truth",
                "status": "not_established",
                "instruction": "capture/config digests and conflict rejection make normalization reproducible, but clock synchronization, hidden rosbag publishers, transport completeness, publisher truth, calibration, and physical agreement remain unverified",
            })
        if not observation["readiness"]["all_required_observations_current"]:
            unresolved.append({
                "topic": "observation_temporal_readiness",
                "status": observation["status"],
                "instruction": "do not describe stale, missing, or static-declaration fallback state as a current observation; inspect every selected stream and fallback entity",
            })
    render_atlas = canonical.get("artifacts", {}).get("semantic_render_atlas")
    if not isinstance(render_atlas, dict):
        unresolved.append({
            "topic": "semantic_visual_grounding",
            "status": "not_generated",
            "instruction": "run export --render when fixed-view semantic projections would help; do not infer unrendered shape from names",
        })
    else:
        if not render_atlas["coverage"]["complete_for_declared_geometry"]:
            unresolved.append({
                "topic": "semantic_render_geometry_coverage",
                "status": "partial",
                "instruction": f"the atlas omits {render_atlas['coverage']['unrendered_geometry_frames']}; do not generalize visible hulls to those frames",
            })
        unresolved.append({
            "topic": "photorealistic_visibility_and_camera_truth",
            "status": "not_established",
            "instruction": "semantic views encode deterministic convex projections only; occlusion, surface appearance, perspective, calibrated camera pixels, and physical-scene agreement are not established",
        })
    motion_atlas = canonical.get("artifacts", {}).get("counterfactual_motion_atlas")
    if not isinstance(motion_atlas, dict):
        unresolved.append({
            "topic": "counterfactual_joint_motion_grounding",
            "status": "not_generated",
            "instruction": "run export --motion-atlas when the question asks what changes, in which direction, or why a joint is a cause rather than only a structural edge",
        })
    else:
        unavailable = (
            motion_atlas["coverage"]["requested_signed_endpoint_count"]
            - motion_atlas["coverage"]["available_signed_endpoint_count"]
        )
        if unavailable:
            unresolved.append({
                "topic": "motion_endpoint_limit_coverage",
                "status": "partial",
                "instruction": f"{unavailable} requested signed counterfactual endpoints are unavailable at feasible limits; preserve the one-sided result instead of inventing motion beyond a limit",
            })
        unresolved.append({
            "topic": "continuous_motion_dynamics_and_hardware_truth",
            "status": "not_established",
            "instruction": "motion-atlas samples are exact finite FK endpoints only; they do not establish intermediate paths, swept volume, velocity, acceleration, effort, controller response, hardware motion, collision safety, or time",
        })
    articulation = canonical.get("artifacts", {}).get("articulation_grammar")
    if not isinstance(articulation, dict):
        unresolved.append({
            "topic": "pose_independent_articulation_law",
            "status": "not_generated",
            "instruction": "generate an articulation grammar before generalizing finite poses or motion-atlas endpoints into a joint law",
        })
    else:
        unresolved.append({
            "topic": "articulation_beyond_supported_source_tree",
            "status": "not_established",
            "instruction": "the grammar is exact for the normalized supported URDF/SDF/MJCF tree subset only; read a supplemental constraint graph before treating that spanning tree as the complete mechanism",
        })
    constraint_graph = canonical.get("artifacts", {}).get("constraint_graph")
    if not isinstance(constraint_graph, dict):
        unresolved.append({
            "topic": "supplemental_mechanism_constraints",
            "status": "not_provided",
            "instruction": "absence of a constraint graph means closed loops, cross-branch attachments, and coordinate couplings are unknown—not absent; do not call the spanning tree the complete mechanism without external evidence",
        })
    else:
        configuration_atlas = canonical.get("artifacts", {}).get("configuration_atlas")
        unresolved.extend([
            {
                "topic": "global_mechanism_configuration_space",
                "status": (
                    "finite_declared_chart_evidence_only"
                    if isinstance(configuration_atlas, dict)
                    else "not_explored_beyond_pose_local_analysis"
                ),
                "instruction": (
                    "the configuration atlas supplies finite multi-seed nodes, proximity components, and rank-drop candidates only; it does not prove exhaustive branch count, certified singularities, global topology, or complete reachability"
                    if isinstance(configuration_atlas, dict)
                    else "constraint satisfaction and numerical rank are pose-conditioned; generate a configuration atlas before making even finite cross-pose branch or rank-drop claims, and never promote finite exploration to global proof"
                ),
            },
            {
                "topic": "constraint_physical_truth",
                "status": "asserted_mechanism_semantics_only",
                "instruction": "the supplemental spec is digest-bound and executable but remains supplied intent; it does not prove assembly, compliance, backlash, contact, calibration, dynamics, or agreement with hardware",
            },
        ])
        if isinstance(configuration_atlas, dict) and configuration_atlas["status"] != "complete_for_declared_sampling":
            unresolved.append({
                "topic": "configuration_atlas_declared_sample_coverage",
                "status": "below_declared_minimum",
                "instruction": "one or more parameter samples found fewer unique satisfying solutions than the declared minimum; report the exact deficient samples and do not treat missing solutions as proof of absence",
            })
    functional_model = canonical.get("artifacts", {}).get("functional_model")
    if not isinstance(functional_model, dict):
        unresolved.append({
            "topic": "component_function_capability_and_affordance",
            "status": "not_provided",
            "instruction": "do not infer what a link, joint, shape, or named component is for; provide a function/affordance spec and query its compiled functional model",
        })
    else:
        if not functional_model["coverage"]["all_declared_capabilities_structurally_grounded"]:
            unresolved.append({
                "topic": "declared_capability_structural_grounding",
                "status": "one_or_more_requirements_not_grounded",
                "instruction": "inspect every capability requirement result; do not treat an ungrounded declaration as supported by the bound structural model",
            })
        unresolved.append({
            "topic": "functional_physical_execution_and_safety_truth",
            "status": "not_established",
            "instruction": "function and affordance intent are project assertions; structural requirement satisfaction does not establish runtime preconditions, physical executability, observed effects, controller/hardware behavior, payload, contact mechanics, or safety",
        })
    for warning in canonical.get("validation", {}).get("warnings", []):
        unresolved.append({"topic": "parser_warning", "status": "warning", "instruction": warning})
    return unresolved


def _guide_markdown(canonical: dict[str, Any]) -> str:
    robot = canonical["robot"]["name"]
    root = canonical["robot"]["root_link"]
    pose = canonical["pose"]["name"]
    return f"""# Agent spatial context guide

This pack is a progressive-disclosure interface for `{robot}` at exported pose `{pose}`.

1. Read `agent-context.json` first. It defines units, transform direction, trust labels, unresolved claims, and artifact digests.
2. For compositional structure, use `query-concepts` against `concept-graph.json`; it returns a typed answer plus the minimal premise closure. Read `concept-language.rsl` only when a compact whole-robot symbolic view is useful.
3. For what a component is for, what capability is declared, or what action it affords, use `query-functions` against `functional-model.json`. Keep project assertions, deterministic structural grounding, unevaluated preconditions, intended effects, and physical execution as distinct modalities.
4. Load one record from `entity-cards.jsonl` by exact typed ID such as `joint/shoulder`, `link/tool0`, `frame/tool0`, `component/gripper`, or `capability/grasp`. Link and frame IDs are intentionally distinct.
4. Treat `summary_cnl` as a compact index, not independent evidence. Follow its `fact_ids` into `facts.jsonl` for provenance and exact/approximate status.
5. Use `retrieve` to verify artifact digests and select only the relevant card/facts. Use the kinematic CLI for a pose, frame pair, or relationship not already exported.
5. Never derive a relative transform by subtracting root-frame positions. Run `transform`; `{root}_from_X` means the pose of X expressed in `{root}`.
6. If a world scene is present, keep `frame/X` (robot-local) distinct from `robot_frame/X` (snapshot-bound), and use `scene-transform` across scene and robot namespaces.
7. If observations are present, read `observation_log/X` before using effective poses. Keep URDF model facts, static-scene declarations, and time-selected source reports as three separate epistemic layers.
8. If the observation log names ROS normalization, read `ros_capture/X` before trusting assembled joints or composed TF poses. Verify capture/config digests and keep transport authority/clock limits visible.
9. If a semantic render atlas is present, read `render_atlas/X` and one exact `render_view/X/view` card before using an image. Treat the SVG and numeric projection as two encodings of the same canonical geometry, then run `verify-render` when consistency matters.
10. Read `articulation_grammar/X` before making a general motion-law claim. Keep `source_binding` distinct from `law_identity`: source bytes/provenance are not the canonical law. Then load one `articulation_variable/X/joint`, `articulation_operator/X/joint`, and `articulation_derivation/X/frame` card; use `evaluate-articulation` for a new pose, `verify-articulation-grammar` for source regeneration/FK checks, and `compare-articulation-grammars` for cross-representation equivalence.
11. If `constraint_graph/X` exists, it is the mechanism layer above the spanning-tree parameterization. Read the graph, exact `attachment/X/name`, and `constraint/X/name` cards before claiming a loop, coupling, full-mechanism mobility, or valid configuration. Use `evaluate-constraints` for typed residuals, `solve-constraints` only as a local branch solver with explicit variables, and `verify-constraint-graph` for binding/regeneration checks.
12. If a configuration atlas is present, read `configuration_atlas/X`, the relevant `configuration_chart/X/chart`, and exact `configuration_node/X/chart/sample/solution` cards. Treat nodes as executable satisfying witnesses, proximity components as finite connectivity hints, and rank drops only as candidates relative to the maximum observed in that declared chart. Run `verify-configuration-atlas` before trusting regeneration or stored nodes.
13. If a counterfactual motion atlas is present, read `motion_atlas/X`, one `motion_driver/X/joint`, and one `motion_view/X/joint/view` card. Distinguish structural causality, finite endpoint displacement, and infinitesimal Jacobian effects; run `verify-motion-atlas` before trusting endpoint/view consistency.

Hard rules:

- lengths are meters, joint angles are radians, and quaternions are xyzw;
- a joint axis is declared in its pre-motion joint frame and may rotate when expressed elsewhere;
- link, joint-pre-motion, visual, collision, and inertial frames are different entities even if coordinates coincide;
- an inertial origin is the URDF-declared center-of-mass and inertia-coordinate frame; use `mass-properties` for aggregation instead of averaging link origins;
- declared mass properties are exact only for the modeled inertials; missing inertial elements do not prove zero physical mass, and no URDF result proves agreement with hardware or payload;
- `gravity-loads` is a gravity-only static model: state the gravity vector/frame and sign convention; it excludes velocity, acceleration, contact, friction, payload, transmission loss, control, and hardware truth;
- `scene-gravity-loads` first rotates the scene-declared world gravity through the declared root mounting; it is still conditional on snapshot provenance and is not a sensor measurement;
- scene results are valid only for their exact `scene_id`, `snapshot_id`, scene digest, robot pose, and contact tolerance; never assume the snapshot remains current;
- observation results are additionally valid only for their log digest, clock domain, query time, maximum-age policy, selected sample IDs, and explicit fallback policy; future samples are never valid evidence for an earlier query;
- ROS-normalized observations are additionally valid only for their capture/config digests, partial-joint assembly age, target TF path age, and visible authority policy; a publisher GID, topic name, or clock-domain label does not prove source truth or synchronization;
- semantic render views are deterministic convex projections with explicit root-to-UV and UV-to-pixel mappings; they do not compute photorealistic visibility, occlusion, perspective, physical camera calibration, or independent geometry truth;
- counterfactual motion views hold every other independent driver fixed and compare exact finite endpoints on one shared screen; they are not trajectories, continuous swept volumes, dynamics, command execution, or physical motion evidence;
- articulation grammar is the pose-independent typed law; FK is one evaluation, Jacobian is a local derivative, and motion atlas is a finite sample. Never substitute one layer for another;
- when a supplemental constraint graph exists, the articulation tree is a coordinate parameterization, not the complete mechanism; evaluate every asserted closure/coupling before calling a pose valid;
- constraint residuals are exact consequences of the embedded tree law plus asserted supplement, but the assertions are not inferred physical observations; numerical local mobility may change at singular configurations and is not global DOF;
- configuration-atlas solution nodes are finite local-solver witnesses whose residuals are re-executed exactly within declared tolerances; seed coverage is not exhaustive, a proximity component is not a branch certificate, and an observed rank drop is not a certified singularity or global topology result;
- `current` means a selected sample passed the query age threshold, not that the source is truthful or calibrated; a static-declaration fallback is not an observation;
- omitted scene objects are unknown, not proof of empty space; `collision_free` means collision-free against the complete declared geometry set for that snapshot and coverage;
- ros2_control, transmission, interface, plugin, and joint-dynamics records are embedded declarations only; they do not prove runtime availability or behavior;
- `evidence.exact=false` means asserted, heuristic, or sampled evidence; read `source_type` and qualifiers;
- mesh filenames do not reveal shape; only `measured` geometry supports shape/bounds claims;
- visual and collision mesh completeness are independent; a collision-only export is not visual-mesh evidence;
- AABB overlap is a broad-phase candidate, not proof of collision;
- SRDF disabled-collision policy never erases a physical pair result; keep physical and policy-filtered statuses distinct;
- semantic roles and project invariants are declared intent, not facts inferred from names;
- component membership, function, capability, condition, intended effect, and relational affordance must come from explicit functional declarations; never infer them from names, topology, or geometry;
- a structurally grounded capability means only that its named typed requirements are supported by the bound concept graph; it does not establish current preconditions, force closure, payload, controller/hardware execution, physical success, or safety;
- an affordance is an actor-action-target-effect relation under named conditions; a declared intended effect is never an observed effect, and absence from even a complete project inventory is not physical impossibility;
- every answer should state pose, reference frame, target frame, units, and the fact IDs or tool result that support it.
"""


def write_agent_context(
    directory: Path,
    canonical: dict[str, Any],
    facts: list[dict[str, Any]],
    facts_path: Path,
    functional_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    facts_by_id = {fact["fact_id"]: fact for fact in facts}
    if len(facts_by_id) != len(facts):
        raise ContextError("facts contain duplicate fact_id values")
    facts_bytes = facts_path.read_bytes()
    expected_facts_bytes = b"".join(_jsonl_line(fact) for fact in facts)
    if facts_bytes != expected_facts_bytes:
        raise ContextError("facts.jsonl serialization does not match the in-memory fact records")

    cards = build_entity_cards(canonical, facts, functional_model)
    cards_path = directory / "entity-cards.jsonl"
    card_offsets: dict[str, dict[str, Any]] = {}
    cards_content = bytearray()
    for card in cards:
        line = _jsonl_line(card)
        card_offsets[card["entity_id"]] = {
            "entity_type": card["entity_type"],
            "name": card["name"],
            "offset_bytes": len(cards_content),
            "length_bytes": len(line),
            "fact_count": len(card["fact_ids"]),
            "trust": card["trust"]["classification"],
        }
        cards_content.extend(line)
    cards_path.write_bytes(bytes(cards_content))

    by_name: dict[str, list[str]] = {}
    for entity_id, record in card_offsets.items():
        by_name.setdefault(record["name"], []).append(entity_id)
    entity_index = {
        "schema_version": ENTITY_INDEX_SCHEMA,
        "cards_sha256": _sha256_bytes(bytes(cards_content)),
        "record_count": len(cards),
        "by_entity_id": card_offsets,
        "by_name": {name: sorted(entity_ids) for name, entity_ids in sorted(by_name.items())},
    }
    entity_index_path = directory / "entity-index.json"
    entity_index_bytes = _json_bytes(entity_index)
    entity_index_path.write_bytes(entity_index_bytes)

    fact_offsets: dict[str, dict[str, Any]] = {}
    by_subject: dict[str, list[str]] = {}
    by_predicate: dict[str, list[str]] = {}
    offset = 0
    for fact in facts:
        line = _jsonl_line(fact)
        fact_offsets[fact["fact_id"]] = {"offset_bytes": offset, "length_bytes": len(line)}
        by_subject.setdefault(fact["subject"], []).append(fact["fact_id"])
        by_predicate.setdefault(fact["predicate"], []).append(fact["fact_id"])
        offset += len(line)
    fact_index = {
        "schema_version": FACT_INDEX_SCHEMA,
        "facts_sha256": _sha256_bytes(facts_bytes),
        "record_count": len(facts),
        "by_fact_id": fact_offsets,
        "by_subject": {key: sorted(value) for key, value in sorted(by_subject.items())},
        "by_predicate": {key: sorted(value) for key, value in sorted(by_predicate.items())},
        "by_entity": {card["entity_id"]: card["fact_ids"] for card in cards},
    }
    fact_index_path = directory / "fact-index.json"
    fact_index_bytes = _json_bytes(fact_index)
    fact_index_path.write_bytes(fact_index_bytes)

    guide_path = directory / "agent-guide.md"
    guide_bytes = _guide_markdown(canonical).encode("utf-8")
    guide_path.write_bytes(guide_bytes)

    exact_count = sum(bool(fact["evidence"]["exact"]) for fact in facts)
    types: dict[str, int] = {}
    for card in cards:
        types[card["entity_type"]] = types.get(card["entity_type"], 0) + 1
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "meaning": "progressive-disclosure robot language: epistemic rules first, structural concept proof query second, explicit function/affordance proof query third, one typed entity card fourth, bound provenance facts fifth, fresh numeric query when required",
        "robot": canonical["robot"],
        "source": canonical["source"],
        "pose": canonical["pose"],
        "units": canonical["units"],
        "load_order": [
            "agent-context.json",
            "query-concepts#task_relevant_proof_closure",
            *(
                ["query-functions#task_relevant_functional_and_structural_proof_closure"]
                if isinstance(canonical.get("artifacts", {}).get("functional_model"), dict)
                else []
            ),
            "entity-cards.jsonl#one_exact_entity",
            "facts.jsonl#bound_fact_ids",
            "model.json#only_if_needed",
        ],
        "identity_grammar": {
            "robot": "robot/<robot-name>",
            "link": "link/<link-name>",
            "joint": "joint/<joint-name>",
            "frame": "frame/<exact-frame-name>",
            "invariant": "invariant/<assertion-id>",
            "ros2_control_system": "ros2_control_system/<system-name>",
            "transmission": "transmission/<transmission-name>",
            "actuator": "actuator/<transmission-name>/<actuator-name>",
            "control_sensor": "control_sensor/<system-name>/<sensor-name>",
            "robot_instance": "robot_instance/<instance-id>",
            "scene_frame": "scene_frame/<name>",
            "scene_object": "scene_object/<object-id>",
            "scene_geometry": "scene_geometry/<object-id>/<geometry-id>",
            "robot_geometry": "robot_geometry/<exact-URDF-geometry-frame-name>",
            "observation_log": "observation_log/<log-id>",
            "ros_capture": "ros_capture/<capture-id>",
            "render_atlas": "render_atlas/<render-id>",
            "render_view": "render_view/<render-id>/<front|side|top|isometric>",
            "motion_atlas": "motion_atlas/<motion-id>",
            "motion_driver": "motion_driver/<motion-id>/<independent-joint-name>",
            "motion_view": "motion_view/<motion-id>/<independent-joint-name>/<front|side|top|isometric>",
            "articulation_grammar": "articulation_grammar/<grammar-id>",
            "articulation_variable": "articulation_variable/<grammar-id>/<independent-joint-name>",
            "articulation_operator": "articulation_operator/<grammar-id>/<physical-joint-name>",
            "articulation_derivation": "articulation_derivation/<grammar-id>/<exact-frame-name>",
            "constraint_graph": "constraint_graph/<constraint-graph-id>",
            "attachment": "attachment/<constraint-graph-id>/<attachment-id>",
            "constraint": "constraint/<constraint-graph-id>/<constraint-id>",
            "configuration_atlas": "configuration_atlas/<configuration-atlas-id>",
            "configuration_chart": "configuration_chart/<configuration-atlas-id>/<chart-id>",
            "configuration_node": "configuration_node/<configuration-atlas-id>/<chart-id>/<sample-index>/<solution-index>",
            "configuration_component": "configuration_component/<configuration-atlas-id>/<chart-id>/<component-index>",
            "concept_graph": "concept_graph/<concept-graph-id>",
            "functional_model": "functional_model/<function-set-name>/<binding-hash>",
            "component": "component/<project-component-id>",
            "function": "function/<project-function-id>",
            "capability": "capability/<project-capability-id>",
            "affordance": "affordance/<project-affordance-id>",
            "condition": "condition/<project-condition-id>",
            "effect": "effect/<project-effect-id>",
            "object_type": "object_type/<project-object-type-id>",
            "note": "link/X and frame/X are distinct even when X is the same URDF link name",
        },
        "epistemic_contract": {
            "exact_true": "deterministically declared, derived, measured, or verified within the stated representation and tolerance",
            "exact_false": "project-asserted, heuristic, or finite-sample evidence; inspect source_type and qualifiers before wording a claim",
            "summary_cnl": "deterministic compact index derived from model.json; cite bound fact IDs or a fresh tool result as evidence",
            "unknown_policy": "absence, not_requested, not_inspected, and indeterminate are not negative facts",
        },
        "answer_contract": [
            "state the evaluated pose or say pose-independent",
            "state reference and target frames with transform direction",
            "state units and quaternion order when numeric",
            "cite fact IDs or the deterministic tool result",
            "state approximation, assertion, or unresolved boundary",
        ],
        "question_router": [
            {"intent": "what is this link/joint/frame", "first_action": "retrieve exact typed entity card"},
            {"intent": "whole structural summary, serial segments, branch points, causal explanation, frame-law explanation, constraint dependency, or proof closure", "tool": "query-concepts", "condition": "submit a strict robot-spatial-concept-query.v1; preserve modality and closed/open-world boundaries from every supporting clause"},
            {"intent": "what is this component for, what capability does it have, what action does it afford, or may it act on an object type", "tool": "query-functions", "condition": "submit a strict robot-spatial-functional-query.v1; preserve project-asserted intent, typed requirement grounding, unevaluated preconditions, intended-effect status, inventory scope, and the physical-execution boundary"},
            {"intent": "ancestry or path", "tool": "chain"},
            {"intent": "which frames move when a joint changes", "tool": "affects"},
            {"intent": "pose between frames", "tool": "transform"},
            {"intent": "joint direction in a requested frame", "tool": "axis"},
            {"intent": "instantaneous target motion", "tool": "jacobian"},
            {"intent": "mass, center of mass, or aggregate inertia", "tool": "mass-properties", "condition": "report declared-model coverage and never promote missing inertials to zero physical mass"},
            {"intent": "gravity torque, gravity force, or static holding effort", "tool": "gravity-loads", "condition": "state gravity vector/frame, sign convention, inertial coverage, and gravity-only boundary"},
            {"intent": "actuator, ros2_control, transmission, command/state interface, plugin, or joint dynamics", "tool": "actuation", "condition": "report declarations only and do not infer runtime or hardware capability"},
            {"intent": "world pose, root mounting, object pose, or robot-to-environment transform", "tool": "scene-transform", "condition": "use typed scene/robot entity IDs and state scene_id plus snapshot_id"},
            {"intent": "world gravity or mounting-aware static loads", "tool": "scene-gravity-loads", "condition": "state snapshot provenance, transformed root gravity, sign convention, and physical-world boundary"},
            {"intent": "robot versus table, shelf, obstacle, or other declared environment collision/clearance", "tool": "scene-collisions", "condition": "report snapshot, tolerance, coverage, indeterminate pairs, and declared-world limitation"},
            {"intent": "what was the robot or object pose at time t", "tool": "observe-transform", "condition": "bind log/query digests, clock, query time, maximum age, selected samples, and declaration fallbacks"},
            {"intent": "what did ROS report or how was JointState/TF normalized", "tool": "ros_observation_adapter.py normalize", "condition": "bind capture/config digests, header/receipt policy, component/edge ages, TF path, authority visibility, and conflict rejection"},
            {"intent": "visualize or ground structure in a front, side, top, or isometric view", "tool": "render", "condition": "read the digest-bound render_atlas/render_view card and keep projection fidelity separate from photorealistic visibility"},
            {"intent": "verify that a view agrees with numeric transforms and geometry", "tool": "verify-render", "condition": "regenerate the atlas from the same model, pose, mesh inspection policy, and semantic highlights"},
            {"intent": "what changes when one joint moves, in which direction, or which frames stay fixed", "tool": "motion-atlas", "condition": "read motion_atlas/motion_driver first; preserve the signed finite step, feasible limits, mimic-driven joints, fixed other drivers, and endpoint-only scope"},
            {"intent": "verify finite joint-cause endpoint deltas and shared-view overlays", "tool": "verify-motion-atlas", "condition": "regenerate with the same model, baseline pose, mesh policy, angular step, and linear step; do not promote endpoints to a trajectory or swept volume"},
            {"intent": "general joint motion law, mimic equation, valid variable domain, or root-to-frame composition", "tool": "evaluate-articulation", "condition": "read the grammar/variable/operator/derivation cards; bind independent drivers only and distinguish the general law from FK, Jacobian, and finite counterfactual samples"},
            {"intent": "verify the pose-independent articulation law", "tool": "verify-articulation-grammar", "condition": "regenerate from the same bound URDF/SDF/MJCF source and execute deterministic fresh all-frame probes; use an independent oracle for parser/FK diversity"},
            {"intent": "compare the same mechanism across URDF, SDF, or MJCF", "tool": "compare-articulation-grammars", "condition": "compare source-binding-free canonical law projections and unseen bindings; require a digest-bound bijective typed correspondence when identifiers differ"},
            {"intent": "closed loop, parallel mechanism, cross-branch attachment, coordinate coupling, valid mechanism pose, or local mobility", "tool": "evaluate-constraints", "condition": "read constraint_graph/attachment/constraint cards; treat the tree as a parameterization, state the evaluated pose and typed tolerances, and keep numerical local mobility distinct from global DOF"},
            {"intent": "find a configuration that satisfies mechanism constraints", "tool": "solve-constraints", "condition": "provide a seed and explicitly choose solved independent drivers; report fixed drivers, convergence, residuals, possible branch dependence, and local-only scope"},
            {"intent": "verify supplemental mechanism semantics and execution", "tool": "verify-constraint-graph", "condition": "regenerate from the exact digest-bound articulation grammar and constraint spec, then reproduce local analysis at an explicit pose"},
            {"intent": "finite mechanism branches, multiple assembly configurations, singularity witnesses, or cross-pose constrained mobility", "tool": "configuration-atlas", "condition": "declare an explicit one-parameter chart, complete seeds, normalized distance/merge thresholds, and minimum solutions per sample; report nodes and deficiencies, and never claim exhaustive global topology"},
            {"intent": "verify finite configuration nodes and atlas regeneration", "tool": "verify-configuration-atlas", "condition": "bind the exact constraint graph, exact atlas spec, and stored atlas; re-execute every node while preserving the finite-evidence boundary"},
            {"intent": "current or observed robot-environment collision", "tool": "observe-collisions", "condition": "report nominal geometry status separately from physical collision and safety, which remain not established"},
            {"intent": "geometry size or placement", "tool": "bounds", "condition": "inspect meshes when geometry_type=mesh"},
            {"intent": "collision", "tool": "surface-collisions", "condition": "do not promote overlaps output to collision"},
            {"intent": "reachable set", "tool": "workspace", "condition": "report finite-sample limitation"},
        ],
        "unresolved_claims": _unresolved_claims(canonical),
        "statistics": {
            "entity_count": len(cards),
            "entity_type_counts": dict(sorted(types.items())),
            "fact_count": len(facts),
            "exact_fact_count": exact_count,
            "nonexact_fact_count": len(facts) - exact_count,
            "predicate_count": len(by_predicate),
        },
        "artifacts": {
            "guide": {"path": "agent-guide.md", "sha256": _sha256_bytes(guide_bytes)},
            "entity_cards": {"path": "entity-cards.jsonl", "sha256": _sha256_bytes(bytes(cards_content)), "schema_version": CARD_SCHEMA},
            "entity_index": {"path": "entity-index.json", "sha256": _sha256_bytes(entity_index_bytes), "schema_version": ENTITY_INDEX_SCHEMA},
            "facts": {"path": "facts.jsonl", "sha256": _sha256_bytes(facts_bytes), "schema_version": "robot-spatial-fact.v1"},
            "fact_index": {"path": "fact-index.json", "sha256": _sha256_bytes(fact_index_bytes), "schema_version": FACT_INDEX_SCHEMA},
            "model": {"path": "model.json", "schema_version": canonical["schema_version"]},
            "overview": {"path": "context.md"},
            **(
                {
                    "concept_graph": {
                        "path": canonical["artifacts"]["concept_graph"]["path"],
                        "sha256": canonical["artifacts"]["concept_graph"]["sha256"],
                        "schema_version": canonical["artifacts"]["concept_graph"]["schema_version"],
                        "concept_graph_id": canonical["artifacts"]["concept_graph"]["concept_graph_id"],
                        "concept_graph_sha256": canonical["artifacts"]["concept_graph"]["concept_graph_sha256"],
                        "language_path": canonical["artifacts"]["concept_graph"]["language_path"],
                        "language_sha256": canonical["artifacts"]["concept_graph"]["language_sha256"],
                        "coverage": canonical["artifacts"]["concept_graph"]["coverage"],
                    }
                }
                if isinstance(canonical.get("artifacts", {}).get("concept_graph"), dict)
                else {}
            ),
            **(
                {
                    "functional_model": {
                        "path": canonical["artifacts"]["functional_model"]["path"],
                        "sha256": canonical["artifacts"]["functional_model"]["sha256"],
                        "schema_version": canonical["artifacts"]["functional_model"]["schema_version"],
                        "functional_model_id": canonical["artifacts"]["functional_model"]["functional_model_id"],
                        "functional_model_sha256": canonical["artifacts"]["functional_model"]["functional_model_sha256"],
                        "function_set_id": canonical["artifacts"]["functional_model"]["function_set_id"],
                        "status": canonical["artifacts"]["functional_model"]["status"],
                        "coverage": canonical["artifacts"]["functional_model"]["coverage"],
                    }
                }
                if isinstance(canonical.get("artifacts", {}).get("functional_model"), dict)
                else {}
            ),
            **(
                {
                    "articulation_grammar": {
                        "path": canonical["artifacts"]["articulation_grammar"]["path"],
                        "sha256": canonical["artifacts"]["articulation_grammar"]["sha256"],
                        "schema_version": canonical["artifacts"]["articulation_grammar"]["schema_version"],
                    }
                }
                if isinstance(canonical.get("artifacts", {}).get("articulation_grammar"), dict)
                else {}
            ),
            **(
                {
                    "constraint_graph": {
                        "path": canonical["artifacts"]["constraint_graph"]["path"],
                        "sha256": canonical["artifacts"]["constraint_graph"]["sha256"],
                        "schema_version": canonical["artifacts"]["constraint_graph"]["schema_version"],
                        "evaluation": canonical["artifacts"]["constraint_graph"]["evaluation"],
                    }
                }
                if isinstance(canonical.get("artifacts", {}).get("constraint_graph"), dict)
                else {}
            ),
            **(
                {
                    "configuration_atlas": {
                        "path": canonical["artifacts"]["configuration_atlas"]["path"],
                        "sha256": canonical["artifacts"]["configuration_atlas"]["sha256"],
                        "schema_version": canonical["artifacts"]["configuration_atlas"]["schema_version"],
                        "status": canonical["artifacts"]["configuration_atlas"]["status"],
                        "coverage": canonical["artifacts"]["configuration_atlas"]["coverage"],
                    }
                }
                if isinstance(canonical.get("artifacts", {}).get("configuration_atlas"), dict)
                else {}
            ),
        },
    }
    manifest_path = directory / "agent-context.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    return {
        "manifest": "agent-context.json",
        "guide": "agent-guide.md",
        "entity_cards": "entity-cards.jsonl",
        "entity_index": "entity-index.json",
        "fact_index": "fact-index.json",
        "schema_version": MANIFEST_SCHEMA,
        "entity_count": len(cards),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContextError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(data, dict):
        raise ContextError(f"{label} must contain a JSON object")
    return data


def _read_at(path: Path, offset: int, length: int, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(length)
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ContextError(f"cannot read indexed {label} from {path}: {error}") from error
    if not isinstance(data, dict):
        raise ContextError(f"indexed {label} must be an object")
    return data


def retrieve_context(
    directory: Path,
    *,
    entity: str | None = None,
    predicate: str | None = None,
    pose: str | None = None,
    evidence: str = "all",
    fact_id: str | None = None,
    list_entities: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    if evidence not in {"all", "exact", "nonexact"}:
        raise ContextError("evidence must be all, exact, or nonexact")
    if limit < 1:
        raise ContextError("limit must be at least 1")
    manifest_path = directory / "agent-context.json"
    manifest = _read_json(manifest_path, "agent context manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ContextError(f"agent context manifest must use {MANIFEST_SCHEMA}")
    artifacts = manifest.get("artifacts", {})
    required = {"entity_cards", "entity_index", "facts", "fact_index"}
    if not required.issubset(artifacts):
        raise ContextError("agent context manifest is missing indexed artifacts")
    paths = {name: directory / artifacts[name]["path"] for name in required}
    integrity = {
        name: {
            "expected_sha256": artifacts[name]["sha256"],
            "actual_sha256": _sha256_path(path),
        }
        for name, path in paths.items()
    }
    if any(record["expected_sha256"] != record["actual_sha256"] for record in integrity.values()):
        raise ContextError(f"agent context artifact digest mismatch: {integrity}")
    index = _read_json(paths["fact_index"], "fact index")
    if index.get("schema_version") != FACT_INDEX_SCHEMA:
        raise ContextError(f"fact index must use {FACT_INDEX_SCHEMA}")
    entity_index_record = _read_json(paths["entity_index"], "entity index")
    if entity_index_record.get("schema_version") != ENTITY_INDEX_SCHEMA:
        raise ContextError(f"entity index must use {ENTITY_INDEX_SCHEMA}")

    entity_index = entity_index_record["by_entity_id"]
    if list_entities:
        return {
            "schema_version": RETRIEVAL_SCHEMA,
            "robot": manifest["robot"],
            "query": {"list_entities": True},
            "integrity": integrity,
            "entities": [{"entity_id": entity_id, **record} for entity_id, record in sorted(entity_index.items())],
            "count": len(entity_index),
        }

    card = None
    if entity is not None:
        resolved = entity
        if entity not in entity_index:
            candidates = entity_index_record["by_name"].get(entity, [])
            if len(candidates) == 1:
                resolved = candidates[0]
            elif candidates:
                raise ContextError(f"entity name {entity!r} is ambiguous; use one of {sorted(candidates)}")
            else:
                raise ContextError(f"unknown entity {entity!r}")
        entity = resolved
        location = entity_index[entity]
        card = _read_at(paths["entity_cards"], location["offset_bytes"], location["length_bytes"], f"entity card {entity}")

    if fact_id is not None:
        fact_ids = [fact_id]
    elif card is not None:
        fact_ids = list(card["fact_ids"])
    elif predicate is not None:
        fact_ids = list(index["by_predicate"].get(predicate, []))
    else:
        raise ContextError("provide --entity, --predicate, --fact-id, or --list-entities")

    selected: list[dict[str, Any]] = []
    for identifier in fact_ids:
        location = index["by_fact_id"].get(identifier)
        if location is None:
            raise ContextError(f"fact ID {identifier!r} is not present in the fact index")
        fact = _read_at(paths["facts"], location["offset_bytes"], location["length_bytes"], f"fact {identifier}")
        if predicate is not None and fact["predicate"] != predicate:
            continue
        if pose is not None and fact["qualifiers"].get("pose") != pose:
            continue
        if evidence == "exact" and not fact["evidence"]["exact"]:
            continue
        if evidence == "nonexact" and fact["evidence"]["exact"]:
            continue
        selected.append(fact)
    selected = selected[:limit]
    return {
        "schema_version": RETRIEVAL_SCHEMA,
        "robot": manifest["robot"],
        "source": manifest["source"],
        "exported_pose": manifest["pose"],
        "query": {
            "entity": entity,
            "predicate": predicate,
            "pose": pose,
            "evidence": evidence,
            "fact_id": fact_id,
            "limit": limit,
        },
        "integrity": integrity,
        "entity_card": card,
        "facts": selected,
        "count": len(selected),
        "truncated": len(selected) == limit and len(fact_ids) > limit,
    }
