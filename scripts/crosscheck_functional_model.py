#!/usr/bin/env python3
"""Dependency-free independent oracle for functional grounding and query boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "robot-spatial-functional-model-crosscheck.v1"
QUERY_SCHEMA = "robot-spatial-functional-query.v1"


def json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json_dump(value), encoding="utf-8")


def run_cli(script: Path, *arguments: object, expected_codes: set[int] | None = None) -> tuple[int, dict[str, Any]]:
    process = subprocess.run(
        [sys.executable, str(script), *(str(argument) for argument in arguments)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    expected = expected_codes or {0}
    if process.returncode not in expected:
        raise RuntimeError(
            f"CLI failed ({process.returncode}): {' '.join(str(argument) for argument in arguments)}; "
            f"stderr={process.stderr.strip()!r}; stdout={process.stdout.strip()!r}"
        )
    payload_text = process.stdout if process.stdout.strip() else process.stderr
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"CLI returned invalid JSON: {error}; output={payload_text!r}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("CLI output must be one JSON object")
    return process.returncode, payload


def generated_urdf(case_index: int, rng: random.Random) -> str:
    link_count = rng.randint(5, 9)
    parents = [0]
    for child in range(1, link_count):
        parents.append(rng.randrange(0, child))
    joint_types = ["root"]
    for child in range(1, link_count):
        joint_types.append(rng.choice(["revolute", "prismatic", "fixed"]))
    joint_types[1] = "revolute"
    movable_by_type: dict[str, list[int]] = {"revolute": [], "prismatic": []}
    mimic: dict[int, tuple[int, float, float]] = {}
    for child in range(1, link_count):
        joint_type = joint_types[child]
        possible = movable_by_type.get(joint_type, [])
        if possible and rng.random() < 0.35:
            source = rng.choice(possible)
            mimic[child] = (source, rng.choice([-0.75, -0.5, 0.5, 1.0]), rng.choice([-0.1, 0.0, 0.1]))
        elif joint_type in movable_by_type:
            movable_by_type[joint_type].append(child)
    lines = [f'<robot name="opaque_{case_index}">']
    lines.extend(f'  <link name="n{index}"/>' for index in range(link_count))
    for child in range(1, link_count):
        joint_type = joint_types[child]
        lines.extend([
            f'  <joint name="e{child}" type="{joint_type}">',
            f'    <parent link="n{parents[child]}"/>',
            f'    <child link="n{child}"/>',
            f'    <origin xyz="{0.1 * child:.6f} {0.03 * (child % 3):.6f} 0" rpy="0 0 {0.1 * (child % 4):.6f}"/>',
        ])
        if joint_type in {"revolute", "prismatic"}:
            axis = "0 0 1" if joint_type == "revolute" else "1 0 0"
            lines.append(f'    <axis xyz="{axis}"/>')
            lines.append('    <limit lower="-1" upper="1" effort="20" velocity="2"/>')
        if child in mimic:
            source, multiplier, offset = mimic[child]
            lines.append(
                f'    <mimic joint="e{source}" multiplier="{multiplier}" offset="{offset}"/>'
            )
        lines.append("  </joint>")
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def parse_tree(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    links = [element.attrib["name"] for element in root.findall("link")]
    joints: dict[str, dict[str, Any]] = {}
    children: dict[str, list[str]] = {link: [] for link in links}
    child_links: set[str] = set()
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        mimic_element = joint.find("mimic")
        mimic = None if mimic_element is None else mimic_element.attrib["joint"]
        joints[name] = {
            "type": joint.attrib["type"],
            "parent": parent,
            "child": child,
            "mimic": mimic,
        }
        children[parent].append(child)
        child_links.add(child)
    roots = sorted(set(links) - child_links)
    if len(roots) != 1:
        raise RuntimeError("oracle generator produced a non-tree URDF")
    return {"links": links, "joints": joints, "children": children, "root": roots[0]}


def descendants(tree: dict[str, Any], link: str) -> set[str]:
    found: set[str] = set()
    pending = [link]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        pending.extend(tree["children"][current])
    return found


def ultimate_driver(joints: dict[str, dict[str, Any]], joint_name: str) -> str:
    seen: set[str] = set()
    current = joint_name
    while joints[current]["mimic"] is not None:
        if current in seen:
            raise RuntimeError("mimic cycle in generated oracle case")
        seen.add(current)
        current = joints[current]["mimic"]
    return current


def expected_driver_effect(tree: dict[str, Any], driver: str) -> tuple[list[str], list[str]]:
    driven = sorted(
        name
        for name, record in tree["joints"].items()
        if record["type"] != "fixed" and ultimate_driver(tree["joints"], name) == driver
    )
    affected: set[str] = set()
    for joint_name in driven:
        affected.update(descendants(tree, tree["joints"][joint_name]["child"]))
    return driven, sorted(affected)


def source_binding(context: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = json.loads((context / "model.json").read_text(encoding="utf-8"))
    concept = json.loads((context / "concept-graph.json").read_text(encoding="utf-8"))
    return {
        "urdf_semantic_sha256": canonical["source"]["semantic_sha256"],
        "articulation_grammar_sha256": canonical["artifacts"]["articulation_grammar"]["sha256"],
        "constraint_graph_sha256": None,
        "configuration_atlas_sha256": None,
    }, concept


def function_spec(
    case_index: int,
    binding: dict[str, Any],
    driver_entity: str,
    driver: str,
    driven_joint: str,
    affected_link: str,
    root_link: str,
    complete: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "robot-spatial-function-affordance-spec.v1",
        "function_set_id": f"function_set/oracle_{case_index}",
        "source_binding": binding,
        "object_types": [{
            "object_type_id": "object_type/target",
            "meaning": "Opaque project target type.",
        }],
        "components": [{
            "component_id": "component/unit",
            "members": [f"link/{affected_link}", f"joint/{driven_joint}"],
            "meaning": "Opaque project grouping with no name-derived purpose.",
        }],
        "functions": [{
            "function_id": "function/declared_use",
            "provided_by": ["component/unit"],
            "verb": "stabilize",
            "object_types": ["object_type/target"],
            "purpose": "Project-declared purpose used only by this test.",
        }],
        "conditions": [{
            "condition_id": "condition/runtime_ready",
            "predicate": "runtime_ready",
            "arguments": ["actor", "target"],
            "truth_source": "runtime_observation_required",
            "meaning": "A runtime source must establish readiness.",
        }],
        "effects": [{
            "effect_id": "effect/stabilized",
            "predicate": "stabilized",
            "arguments": ["target"],
            "meaning": "Intended stabilization effect.",
        }],
        "capabilities": [{
            "capability_id": "capability/declared_action",
            "provided_by": ["component/unit"],
            "realizes_functions": ["function/declared_use"],
            "enabling_requirements": [
                {
                    "requirement_id": "requirement/entity_present",
                    "type": "entity_exists",
                    "parameters": {"entity": f"link/{affected_link}"},
                },
                {
                    "requirement_id": "requirement/entity_missing",
                    "type": "entity_exists",
                    "parameters": {"entity": "link/oracle_missing"},
                },
                {
                    "requirement_id": "requirement/path_present",
                    "type": "kinematic_path_exists",
                    "parameters": {"from_link": f"link/{root_link}", "to_link": f"link/{affected_link}"},
                },
                {
                    "requirement_id": "requirement/driver_moves_target",
                    "type": "driver_affects_frame",
                    "parameters": {"driver": driver_entity, "frame": f"frame/{affected_link}"},
                },
                {
                    "requirement_id": "requirement/driver_does_not_move_root",
                    "type": "driver_affects_frame",
                    "parameters": {"driver": driver_entity, "frame": f"frame/{root_link}"},
                },
                {
                    "requirement_id": "requirement/driver_drives_joint",
                    "type": "driver_drives_joint",
                    "parameters": {"driver": driver_entity, "joint": f"joint/{driven_joint}"},
                },
                {
                    "requirement_id": "requirement/role_unasserted",
                    "type": "frame_has_asserted_role",
                    "parameters": {"frame": f"frame/{affected_link}", "role": "oracle_role"},
                },
            ],
            "condition_refs": ["condition/runtime_ready"],
            "limitations": ["No runtime, physical, hardware, success, or safety truth is established."],
        }],
        "affordances": [{
            "affordance_id": "affordance/declared_action",
            "offered_by": ["component/unit"],
            "action_verb": "act_on",
            "target_object_types": ["object_type/target"],
            "capability_refs": ["capability/declared_action"],
            "precondition_refs": ["condition/runtime_ready"],
            "effect_refs": ["effect/stabilized"],
            "meaning": f"Opaque declared relation for independent driver {driver}.",
        }],
        "inventory_completeness": (
            [{
                "subject": "component/unit",
                "inventories": ["affordances"],
                "scope": "Only the generated oracle project spec.",
            }]
            if complete
            else []
        ),
    }


def query(script: Path, model: Path, directory: Path, query_id: str, intent: str, parameters: dict[str, Any]) -> dict[str, Any]:
    query_path = directory / f"query-{query_id}.json"
    write_json(query_path, {
        "schema_version": QUERY_SCHEMA,
        "query_id": query_id,
        "intent": intent,
        "parameters": parameters,
    })
    _, result = run_cli(script, "query-functions", model, query_path, "--compact")
    return result


def run_crosscheck(script: Path, case_count: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    discrepancies: list[dict[str, Any]] = []
    cli_failures: list[dict[str, Any]] = []
    counts = {
        "case_count": case_count,
        "requirement_result_count": 0,
        "satisfied_requirement_count": 0,
        "exact_closed_world_negative_count": 0,
        "open_world_unknown_count": 0,
        "query_count": 0,
        "complete_inventory_control_count": 0,
        "incomplete_inventory_control_count": 0,
        "exact_verification_count": 0,
    }
    with tempfile.TemporaryDirectory(prefix="robot-functional-oracle-") as temporary:
        root = Path(temporary)
        for case_index in range(case_count):
            case_root = root / f"case-{case_index:04d}"
            case_root.mkdir()
            urdf_path = case_root / "robot.urdf"
            urdf_path.write_text(generated_urdf(case_index, rng), encoding="utf-8")
            try:
                tree = parse_tree(urdf_path)
                context = case_root / "context"
                run_cli(script, "export", urdf_path, "--workspace-samples", "0", "--out", context)
                binding, concept = source_binding(context)
                drivers = concept["projections"]["articulation"]["drivers"]
                if not drivers:
                    raise RuntimeError("generated case has no independent driver")
                driver_record = rng.choice(drivers)
                driver_name = driver_record["driver_name"]
                driven, affected = expected_driver_effect(tree, driver_name)
                if not driven or not affected:
                    raise RuntimeError("independent oracle found an empty driver effect")
                driven_joint = rng.choice(driven)
                affected_link = rng.choice(affected)
                complete = case_index % 2 == 0
                spec = function_spec(
                    case_index,
                    binding,
                    driver_record["driver_entity"],
                    driver_name,
                    driven_joint,
                    affected_link,
                    tree["root"],
                    complete,
                )
                spec_path = case_root / "function-spec.json"
                model_path = case_root / "functional-model.json"
                write_json(spec_path, spec)
                run_cli(script, "functional-model", context, spec_path, "--out", model_path)
                model = json.loads(model_path.read_text(encoding="utf-8"))
                capability = model["projections"]["capabilities"][0]
                actual = {
                    item["requirement_id"]: (item["status"], item["satisfied"], item["evidence"]["closure_basis"])
                    for item in capability["requirements"]
                }
                expected = {
                    "requirement/entity_present": ("satisfied", True, "complete concept entity inventory"),
                    "requirement/entity_missing": ("not_satisfied_exact_closed_world", False, "complete concept entity inventory"),
                    "requirement/path_present": ("satisfied", True, "complete canonical tree links and edges"),
                    "requirement/driver_moves_target": ("satisfied", True, "complete articulation driver projection"),
                    "requirement/driver_does_not_move_root": ("not_satisfied_exact_closed_world", False, "complete articulation driver projection"),
                    "requirement/driver_drives_joint": ("satisfied", True, "complete articulation driver projection"),
                    "requirement/role_unasserted": ("not_established_open_world", False, "open-world project semantic role assertions"),
                }
                counts["requirement_result_count"] += len(actual)
                counts["satisfied_requirement_count"] += sum(status[1] for status in actual.values())
                counts["exact_closed_world_negative_count"] += sum(
                    status[0] == "not_satisfied_exact_closed_world" for status in actual.values()
                )
                counts["open_world_unknown_count"] += sum(
                    status[0] == "not_established_open_world" for status in actual.values()
                )
                if actual != expected:
                    discrepancies.append({"case": case_index, "check": "requirement_results", "expected": expected, "actual": actual})
                if capability["grounding_status"] != "one_or_more_declared_requirements_not_grounded":
                    discrepancies.append({"case": case_index, "check": "capability_grounding", "actual": capability["grounding_status"]})
                if capability["physical_capability_verified"] is not False:
                    discrepancies.append({"case": case_index, "check": "physical_capability_boundary"})

                explained = query(
                    script,
                    model_path,
                    case_root,
                    "capability",
                    "explain_capability",
                    {"capability": "declared_action"},
                )
                positive = query(
                    script,
                    model_path,
                    case_root,
                    "positive",
                    "can_perform_action",
                    {"offered_by": "unit", "action_verb": "act_on", "target_object_type": "target"},
                )
                negative = query(
                    script,
                    model_path,
                    case_root,
                    "negative",
                    "can_perform_action",
                    {"offered_by": "unit", "action_verb": "not_declared", "target_object_type": "target"},
                )
                counts["query_count"] += 3
                if not explained["structural_supporting_clauses"]:
                    discrepancies.append({"case": case_index, "check": "missing_structural_proof_closure"})
                if positive["answer"]["conclusion"] != "declared_affordance_with_ungrounded_capability_requirements":
                    discrepancies.append({"case": case_index, "check": "ungrounded_positive_boundary", "actual": positive["answer"]})
                if positive["answer"]["physical_executability"] != "not_established":
                    discrepancies.append({"case": case_index, "check": "physical_execution_boundary"})
                expected_negative = (
                    "not_declared_in_complete_project_inventory"
                    if complete
                    else "unknown_not_in_incomplete_inventory"
                )
                if negative["answer"]["conclusion"] != expected_negative or negative["answer"]["physical_impossibility"] != "not_established":
                    discrepancies.append({"case": case_index, "check": "inventory_boundary", "expected": expected_negative, "actual": negative["answer"]})
                if complete:
                    counts["complete_inventory_control_count"] += 1
                else:
                    counts["incomplete_inventory_control_count"] += 1
                _, verification = run_cli(
                    script,
                    "verify-functional-model",
                    context,
                    spec_path,
                    "--model",
                    model_path,
                )
                if verification["status"] != "passed" or not verification["exact_regeneration_match"]:
                    discrepancies.append({"case": case_index, "check": "exact_verification", "actual": verification})
                else:
                    counts["exact_verification_count"] += 1
            except Exception as error:  # report the complete randomized run instead of losing later cases
                cli_failures.append({"case": case_index, "error": str(error)})
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if not discrepancies and not cli_failures else "failed",
        "seed": seed,
        "counts": counts,
        "discrepancies": discrepancies,
        "cli_failures": cli_failures,
        "independence": {
            "production_modules_imported": [],
            "source_parser": "xml.etree.ElementTree direct raw URDF parsing",
            "oracle_method": "independent tree/mimic closure plus expected modal requirement and inventory-query outcomes",
            "candidate_interface": "public robot_spatial.py CLI only",
        },
        "exclusions": [
            "project declarations are treated as supplied intent rather than independently true function",
            "no physical capability, contact mechanics, payload, runtime, hardware, action-success, or safety validation",
            "requirement coverage is limited to entity, tree path, driver/joint/frame, and absent semantic-role controls",
            "no supplemental constraint or finite configuration requirement cases",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).resolve().with_name("robot_spatial.py"),
        help="public robot_spatial.py entrypoint",
    )
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.cases <= 0:
        raise SystemExit("--cases must be positive")
    report = run_crosscheck(args.script.resolve(), args.cases, args.seed)
    serialized = json_dump(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
