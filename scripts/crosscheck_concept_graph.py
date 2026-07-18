#!/usr/bin/env python3
"""Independent black-box oracle for robot spatial concept graphs.

This script deliberately imports no robot_spatial production module. It creates raw
URDF trees, parses them independently with ElementTree, derives structural and
mimic-causal expectations, and compares them with the public CLI artifacts/queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "robot-spatial-concept-independent-oracle.v1"
QUERY_SCHEMA = "robot-spatial-concept-query.v1"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _canonical_digest(value: dict[str, Any]) -> str:
    body = {key: child for key, child in value.items() if key != "concept_graph_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run(command: list[str], *, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def _joint_xml(
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    index: int,
    mimic: dict[str, Any] | None,
) -> list[str]:
    axis = ("0 0 1", "0 1 0", "1 0 0")[index % 3]
    lines = [
        f'  <joint name="{name}" type="{joint_type}">',
        f'    <parent link="{parent}"/>',
        f'    <child link="{child}"/>',
        f'    <origin xyz="{0.01 * (index + 1):.8f} {0.02 * (index % 2):.8f} 0" rpy="0 0 0"/>',
    ]
    if joint_type != "fixed":
        lines.append(f'    <axis xyz="{axis}"/>')
    if joint_type == "revolute":
        lines.append('    <limit lower="-3" upper="3" effort="50" velocity="5"/>')
    elif joint_type == "prismatic":
        lines.append('    <limit lower="-2" upper="2" effort="50" velocity="5"/>')
    if mimic is not None:
        lines.append(
            f'    <mimic joint="{mimic["joint"]}" multiplier="{mimic["multiplier"]}" '
            f'offset="{mimic["offset"]}"/>'
        )
    lines.append("  </joint>")
    return lines


def _generate_urdf(rng: random.Random, case_index: int) -> str:
    link_count = rng.randint(4, 10)
    links = [f"body_{case_index}_{index}" for index in range(link_count)]
    joint_records: list[dict[str, Any]] = []
    movable_by_unit: dict[str, list[str]] = {"angular": [], "linear": []}
    for child_index in range(1, link_count):
        if child_index == 1:
            parent_index = 0
        elif case_index % 4 == 1:
            parent_index = child_index - 1
        elif child_index == 2 and case_index % 3 == 0:
            parent_index = 0
        else:
            parent_index = rng.randrange(child_index)
        parent = links[parent_index]
        child = links[child_index]
        joint_name = f"hinge_{case_index}_{child_index}"
        joint_type = rng.choices(
            ["fixed", "revolute", "continuous", "prismatic"],
            weights=[2, 4, 2, 3],
            k=1,
        )[0]
        mimic: dict[str, Any] | None = None
        if case_index % 5 == 0 and child_index == 1:
            joint_type = "continuous"
        elif case_index % 5 == 0 and child_index in {2, 3}:
            joint_type = "continuous"
            mimic = {
                "joint": f"hinge_{case_index}_{child_index - 1}",
                "multiplier": -1.25 if child_index == 2 else 0.5,
                "offset": 0.0,
            }
        unit = "linear" if joint_type == "prismatic" else "angular"
        if mimic is None and joint_type != "fixed" and movable_by_unit[unit] and rng.random() < 0.32:
            target = rng.choice(movable_by_unit[unit])
            joint_type = "prismatic" if unit == "linear" else rng.choice(["revolute", "continuous"])
            mimic = {
                "joint": target,
                "multiplier": rng.choice([-1.25, -0.75, 0.5, 1.0, 1.5]),
                "offset": rng.choice([-0.2, 0.0, 0.15]),
            }
        if joint_type != "fixed":
            movable_by_unit[unit].append(joint_name)
        joint_records.append({
            "name": joint_name,
            "type": joint_type,
            "parent": parent,
            "child": child,
            "mimic": mimic,
        })
    if not any(record["type"] != "fixed" and record["mimic"] is None for record in joint_records):
        joint_records[0]["type"] = "continuous"
        joint_records[0]["mimic"] = None

    lines = [f'<robot name="oracle_robot_{case_index}">']
    for link in links:
        lines.append(f'  <link name="{link}"/>')
    for index, record in enumerate(joint_records):
        lines.extend(_joint_xml(
            record["name"], record["type"], record["parent"], record["child"], index, record["mimic"]
        ))
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def _parse_raw_urdf(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    links = sorted(element.attrib["name"] for element in root.findall("link"))
    joints: dict[str, dict[str, Any]] = {}
    children: dict[str, list[tuple[str, str]]] = {link: [] for link in links}
    incoming: dict[str, tuple[str, str]] = {}
    for element in root.findall("joint"):
        name = element.attrib["name"]
        parent = element.find("parent").attrib["link"]  # type: ignore[union-attr]
        child = element.find("child").attrib["link"]  # type: ignore[union-attr]
        mimic_element = element.find("mimic")
        mimic = None if mimic_element is None else {
            "joint": mimic_element.attrib["joint"],
            "multiplier": float(mimic_element.attrib.get("multiplier", "1")),
            "offset": float(mimic_element.attrib.get("offset", "0")),
        }
        joints[name] = {
            "name": name,
            "type": element.attrib["type"],
            "parent": parent,
            "child": child,
            "mimic": mimic,
        }
        children[parent].append((name, child))
        incoming[child] = (name, parent)
    for values in children.values():
        values.sort()
    roots = sorted(set(links) - set(incoming))
    if len(roots) != 1:
        raise ValueError(f"expected one root, got {roots}")
    tree_root = roots[0]

    driver_cache: dict[str, str | None] = {}

    def driver_for(joint_name: str) -> str | None:
        if joint_name in driver_cache:
            return driver_cache[joint_name]
        joint = joints[joint_name]
        if joint["type"] == "fixed":
            result = None
        elif joint["mimic"] is None:
            result = joint_name
        else:
            result = driver_for(joint["mimic"]["joint"])
        driver_cache[joint_name] = result
        return result

    for joint_name in joints:
        driver_for(joint_name)
    drivers = sorted({driver for driver in driver_cache.values() if driver is not None})
    physical_joints_by_driver = {
        driver: sorted(joint for joint, resolved in driver_cache.items() if resolved == driver)
        for driver in drivers
    }

    path_joints: dict[str, list[str]] = {tree_root: []}
    queue = deque([tree_root])
    while queue:
        parent = queue.popleft()
        for joint, child in children[parent]:
            path_joints[child] = [*path_joints[parent], joint]
            queue.append(child)

    frame_dependencies: dict[str, set[str]] = {}
    for link in links:
        frame_dependencies[f"frame/{link}"] = {
            resolved for joint in path_joints[link]
            if (resolved := driver_cache[joint]) is not None
        }
    for name, joint in joints.items():
        frame_dependencies[f"frame/joint/{name}"] = {
            resolved for ancestor_joint in path_joints[joint["parent"]]
            if (resolved := driver_cache[ancestor_joint]) is not None
        }
    affected_frames_by_driver = {
        driver: sorted(frame for frame, dependencies in frame_dependencies.items() if driver in dependencies)
        for driver in drivers
    }

    branches = sorted(link for link, values in children.items() if len(values) > 1)
    leaves = sorted(link for link, values in children.items() if not values)
    boundaries = {tree_root, *branches, *leaves}
    segments: list[dict[str, Any]] = []
    for start in sorted(boundaries):
        for first_joint, first_child in children[start]:
            ordered_links = [start, first_child]
            ordered_joints = [first_joint]
            current = first_child
            while current not in boundaries and len(children[current]) == 1:
                next_joint, next_child = children[current][0]
                ordered_joints.append(next_joint)
                ordered_links.append(next_child)
                current = next_child
            segments.append({
                "start_link": f"link/{start}",
                "end_link": f"link/{current}",
                "ordered_links": [f"link/{link}" for link in ordered_links],
                "ordered_joints": [f"joint/{joint}" for joint in ordered_joints],
                "start_boundary": "root" if start == tree_root else "branch_point",
                "end_boundary": "leaf" if current in leaves else "branch_point",
            })

    descendants: dict[tuple[str, str], list[str]] = {}
    for descendant in links:
        current = descendant
        reverse_joints: list[str] = []
        while current != tree_root:
            joint, parent = incoming[current]
            reverse_joints.append(joint)
            descendants[(parent, descendant)] = list(reversed(reverse_joints))
            current = parent

    return {
        "robot_name": root.attrib["name"],
        "root": tree_root,
        "links": links,
        "joints": joints,
        "children": children,
        "path_joints": path_joints,
        "drivers": drivers,
        "physical_joints_by_driver": physical_joints_by_driver,
        "affected_frames_by_driver": affected_frames_by_driver,
        "branches": branches,
        "leaves": leaves,
        "segments": segments,
        "descendants": descendants,
    }


def _query(
    python: str,
    cli: Path,
    graph_path: Path,
    work: Path,
    query_id: str,
    intent: str,
    parameters: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    query_path = work / f"query-{query_id}.json"
    query_path.write_bytes(_json_bytes({
        "schema_version": QUERY_SCHEMA,
        "query_id": query_id,
        "intent": intent,
        "parameters": parameters,
    }))
    result = _run([python, str(cli), "query-concepts", str(graph_path), str(query_path)])
    if result.returncode != 0:
        return None, result.stderr.strip() or result.stdout.strip()
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError as error:
        return None, f"invalid query JSON: {error}"


def _normalized_segments(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("start_link", "end_link", "ordered_links", "ordered_joints", "start_boundary", "end_boundary")
    return [{key: record[key] for key in keys} for record in records]


def _evaluate_case(
    python: str,
    cli: Path,
    case_root: Path,
    case_index: int,
    urdf_text: str,
) -> tuple[dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
    counts = {
        "topology_edges_checked": 0,
        "descendant_relations_checked": 0,
        "segments_checked": 0,
        "drivers_checked": 0,
        "mimic_physical_joints_checked": 0,
        "affected_frames_checked": 0,
        "queries_checked": 0,
        "negative_controls_checked": 0,
    }
    discrepancies: list[dict[str, Any]] = []
    cli_failures: list[dict[str, Any]] = []
    urdf_path = case_root / "robot.urdf"
    urdf_path.write_text(urdf_text, encoding="utf-8")
    expected = _parse_raw_urdf(urdf_path)
    context = case_root / "context"
    export = _run([
        python, str(cli), "export", str(urdf_path), "--workspace-samples", "0", "--out", str(context)
    ])
    if export.returncode != 0:
        cli_failures.append({"case": case_index, "command": "export", "stderr": export.stderr[-2000:]})
        return counts, discrepancies, cli_failures
    try:
        graph = json.loads((context / "concept-graph.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        cli_failures.append({"case": case_index, "command": "read concept graph", "stderr": str(error)})
        return counts, discrepancies, cli_failures

    def check(label: str, actual: Any, wanted: Any) -> None:
        if actual != wanted:
            discrepancies.append({"case": case_index, "check": label, "expected": wanted, "actual": actual})

    topology = graph["projections"]["topology"]
    check("root", topology["root_link"], f"link/{expected['root']}")
    check("complete links", topology["links"], [f"link/{name}" for name in expected["links"]])
    check("complete joints", topology["joints"], [f"joint/{name}" for name in sorted(expected["joints"])])
    expected_edges = sorted(
        ({
            "joint": f"joint/{name}",
            "parent_link": f"link/{joint['parent']}",
            "child_link": f"link/{joint['child']}",
            "joint_type": joint["type"],
        } for name, joint in expected["joints"].items()),
        key=lambda value: value["joint"],
    )
    actual_edges = sorted(
        ({key: edge[key] for key in ("joint", "parent_link", "child_link", "joint_type")} for edge in topology["edges"]),
        key=lambda value: value["joint"],
    )
    check("complete typed edges", actual_edges, expected_edges)
    counts["topology_edges_checked"] += len(expected_edges)
    check("branch points", topology["branch_points"], [f"link/{name}" for name in expected["branches"]])
    check("structural leaves", topology["structural_leaves"], [f"link/{name}" for name in expected["leaves"]])
    check("maximal serial segments", _normalized_segments(topology["maximal_serial_segments"]), expected["segments"])
    counts["segments_checked"] += len(expected["segments"])

    descendant_clauses = [
        clause for clause in graph["clauses"] if clause["predicate"] == "is_descendant_of"
    ]
    actual_descendants = {
        (clause["object"]["ancestor"].removeprefix("link/"), clause["subject"].removeprefix("link/")):
        [joint.removeprefix("joint/") for joint in clause["object"]["ordered_joint_path"]]
        for clause in descendant_clauses
    }
    check("descendant closure", actual_descendants, expected["descendants"])
    counts["descendant_relations_checked"] += len(expected["descendants"])

    drivers = {record["driver_name"]: record for record in graph["projections"]["articulation"]["drivers"]}
    check("independent drivers", sorted(drivers), expected["drivers"])
    for driver_name in expected["drivers"]:
        if driver_name not in drivers:
            continue
        record = drivers[driver_name]
        wanted_physical = [f"joint/{name}" for name in expected["physical_joints_by_driver"][driver_name]]
        wanted_frames = expected["affected_frames_by_driver"][driver_name]
        check(f"driver {driver_name} physical joints", sorted(record["physical_joints_driven"]), wanted_physical)
        check(f"driver {driver_name} affected frames", sorted(record["affected_frames"]), wanted_frames)
        counts["drivers_checked"] += 1
        counts["mimic_physical_joints_checked"] += len(wanted_physical)
        counts["affected_frames_checked"] += len(wanted_frames)

    summary, error = _query(python, cli, context / "concept-graph.json", case_root, f"{case_index}-summary", "structural_summary", {})
    counts["queries_checked"] += 1
    if error:
        cli_failures.append({"case": case_index, "command": "structural_summary", "stderr": error})
    else:
        check("query root", summary["answer"]["root_link"], f"link/{expected['root']}")
        check("query leaves", summary["answer"]["structural_leaves"], [f"link/{name}" for name in expected["leaves"]])

    links = expected["links"]
    from_link, to_link = links[-1], links[0]
    path_answer, error = _query(
        python, cli, context / "concept-graph.json", case_root, f"{case_index}-path",
        "trace_kinematic_path", {"from_link": from_link, "to_link": to_link},
    )
    counts["queries_checked"] += 1
    if error:
        cli_failures.append({"case": case_index, "command": "trace_kinematic_path", "stderr": error})
    elif path_answer["answer"]["joint_count"] != len(expected["path_joints"][from_link]):
        discrepancies.append({
            "case": case_index,
            "check": "reverse path joint count",
            "expected": len(expected["path_joints"][from_link]),
            "actual": path_answer["answer"]["joint_count"],
        })

    if expected["drivers"]:
        driver_name = expected["drivers"][0]
        positive_frames = expected["affected_frames_by_driver"][driver_name]
        if positive_frames:
            positive, error = _query(
                python, cli, context / "concept-graph.json", case_root, f"{case_index}-positive",
                "explain_driver_effect", {"driver": driver_name, "target_frame": positive_frames[-1]},
            )
            counts["queries_checked"] += 1
            if error:
                cli_failures.append({"case": case_index, "command": "positive driver effect", "stderr": error})
            else:
                check("positive causal query", positive["answer"]["target_pose_can_change_relative_to_root"], True)
        negative, error = _query(
            python, cli, context / "concept-graph.json", case_root, f"{case_index}-negative",
            "explain_driver_effect", {"driver": driver_name, "target_frame": expected["root"]},
        )
        counts["queries_checked"] += 1
        counts["negative_controls_checked"] += 1
        if error:
            cli_failures.append({"case": case_index, "command": "negative driver effect", "stderr": error})
        else:
            check("exact negative causal query", negative["answer"]["target_pose_can_change_relative_to_root"], False)
            check("exact negative has no unknowns", negative["unknowns"], [])

    leaf = expected["leaves"][0]
    leaf_answer, error = _query(
        python, cli, context / "concept-graph.json", case_root, f"{case_index}-leaf",
        "describe_entity", {"entity": f"link/{leaf}"},
    )
    counts["queries_checked"] += 1
    counts["negative_controls_checked"] += 1
    if error:
        cli_failures.append({"case": case_index, "command": "leaf boundary", "stderr": error})
    else:
        leaf_clauses = [
            clause for clause in leaf_answer["supporting_clauses"]
            if clause["predicate"] == "is_structural_leaf"
        ]
        check("leaf boundary clause count", len(leaf_clauses), 1)
        if leaf_clauses:
            check(
                "leaf boundary wording",
                "does not by itself assert end-effector" in leaf_clauses[0]["cnl"],
                True,
            )

    verify = _run([
        python, str(cli), "verify-concept-graph", str(context),
        "--concept", str(context / "concept-graph.json"),
        "--language", str(context / "concept-language.rsl"),
    ])
    if verify.returncode != 0:
        cli_failures.append({"case": case_index, "command": "verify-concept-graph", "stderr": verify.stderr[-2000:]})
    else:
        check("exact verifier", json.loads(verify.stdout)["status"], "passed")

    if case_index == 0 and graph["projections"]["articulation"]["drivers"]:
        tampered = json.loads(json.dumps(graph))
        tampered["projections"]["articulation"]["drivers"][0]["affected_frames"] = []
        tampered["concept_graph_sha256"] = _canonical_digest(tampered)
        tampered_path = case_root / "tampered-concept-graph.json"
        tampered_path.write_bytes(_json_bytes(tampered))
        query_path = case_root / "tamper-query.json"
        query_path.write_bytes(_json_bytes({
            "schema_version": QUERY_SCHEMA,
            "query_id": "tamper-negative-control",
            "intent": "structural_summary",
            "parameters": {},
        }))
        rejection = _run([python, str(cli), "query-concepts", str(tampered_path), str(query_path)])
        counts["negative_controls_checked"] += 1
        check("self-consistent projection tamper rejected", rejection.returncode, 2)

    rsl = (context / "concept-language.rsl").read_text(encoding="utf-8")
    check("RSL identity boundary", "IDENTITY_RULE" in rsl, True)
    check("RSL epistemic boundary", "finite component is not a certified global branch" in rsl, True)
    return counts, discrepancies, cli_failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cli", type=Path, default=Path(__file__).with_name("robot_spatial.py"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    rng = random.Random(args.seed)
    totals = {
        "topology_edges_checked": 0,
        "descendant_relations_checked": 0,
        "segments_checked": 0,
        "drivers_checked": 0,
        "mimic_physical_joints_checked": 0,
        "affected_frames_checked": 0,
        "queries_checked": 0,
        "negative_controls_checked": 0,
    }
    discrepancies: list[dict[str, Any]] = []
    cli_failures: list[dict[str, Any]] = []
    generated_mimic_joints = 0
    generated_negative_mimic_joints = 0
    generated_nested_mimic_joints = 0
    generated_branch_cases = 0
    generated_serial_cases = 0
    generated_joint_types: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="concept-oracle-") as temp_dir:
        root = Path(temp_dir)
        for case_index in range(args.cases):
            case_root = root / f"case-{case_index:04d}"
            case_root.mkdir()
            urdf_text = _generate_urdf(rng, case_index)
            parsed_root = ET.fromstring(urdf_text)
            joint_elements = parsed_root.findall("joint")
            joint_elements_by_name = {joint.attrib["name"]: joint for joint in joint_elements}
            for joint in joint_elements:
                joint_type = joint.attrib["type"]
                generated_joint_types[joint_type] = generated_joint_types.get(joint_type, 0) + 1
                mimic = joint.find("mimic")
                if mimic is None:
                    continue
                generated_mimic_joints += 1
                if float(mimic.attrib.get("multiplier", "1")) < 0.0:
                    generated_negative_mimic_joints += 1
                if joint_elements_by_name[mimic.attrib["joint"]].find("mimic") is not None:
                    generated_nested_mimic_joints += 1
            parents = [joint.find("parent").attrib["link"] for joint in joint_elements]  # type: ignore[union-attr]
            if len(parents) != len(set(parents)):
                generated_branch_cases += 1
            else:
                generated_serial_cases += 1
            counts, case_discrepancies, case_failures = _evaluate_case(
                args.python, args.cli.resolve(), case_root, case_index, urdf_text
            )
            for key, value in counts.items():
                totals[key] += value
            discrepancies.extend(case_discrepancies)
            cli_failures.extend(case_failures)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if not discrepancies and not cli_failures else "failed",
        "method": {
            "production_modules_imported": [],
            "production_interface": "public robot_spatial.py CLI only",
            "independent_parser": "xml.etree.ElementTree",
            "independent_derivations": [
                "rooted tree and complete typed edges",
                "branch points, structural leaves, and maximal serial segments",
                "transitive descendant paths",
                "recursive mimic-to-independent-driver resolution",
                "driver effects over link and joint pre-motion frame derivations",
            ],
        },
        "seed": args.seed,
        "requested_cases": args.cases,
        "completed_cases": args.cases - len({failure["case"] for failure in cli_failures}),
        "generated_coverage": {
            "serial_case_count": generated_serial_cases,
            "branch_cases": generated_branch_cases,
            "joint_type_counts": generated_joint_types,
            "mimic_joint_count": generated_mimic_joints,
            "negative_multiplier_mimic_joint_count": generated_negative_mimic_joints,
            "nested_mimic_joint_count": generated_nested_mimic_joints,
            **totals,
        },
        "discrepancy_count": len(discrepancies),
        "cli_failure_count": len(cli_failures),
        "discrepancies": discrepancies,
        "cli_failures": cli_failures,
        "exclusions": [
            "physical construction and calibration truth",
            "functional roles, affordances, and action semantics",
            "supplemental constraints and global configuration topology",
            "numeric forward kinematics, dynamics, control, hardware, and safety",
        ],
    }
    encoded = _json_bytes(report)
    if args.out is None:
        sys.stdout.buffer.write(encoded)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(encoded)
        print(json.dumps({"status": report["status"], "report": str(args.out.resolve())}, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
