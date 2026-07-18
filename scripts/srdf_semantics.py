#!/usr/bin/env python3
"""Strict SRDF semantic parser for the robot-spatial canonical model."""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


class SRDFError(ValueError):
    """An invalid or unsupported SRDF semantic declaration."""


def _required(element: ET.Element, attribute: str, context: str) -> str:
    value = element.get(attribute)
    if not value:
        raise SRDFError(f"{context} is missing required {attribute!r}")
    return value


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def parse_srdf(path: Path | None, robot: Any) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        raw = path.read_bytes()
        root = ET.fromstring(raw)
    except (OSError, ET.ParseError) as error:
        raise SRDFError(f"cannot read SRDF {path}: {error}") from error
    if root.tag != "robot":
        raise SRDFError("SRDF root element must be <robot>")
    robot_name = _required(root, "name", "SRDF robot")
    if robot_name != robot.name:
        raise SRDFError(f"SRDF robot name {robot_name!r} does not match URDF robot name {robot.name!r}")

    raw_groups: dict[str, dict[str, Any]] = {}
    for element in root.findall("group"):
        name = _required(element, "name", "SRDF group")
        if name in raw_groups:
            raise SRDFError(f"duplicate SRDF group name: {name!r}")
        raw_groups[name] = {
            "explicit_joints": [_required(child, "name", f"SRDF group {name!r} joint") for child in element.findall("joint")],
            "explicit_passive_joints": [
                _required(child, "name", f"SRDF group {name!r} passive_joint")
                for child in element.findall("passive_joint")
            ],
            "explicit_links": [_required(child, "name", f"SRDF group {name!r} link") for child in element.findall("link")],
            "chains": [
                {
                    "base_link": _required(child, "base_link", f"SRDF group {name!r} chain"),
                    "tip_link": _required(child, "tip_link", f"SRDF group {name!r} chain"),
                }
                for child in element.findall("chain")
            ],
            "subgroups": [_required(child, "name", f"SRDF group {name!r} subgroup") for child in element.findall("group")],
        }
    for group_name, group in raw_groups.items():
        unknown_joints = sorted(
            (set(group["explicit_joints"]) | set(group["explicit_passive_joints"])) - set(robot.joints)
        )
        unknown_links = sorted(set(group["explicit_links"]) - set(robot.links))
        unknown_subgroups = sorted(set(group["subgroups"]) - set(raw_groups))
        if unknown_joints:
            raise SRDFError(f"SRDF group {group_name!r} references unknown joints: {unknown_joints}")
        if unknown_links:
            raise SRDFError(f"SRDF group {group_name!r} references unknown links: {unknown_links}")
        if unknown_subgroups:
            raise SRDFError(f"SRDF group {group_name!r} references unknown subgroups: {unknown_subgroups}")

    resolved_groups: dict[str, dict[str, Any]] = {}
    active: set[str] = set()

    def resolve_group(name: str) -> dict[str, Any]:
        if name in resolved_groups:
            return resolved_groups[name]
        if name in active:
            raise SRDFError(f"SRDF subgroup cycle detected at group {name!r}")
        active.add(name)
        raw_group = raw_groups[name]
        joints = list(raw_group["explicit_joints"]) + list(raw_group["explicit_passive_joints"])
        links = list(raw_group["explicit_links"])
        chain_records: list[dict[str, Any]] = []
        for chain in raw_group["chains"]:
            try:
                path_record = robot.chain(chain["base_link"], chain["tip_link"])
            except ValueError as error:
                raise SRDFError(f"invalid chain in SRDF group {name!r}: {error}") from error
            if any(step["traversal"] != "parent_to_child" for step in path_record["steps"]):
                raise SRDFError(f"SRDF group {name!r} chain tip {chain['tip_link']!r} is not downstream of base {chain['base_link']!r}")
            joints.extend(step["joint"] for step in path_record["steps"])
            links.extend(path_record["links"])
            chain_records.append({**chain, "resolved_links": path_record["links"], "resolved_joints": [step["joint"] for step in path_record["steps"]]})
        for subgroup_name in raw_group["subgroups"]:
            subgroup = resolve_group(subgroup_name)
            joints.extend(subgroup["expanded_joints"])
            links.extend(subgroup["expanded_links"])
        active.remove(name)
        resolved = {
            **raw_group,
            "chains": chain_records,
            "expanded_joints": _ordered_unique(joints),
            "expanded_links": _ordered_unique(links),
        }
        resolved_groups[name] = resolved
        return resolved

    for group_name in raw_groups:
        resolve_group(group_name)

    group_states: dict[str, dict[str, Any]] = {}
    for element in root.findall("group_state"):
        state_name = _required(element, "name", "SRDF group_state")
        group_name = _required(element, "group", f"SRDF group_state {state_name!r}")
        if group_name not in resolved_groups:
            raise SRDFError(f"SRDF group_state {state_name!r} references unknown group {group_name!r}")
        joint_values: dict[str, float] = {}
        for joint_element in element.findall("joint"):
            joint_name = _required(joint_element, "name", f"SRDF group_state {state_name!r} joint")
            raw_value = _required(joint_element, "value", f"SRDF group_state {state_name!r} joint {joint_name!r}")
            if joint_name not in robot.joints:
                raise SRDFError(f"SRDF group_state {state_name!r} references unknown joint {joint_name!r}")
            if joint_name not in resolved_groups[group_name]["expanded_joints"]:
                raise SRDFError(f"joint {joint_name!r} in SRDF group_state {state_name!r} is not in group {group_name!r}")
            values = raw_value.split()
            if len(values) != 1:
                raise SRDFError(f"multi-DOF SRDF group_state values are not supported for joint {joint_name!r}")
            try:
                value = float(values[0])
            except ValueError as error:
                raise SRDFError(f"non-numeric SRDF group_state value for joint {joint_name!r}") from error
            if not math.isfinite(value):
                raise SRDFError(f"non-finite SRDF group_state value for joint {joint_name!r}")
            joint = robot.joints[joint_name]
            if joint.type == "fixed":
                raise SRDFError(f"SRDF group_state cannot assign fixed joint {joint_name!r}")
            lower, upper = joint.limit["lower"], joint.limit["upper"]
            if lower is not None and value < lower:
                raise SRDFError(f"SRDF group_state {state_name!r} value for {joint_name!r} is below its lower limit")
            if upper is not None and value > upper:
                raise SRDFError(f"SRDF group_state {state_name!r} value for {joint_name!r} is above its upper limit")
            if joint_name in joint_values:
                raise SRDFError(f"duplicate joint {joint_name!r} in SRDF group_state {state_name!r}")
            joint_values[joint_name] = value
        try:
            resolved_pose = robot.resolve_pose(joint_values)
        except ValueError as error:
            raise SRDFError(f"invalid SRDF group_state {state_name!r}: {error}") from error
        for joint_name, declared_value in joint_values.items():
            if robot.joints[joint_name].mimic and not math.isclose(
                resolved_pose[joint_name], declared_value, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise SRDFError(
                    f"SRDF group_state {state_name!r} assigns mimic joint {joint_name!r}={declared_value}, "
                    f"but its declared mimic relation resolves to {resolved_pose[joint_name]}"
                )
        key = f"{group_name}/{state_name}"
        if key in group_states:
            raise SRDFError(f"duplicate SRDF group_state: {key!r}")
        group_states[key] = {"name": state_name, "group": group_name, "joints": joint_values}

    passive_joints: list[str] = []
    for element in root.iter("passive_joint"):
        name = _required(element, "name", "SRDF passive_joint")
        if name not in robot.joints:
            raise SRDFError(f"SRDF passive_joint references unknown joint {name!r}")
        if robot.joints[name].type == "fixed":
            raise SRDFError(f"SRDF passive_joint {name!r} cannot be fixed")
        passive_joints.append(name)

    virtual_joints: dict[str, dict[str, str]] = {}
    for element in root.findall("virtual_joint"):
        name = _required(element, "name", "SRDF virtual_joint")
        child_link = _required(element, "child_link", f"SRDF virtual_joint {name!r}")
        if child_link not in robot.links:
            raise SRDFError(f"SRDF virtual_joint {name!r} references unknown child link {child_link!r}")
        joint_type = _required(element, "type", f"SRDF virtual_joint {name!r}")
        if joint_type not in {"fixed", "floating", "planar"}:
            raise SRDFError(f"SRDF virtual_joint {name!r} has unsupported type {joint_type!r}")
        virtual_joints[name] = {
            "type": joint_type,
            "parent_frame": _required(element, "parent_frame", f"SRDF virtual_joint {name!r}"),
            "child_link": child_link,
        }

    end_effectors: dict[str, dict[str, str | None]] = {}
    for element in root.findall("end_effector"):
        name = _required(element, "name", "SRDF end_effector")
        parent_link = _required(element, "parent_link", f"SRDF end_effector {name!r}")
        group_name = _required(element, "group", f"SRDF end_effector {name!r}")
        parent_group = element.get("parent_group")
        if parent_link not in robot.links:
            raise SRDFError(f"SRDF end_effector {name!r} references unknown parent link {parent_link!r}")
        if group_name not in resolved_groups:
            raise SRDFError(f"SRDF end_effector {name!r} references unknown group {group_name!r}")
        if parent_group is not None and parent_group not in resolved_groups:
            raise SRDFError(f"SRDF end_effector {name!r} references unknown parent group {parent_group!r}")
        if parent_group is not None and parent_link not in resolved_groups[parent_group]["expanded_links"]:
            raise SRDFError(f"SRDF end_effector {name!r} parent link {parent_link!r} is not in parent group {parent_group!r}")
        end_effectors[name] = {"parent_link": parent_link, "component_group": group_name, "parent_group": parent_group}

    disabled_collisions: list[dict[str, str]] = []
    for element in root.findall("disable_collisions"):
        link1 = _required(element, "link1", "SRDF disable_collisions")
        link2 = _required(element, "link2", "SRDF disable_collisions")
        if link1 not in robot.links or link2 not in robot.links:
            raise SRDFError(f"SRDF disable_collisions references unknown links: {link1!r}, {link2!r}")
        if link1 == link2:
            raise SRDFError(f"SRDF disable_collisions cannot name the same link twice: {link1!r}")
        disabled_collisions.append({"link1": link1, "link2": link2, "reason": element.get("reason", "")})

    return {
        "status": "parsed_and_validated",
        "schema_version": "robot-srdf-semantics.v1",
        "source": {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()},
        "robot_name": robot_name,
        "groups": {name: resolved_groups[name] for name in sorted(resolved_groups)},
        "named_poses": {name: group_states[name] for name in sorted(group_states)},
        "end_effectors": end_effectors,
        "passive_joints": sorted(set(passive_joints)),
        "virtual_joints": virtual_joints,
        "disabled_collisions": sorted(disabled_collisions, key=lambda item: (item["link1"], item["link2"])),
    }


def resolve_named_pose(srdf: dict[str, Any], name: str) -> tuple[str, dict[str, float]]:
    poses = srdf["named_poses"]
    if name in poses:
        record = poses[name]
        return name, dict(record["joints"])
    candidates = [key for key, record in poses.items() if record["name"] == name]
    if len(candidates) == 1:
        key = candidates[0]
        return key, dict(poses[key]["joints"])
    if not candidates:
        raise SRDFError(f"unknown SRDF named pose {name!r}; available: {sorted(poses)}")
    raise SRDFError(f"ambiguous SRDF named pose {name!r}; use one of {sorted(candidates)}")
