#!/usr/bin/env python3
"""Generate and grade grounded robot-spatial competency questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


class EvaluationError(ValueError):
    """An invalid evaluation artifact or answer submission."""


def json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def jsonl_dump(records: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(data, dict):
        raise EvaluationError(f"JSON artifact {path} must contain an object")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvaluationError(f"cannot read JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationError(f"invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise EvaluationError(f"JSONL record at {path}:{line_number} must be an object")
        records.append(record)
    return records


def matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][index] * b[index][column] for index in range(4)) for column in range(4)] for row in range(4)]


def inverse_rigid(transform: list[list[float]]) -> list[list[float]]:
    result = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    for row in range(3):
        for column in range(3):
            result[row][column] = transform[column][row]
        result[row][3] = -sum(transform[column][row] * transform[column][3] for column in range(3))
    return result


def quaternion_xyzw(transform: list[list[float]]) -> list[float]:
    trace = transform[0][0] + transform[1][1] + transform[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (transform[2][1] - transform[1][2]) / scale
        qy = (transform[0][2] - transform[2][0]) / scale
        qz = (transform[1][0] - transform[0][1]) / scale
    elif transform[0][0] > transform[1][1] and transform[0][0] > transform[2][2]:
        scale = math.sqrt(1.0 + transform[0][0] - transform[1][1] - transform[2][2]) * 2.0
        qw = (transform[2][1] - transform[1][2]) / scale
        qx = 0.25 * scale
        qy = (transform[0][1] + transform[1][0]) / scale
        qz = (transform[0][2] + transform[2][0]) / scale
    elif transform[1][1] > transform[2][2]:
        scale = math.sqrt(1.0 + transform[1][1] - transform[0][0] - transform[2][2]) * 2.0
        qw = (transform[0][2] - transform[2][0]) / scale
        qx = (transform[0][1] + transform[1][0]) / scale
        qy = 0.25 * scale
        qz = (transform[1][2] + transform[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + transform[2][2] - transform[0][0] - transform[1][1]) * 2.0
        qw = (transform[1][0] - transform[0][1]) / scale
        qx = (transform[0][2] + transform[2][0]) / scale
        qy = (transform[1][2] + transform[2][1]) / scale
        qz = 0.25 * scale
    quaternion = [qx, qy, qz, qw]
    if quaternion[3] < 0.0:
        quaternion = [-value for value in quaternion]
    return [0.0 if abs(value) < 1e-12 else round(value, 12) for value in quaternion]


def relative_pose(model: dict[str, Any], reference: str, target: str) -> dict[str, Any]:
    frames = model["frames"]
    reference_matrix = frames[reference]["world_from_frame"]["matrix_4x4_rowmajor"]
    target_matrix = frames[target]["world_from_frame"]["matrix_4x4_rowmajor"]
    transform = matrix_multiply(inverse_rigid(reference_matrix), target_matrix)
    return {
        "translation_xyz_m": [0.0 if abs(transform[index][3]) < 1e-12 else round(transform[index][3], 12) for index in range(3)],
        "quaternion_xyzw": quaternion_xyzw(transform),
    }


def answer_shape(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def spatial_truth_sha256(model: dict[str, Any]) -> str:
    def portable(value: Any) -> Any:
        if isinstance(value, list):
            return [portable(item) for item in value]
        if not isinstance(value, dict):
            return value
        content_bound_location = isinstance(value.get("sha256"), str)
        return {
            key: portable(item)
            for key, item in value.items()
            if key != "artifacts"
            and not (content_bound_location and key in {"path", "urdf"})
        }

    truth = portable(model)
    return hashlib.sha256(json.dumps(truth, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def generate_records(
    model: dict[str, Any],
    facts: list[dict[str, Any]],
    concept_graph: dict[str, Any] | None = None,
    functional_model: dict[str, Any] | None = None,
    action_assurance: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if model.get("schema_version") != "robot-spatial.v2":
        raise EvaluationError("evaluation generator requires robot-spatial.v2 model.json")
    robot_name = model["robot"]["name"]
    root_link = model["robot"]["root_link"]
    pose_name = model["pose"]["name"]
    source_sha256 = model["source"]["sha256"]
    truth_sha256 = spatial_truth_sha256(model)
    questions: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []

    def add(
        capability: str,
        task: str,
        query_identity: dict[str, Any],
        prompt_en: str,
        prompt_zh_tw: str,
        answer: Any,
        *,
        evidence_type: str,
        evidence_locator: str,
        exact: bool,
        difficulty: str = "direct",
        list_order: str = "ordered",
        absolute_tolerance: float = 1e-6,
        context: dict[str, Any] | None = None,
        evidence_sha256: str | None = None,
        submission_answer_contract: Any | None = None,
    ) -> None:
        identity = {"capability": capability, "task": task, **query_identity}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()[:20]
        question_id = f"spatial-{digest}"
        shared = {
            "question_id": question_id,
            "capability": capability,
            "task": task,
            "difficulty": difficulty,
        }
        questions.append({
            "schema_version": "robot-spatial-question.v1",
            **shared,
            "robot": robot_name,
            "pose": pose_name,
            "prompt": {"en": prompt_en, "zh_tw": prompt_zh_tw},
            "submission_contract": {
                "record": {
                    "question_id": question_id,
                    "answer": (
                        submission_answer_contract
                        if submission_answer_contract is not None
                        else f"<{answer_shape(answer)}>"
                    ),
                },
                "instruction": (
                    "Return one JSONL record with exactly question_id and answer; do not include explanation inside answer. "
                    "When answer is an object contract, preserve every shown key exactly; placeholders declare value types, not expected values or array lengths."
                ),
            },
            "context": context or {},
        })
        keys.append({
            "schema_version": "robot-spatial-answer-key.v1",
            **shared,
            "answer": answer,
            "comparison": {
                "absolute_tolerance": absolute_tolerance,
                "relative_tolerance": 1e-9,
                "list_order": list_order,
            },
            "evidence": {
                "source_type": evidence_type,
                "source_sha256": evidence_sha256 or source_sha256,
                "spatial_truth_sha256": truth_sha256,
                "locator": evidence_locator,
                "exact": exact,
            },
        })

    add(
        "topology",
        "identify_root_link",
        {"robot": robot_name},
        f"Which link is the kinematic root of robot {robot_name}?",
        f"機器人 {robot_name} 的運動學根連桿是哪一個？",
        root_link,
        evidence_type="urdf_declared_and_tree_validated",
        evidence_locator="robot.root_link",
        exact=True,
    )
    for joint_name, joint in sorted(model["joints"].items()):
        add(
            "topology",
            "identify_joint_connection",
            {"joint": joint_name},
            f"What joint type and parent-to-child link connection are declared for {joint_name}?",
            f"關節 {joint_name} 宣告的類型與父連桿到子連桿關係是什麼？",
            {"joint_type": joint["type"], "parent_link": joint["parent_link"], "child_link": joint["child_link"]},
            evidence_type="urdf_declared",
            evidence_locator=f"joints.{joint_name}",
            exact=True,
        )
    for frame_name, frame in sorted(model["frames"].items()):
        add(
            "frame_semantics",
            "classify_frame",
            {"frame": frame_name},
            f"Classify frame {frame_name} and name its immediate parent frame.",
            f"請分類 frame {frame_name}，並指出它的直接父 frame。",
            {"semantic_type": frame["type"], "parent_frame": frame["parent_frame"]},
            evidence_type="urdf_frame_derivation",
            evidence_locator=f"frames.{frame_name}",
            exact=True,
        )
    mass_properties = model.get("physical_analysis", {}).get("declared_mass_properties")
    if mass_properties:
        coverage = mass_properties["coverage"]
        add(
            "mass_properties",
            "report_declared_aggregate_mass_properties",
            {"pose": pose_name, "selection": mass_properties["selection"], "expressed_in_frame": mass_properties["expressed_in_frame"]},
            f"At pose {pose_name}, report the aggregate URDF-declared mass, center of mass, and inertia about that center expressed in {mass_properties['expressed_in_frame']}. Also report missing inertial links and whether physical-world completeness is established.",
            f"在姿態 {pose_name} 下，請回報以 {mass_properties['expressed_in_frame']} 表示的 URDF 宣告總質量、重心與關於該重心的慣量；另列出未宣告 inertial 的 links，並說明是否已建立實體世界完整性。",
            {
                "status": mass_properties["status"],
                "declared_mass_kg": mass_properties["declared_mass_kg"],
                "center_of_mass_in_expressed_frame_m": mass_properties["center_of_mass_in_expressed_frame_m"],
                "inertia_about_center_of_mass_in_expressed_frame_kg_m2": mass_properties["inertia_about_center_of_mass_in_expressed_frame_kg_m2"],
                "missing_inertial_links": coverage["missing_inertial_links"],
                "physical_world_completeness": coverage["physical_world_completeness"],
            },
            evidence_type="urdf_declared_inertials_forward_kinematics_parallel_axis_theorem",
            evidence_locator="physical_analysis.declared_mass_properties",
            exact=True,
            difficulty="derived_numeric_epistemic",
        )
        for link_name, link in sorted(model["links"].items()):
            inertial = link["inertial"]
            if inertial is None:
                continue
            add(
                "mass_properties",
                "interpret_link_inertial_declaration",
                {"link": link_name},
                f"For link {link_name}, report the inertial validation status, inertial frame, declared mass, link-from-inertial origin, tensor components in that inertial frame, and principal moments.",
                f"請回報 link {link_name} 的 inertial 驗證狀態、inertial frame、宣告質量、link-from-inertial 原點、以該 inertial frame 表示的 tensor 分量與 principal moments。",
                {
                    "validation_status": inertial["validation"]["status"],
                    "inertial_frame": inertial["frame"],
                    "declared_mass_kg": inertial["mass_kg"],
                    "link_from_inertial_origin_xyz_m": inertial["origin_xyz_m"],
                    "link_from_inertial_origin_rpy_rad": inertial["origin_rpy_rad"],
                    "inertia_kg_m2": inertial["inertia_kg_m2"],
                    "principal_moments_kg_m2": inertial["validation"]["principal_moments_kg_m2"],
                },
                evidence_type="urdf_declared_and_inertial_tensor_validated",
                evidence_locator=f"links.{link_name}.inertial",
                exact=True,
                difficulty="frame_and_tensor_semantics",
            )
    gravity_loads = model.get("physical_analysis", {}).get(
        "declared_static_gravity_loads_under_root_frame_convention"
    )
    if gravity_loads:
        add(
            "static_gravity_loads",
            "report_gravity_model_and_driver_loads",
            {"pose": pose_name, "gravity": gravity_loads["gravity"]},
            f"At pose {pose_name}, under the exported gravity convention, report the modeled generalized gravity force and opposite ideal static holding effort for every independent driver, including units and inertial coverage.",
            f"在姿態 {pose_name} 與匯出的重力慣例下，請回報每個獨立驅動關節的模型 generalized gravity force、反向理想靜態 holding effort、單位與 inertial 覆蓋範圍。",
            {
                "status": gravity_loads["status"],
                "gravity": gravity_loads["gravity"],
                "independent_driver_loads": {
                    driver: {
                        key: record[key]
                        for key in (
                            "unit",
                            "generalized_gravity_force",
                            "ideal_static_holding_effort",
                        )
                    }
                    for driver, record in (gravity_loads.get("independent_driver_loads") or {}).items()
                },
                "coverage": gravity_loads["coverage"],
            },
            evidence_type="urdf_declared_inertials_forward_kinematics_gravity_projection_with_mimic_chain_rule",
            evidence_locator="physical_analysis.declared_static_gravity_loads_under_root_frame_convention",
            exact=True,
            difficulty="derived_numeric_epistemic",
        )
        for driver, load in (gravity_loads.get("independent_driver_loads") or {}).items():
            add(
                "static_gravity_loads",
                "interpret_driver_gravity_sign",
                {"pose": pose_name, "joint": driver, "gravity": gravity_loads["gravity"]},
                f"For independent joint {driver} at pose {pose_name}, what modeled gravity generalized force acts along positive joint motion, and what opposite ideal holding effort balances it? Include the unit.",
                f"獨立關節 {driver} 在姿態 {pose_name} 下，沿正向關節運動的模型重力 generalized force 是多少？反向平衡它的理想 holding effort 是多少？請附單位。",
                {
                    "generalized_gravity_force": load["generalized_gravity_force"],
                    "ideal_static_holding_effort": load["ideal_static_holding_effort"],
                    "unit": load["unit"],
                },
                evidence_type="urdf_declared_inertials_forward_kinematics_gravity_projection_with_mimic_chain_rule",
                evidence_locator=f"physical_analysis.declared_static_gravity_loads_under_root_frame_convention.independent_driver_loads.{driver}",
                exact=True,
                difficulty="derived_numeric",
            )
        add(
            "static_gravity_loads",
            "reject_hardware_effort_claim_from_gravity_model",
            {"pose": pose_name},
            "Does a computed URDF gravity-only holding effort prove the real actuator can hold the robot at that pose?",
            "URDF 重力模型算出的 holding effort，是否足以證明真實 actuator 能在該姿態維持住機器人？",
            False,
            evidence_type="static_gravity_load_epistemic_contract",
            evidence_locator="physical_analysis.declared_static_gravity_loads_under_root_frame_convention.epistemic_scope",
            exact=True,
            difficulty="epistemic",
        )
    actuation = model.get("actuation", {})
    if actuation:
        add(
            "actuation_declarations",
            "report_embedded_actuation_coverage",
            {},
            "Which ros2_control systems and legacy transmissions are embedded in this expanded URDF, and which movable joints lack a declared ros2 command interface?",
            "此展開 URDF 內嵌了哪些 ros2_control systems 與 legacy transmissions？哪些可動關節沒有宣告 ros2 command interface？",
            {
                "ros2_control_systems": sorted(actuation["ros2_control_systems"]),
                "legacy_transmissions": sorted(actuation["legacy_transmissions"]),
                "movable_joints_without_declared_command_interface": actuation["coverage"]["movable_joints_without_declared_command_interface"],
            },
            evidence_type="expanded_urdf_actuation_declaration_transcription_and_reference_validation",
            evidence_locator="actuation",
            exact=True,
            difficulty="declaration_grounding",
        )
        for joint_name, binding in sorted(actuation.get("joint_bindings", {}).items()):
            add(
                "actuation_declarations",
                "report_joint_control_bindings",
                {"joint": joint_name},
                f"What embedded ros2_control command/state interfaces and legacy transmission bindings are declared for joint {joint_name}?",
                f"關節 {joint_name} 內嵌宣告了哪些 ros2_control command/state interfaces 與 legacy transmission bindings？",
                binding,
                evidence_type="expanded_urdf_actuation_declaration_transcription_and_reference_validation",
                evidence_locator=f"actuation.joint_bindings.{joint_name}",
                exact=True,
                difficulty="declaration_grounding",
            )
        for system_name, system in sorted(actuation.get("ros2_control_systems", {}).items()):
            add(
                "actuation_declarations",
                "report_ros2_control_system_declaration",
                {"system": system_name},
                f"For embedded ros2_control system {system_name}, report its type, declared hardware plugin/parameters, joint names, sensor names, and GPIO names.",
                f"請回報內嵌 ros2_control system {system_name} 的 type、宣告的 hardware plugin/parameters、joint 名稱、sensor 名稱與 GPIO 名稱。",
                {
                    "type": system["type"],
                    "hardware": system["hardware"],
                    "joints": sorted(system["joints"]),
                    "sensors": sorted(system["sensors"]),
                    "gpios": sorted(system["gpios"]),
                },
                evidence_type="expanded_urdf_ros2_control_declaration",
                evidence_locator=f"actuation.ros2_control_systems.{system_name}",
                exact=True,
                difficulty="declaration_grounding",
            )
        for transmission_name, transmission in sorted(actuation.get("legacy_transmissions", {}).items()):
            add(
                "actuation_declarations",
                "report_legacy_transmission_declaration",
                {"transmission": transmission_name},
                f"Report the embedded legacy transmission {transmission_name}: type, joint records, actuator records, and declared mechanical reductions.",
                f"請回報內嵌 legacy transmission {transmission_name} 的 type、joint records、actuator records 與宣告 mechanical reductions。",
                transmission,
                evidence_type="expanded_urdf_legacy_transmission_declaration",
                evidence_locator=f"actuation.legacy_transmissions.{transmission_name}",
                exact=True,
                difficulty="declaration_grounding",
            )
        for joint_name, joint in sorted(model.get("joints", {}).items()):
            if joint.get("dynamics") is None:
                continue
            add(
                "actuation_declarations",
                "interpret_joint_dynamics_declaration",
                {"joint": joint_name},
                f"For joint {joint_name}, report standard URDF dynamics values and uninterpreted extension attributes. Do those declarations alone establish the real dynamic response?",
                f"請回報關節 {joint_name} 的標準 URDF dynamics 值與未解釋 extension attributes；這些宣告本身是否足以建立真實 dynamic response？",
                {"dynamics": joint["dynamics"], "establishes_real_dynamic_response": False},
                evidence_type="urdf_joint_dynamics_declaration",
                evidence_locator=f"joints.{joint_name}.dynamics",
                exact=True,
                difficulty="declaration_epistemics",
            )
        add(
            "actuation_declarations",
            "reject_runtime_capability_claim_from_declaration",
            {},
            "Do embedded ros2_control or transmission declarations by themselves prove that plugins, controllers, interface claiming, and connected hardware are operational?",
            "僅憑內嵌 ros2_control 或 transmission 宣告，是否能證明 plugins、controllers、interface claiming 與連接硬體都可運作？",
            False,
            evidence_type="actuation_declaration_epistemic_contract",
            evidence_locator="actuation.epistemic_scope",
            exact=True,
            difficulty="epistemic",
        )
    render_atlas = model.get("artifacts", {}).get("semantic_render_atlas")
    if isinstance(render_atlas, dict):
        render_id = render_atlas["render_id"]
        manifest_sha256 = render_atlas["manifest_sha256"]
        add(
            "semantic_visual_grounding",
            "report_render_atlas_binding_and_coverage",
            {"render_id": render_id},
            "Report the semantic render atlas ID/input digest, exact model/pose binding, coordinate contract, geometry coverage, and the four available view IDs.",
            "請回報 semantic render atlas 的 ID/input digest、精確 model/pose binding、coordinate contract、geometry coverage，以及四個可用 view IDs。",
            {
                "render_id": render_id,
                "render_input_sha256": render_atlas["render_input_sha256"],
                "model_semantic_sha256": model["source"]["semantic_sha256"],
                "pose_binding": render_atlas["pose_binding"],
                "coordinate_contract": render_atlas["coordinate_contract"],
                "coverage": render_atlas["coverage"],
                "view_ids": sorted(render_atlas["views"]),
            },
            evidence_type="digest_bound_semantic_render_atlas_manifest",
            evidence_locator="artifacts.semantic_render_atlas",
            evidence_sha256=manifest_sha256,
            exact=True,
            difficulty="visual_provenance",
        )
        contract_view_id = "isometric" if "isometric" in render_atlas["views"] else sorted(render_atlas["views"])[0]
        contract_view = render_atlas["views"][contract_view_id]
        add(
            "semantic_visual_grounding",
            "report_view_projection_and_pixel_contract",
            {"render_id": render_id, "view": contract_view_id},
            f"For render view {contract_view_id}, report the root-XYZ-to-UV projection, depth convention, UV-to-pixel mapping, scene projection bounds, and SVG artifact digest.",
            f"請回報 render view {contract_view_id} 的 root-XYZ-to-UV projection、depth convention、UV-to-pixel mapping、scene projection bounds 與 SVG artifact digest。",
            {
                "view_entity": f"render_view/{render_id}/{contract_view_id}",
                "projection": contract_view["projection"],
                "screen": contract_view["screen"],
                "scene_projection_bounds_uv_m": contract_view["scene_projection_bounds_uv_m"],
                "artifact": contract_view["artifact"],
            },
            evidence_type="deterministic_root_xyz_to_uv_to_pixel_projection",
            evidence_locator=f"artifacts.semantic_render_atlas.views.{contract_view_id}",
            evidence_sha256=manifest_sha256,
            exact=True,
            difficulty="view_numeric_grounding",
        )
        representative_frames = contract_view.get("link_frames", [])
        if representative_frames:
            representative = max(
                representative_frames,
                key=lambda record: (
                    sum(float(value) ** 2 for value in record["origin_root_xyz_m"]),
                    record["entity_id"],
                ),
            )
            frame_entity = representative["entity_id"]
            per_view: dict[str, Any] = {}
            for view_id, view in sorted(render_atlas["views"].items()):
                match = next(record for record in view["link_frames"] if record["entity_id"] == frame_entity)
                per_view[view_id] = {
                    "projected_uv_m": match["projected_uv_m"],
                    "pixel_xy": match["pixel_xy"],
                }
            add(
                "semantic_visual_grounding",
                "cross_ground_frame_origin_across_views",
                {"render_id": render_id, "frame": frame_entity},
                f"For {frame_entity} at pose {pose_name}, report its root-frame XYZ origin and its projected UV and pixel XY in every semantic render view.",
                f"對 {frame_entity} 在姿態 {pose_name} 下，請回報其 root-frame XYZ origin，以及它在每個 semantic render view 的 projected UV 與 pixel XY。",
                {
                    "frame_entity": frame_entity,
                    "origin_root_xyz_m": representative["origin_root_xyz_m"],
                    "by_view": per_view,
                },
                evidence_type="deterministic_cross_view_frame_origin_projection",
                evidence_locator=f"artifacts.semantic_render_atlas.views.*.link_frames[{frame_entity}]",
                evidence_sha256=manifest_sha256,
                exact=True,
                difficulty="cross_view_numeric_grounding",
            )
        geometry_records = contract_view.get("geometry", [])
        if geometry_records:
            exact_records = [
                record for record in geometry_records
                if record["projection_support"]["convex_hull"].startswith("exact_")
            ]
            representative_geometry = sorted(exact_records or geometry_records, key=lambda record: record["entity_id"])[0]
            add(
                "semantic_visual_grounding",
                "report_geometry_projection_support_bounds_and_depth",
                {
                    "render_id": render_id,
                    "view": contract_view_id,
                    "geometry": representative_geometry["entity_id"],
                },
                f"In render view {contract_view_id}, report the projection support/fidelity, UV hull bounds, pixel hull bounds, and depth interval for {representative_geometry['entity_id']}.",
                f"在 render view {contract_view_id} 中，請回報 {representative_geometry['entity_id']} 的 projection support/fidelity、UV hull bounds、pixel hull bounds 與 depth interval。",
                {
                    "view_entity": f"render_view/{render_id}/{contract_view_id}",
                    "geometry_entity": representative_geometry["entity_id"],
                    "kind": representative_geometry["kind"],
                    "geometry_type": representative_geometry["geometry_type"],
                    "projection_support": representative_geometry["projection_support"],
                    "projection_bounds_uv_m": representative_geometry["projection_bounds_uv_m"],
                    "pixel_bounds_xy": representative_geometry["pixel_bounds_xy"],
                    "depth_interval_m": representative_geometry["depth_interval_m"],
                },
                evidence_type="deterministic_semantic_geometry_convex_projection",
                evidence_locator=f"artifacts.semantic_render_atlas.views.{contract_view_id}.geometry[{representative_geometry['entity_id']}]",
                evidence_sha256=manifest_sha256,
                exact=representative_geometry["projection_support"]["convex_hull"].startswith("exact_"),
                difficulty="view_geometry_grounding",
            )
        add(
            "semantic_visual_grounding",
            "reject_independent_visibility_or_physical_truth_claim",
            {"render_id": render_id},
            "Does a verified semantic render atlas independently establish photorealistic visibility/occlusion, calibrated camera pixels, or agreement with the physical robot and world?",
            "通過驗證的 semantic render atlas，是否能獨立建立 photorealistic visibility/occlusion、calibrated camera pixels，或與實體機器人及世界的一致性？",
            False,
            evidence_type="semantic_render_atlas_epistemic_contract",
            evidence_locator="artifacts.semantic_render_atlas.epistemic_scope",
            evidence_sha256=manifest_sha256,
            exact=True,
            difficulty="epistemic",
        )
    articulation = model.get("artifacts", {}).get("articulation_grammar")
    if isinstance(articulation, dict):
        grammar_id = articulation["grammar_id"]
        grammar_sha256 = articulation["sha256"]
        add(
            "articulation_grammar_understanding",
            "report_grammar_binding_language_layers_and_coverage",
            {"grammar_id": grammar_id},
            "Report the articulation grammar ID/digests, source binding, source-binding-free law identity, coordinate/language/layer contracts, coverage, and artifact path.",
            "請回報 articulation grammar 的 ID/digests、source binding、排除 source binding 的 law identity、coordinate/language/layer contracts、coverage 與 artifact path。",
            {
                "grammar_entity": f"articulation_grammar/{grammar_id}",
                "grammar_id": grammar_id,
                "grammar_input_sha256": articulation["grammar_input_sha256"],
                "artifact_path": articulation["path"],
                "artifact_sha256": grammar_sha256,
                "law_identity": articulation["law_identity"],
                "source_binding": articulation["source_binding"],
                "coordinate_contract": articulation["coordinate_contract"],
                "language_contract": articulation["language_contract"],
                "layer_contract": articulation["layer_contract"],
                "coverage": articulation["coverage"],
            },
            evidence_type="digest_bound_pose_independent_articulation_grammar",
            evidence_locator="artifacts.articulation_grammar",
            evidence_sha256=grammar_sha256,
            exact=True,
            difficulty="grammar_provenance",
        )
        variables = articulation["independent_variables"]
        representative_driver = None
        if variables:
            representative_driver = max(
                sorted(variables),
                key=lambda name: (len(variables[name]["physical_joints_driven"]), name),
            )
            variable = variables[representative_driver]
            add(
                "articulation_grammar_understanding",
                "explain_independent_variable_domain_and_causality",
                {"grammar_id": grammar_id, "driver": representative_driver},
                f"For articulation variable {representative_driver}, report its unit/type/default, full mimic-constrained feasible domain and constraints, physical joints driven, and structural causality.",
                f"對 articulation variable {representative_driver}，請回報 unit/type/default、完整 mimic-constrained feasible domain 與 constraints、physical joints driven，以及 structural causality。",
                {
                    "variable_entity": f"articulation_variable/{grammar_id}/{representative_driver}",
                    "driver_joint": representative_driver,
                    **variable,
                },
                evidence_type="urdf_limits_affine_mimic_constraints_and_tree_causality",
                evidence_locator=f"artifacts.articulation_grammar.independent_variables.{representative_driver}",
                evidence_sha256=grammar_sha256,
                exact=True,
                difficulty="variable_domain",
            )
        else:
            add(
                "articulation_grammar_understanding",
                "report_zero_independent_variable_contract",
                {"grammar_id": grammar_id},
                "Report the grammar's zero-independent-variable contract and the coverage counts that establish it as a constant kinematic tree.",
                "請回報此 grammar 的零 independent-variable contract，以及證明它是 constant kinematic tree 的 coverage counts。",
                {
                    "independent_variables": variables,
                    "independent_driver_count": articulation["coverage"]["independent_driver_count"],
                    "physical_joint_count": articulation["coverage"]["physical_joint_count"],
                    "fixed_joint_count": articulation["coverage"]["fixed_joint_count"],
                },
                evidence_type="pose_independent_constant_tree_variable_contract",
                evidence_locator="artifacts.articulation_grammar.independent_variables",
                evidence_sha256=grammar_sha256,
                exact=True,
                difficulty="variable_domain",
            )
        position_rules = articulation["joint_position_rules"]
        dependent_candidates = sorted(
            name for name, rule in position_rules.items()
            if rule["type"] == "affine_driver_dependency"
        )
        position_joint = (
            dependent_candidates[0]
            if dependent_candidates
            else representative_driver or (sorted(position_rules)[0] if position_rules else None)
        )
        if position_joint is not None:
            add(
                "articulation_grammar_understanding",
                "report_joint_position_equation",
                {"grammar_id": grammar_id, "joint": position_joint},
                f"Report the complete typed position rule for physical joint {position_joint}, including its independent driver, multiplier, offset, unit, mimic chain, and equation.",
                f"請回報 physical joint {position_joint} 的完整 typed position rule，包括 independent driver、multiplier、offset、unit、mimic chain 與 equation。",
                {
                    "operator_entity": f"articulation_operator/{grammar_id}/{position_joint}",
                    "joint": position_joint,
                    "position_rule": position_rules[position_joint],
                },
                evidence_type="normalized_affine_joint_position_law",
                evidence_locator=f"artifacts.articulation_grammar.joint_position_rules.{position_joint}",
                evidence_sha256=grammar_sha256,
                exact=True,
                difficulty="mimic_equation",
            )
        else:
            add(
                "articulation_grammar_understanding",
                "report_zero_joint_position_rule_contract",
                {"grammar_id": grammar_id},
                "Report the empty physical-joint position-rule contract for this single-link grammar.",
                "請回報此 single-link grammar 的空 physical-joint position-rule contract。",
                {"joint_position_rules": position_rules, "physical_joint_count": 0},
                evidence_type="single_link_zero_joint_position_law",
                evidence_locator="artifacts.articulation_grammar.joint_position_rules",
                evidence_sha256=grammar_sha256,
                exact=True,
                difficulty="mimic_equation",
            )
        joint_operators = articulation["joint_operators"]
        movable_operators = {
            name: operator for name, operator in joint_operators.items()
            if operator["joint_type"] != "fixed"
        }
        operator_candidates = movable_operators or joint_operators
        if operator_candidates:
            representative_operator = max(
                sorted(operator_candidates),
                key=lambda name: (len(model["joints"][name]["child_link"]), name),
            )
            add(
                "articulation_grammar_understanding",
                "explain_typed_joint_operator",
                {"grammar_id": grammar_id, "joint": representative_operator},
                f"Explain the executable typed operator for {representative_operator}: edge, pre-motion constant, motion type/axis/parameter source, post-motion-to-child-zero constant, composition rule, and own-pre-motion-frame causality.",
                f"請說明 {representative_operator} 的 executable typed operator：edge、pre-motion constant、motion type/axis/parameter source、post-motion-to-child-zero constant、composition rule，以及 own-pre-motion-frame causality。",
                {
                    "operator_entity": f"articulation_operator/{grammar_id}/{representative_operator}",
                    "joint_position_rule": position_rules[representative_operator],
                "operator": operator_candidates[representative_operator],
                },
                evidence_type="typed_constant_plus_joint_motion_operator",
                evidence_locator=f"artifacts.articulation_grammar.joint_operators.{representative_operator}",
                evidence_sha256=grammar_sha256,
                exact=True,
                difficulty="operator_semantics",
            )
        else:
            add(
                "articulation_grammar_understanding",
                "report_zero_joint_operator_contract",
                {"grammar_id": grammar_id},
                "Report the empty joint-operator contract and identity-only root frame semantics for this single-link grammar.",
                "請回報此 single-link grammar 的空 joint-operator contract 與 identity-only root frame semantics。",
                {
                    "joint_operators": joint_operators,
                    "physical_joint_count": 0,
                    "root_frame": articulation["coordinate_contract"]["root_frame"],
                },
                evidence_type="single_link_zero_joint_operator_law",
                evidence_locator="artifacts.articulation_grammar.joint_operators",
                evidence_sha256=grammar_sha256,
                exact=True,
                difficulty="operator_semantics",
            )
        derivations = articulation["frame_derivations"]
        representative_frame = max(
            sorted(derivations),
            key=lambda name: (
                len(derivations[name]["ordered_operator_refs"]),
                len(derivations[name]["independent_driver_dependencies"]),
                name,
            ),
        )
        add(
            "articulation_grammar_understanding",
            "compose_root_to_frame_derivation",
            {"grammar_id": grammar_id, "frame": representative_frame},
            f"Report the complete pose-independent root-to-frame derivation for {representative_frame}: semantic identity, attachment link, ordered operator references, terminal constant, independent dependencies, tokens, and expression.",
            f"請回報 {representative_frame} 完整的 pose-independent root-to-frame derivation：semantic identity、attachment link、ordered operator references、terminal constant、independent dependencies、tokens 與 expression。",
            {
                "derivation_entity": f"articulation_derivation/{grammar_id}/{representative_frame}",
                **derivations[representative_frame],
            },
            evidence_type="validated_tree_path_typed_operator_composition",
            evidence_locator=f"artifacts.articulation_grammar.frame_derivations.{representative_frame}",
            evidence_sha256=grammar_sha256,
            exact=True,
            difficulty="grammar_composition",
        )
        add(
            "articulation_grammar_understanding",
            "distinguish_law_fk_jacobian_motion_atlas_and_reject_extra_claims",
            {"grammar_id": grammar_id},
            "Report the exact distinction among articulation grammar, FK, geometric Jacobian, and motion atlas. Does this grammar establish dynamics, a trajectory/swept volume, closed-loop constraints, control/hardware behavior, calibration, or physical truth?",
            "請精確區分 articulation grammar、FK、geometric Jacobian 與 motion atlas。這份 grammar 是否能建立 dynamics、trajectory/swept volume、closed-loop constraints、control/hardware behavior、calibration 或 physical truth？",
            {
                "layer_contract": articulation["layer_contract"],
                "establishes_dynamics_trajectory_swept_volume_closed_loops_control_hardware_calibration_or_physical_truth": False,
                "epistemic_scope": articulation["epistemic_scope"],
            },
            evidence_type="articulation_grammar_epistemic_contract",
            evidence_locator="artifacts.articulation_grammar.layer_contract",
            evidence_sha256=grammar_sha256,
            exact=True,
            difficulty="epistemic",
        )
    constraint_graph = model.get("artifacts", {}).get("constraint_graph")
    if isinstance(constraint_graph, dict):
        graph_id = constraint_graph["constraint_graph_id"]
        graph_sha256 = constraint_graph["sha256"]
        evaluation = constraint_graph["evaluation"]
        add(
            "supplemental_mechanism_understanding",
            "report_graph_binding_topology_and_tree_boundary",
            {"constraint_graph_id": graph_id},
            "Report the supplemental constraint graph identity/digests, source binding, structural graph, coverage, and whether the articulation tree is only a parameterization rather than the complete mechanism.",
            "請回報 supplemental constraint graph 的 identity/digests、source binding、structural graph、coverage，以及 articulation tree 是否只是參數化而非完整機構。",
            {
                "constraint_graph_entity": f"constraint_graph/{graph_id}",
                "constraint_graph_id": graph_id,
                "constraint_graph_sha256": constraint_graph["constraint_graph_sha256"],
                "artifact_sha256": graph_sha256,
                "source_binding": constraint_graph["source_binding"],
                "structural_graph": constraint_graph["structural_graph"],
                "coverage": constraint_graph["coverage"],
            },
            evidence_type="deterministic_compilation_of_digest_bound_articulation_and_asserted_constraint_spec",
            evidence_locator="artifacts.constraint_graph",
            evidence_sha256=graph_sha256,
            exact=True,
            difficulty="mechanism_topology_epistemics",
        )
        if constraint_graph["attachments"]:
            attachment = sorted(constraint_graph["attachments"], key=lambda record: record["attachment_id"])[0]
            add(
                "supplemental_mechanism_understanding",
                "report_asserted_attachment_and_evaluated_pose",
                {"constraint_graph_id": graph_id, "attachment_id": attachment["attachment_id"]},
                f"For supplemental attachment {attachment['attachment_id']}, report its typed parent transform/role and its evaluated root pose. Is that attachment semantics asserted or physically observed?",
                f"對 supplemental attachment {attachment['attachment_id']}，請回報 typed parent transform/role 與 evaluated root pose；此 attachment semantics 是 asserted 還是 physically observed？",
                {
                    "attachment_entity": f"attachment/{graph_id}/{attachment['attachment_id']}",
                    "declaration": attachment,
                    "root_pose_at_export_binding": evaluation["attachments"][attachment["frame_id"]],
                    "semantics_are_asserted": True,
                    "physically_observed": False,
                },
                evidence_type="asserted_rigid_attachment_plus_deterministic_pose_composition",
                evidence_locator=f"artifacts.constraint_graph.attachments[{attachment['attachment_id']}]",
                evidence_sha256=graph_sha256,
                exact=False,
                difficulty="assertion_and_derivation_separation",
            )
        representative_constraint = sorted(
            constraint_graph["constraints"], key=lambda record: record["constraint_id"]
        )[0]
        evaluated_constraint = next(
            record for record in evaluation["constraints"]
            if record["constraint_id"] == representative_constraint["constraint_id"]
        )
        add(
            "supplemental_mechanism_understanding",
            "report_declared_constraint_and_typed_residual",
            {"constraint_graph_id": graph_id, "constraint_id": representative_constraint["constraint_id"]},
            f"For constraint {representative_constraint['constraint_id']}, report the asserted typed relation separately from every export-pose residual component, tolerance, normalized value, and satisfaction status.",
            f"對 constraint {representative_constraint['constraint_id']}，請將 asserted typed relation 與 export pose 的每個 residual component、tolerance、normalized value 及 satisfaction status 分開回報。",
            {
                "constraint_entity": f"constraint/{graph_id}/{representative_constraint['constraint_id']}",
                "asserted_relation": representative_constraint,
                "evaluation_pose": evaluation["pose"],
                "typed_residual": evaluated_constraint,
                "relation_is_asserted": True,
                "residual_is_deterministically_evaluated": True,
            },
            evidence_type="asserted_mechanism_relation_plus_deterministic_typed_residual",
            evidence_locator=f"artifacts.constraint_graph.constraints[{representative_constraint['constraint_id']}]",
            evidence_sha256=graph_sha256,
            exact=False,
            difficulty="constraint_residual_grounding",
        )
        local = evaluation.get("local_constraint_analysis")
        if isinstance(local, dict):
            add(
                "supplemental_mechanism_understanding",
                "distinguish_tree_variables_local_rank_mobility_and_global_dof",
                {"constraint_graph_id": graph_id, "pose": evaluation["pose"]["name"]},
                "At the export pose, report the tree independent-variable count, numerical residual-Jacobian rank, local mobility estimate, method/tolerances, and whether this proves global mechanism DOF.",
                "在 export pose 下，請回報 tree independent-variable count、數值 residual-Jacobian rank、local mobility estimate、method/tolerances，以及這是否能證明全域機構 DOF。",
                {
                    "pose": evaluation["pose"],
                    "tree_independent_variable_count": local["tree_independent_variable_count"],
                    "local_constraint_rank": local["local_constraint_rank"],
                    "local_mobility_estimate": local["local_mobility_estimate"],
                    "analysis_type": local["analysis_type"],
                    "finite_difference_step": local["finite_difference_step"],
                    "rank_relative_tolerance": local["rank_relative_tolerance"],
                    "singularity_warning": local["singularity_warning"],
                    "proves_global_mechanism_dof": False,
                },
                evidence_type="pose_conditioned_normalized_residual_jacobian_rank",
                evidence_locator="artifacts.constraint_graph.evaluation.local_constraint_analysis",
                evidence_sha256=graph_sha256,
                exact=True,
                difficulty="local_mechanism_analysis",
            )
        add(
            "supplemental_mechanism_understanding",
            "reject_physical_global_or_unique_solution_claims",
            {"constraint_graph_id": graph_id},
            "Does a satisfied constraint evaluation or converged local solve prove global feasibility/uniqueness, complete configuration space, physical assembly/calibration, compliance/contact, dynamics, hardware behavior, or safety?",
            "constraint evaluation satisfied 或 local solve converged，是否能證明全域 feasibility/uniqueness、完整 configuration space、實體 assembly/calibration、compliance/contact、dynamics、hardware behavior 或 safety？",
            False,
            evidence_type="supplemental_constraint_graph_epistemic_contract",
            evidence_locator="artifacts.constraint_graph.epistemic_scope",
            evidence_sha256=graph_sha256,
            exact=True,
            difficulty="epistemic",
        )
    configuration_atlas = model.get("artifacts", {}).get("configuration_atlas")
    if isinstance(configuration_atlas, dict):
        atlas_id = configuration_atlas["configuration_atlas_id"]
        atlas_sha256 = configuration_atlas["sha256"]
        add(
            "finite_configuration_space_understanding",
            "report_atlas_binding_contract_status_and_coverage",
            {"configuration_atlas_id": atlas_id},
            "Report the configuration atlas identity/digests, exact graph/spec bindings, declared-sampling status, coverage, and finite-evidence scope.",
            "請回報 configuration atlas 的 identity/digests、精確 graph/spec bindings、declared-sampling status、coverage 與有限證據範圍。",
            {
                "configuration_atlas_entity": f"configuration_atlas/{atlas_id}",
                "configuration_atlas_id": atlas_id,
                "configuration_atlas_sha256": configuration_atlas["configuration_atlas_sha256"],
                "artifact_sha256": atlas_sha256,
                "status": configuration_atlas["status"],
                "source_binding": configuration_atlas["source_binding"],
                "coverage": configuration_atlas["coverage"],
                "epistemic_scope": configuration_atlas["epistemic_scope"],
            },
            evidence_type="deterministic_digest_bound_multi_seed_configuration_atlas",
            evidence_locator="artifacts.configuration_atlas",
            evidence_sha256=atlas_sha256,
            exact=True,
            difficulty="configuration_space_contract",
        )
        representative_chart = sorted(
            configuration_atlas["charts"], key=lambda record: record["chart_id"]
        )[0]
        richest_sample = max(
            representative_chart["samples"],
            key=lambda record: (record["unique_solution_count"], -record["sample_index"]),
        )
        add(
            "finite_configuration_space_understanding",
            "report_multiple_satisfying_nodes_at_one_declared_sample",
            {
                "configuration_atlas_id": atlas_id,
                "chart_id": representative_chart["chart_id"],
                "sample_index": richest_sample["sample_index"],
            },
            f"For chart {representative_chart['chart_id']} sample {richest_sample['sample_index']}, report the fixed parameter binding, coverage status, and every unique satisfying configuration node with residual maximum. Does this enumerate every physical assembly configuration?",
            f"對 chart {representative_chart['chart_id']} 的 sample {richest_sample['sample_index']}，請回報固定 parameter binding、coverage status，以及每個 unique satisfying configuration node 與 residual maximum；這是否列舉了所有實體 assembly configuration？",
            {
                "configuration_chart_entity": f"configuration_chart/{atlas_id}/{representative_chart['chart_id']}",
                "sample_index": richest_sample["sample_index"],
                "parameter_driver": richest_sample["parameter_driver"],
                "parameter_value": richest_sample["parameter_value"],
                "coverage_status": richest_sample["coverage_status"],
                "minimum_solutions_required": richest_sample["minimum_solutions_required"],
                "unique_solution_count": richest_sample["unique_solution_count"],
                "solutions": [
                    {
                        "node_id": node["node_id"],
                        "independent_driver_positions": node["independent_driver_positions"],
                        "constraint_status": node["constraint_status"],
                        "maximum_normalized_abs": node["maximum_normalized_abs"],
                    }
                    for node in richest_sample["solutions"]
                ],
                "exhaustive_physical_assembly_enumeration": False,
            },
            evidence_type="finite_multi_seed_local_solve_nodes_with_exact_constraint_reexecution",
            evidence_locator=f"artifacts.configuration_atlas.charts[{representative_chart['chart_id']}].samples[{richest_sample['sample_index']}]",
            evidence_sha256=atlas_sha256,
            exact=False,
            difficulty="finite_branch_witnesses",
        )
        singular_nodes = [
            node
            for chart in configuration_atlas["charts"]
            for sample in chart["samples"]
            for node in sample["solutions"]
            if node["singularity_witness"]["mechanism_rank_drop_candidate"]
            or node["singularity_witness"]["chart_parameterization_rank_drop_candidate"]
        ]
        if singular_nodes:
            singular_node = sorted(singular_nodes, key=lambda record: record["node_id"])[0]
            add(
                "finite_configuration_space_understanding",
                "interpret_observed_rank_drop_candidate",
                {"configuration_atlas_id": atlas_id, "node_id": singular_node["node_id"]},
                f"For configuration node {singular_node['node_id']}, report the binding, full/passive singular diagnostics, observed chart rank references, candidate labels, and whether this is a certified singularity.",
                f"對 configuration node {singular_node['node_id']}，請回報 binding、full/passive singular diagnostics、observed chart rank references、candidate labels，以及這是否為 certified singularity。",
                {
                    "node_id": singular_node["node_id"],
                    "independent_driver_positions": singular_node["independent_driver_positions"],
                    "full_constraint_jacobian": singular_node["full_constraint_jacobian"],
                    "chart_passive_jacobian": singular_node["chart_passive_jacobian"],
                    "singularity_witness": singular_node["singularity_witness"],
                    "certified_singularity": False,
                },
                evidence_type="finite_difference_normalized_jacobian_singular_values_relative_to_observed_chart_rank",
                evidence_locator=f"artifacts.configuration_atlas.node[{singular_node['node_id']}]",
                evidence_sha256=atlas_sha256,
                exact=False,
                difficulty="singularity_epistemics",
            )
        add(
            "finite_configuration_space_understanding",
            "reject_global_topology_from_finite_atlas",
            {"configuration_atlas_id": atlas_id},
            "Do complete declared sample minima, proximity components, and observed rank-drop candidates prove exhaustive branch coverage, global topology/connectivity, certified singularities, complete reachability, physical truth, or safety?",
            "declared sample minima 完成、proximity components 與 observed rank-drop candidates，是否能證明 exhaustive branch coverage、全域 topology/connectivity、certified singularities、完整 reachability、physical truth 或 safety？",
            False,
            evidence_type="configuration_atlas_epistemic_contract",
            evidence_locator="artifacts.configuration_atlas.epistemic_scope",
            evidence_sha256=atlas_sha256,
            exact=True,
            difficulty="epistemic",
        )
    concept_artifact = model.get("artifacts", {}).get("concept_graph")
    if concept_graph is not None:
        if not isinstance(concept_artifact, dict):
            raise EvaluationError("concept graph was supplied but model.json has no concept_graph artifact binding")
        if concept_graph.get("schema_version") != "robot-spatial-concept-graph.v1":
            raise EvaluationError("concept graph must use robot-spatial-concept-graph.v1")
        if concept_graph.get("concept_graph_id") != concept_artifact.get("concept_graph_id"):
            raise EvaluationError("concept graph identity does not match model artifact binding")
        if concept_graph.get("concept_graph_sha256") != concept_artifact.get("concept_graph_sha256"):
            raise EvaluationError("concept graph semantic digest does not match model artifact binding")
        concept_sha256 = concept_artifact["sha256"]
        topology = concept_graph["projections"]["topology"]
        articulation_projection = concept_graph["projections"]["articulation"]
        mechanism_projection = concept_graph["projections"]["mechanism"]
        configuration_projection = concept_graph["projections"]["configuration"]
        add(
            "proof_carrying_concept_understanding",
            "report_structural_abstraction_and_closed_world_contract",
            {"concept_graph_id": concept_graph["concept_graph_id"]},
            "Report the concept graph binding, root, branch points, structural leaves, maximal serial segments, independent drivers, and its exact-negative closed-world rule. Do structural leaves automatically mean end effectors?",
            "請回報 concept graph binding、root、branch points、structural leaves、maximal serial segments、independent drivers，以及 exact-negative closed-world rule；structural leaf 是否自動等於 end effector？",
            {
                "concept_graph_entity": concept_graph["concept_graph_id"],
                "concept_graph_sha256": concept_graph["concept_graph_sha256"],
                "root_link": topology["root_link"],
                "branch_points": topology["branch_points"],
                "structural_leaves": topology["structural_leaves"],
                "maximal_serial_segments": [
                    {
                        "segment_entity": segment["segment_entity"],
                        "start_link": segment["start_link"],
                        "end_link": segment["end_link"],
                        "ordered_links": segment["ordered_links"],
                        "ordered_joints": segment["ordered_joints"],
                    }
                    for segment in topology["maximal_serial_segments"]
                ],
                "independent_drivers": [
                    driver["driver_entity"] for driver in articulation_projection["drivers"]
                ],
                "negative_answer_rule": concept_graph["language_contract"]["negative_answer_rule"],
                "structural_leaf_implies_end_effector": False,
            },
            evidence_type="digest_bound_proof_carrying_concept_graph",
            evidence_locator="concept-graph.json#/projections/topology",
            evidence_sha256=concept_sha256,
            exact=True,
            difficulty="conceptual_abstraction",
        )
        if articulation_projection["drivers"]:
            representative_driver = max(
                articulation_projection["drivers"],
                key=lambda record: (len(record["affected_frames"]), record["driver_entity"]),
            )
            affected_target = representative_driver["affected_frames"][-1]
            frame_entities = [f"frame/{name}" for name in sorted(model["frames"])]
            unaffected_targets = [
                frame for frame in frame_entities
                if frame not in representative_driver["affected_frames"]
            ]
            unaffected_target = unaffected_targets[0] if unaffected_targets else None
            add(
                "proof_carrying_concept_understanding",
                "explain_driver_effect_with_exact_positive_and_negative",
                {
                    "concept_graph_id": concept_graph["concept_graph_id"],
                    "driver": representative_driver["driver_entity"],
                    "affected_target": affected_target,
                    "unaffected_target": unaffected_target,
                },
                f"For driver {representative_driver['driver_entity']}, report its domain and physical joints, whether it can change {affected_target} relative to root, and whether it can change {unaffected_target} relative to root while other independent drivers stay fixed. State why a negative is permitted.",
                f"對 driver {representative_driver['driver_entity']}，請回報 domain 與 physical joints；在其他 independent drivers 固定時，它是否能改變 {affected_target} 與 {unaffected_target} 相對 root 的 pose？並說明為何可以給 exact negative。",
                {
                    "driver_entity": representative_driver["driver_entity"],
                    "domain": representative_driver["domain"],
                    "physical_joints_driven": representative_driver["physical_joints_driven"],
                    "affected_target_frame": affected_target,
                    "affected_target_pose_can_change_relative_to_root": True,
                    "unaffected_target_frame": unaffected_target,
                    "unaffected_target_pose_can_change_relative_to_root": False if unaffected_target is not None else None,
                    "other_independent_drivers_held_fixed": True,
                    "negative_is_supported_by_complete_frame_dependency_projection": articulation_projection["coverage"]["negative_driver_frame_effect_answerable_from_complete_dependency_sets"],
                },
                evidence_type="complete_executable_articulation_dependency_projection",
                evidence_locator=f"concept-graph.json#/projections/articulation/drivers/{representative_driver['driver_entity']}",
                evidence_sha256=concept_sha256,
                exact=True,
                difficulty="causal_proof",
            )
        if mechanism_projection["constraints"]:
            representative_constraint = mechanism_projection["constraints"][0]
            add(
                "proof_carrying_concept_understanding",
                "distinguish_asserted_mechanism_relation_from_derived_dependency",
                {
                    "concept_graph_id": concept_graph["concept_graph_id"],
                    "constraint": representative_constraint["constraint_entity"],
                },
                f"For {representative_constraint['constraint_entity']}, report its type, role, driver dependencies, relation modality, and whether the relation itself is observed physical truth.",
                f"對 {representative_constraint['constraint_entity']}，請回報 type、role、driver dependencies、relation modality，以及此 relation 本身是否為 observed physical truth。",
                {
                    "constraint_entity": representative_constraint["constraint_entity"],
                    "type": representative_constraint["type"],
                    "role": representative_constraint["role"],
                    "driver_dependencies": representative_constraint["driver_dependencies"],
                    "relation_modality": "supplemental_asserted_relation",
                    "dependency_modality": "derived_exact_from_asserted_relation",
                    "observed_physical_truth": False,
                },
                evidence_type="asserted_relation_plus_exact_articulation_dependency_derivation",
                evidence_locator=f"concept-graph.json#/projections/mechanism/constraints/{representative_constraint['constraint_entity']}",
                evidence_sha256=concept_sha256,
                exact=False,
                difficulty="epistemic_composition",
            )
        add(
            "proof_carrying_concept_understanding",
            "reject_symbolic_abstraction_overclaim",
            {"concept_graph_id": concept_graph["concept_graph_id"], "task": "epistemic_boundary"},
            "Does a regenerated concept graph prove inferred component function, undeclared semantic roles, physical construction/calibration, runtime or hardware behavior, global configuration branches/topology, certified singularities, or safety?",
            "可精確重建的 concept graph 是否能證明 inferred component function、未宣告的 semantic roles、實體 construction/calibration、runtime 或 hardware behavior、全域 configuration branches/topology、certified singularities 或 safety？",
            {
                "inferred_component_function": False,
                "undeclared_semantic_roles": False,
                "physical_construction_or_calibration": False,
                "runtime_or_hardware_behavior": False,
                "global_configuration_branches_or_topology": False,
                "certified_singularities": False,
                "safety": False,
                "finite_component_is_global_branch": configuration_projection.get("finite_proximity_component_is_global_branch", False),
            },
            evidence_type="concept_language_open_world_and_epistemic_contract",
            evidence_locator="concept-graph.json#/language_contract",
            evidence_sha256=concept_sha256,
            exact=True,
            difficulty="epistemic",
        )
    motion_atlas = model.get("artifacts", {}).get("counterfactual_motion_atlas")
    if isinstance(motion_atlas, dict):
        motion_id = motion_atlas["motion_id"]
        manifest_sha256 = motion_atlas["manifest_sha256"]
        drivers = motion_atlas["drivers"]
        add(
            "counterfactual_motion_understanding",
            "report_motion_atlas_binding_policy_and_coverage",
            {"motion_id": motion_id},
            "Report the motion-atlas ID/input digest, exact model/baseline-pose binding, signed perturbation policy, coordinate contract, coverage, and independent driver IDs.",
            "請回報 motion atlas 的 ID/input digest、精確 model/baseline-pose binding、signed perturbation policy、coordinate contract、coverage 與 independent driver IDs。",
            {
                "motion_id": motion_id,
                "motion_input_sha256": motion_atlas["motion_input_sha256"],
                "model_semantic_sha256": model["source"]["semantic_sha256"],
                "baseline_pose_binding": motion_atlas["baseline_pose_binding"],
                "perturbation_policy": motion_atlas["perturbation_policy"],
                "coordinate_contract": motion_atlas["coordinate_contract"],
                "coverage": motion_atlas["coverage"],
                "driver_ids": sorted(drivers),
            },
            evidence_type="digest_bound_counterfactual_motion_atlas_manifest",
            evidence_locator="artifacts.counterfactual_motion_atlas",
            evidence_sha256=manifest_sha256,
            exact=True,
            difficulty="motion_provenance",
        )
        representative_driver = max(
            sorted(drivers),
            key=lambda name: (
                len(drivers[name]["physical_joints_driven"]),
                len(drivers[name]["structural_causality"]["affected_links"]),
                name,
            ),
        )
        representative_record = drivers[representative_driver]
        add(
            "counterfactual_motion_understanding",
            "explain_driver_structure_limits_and_mimic_causality",
            {"motion_id": motion_id, "driver": representative_driver},
            f"For independent driver {representative_driver}, report its type/unit, baseline and nominal finite step, mimic-constrained feasible interval, physical joints driven, affected links/frames, and whether its own pre-motion frame is affected.",
            f"對 independent driver {representative_driver}，請回報 type/unit、baseline 與 nominal finite step、受 mimic 約束的 feasible interval、實際被驅動的 physical joints、affected links/frames，以及它自己的 pre-motion frame 是否受影響。",
            {
                "motion_driver_entity": f"motion_driver/{motion_id}/{representative_driver}",
                "driver_joint": representative_driver,
                "joint_type": representative_record["joint_type"],
                "joint_position_unit": representative_record["joint_position_unit"],
                "baseline_position": representative_record["baseline_position"],
                "nominal_step": representative_record["nominal_step"],
                "feasible_interval": representative_record["feasible_interval"],
                "physical_joints_driven": representative_record["physical_joints_driven"],
                "affected_links": representative_record["structural_causality"]["affected_links"],
                "affected_frames": representative_record["structural_causality"]["affected_frames"],
                "pre_motion_frame": representative_record["structural_causality"]["pre_motion_frame"],
                "pre_motion_frame_is_affected_by_own_motion": representative_record["structural_causality"]["pre_motion_frame_is_affected_by_own_motion"],
            },
            evidence_type="declared_limits_mimic_affine_constraints_and_kinematic_tree",
            evidence_locator=f"artifacts.counterfactual_motion_atlas.drivers.{representative_driver}",
            evidence_sha256=manifest_sha256,
            exact=True,
            difficulty="causal_structure",
        )
        available_directions = [
            direction for direction in ("minus", "plus")
            if "link_frame_deltas" in representative_record["endpoints"][direction]
        ]
        if available_directions:
            affected_links = representative_record["structural_causality"]["affected_links"]
            representative_link = max(
                sorted(affected_links),
                key=lambda link: (
                    sum(
                        representative_record["endpoints"][direction]["link_frame_deltas"][link]["origin_displacement_norm_m"]
                        + representative_record["endpoints"][direction]["link_frame_deltas"][link]["baseline_frame_from_endpoint_frame"]["angle_rad"]
                        for direction in available_directions
                    ),
                    link,
                ),
            )
            unaffected_links = sorted(set(model["links"]) - set(affected_links))
            unaffected_link = root_link if root_link in unaffected_links else unaffected_links[0]

            def endpoint_frame_summary(direction: str, link: str) -> dict[str, Any]:
                delta = representative_record["endpoints"][direction]["link_frame_deltas"][link]
                return {
                    "origin_displacement_root_xyz_m": delta["origin_displacement_root_xyz_m"],
                    "origin_displacement_norm_m": delta["origin_displacement_norm_m"],
                    "baseline_frame_from_endpoint_frame": delta["baseline_frame_from_endpoint_frame"],
                    "origin_moved": delta["origin_moved"],
                    "orientation_changed": delta["orientation_changed"],
                    "frame_changed": delta["frame_changed"],
                }

            add(
                "counterfactual_motion_understanding",
                "compare_affected_and_upstream_frames_under_signed_endpoints",
                {
                    "motion_id": motion_id,
                    "driver": representative_driver,
                    "affected_frame": representative_link,
                    "unaffected_frame": unaffected_link,
                },
                f"Holding other independent drivers fixed, compare frame/{representative_link} with upstream frame/{unaffected_link} under every available signed endpoint of {representative_driver}; report the applied joint delta, each frame's finite SE(3) delta, and the causality checks.",
                f"固定其他 independent drivers，請比較 frame/{representative_link} 與 upstream frame/{unaffected_link} 在 {representative_driver} 每個可用 signed endpoint 下的結果；回報 applied joint delta、兩個 frame 的 finite SE(3) delta 與 causality checks。",
                {
                    "motion_driver_entity": f"motion_driver/{motion_id}/{representative_driver}",
                    "affected_frame": f"frame/{representative_link}",
                    "unaffected_frame": f"frame/{unaffected_link}",
                    "by_direction": {
                        direction: {
                            "applied_delta": representative_record["endpoints"][direction]["applied_delta"],
                            "joint_position_unit": representative_record["endpoints"][direction]["joint_position_unit"],
                            "affected_frame_delta": endpoint_frame_summary(direction, representative_link),
                            "unaffected_frame_delta": endpoint_frame_summary(direction, unaffected_link),
                            "causality_check": representative_record["endpoints"][direction]["causality_check"],
                        }
                        for direction in available_directions
                    },
                },
                evidence_type="finite_endpoint_forward_kinematics_and_structural_causality_composition",
                evidence_locator=f"artifacts.counterfactual_motion_atlas.drivers.{representative_driver}.endpoints",
                evidence_sha256=manifest_sha256,
                exact=True,
                difficulty="compositional_causality",
            )
            view_id = "isometric" if "isometric" in representative_record["views"] else sorted(representative_record["views"])[0]
            motion_view = representative_record["views"][view_id]
            samples_answer: dict[str, Any] = {}
            for sample_name in ["baseline", *available_directions]:
                frame_record = next(
                    record for record in motion_view["samples"][sample_name]["link_frames"]
                    if record["frame_name"] == representative_link
                )
                samples_answer[sample_name] = {
                    "origin_root_xyz_m": frame_record["origin_root_xyz_m"],
                    "projected_uv_m": frame_record["projected_uv_m"],
                    "pixel_xy": frame_record["pixel_xy"],
                }
            vectors = {
                record["direction"]: {
                    key: record[key]
                    for key in (
                        "projected_displacement_uv_m",
                        "projected_displacement_norm_m",
                        "pixel_displacement_xy",
                        "pixel_displacement_norm",
                        "root_origin_displacement_xyz_m",
                        "root_origin_displacement_norm_m",
                        "orientation_change_rad",
                    )
                }
                for record in motion_view["motion_vectors"]
                if record["frame_name"] == representative_link and record["direction"] in available_directions
            }
            add(
                "counterfactual_motion_understanding",
                "cross_ground_finite_endpoint_motion_in_shared_view",
                {
                    "motion_id": motion_id,
                    "driver": representative_driver,
                    "view": view_id,
                    "frame": representative_link,
                },
                f"In the shared-screen {view_id} motion view for {representative_driver}, cross-ground frame/{representative_link}: report the common projection/screen contract, baseline and available endpoint root/UV/pixel positions, signed motion vectors, and SVG artifact.",
                f"在 {representative_driver} 共用 screen 的 {view_id} motion view 中，請交叉對應 frame/{representative_link}：回報共同 projection/screen contract、baseline 與可用 endpoints 的 root/UV/pixel positions、signed motion vectors 與 SVG artifact。",
                {
                    "motion_view_entity": f"motion_view/{motion_id}/{representative_driver}/{view_id}",
                    "frame_entity": f"frame/{representative_link}",
                    "projection": motion_view["projection"],
                    "screen": motion_view["screen"],
                    "samples": samples_answer,
                    "motion_vectors": vectors,
                    "artifact": motion_view["artifact"],
                },
                evidence_type="deterministic_finite_endpoint_shared_screen_projection",
                evidence_locator=f"artifacts.counterfactual_motion_atlas.drivers.{representative_driver}.views.{view_id}",
                evidence_sha256=manifest_sha256,
                exact=True,
                difficulty="cross_modal_motion_grounding",
            )
        add(
            "counterfactual_motion_understanding",
            "report_signed_limit_clipping_and_endpoint_availability",
            {"motion_id": motion_id, "drivers": sorted(drivers)},
            "For every independent driver, report minus/plus endpoint status, requested and applied delta, endpoint position, unit, and whether each endpoint is available, clipped, or blocked by the feasible limit.",
            "對每個 independent driver，請回報 minus/plus endpoint status、requested/applied delta、endpoint position、unit，以及各 endpoint 是 available、clipped 或被 feasible limit 阻擋。",
            {
                driver: {
                    direction: {
                        key: record["endpoints"][direction][key]
                        for key in ("status", "requested_delta", "applied_delta", "joint_position", "joint_position_unit")
                    }
                    for direction in ("minus", "plus")
                }
                for driver, record in sorted(drivers.items())
            },
            evidence_type="declared_and_mimic_constrained_signed_endpoint_policy",
            evidence_locator="artifacts.counterfactual_motion_atlas.drivers.*.endpoints",
            evidence_sha256=manifest_sha256,
            exact=True,
            difficulty="limit_reasoning",
        )
        add(
            "counterfactual_motion_understanding",
            "reject_trajectory_dynamics_swept_volume_and_hardware_claim",
            {"motion_id": motion_id},
            "Does a verified counterfactual motion atlas by itself establish the intermediate trajectory, continuous swept volume/collision, velocity, acceleration, effort, controller response, hardware motion, or safety?",
            "通過驗證的 counterfactual motion atlas，是否能單獨建立 intermediate trajectory、continuous swept volume/collision、velocity、acceleration、effort、controller response、hardware motion 或 safety？",
            False,
            evidence_type="counterfactual_motion_atlas_epistemic_contract",
            evidence_locator="artifacts.counterfactual_motion_atlas.epistemic_scope",
            evidence_sha256=manifest_sha256,
            exact=True,
            difficulty="epistemic",
        )
    world_scene = model.get("world_scene", {})
    if world_scene.get("status") == "parsed_validated_and_bound":
        scene_sha256 = world_scene["source"]["sha256"]
        mount = world_scene["robot_mount"]
        add(
            "world_scene",
            "report_snapshot_and_root_mount",
            {"scene_id": world_scene["scene_id"], "snapshot_id": world_scene["snapshot"]["id"]},
            "Report the bound scene ID, snapshot ID/time semantics, typed parent/root entities, world-from-robot-root pose, and root-placement provenance.",
            "請回報綁定的 scene ID、snapshot ID/time semantics、typed parent/root entities、world-from-robot-root pose 與 root placement provenance。",
            {
                "scene_id": world_scene["scene_id"],
                "snapshot": world_scene["snapshot"],
                "parent_entity": mount["parent_entity"],
                "root_entity": mount["root_entity"],
                "world_from_robot_root": mount["world_from_robot_root"],
                "placement_source": mount["source"],
            },
            evidence_type="validated_world_scene_root_binding",
            evidence_locator="world_scene.robot_mount",
            evidence_sha256=scene_sha256,
            exact=True,
            difficulty="scene_frame_grounding",
        )
        for object_id, object_record in sorted(world_scene.get("objects", {}).items()):
            add(
                "world_scene",
                "report_scene_object",
                {"scene_id": world_scene["scene_id"], "snapshot_id": world_scene["snapshot"]["id"], "object": object_id},
                f"For scene object {object_id}, report its typed ID, parent scene frame, world pose, asserted semantics, placement provenance, and collision geometry IDs/declarations.",
                f"請回報 scene object {object_id} 的 typed ID、parent scene frame、world pose、asserted semantics、placement provenance 與 collision geometry IDs/declarations。",
                {
                    "entity_id": object_record["entity_id"],
                    "parent_frame": object_record["parent_frame"],
                    "world_from_object": object_record["world_from_object"],
                    "semantics": object_record["semantics"],
                    "source": object_record["source"],
                    "collision_geometries": [
                        {
                            "entity_id": geometry["entity_id"],
                            "pose_in_object": geometry["pose_in_object"],
                            "geometry": geometry["geometry"],
                        }
                        for geometry in object_record["collision_geometries"]
                    ],
                },
                evidence_type="validated_world_scene_object_declaration",
                evidence_locator=f"world_scene.objects.{object_id}",
                evidence_sha256=scene_sha256,
                exact=True,
                difficulty="scene_object_grounding",
            )
        scene_gravity = model.get("physical_analysis", {}).get("declared_static_gravity_loads_under_scene_gravity", {})
        add(
            "world_scene_gravity",
            "report_world_to_root_gravity_conversion",
            {"scene_id": world_scene["scene_id"], "snapshot_id": world_scene["snapshot"]["id"], "pose": pose_name},
            "Report the scene-declared gravity, world-from-root pose, gravity vector converted into the robot root, and the resulting per-driver generalized gravity/holding efforts with units.",
            "請回報 scene 宣告的 gravity、world-from-root pose、轉換到 robot root 的 gravity vector，以及每個 driver 的 generalized gravity/holding efforts 與單位。",
            {
                "status": scene_gravity["status"],
                "scene_gravity": scene_gravity.get("scene_gravity"),
                "independent_driver_loads": {
                    driver: {
                        key: record[key]
                        for key in ("unit", "generalized_gravity_force", "ideal_static_holding_effort")
                    }
                    for driver, record in (((scene_gravity.get("loads") or {}).get("independent_driver_loads") or {}).items())
                },
                "coverage": None if scene_gravity.get("loads") is None else scene_gravity["loads"]["coverage"],
            },
            evidence_type="scene_world_gravity_rotation_through_root_mount_then_urdf_inertial_projection",
            evidence_locator="physical_analysis.declared_static_gravity_loads_under_scene_gravity",
            evidence_sha256=scene_sha256,
            exact=True,
            difficulty="multi_frame_physical_derivation",
        )
        scene_collision = world_scene["robot_environment_collision"]
        add(
            "robot_environment_collision",
            "report_scene_collision_and_minimum_separation",
            {"scene_id": world_scene["scene_id"], "snapshot_id": world_scene["snapshot"]["id"], "pose": pose_name},
            "For this scene snapshot and robot pose, report the robot/environment collision status, minimum-separation record, contact tolerance, and geometry/pair coverage including indeterminate pairs.",
            "對此 scene snapshot 與 robot pose，請回報 robot/environment collision status、minimum-separation record、contact tolerance，以及包含 indeterminate pairs 的 geometry/pair coverage。",
            {
                "status": scene_collision["status"],
                "minimum_separation": scene_collision["minimum_separation"],
                "contact_tolerance_m": scene_collision["contact_tolerance_m"],
                "coverage": scene_collision["coverage"],
            },
            evidence_type="all_declared_robot_environment_geometry_pair_analysis",
            evidence_locator="world_scene.robot_environment_collision",
            evidence_sha256=scene_sha256,
            exact=True,
            difficulty="collision_coverage_epistemics",
        )
        for pair_index, pair in enumerate(scene_collision["pair_results"]):
            add(
                "robot_environment_collision",
                "report_robot_environment_pair",
                {
                    "scene_id": world_scene["scene_id"],
                    "snapshot_id": world_scene["snapshot"]["id"],
                    "robot_geometry": pair["robot_geometry"],
                    "environment_geometry": pair["environment_geometry"],
                },
                f"Report the status, exact separation or conservative lower bound, method, and any unresolved reason for {pair['robot_geometry']} versus {pair['environment_geometry']}.",
                f"請回報 {pair['robot_geometry']} 對 {pair['environment_geometry']} 的 status、exact separation 或 conservative lower bound、method，以及任何 unresolved reason。",
                {
                    key: pair.get(key)
                    for key in (
                        "robot_geometry",
                        "robot_link",
                        "environment_geometry",
                        "environment_object",
                        "status",
                        "separation_m",
                        "separation_lower_bound_m",
                        "surface_distance_m",
                        "method",
                        "reason",
                    )
                    if key in pair
                },
                evidence_type="snapshot_bound_robot_environment_pair_analysis",
                evidence_locator=f"world_scene.robot_environment_collision.pair_results[{pair_index}]",
                evidence_sha256=scene_sha256,
                exact=True,
                difficulty="pairwise_collision_reasoning",
            )
        add(
            "world_scene",
            "reject_current_complete_world_claim",
            {"scene_id": world_scene["scene_id"], "snapshot_id": world_scene["snapshot"]["id"]},
            "Does an internally valid static scene snapshot prove that its calibration is accurate, that it is still current, and that every physical obstacle is represented?",
            "內部有效的 static scene snapshot，是否足以證明其 calibration 正確、仍是最新狀態，而且所有實體障礙物都已被表示？",
            False,
            evidence_type="world_scene_epistemic_contract",
            evidence_locator="world_scene.epistemic_scope",
            evidence_sha256=scene_sha256,
            exact=True,
            difficulty="epistemic",
        )
        add(
            "robot_environment_collision",
            "reject_unbounded_collision_free_claim",
            {"scene_id": world_scene["scene_id"], "snapshot_id": world_scene["snapshot"]["id"]},
            "If scene collision analysis reports collision_free with complete declared-pair coverage, does that prove the robot is collision-free against omitted or future physical objects?",
            "若 scene collision analysis 在宣告 pair coverage 完整時回報 collision_free，是否能證明機器人對未列入或未來出現的實體物件也 collision-free？",
            False,
            evidence_type="world_scene_collision_epistemic_contract",
            evidence_locator="world_scene.robot_environment_collision.epistemic_scope",
            evidence_sha256=scene_sha256,
            exact=True,
            difficulty="epistemic",
        )
    observed_world = model.get("observed_world", {})
    observation = observed_world.get("observation") if isinstance(observed_world, dict) else None
    if isinstance(observation, dict):
        log = observation["observation_log"]
        query = observation["query"]
        add(
            "temporal_observation",
            "report_time_selection_and_readiness",
            {"log": log["id"], "query": query["query_id"], "time_ns": query["time_ns"]},
            "Report the observation clock/query time, age limits, selected joint/root/object sample IDs with ages/statuses and ignored-future counts, plus readiness and declaration-fallback entities.",
            "請回報 observation clock/query time、age limits、選中的 joint/root/object sample IDs（含 ages/statuses 與 ignored-future counts），以及 readiness 與 declaration-fallback entities。",
            {
                "clock": log["clock"],
                "query": query,
                "selections": observation["selections"],
                "readiness": observation["readiness"],
            },
            evidence_type="bound_timestamp_policy_selection",
            evidence_locator="observed_world.observation",
            evidence_sha256=log["sha256"],
            exact=True,
            difficulty="temporal_grounding",
        )
        add(
            "temporal_observation",
            "distinguish_model_scene_and_observation_layers",
            {"log": log["id"], "query": query["query_id"]},
            "For this query, report the three epistemic layers and every effective joint/root/object source layer. A static-scene fallback must not be called an observation.",
            "對此 query，請回報三種 epistemic layers 與每個 effective joint/root/object 的 source layer；static-scene fallback 不可稱為 observation。",
            {
                "epistemic_layers": observation["epistemic_layers"],
                "effective_sources": observation["effective_state"]["sources"],
            },
            evidence_type="observation_layer_contract_and_selected_sources",
            evidence_locator="observed_world.observation.epistemic_layers|effective_state.sources",
            evidence_sha256=log["sha256"],
            exact=True,
            difficulty="epistemic",
        )
        add(
            "temporal_observation",
            "reject_future_sample_for_past_query",
            {"log": log["id"], "query": query["query_id"]},
            "May a sample timestamped after the query time be used to answer state at that earlier time under this version's selection contract?",
            "依照此版本的 selection contract，時間戳晚於 query time 的樣本，是否可用來回答較早時間點的狀態？",
            False,
            evidence_type="latest_past_zero_order_hold_contract",
            evidence_locator="observed_world.observation.selection_method.future_samples_consumed",
            evidence_sha256=log["sha256"],
            exact=True,
            difficulty="epistemic",
        )
        add(
            "temporal_observation",
            "reject_physical_truth_from_current_age",
            {"log": log["id"], "query": query["query_id"]},
            "If all required observations are marked current, does that alone establish sensor truth, calibration, omitted-object absence, and physical safety?",
            "若所有 required observations 都標為 current，是否僅憑此就能建立 sensor truth、calibration、omitted-object absence 與 physical safety？",
            False,
            evidence_type="observation_epistemic_contract",
            evidence_locator="observed_world.observation.epistemic_scope",
            evidence_sha256=log["sha256"],
            exact=True,
            difficulty="epistemic",
        )
        normalization = log.get("normalization")
        if isinstance(normalization, dict):
            add(
                "ros_observation_normalization",
                "report_capture_config_and_normalization_policy",
                {"log": log["id"], "capture": normalization["capture_id"]},
                "Report the exact ROS capture/config digests, adapter/method, clock policy, authority policy, and TF reconstruction policy that produced this observation log.",
                "請回報產生此 observation log 的 ROS capture/config 精確 digests、adapter/method、clock policy、authority policy 與 TF reconstruction policy。",
                {
                    "capture_id": normalization["capture_id"],
                    "capture_sha256": normalization["capture_sha256"],
                    "adapter_id": normalization["adapter_id"],
                    "config_sha256": normalization["config_sha256"],
                    "method": normalization["method"],
                    "clock_policy": normalization["clock_policy"],
                    "authority_policy": normalization["authority_policy"],
                    "tf_policy": normalization["tf_policy"],
                },
                evidence_type="digest_bound_ros_capture_normalization_provenance",
                evidence_locator="observed_world.observation.observation_log.normalization",
                evidence_sha256=log["sha256"],
                exact=True,
                difficulty="transport_provenance",
            )
            add(
                "ros_observation_normalization",
                "reject_source_truth_from_transport_identity",
                {"log": log["id"], "capture": normalization["capture_id"]},
                "Do matching clock-domain labels plus a unique ROS topic or publisher GID prove clock synchronization, publisher truth, calibration, and physical completeness?",
                "相符的 clock-domain labels 加上唯一 ROS topic 或 publisher GID，是否足以證明 clock synchronization、publisher truth、calibration 與 physical completeness？",
                False,
                evidence_type="ros_normalization_epistemic_contract",
                evidence_locator="observed_world.observation.observation_log.normalization.meaning",
                evidence_sha256=log["sha256"],
                exact=True,
                difficulty="epistemic",
            )
        observed_analysis = observed_world.get("analysis", {})
        if observed_analysis.get("robot_environment_collision") is not None:
            nominal = observed_analysis["robot_environment_collision"]
            add(
                "temporal_observation_collision",
                "report_nominal_vs_physical_collision_status",
                {"log": log["id"], "query": query["query_id"]},
                "Report the nominal declared-geometry collision status under selected poses separately from the physical-world truth and safety conclusion.",
                "請把 selected poses 下的 nominal declared-geometry collision status，與 physical-world truth 及 safety conclusion 分開回報。",
                {
                    "analysis_status": observed_analysis["status"],
                    "nominal_collision_status": nominal["status"],
                    "physical_world_truth": observed_analysis["physical_world_truth"],
                    "safety_conclusion": observed_analysis["safety_conclusion"],
                },
                evidence_type="selected_observation_poses_and_declared_geometry_analysis",
                evidence_locator="observed_world.analysis",
                evidence_sha256=log["sha256"],
                exact=True,
                difficulty="collision_epistemics",
            )
    child_joint = {joint["child_link"]: (name, joint) for name, joint in model["joints"].items()}
    for target_frame, analysis in sorted(model.get("kinematic_analysis", {}).get("targets", {}).items()):
        jacobian = analysis["geometric_jacobian"]
        attachment = jacobian["target_attachment_link"]
        reversed_links = [attachment]
        reversed_joints: list[str] = []
        cursor = attachment
        while cursor != root_link:
            if cursor not in child_joint:
                raise EvaluationError(f"cannot reconstruct root path for target {target_frame!r}")
            joint_name, joint = child_joint[cursor]
            reversed_joints.append(joint_name)
            cursor = joint["parent_link"]
            reversed_links.append(cursor)
        add(
            "topology",
            "trace_root_to_target_chain",
            {"target_frame": target_frame},
            f"List the ordered link and joint path from {root_link} to the attachment link of target frame {target_frame}.",
            f"列出從 {root_link} 到目標 frame {target_frame} 所附著連桿的有序 link 與 joint 路徑。",
            {"links": list(reversed(reversed_links)), "joints": list(reversed(reversed_joints))},
            evidence_type="kinematic_tree_derivation",
            evidence_locator=f"kinematic_analysis.targets.{target_frame}.geometric_jacobian.target_attachment_link",
            exact=True,
            difficulty="derived",
        )
        target_pose = model["frames"][target_frame]["world_from_frame"]
        add(
            "pose_transform",
            "locate_target_in_root",
            {"target_frame": target_frame, "pose": pose_name},
            f"At pose {pose_name}, what is {root_link}_from_{target_frame}? Return translation in meters and quaternion in xyzw order.",
            f"在姿態 {pose_name} 下，{root_link}_from_{target_frame} 是多少？請回傳公尺平移與 xyzw 四元數。",
            {"translation_xyz_m": target_pose["translation_xyz_m"], "quaternion_xyzw": target_pose["quaternion_xyzw"]},
            evidence_type="forward_kinematics",
            evidence_locator=f"frames.{target_frame}.world_from_frame",
            exact=True,
            difficulty="numeric",
        )
        for column in jacobian["columns"]:
            joint_name = column["joint"]
            add(
                "instantaneous_motion",
                "predict_target_twist_from_joint_rate",
                {"target_frame": target_frame, "joint": joint_name, "pose": pose_name},
                f"At pose {pose_name}, what linear and angular velocity vectors of {target_frame} are produced per unit rate of independent joint {joint_name}? Express components in {jacobian['components_expressed_in_orientation_of_frame']}.",
                f"在姿態 {pose_name} 下，獨立關節 {joint_name} 每單位速度會讓 {target_frame} 產生什麼線速度與角速度向量？分量請用 {jacobian['components_expressed_in_orientation_of_frame']} 表示。",
                {"linear_xyz_per_joint_unit": column["linear_xyz_per_joint_unit"], "angular_xyz_per_joint_unit": column["angular_xyz_per_joint_unit"]},
                evidence_type="analytic_geometric_jacobian",
                evidence_locator=f"kinematic_analysis.targets.{target_frame}.geometric_jacobian.columns.{joint_name}",
                exact=True,
                difficulty="derived_numeric",
            )
        workspace = analysis.get("sampled_workspace")
        if workspace:
            add(
                "workspace_epistemics",
                "report_sampled_workspace_observation",
                {"target_frame": target_frame, "sample_sha256": workspace["sampling"]["sample_sha256"]},
                f"For target {target_frame}, report the observed root-frame target-origin AABB and evaluated count from the deterministic workspace sample.",
                f"請回報目標 {target_frame} 在決定性 workspace 取樣中觀察到的 root-frame 原點 AABB 與實際樣本數。",
                {"observed_aabb": workspace["observed_target_origin_aabb_in_root"], "evaluated_sample_count": workspace["sampling"]["evaluated_sample_count"]},
                evidence_type="deterministic_joint_space_sampling",
                evidence_locator=f"kinematic_analysis.targets.{target_frame}.sampled_workspace",
                exact=False,
                difficulty="sampled_numeric",
            )
            add(
                "workspace_epistemics",
                "reject_complete_reachability_claim",
                {"target_frame": target_frame},
                f"Does the sampled AABB for {target_frame} prove both that every point inside is reachable and that no reachable point exists outside?",
                f"{target_frame} 的取樣 AABB 是否能同時證明內部每個點皆可達，而且外部不存在任何可達點？",
                False,
                evidence_type="workspace_method_contract",
                evidence_locator=f"kinematic_analysis.targets.{target_frame}.sampled_workspace.meaning",
                exact=True,
                difficulty="epistemic",
            )
    for joint_name, joint in sorted(model["joints"].items()):
        if joint["axis_in_root_frame_at_pose"] is None:
            continue
        add(
            "joint_axis",
            "express_joint_axis_in_root",
            {"joint": joint_name, "pose": pose_name},
            f"At pose {pose_name}, what unit axis vector does joint {joint_name} have when expressed in {root_link}?",
            f"在姿態 {pose_name} 下，關節 {joint_name} 用 {root_link} 表示的單位軸向量是什麼？",
            joint["axis_in_root_frame_at_pose"],
            evidence_type="forward_kinematics",
            evidence_locator=f"joints.{joint_name}.axis_in_root_frame_at_pose",
            exact=True,
            difficulty="numeric",
        )
    affected: dict[str, list[str]] = {}
    for fact in facts:
        if fact.get("predicate") == "can_change_pose_of_link" and isinstance(fact.get("subject"), str):
            affected.setdefault(fact["subject"].removeprefix("joint/"), []).append(fact["object"])
    for joint_name, links in sorted(affected.items()):
        add(
            "kinematic_causality",
            "identify_links_affected_by_driver",
            {"joint": joint_name},
            f"Which link frames may change relative to {root_link} when independent driver joint {joint_name} changes and other independent joints stay fixed?",
            f"當獨立驅動關節 {joint_name} 改變且其他獨立關節固定時，哪些 link frame 相對 {root_link} 可能改變？",
            sorted(set(links)),
            evidence_type="kinematic_tree_and_mimic_derivation",
            evidence_locator=f"facts[joint/{joint_name},can_change_pose_of_link]",
            exact=True,
            difficulty="derived",
            list_order="unordered",
        )
    for geometry_frame, geometry in sorted(model["geometry_analysis"].items()):
        if geometry.get("status") != "measured":
            continue
        add(
            "geometry",
            "locate_geometry_bounds",
            {"geometry_frame": geometry_frame, "pose": pose_name},
            f"At pose {pose_name}, what root-frame AABB was measured for geometry frame {geometry_frame}?",
            f"在姿態 {pose_name} 下，幾何 frame {geometry_frame} 量測到的 root-frame AABB 是什麼？",
            geometry["bounds_in_root_frame_at_pose"],
            evidence_type="measured_mesh" if geometry["geometry_type"] == "mesh" else "analytic_urdf_primitive",
            evidence_locator=f"geometry_analysis.{geometry_frame}.bounds_in_root_frame_at_pose",
            exact=True,
            difficulty="numeric",
            evidence_sha256=geometry.get("source", {}).get("sha256"),
        )
    broadphase = model["collision_broadphase"]
    add(
        "collision_epistemics",
        "report_aabb_broadphase",
        {"pose": pose_name},
        f"At pose {pose_name}, is AABB broad-phase complete for declared collision geometry, and how many candidate pairs were reported?",
        f"在姿態 {pose_name} 下，AABB broad-phase 是否涵蓋所有已宣告 collision geometry，並回報了幾組候選 pair？",
        {"complete_for_declared_collision_geometry": broadphase["complete_for_declared_collision_geometry"], "candidate_pair_count": len(broadphase["overlap_pairs"])},
        evidence_type="aabb_broadphase",
        evidence_locator="collision_broadphase",
        exact=True,
        difficulty="derived",
    )
    add(
        "collision_epistemics",
        "reject_triangle_collision_claim",
        {"pose": pose_name},
        "Does an AABB overlap candidate by itself prove triangle-level surface contact?",
        "單憑一組 AABB 重疊候選，是否能證明三角網格表面實際接觸？",
        False,
        evidence_type="aabb_broadphase_method_contract",
        evidence_locator="collision_broadphase.meaning",
        exact=True,
        difficulty="epistemic",
    )
    collision_surface = model.get("collision_surface", {"status": "not_requested"})
    if collision_surface.get("status") != "not_requested":
        add(
            "collision_surface",
            "report_self_collision_status",
            {"pose": pose_name},
            f"At pose {pose_name}, what self-collision status did exact triangle-surface and closed-solid analysis report?",
            f"在姿態 {pose_name} 下，精確三角形表面與封閉實體分析回報的 self-collision 狀態為何？",
            collision_surface["self_collision_status"],
            evidence_type="triangle_surface_and_closed_solid_analysis",
            evidence_locator="collision_surface.self_collision_status",
            exact=True,
            difficulty="derived",
        )
        if collision_surface.get("srdf_policy_provided"):
            add(
                "collision_surface",
                "report_srdf_policy_filtered_self_collision_status",
                {"pose": pose_name},
                f"At pose {pose_name}, what self-collision status remains after applying the SRDF disabled-collision policy, without changing the physical pair results?",
                f"在姿態 {pose_name} 下，套用 SRDF disabled-collision policy、但不改寫實體 pair 結果後，剩餘的 self-collision 狀態為何？",
                collision_surface["srdf_policy_filtered_self_collision_status"],
                evidence_type="triangle_surface_analysis_with_srdf_policy_filter",
                evidence_locator="collision_surface.srdf_policy_filtered_self_collision_status",
                exact=True,
                difficulty="derived",
            )
        add(
            "collision_surface",
            "reject_surface_distance_only_separation_claim",
            {"pose": pose_name},
            "If two closed collision surfaces have positive surface distance, does that fact alone prove their solids do not collide?",
            "若兩個封閉 collision 表面的表面距離為正，單憑這件事是否足以證明兩個實體沒有碰撞？",
            False,
            evidence_type="closed_surface_containment_method_contract",
            evidence_locator="collision_surface.method",
            exact=True,
            difficulty="epistemic",
        )
        for pair in collision_surface["candidate_results"]:
            answer = {
                "status": pair["status"],
                "surface_distance_m": pair.get("surface_distance_m"),
                "containment_detected": bool(pair.get("containment")),
                "disabled_by_srdf": pair["disabled_by_srdf"],
            }
            add(
                "collision_surface",
                "report_candidate_collision_result",
                {"pose": pose_name, "geometry_a": pair["geometry_a"], "geometry_b": pair["geometry_b"]},
                f"At pose {pose_name}, report the verified collision status, triangle-surface distance, containment detection, and SRDF policy flag for {pair['geometry_a']} versus {pair['geometry_b']}.",
                f"在姿態 {pose_name} 下，請回報 {pair['geometry_a']} 與 {pair['geometry_b']} 的驗證碰撞狀態、三角形表面距離、包覆偵測與 SRDF policy 標記。",
                answer,
                evidence_type="triangle_surface_and_closed_solid_analysis",
                evidence_locator=f"collision_surface.candidate_results[{pair['geometry_a']}|{pair['geometry_b']}]",
                exact=True,
                difficulty="derived_numeric",
            )
    invariant_validation = model.get("invariant_validation", {"status": "not_provided"})
    if invariant_validation["status"] != "not_provided":
        add(
            "project_intent",
            "report_invariant_contract_status",
            {"contract_sha256": invariant_validation["contract_source"]["sha256"]},
            "Did every declared project spatial invariant pass, and how many assertions failed?",
            "所有專案空間 invariant 是否都通過？共有幾項 assertion 失敗？",
            {"status": invariant_validation["status"], "failed_count": invariant_validation["failed_count"]},
            evidence_type="project_spatial_invariant_contract_evaluation",
            evidence_locator="invariant_validation",
            exact=True,
            difficulty="derived",
            evidence_sha256=invariant_validation["contract_source"]["sha256"],
        )
        for invariant in invariant_validation["results"]:
            add(
                "project_intent",
                "report_spatial_invariant_result",
                {"invariant_id": invariant["id"], "contract_sha256": invariant_validation["contract_source"]["sha256"]},
                f"For project spatial invariant {invariant['id']}, report its assertion type, evaluated pose, and pass/fail status.",
                f"請回報專案空間 invariant {invariant['id']} 的 assertion 類型、評估姿態與通過／失敗狀態。",
                {"type": invariant["type"], "pose": invariant["pose"], "status": invariant["status"]},
                evidence_type="project_spatial_invariant_contract_evaluation",
                evidence_locator=f"invariant_validation.results.{invariant['id']}",
                exact=True,
                difficulty="direct",
                evidence_sha256=invariant_validation["contract_source"]["sha256"],
            )
    semantics = model["semantics"]
    if semantics["status"] != "not_provided":
        semantic_sha = semantics["source"]["sha256"]
        for frame_name, annotation in sorted(semantics["frames"].items()):
            add(
                "semantic_grounding",
                "identify_asserted_frame_roles",
                {"frame": frame_name, "semantic_sha256": semantic_sha},
                f"Which project-asserted semantic roles are assigned to frame {frame_name}?",
                f"專案明確宣告 frame {frame_name} 具有哪些語意角色？",
                annotation["roles"],
                evidence_type="user_or_project_semantic_assertion",
                evidence_locator=f"semantics.frames.{frame_name}.roles",
                exact=False,
                list_order="unordered",
                evidence_sha256=semantic_sha,
            )
        for end_effector_name, end_effector in sorted(semantics["end_effectors"].items()):
            add(
                "semantic_grounding",
                "identify_end_effector_frames",
                {"end_effector": end_effector_name, "semantic_sha256": semantic_sha},
                f"What mount and TCP frames are asserted for end effector {end_effector_name}?",
                f"末端執行器 {end_effector_name} 明確宣告的 mount frame 與 TCP frame 是什麼？",
                end_effector,
                evidence_type="user_or_project_semantic_assertion",
                evidence_locator=f"semantics.end_effectors.{end_effector_name}",
                exact=False,
                evidence_sha256=semantic_sha,
            )
            add(
                "pose_transform",
                "locate_tcp_relative_to_mount",
                {"end_effector": end_effector_name, "pose": pose_name},
                f"At pose {pose_name}, what is {end_effector['mount_frame']}_from_{end_effector['tcp_frame']}? Return translation in meters and quaternion xyzw.",
                f"在姿態 {pose_name} 下，{end_effector['mount_frame']}_from_{end_effector['tcp_frame']} 是多少？請回傳公尺平移與 xyzw 四元數。",
                relative_pose(model, end_effector["mount_frame"], end_effector["tcp_frame"]),
                evidence_type="forward_kinematics",
                evidence_locator=f"relative(frames.{end_effector['mount_frame']},frames.{end_effector['tcp_frame']})",
                exact=True,
                difficulty="derived_numeric",
            )
    srdf = model["srdf"]
    if srdf["status"] != "not_provided":
        srdf_sha = srdf["source"]["sha256"]
        for group_name, group in sorted(srdf["groups"].items()):
            add(
                "semantic_grounding",
                "identify_srdf_group_membership",
                {"group": group_name, "srdf_sha256": srdf_sha},
                f"Which ordered joints and links are expanded for SRDF group {group_name}?",
                f"SRDF 群組 {group_name} 展開後包含哪些有序 joints 與 links？",
                {"expanded_joints": group["expanded_joints"], "expanded_links": group["expanded_links"]},
                evidence_type="srdf_declared_and_validated",
                evidence_locator=f"srdf.groups.{group_name}",
                exact=True,
                evidence_sha256=srdf_sha,
            )
        for pose_key, pose_record in sorted(srdf["named_poses"].items()):
            add(
                "semantic_grounding",
                "identify_srdf_named_pose",
                {"named_pose": pose_key, "srdf_sha256": srdf_sha},
                f"What joint assignments are declared for SRDF named pose {pose_key}?",
                f"SRDF named pose {pose_key} 宣告了哪些關節值？",
                pose_record["joints"],
                evidence_type="srdf_declared_and_validated",
                evidence_locator=f"srdf.named_poses.{pose_key}.joints",
                exact=True,
                evidence_sha256=srdf_sha,
            )
        for end_effector_name, end_effector in sorted(srdf["end_effectors"].items()):
            add(
                "semantic_grounding",
                "identify_srdf_end_effector",
                {"end_effector": end_effector_name, "srdf_sha256": srdf_sha},
                f"What parent link, component group, and parent group are declared for SRDF end effector {end_effector_name}?",
                f"SRDF 末端執行器 {end_effector_name} 宣告的 parent link、component group 與 parent group 是什麼？",
                end_effector,
                evidence_type="srdf_declared_and_validated",
                evidence_locator=f"srdf.end_effectors.{end_effector_name}",
                exact=True,
                evidence_sha256=srdf_sha,
            )
    functional_artifact = model.get("artifacts", {}).get("functional_model")
    if functional_model is not None:
        if not isinstance(functional_artifact, dict):
            raise EvaluationError("functional model was supplied but model.json has no functional_model artifact binding")
        if functional_model.get("schema_version") != "robot-spatial-functional-model.v1":
            raise EvaluationError("functional model uses an unsupported schema")
        if functional_model.get("functional_model_id") != functional_artifact.get("functional_model_id"):
            raise EvaluationError("functional model ID does not match model.json")
        if functional_model.get("functional_model_sha256") != functional_artifact.get("functional_model_sha256"):
            raise EvaluationError("functional model semantic digest does not match model.json")
        functional_sha256 = functional_artifact["sha256"]
        projections = functional_model["projections"]
        if projections["components"]:
            component = projections["components"][0]
            component_id = component["component_id"]
            functions = [item for item in projections["functions"] if component_id in item["provided_by"]]
            capabilities = [item for item in projections["capabilities"] if component_id in item["provided_by"]]
            affordances = [item for item in projections["affordances"] if component_id in item["offered_by"]]
            add(
                "function_affordance_understanding",
                "explain_explicit_component_function_without_name_inference",
                {"functional_model_id": functional_model["functional_model_id"], "component": component_id},
                f"Using only the explicit functional model, report the members and meaning of {component_id}, plus its declared function, capability, and affordance IDs. Did this answer infer purpose from URDF names or geometry?",
                f"只能使用明確的 functional model：請回報 {component_id} 的 members 與 meaning，以及它宣告的 function、capability、affordance IDs。此答案是否從 URDF 名稱或幾何推測用途？",
                {
                    "component_id": component_id,
                    "members": component["members"],
                    "meaning": component["meaning"],
                    "function_ids": [item["function_id"] for item in functions],
                    "capability_ids": [item["capability_id"] for item in capabilities],
                    "affordance_ids": [item["affordance_id"] for item in affordances],
                    "name_or_geometry_inference_used": False,
                },
                evidence_type="project_asserted_function_knowledge",
                evidence_locator=f"functional-model.json#/projections/components/{component_id}",
                evidence_sha256=functional_sha256,
                exact=False,
                difficulty="functional_semantic_grounding",
                submission_answer_contract={
                    "component_id": "<string>",
                    "members": ["<string>"],
                    "meaning": "<string>",
                    "function_ids": ["<string>"],
                    "capability_ids": ["<string>"],
                    "affordance_ids": ["<string>"],
                    "name_or_geometry_inference_used": "<boolean>",
                },
            )
        if projections["capabilities"]:
            capability = max(
                projections["capabilities"],
                key=lambda item: (len(item["requirements"]), item["capability_id"]),
            )
            add(
                "function_affordance_understanding",
                "separate_capability_declaration_structural_grounding_and_physical_truth",
                {"functional_model_id": functional_model["functional_model_id"], "capability": capability["capability_id"]},
                f"For {capability['capability_id']}, report providers, realized functions, every typed enabling requirement with status/modality/closure basis, overall grounding status, limitations, and whether physical capability is verified.",
                f"對 {capability['capability_id']}，請回報 providers、realized functions、每個 typed enabling requirement 的 status/modality/closure basis、整體 grounding status、limitations，以及是否已驗證 physical capability。",
                {
                    "capability_id": capability["capability_id"],
                    "provided_by": capability["provided_by"],
                    "realizes_functions": capability["realizes_functions"],
                    "requirements": [
                        {
                            "requirement_id": requirement["requirement_id"],
                            "type": requirement["type"],
                            "status": requirement["status"],
                            "satisfied": requirement["satisfied"],
                            "modality": requirement["evidence"]["modality"],
                            "closure_basis": requirement["evidence"]["closure_basis"],
                            "concept_clause_ids": requirement["evidence"]["concept_clause_ids"],
                        }
                        for requirement in capability["requirements"]
                    ],
                    "grounding_status": capability["grounding_status"],
                    "limitations": capability["limitations"],
                    "physical_capability_verified": capability["physical_capability_verified"],
                },
                evidence_type="project_capability_assertion_plus_deterministic_structural_requirement_evaluation",
                evidence_locator=f"functional-model.json#/projections/capabilities/{capability['capability_id']}",
                evidence_sha256=functional_sha256,
                exact=False,
                difficulty="modal_proof_composition",
                submission_answer_contract={
                    "capability_id": "<string>",
                    "provided_by": ["<string>"],
                    "realizes_functions": ["<string>"],
                    "requirements": [{
                        "requirement_id": "<string>",
                        "type": "<string>",
                        "status": "<string>",
                        "satisfied": "<boolean>",
                        "modality": "<string>",
                        "closure_basis": "<string>",
                        "concept_clause_ids": ["<string>"],
                    }],
                    "grounding_status": "<string>",
                    "limitations": ["<string>"],
                    "physical_capability_verified": "<boolean>",
                },
            )
        if projections["affordances"]:
            affordance = projections["affordances"][0]
            add(
                "function_affordance_understanding",
                "explain_relational_affordance_preconditions_and_intended_effects",
                {"functional_model_id": functional_model["functional_model_id"], "affordance": affordance["affordance_id"]},
                f"For {affordance['affordance_id']}, report the actor/provider, action, target object types, capabilities, named preconditions with truth sources, intended effects, and the current-precondition/physical-executability status.",
                f"對 {affordance['affordance_id']}，請回報 actor/provider、action、target object types、capabilities、具 truth source 的 named preconditions、intended effects，以及 current-precondition/physical-executability status。",
                {
                    "affordance_id": affordance["affordance_id"],
                    "offered_by": affordance["offered_by"],
                    "action_verb": affordance["action_verb"],
                    "target_object_types": affordance["target_object_types"],
                    "capability_refs": affordance["capability_refs"],
                    "preconditions": [
                        {
                            "condition_id": condition_id,
                            "truth_source": next(
                                condition["truth_source"]
                                for condition in projections["conditions"]
                                if condition["condition_id"] == condition_id
                            ),
                        }
                        for condition_id in affordance["precondition_refs"]
                    ],
                    "intended_effect_refs": affordance["effect_refs"],
                    "effects_are_observed": False,
                    "current_preconditions_satisfied": affordance["current_preconditions_satisfied"],
                    "physical_executability": affordance["physical_executability"],
                },
                evidence_type="project_asserted_relational_affordance",
                evidence_locator=f"functional-model.json#/projections/affordances/{affordance['affordance_id']}",
                evidence_sha256=functional_sha256,
                exact=False,
                difficulty="relational_affordance_semantics",
                submission_answer_contract={
                    "affordance_id": "<string>",
                    "offered_by": ["<string>"],
                    "action_verb": "<string>",
                    "target_object_types": ["<string>"],
                    "capability_refs": ["<string>"],
                    "preconditions": [{
                        "condition_id": "<string>",
                        "truth_source": "<string>",
                    }],
                    "intended_effect_refs": ["<string>"],
                    "effects_are_observed": "<boolean>",
                    "current_preconditions_satisfied": "<string>",
                    "physical_executability": "<string>",
                },
            )
            provider = affordance["offered_by"][0]
            target_type = affordance["target_object_types"][0]
            capability_grounding = {
                capability_id: next(
                    item["grounding_status"]
                    for item in projections["capabilities"]
                    if item["capability_id"] == capability_id
                )
                for capability_id in affordance["capability_refs"]
            }
            structurally_grounded = all(
                status == "all_declared_requirements_grounded"
                for status in capability_grounding.values()
            )
            action_conclusion = (
                "declared_possible_if_preconditions_hold"
                if structurally_grounded
                else "declared_affordance_with_ungrounded_capability_requirements"
            )
            add(
                "function_affordance_understanding",
                "apply_declared_action_contract_without_claiming_execution",
                {
                    "functional_model_id": functional_model["functional_model_id"],
                    "provider": provider,
                    "action": affordance["action_verb"],
                    "target_object_type": target_type,
                },
                f"For {provider} performing {affordance['action_verb']!r} on {target_type}, report the functional query conclusion, matching and structurally grounded affordances, capability grounding, named preconditions, intended effects, and the current-precondition/physical-executability status.",
                f"對 {provider} 在 {target_type} 執行 {affordance['action_verb']!r}：請回報 functional query conclusion、matching 與 structurally grounded affordances、capability grounding、named preconditions、intended effects，以及 current-precondition/physical-executability status。",
                {
                    "matching_affordances": [affordance["affordance_id"]],
                    "structurally_grounded_matching_affordances": (
                        [affordance["affordance_id"]] if structurally_grounded else []
                    ),
                    "capability_grounding": capability_grounding,
                    "conclusion": action_conclusion,
                    "precondition_refs": affordance["precondition_refs"],
                    "effect_refs": affordance["effect_refs"],
                    "current_preconditions_satisfied": "not_evaluated",
                    "physical_executability": "not_established",
                },
                evidence_type="project_affordance_match_plus_structural_capability_grounding",
                evidence_locator=f"functional-model.json#/projections/affordances/{affordance['affordance_id']}",
                evidence_sha256=functional_sha256,
                exact=False,
                difficulty="action_contract_composition",
                submission_answer_contract={
                    "matching_affordances": ["<string>"],
                    "structurally_grounded_matching_affordances": ["<string>"],
                    "capability_grounding": {"<capability_id>": "<grounding_status>"},
                    "conclusion": "<string>",
                    "precondition_refs": ["<string>"],
                    "effect_refs": ["<string>"],
                    "current_preconditions_satisfied": "<string>",
                    "physical_executability": "<string>",
                },
            )
            declared_actions = {item["action_verb"] for item in projections["affordances"] if provider in item["offered_by"]}
            undeclared_action = "__undeclared_control_action__"
            while undeclared_action in declared_actions:
                undeclared_action += "_x"
            complete = next((
                item for item in projections["inventory_completeness"]
                if item["subject"] == provider and "affordances" in item["inventories"]
            ), None)
            add(
                "function_affordance_understanding",
                "apply_inventory_completeness_without_claiming_physical_impossibility",
                {
                    "functional_model_id": functional_model["functional_model_id"],
                    "provider": provider,
                    "action": undeclared_action,
                    "target_object_type": target_type,
                },
                f"The functional model has no {undeclared_action!r} affordance for {provider} on {target_type}. Report the correct project-inventory conclusion and whether physical impossibility is established.",
                f"Functional model 中沒有 {provider} 對 {target_type} 執行 {undeclared_action!r} 的 affordance。請回報正確的 project-inventory conclusion，以及是否已建立 physical impossibility。",
                {
                    "conclusion": (
                        "not_declared_in_complete_project_inventory"
                        if complete is not None
                        else "unknown_not_in_incomplete_inventory"
                    ),
                    "inventory_scope": None if complete is None else complete["scope"],
                    "physical_impossibility": "not_established",
                },
                evidence_type="project_asserted_inventory_completeness_boundary",
                evidence_locator="functional-model.json#/projections/inventory_completeness",
                evidence_sha256=functional_sha256,
                exact=False,
                difficulty="closed_open_world_boundary",
                submission_answer_contract={
                    "conclusion": "<string>",
                    "inventory_scope": "<string-or-null>",
                    "physical_impossibility": "<string>",
                },
            )
    if action_assurance is not None:
        if functional_model is None:
            raise EvaluationError("action assurance requires its bound functional model")
        if action_assurance.get("schema_version") != "robot-spatial-action-assurance.v1":
            raise EvaluationError("action assurance uses an unsupported schema")
        functional_binding = action_assurance.get("functional_model_binding")
        if not isinstance(functional_binding, dict):
            raise EvaluationError("action assurance has no functional model binding")
        if functional_binding.get("functional_model_id") != functional_model.get("functional_model_id"):
            raise EvaluationError("action assurance functional model ID does not match the supplied functional model")
        if functional_binding.get("functional_model_sha256") != functional_model.get("functional_model_sha256"):
            raise EvaluationError("action assurance functional model digest does not match the supplied functional model")
        projections = action_assurance.get("projections")
        if not isinstance(projections, dict):
            raise EvaluationError("action assurance projections are missing")
        action = projections["declared_action"]
        assurance_id = action_assurance["assurance_id"]
        assurance_sha256 = action_assurance["assurance_sha256"]
        preconditions = projections["preconditions"]
        add(
            "action_execution_evidence",
            "evaluate_action_readiness_at_decision_time",
            {"assurance_id": assurance_id, "action_instance_id": action["action_instance_id"]},
            "At the declared decision time, report each precondition's selected evidence result and the bounded readiness conclusion. Do not turn readiness into dispatch authorization, physical executability, or safety proof.",
            "在宣告的 decision time，回報每個 precondition 的證據選取結果與有界的 readiness conclusion。不可把 readiness 當成 dispatch authorization、physical executability 或 safety proof。",
            {
                "action_instance_id": action["action_instance_id"],
                "decision_time_ns": projections["precondition_summary"]["decision_time_ns"],
                "preconditions": [
                    {
                        "condition_id": item["condition_id"],
                        "status": item["status"],
                        "truth": item["truth"],
                        "selected_record_ids": item["selected_record_ids"],
                        "reference_time_ns": item["reference_time_ns"],
                    }
                    for item in preconditions
                ],
                "readiness_conclusion": projections["readiness"]["conclusion"],
                "authorization_to_dispatch": projections["readiness"]["authorization_to_dispatch"],
                "physical_executability": projections["readiness"]["physical_executability"],
                "safety": projections["readiness"]["safety"],
            },
            evidence_type="digest_bound_time_qualified_action_condition_evidence",
            evidence_locator="action-assurance.json#/projections/preconditions",
            evidence_sha256=assurance_sha256,
            exact=False,
            difficulty="time_qualified_evidence_selection",
            list_order="unordered",
            submission_answer_contract={
                "action_instance_id": "<string>",
                "decision_time_ns": "<number>",
                "preconditions": [{
                    "condition_id": "<string>",
                    "status": "<string>",
                    "truth": "<string>",
                    "selected_record_ids": ["<string>"],
                    "reference_time_ns": "<number>",
                }],
                "readiness_conclusion": "<string>",
                "authorization_to_dispatch": "<string>",
                "physical_executability": "<string>",
                "safety": "<string>",
            },
        )
        lifecycle = projections["lifecycle"]
        add(
            "action_execution_evidence",
            "separate_action_server_lifecycle_from_physical_execution",
            {"assurance_id": assurance_id, "lifecycle": lifecycle["status"]},
            "Report the observed action-server lifecycle, its consistency, and the execution-start observation. Does this protocol evidence independently verify physical execution?",
            "回報觀測到的 action-server lifecycle、其一致性與 execution-start observation。此協定證據是否能獨立驗證 physical execution？",
            {
                "status": lifecycle["status"],
                "consistency": lifecycle["consistency"],
                "goal_response": lifecycle["goal_response"],
                "latest_observed_status": lifecycle["latest_observed_status"],
                "terminal_result": lifecycle["terminal_result"],
                "execution_started_observed": lifecycle["execution_started_observed"],
                "execution_started_at_ns": lifecycle["execution_started_at_ns"],
                "issue_codes": [item["code"] for item in lifecycle["issues"]],
                "independent_physical_verification": lifecycle[
                    "action_server_reports_are_independent_physical_verification"
                ],
            },
            evidence_type="observed_action_server_protocol_evidence",
            evidence_locator="action-assurance.json#/projections/lifecycle",
            evidence_sha256=assurance_sha256,
            exact=False,
            difficulty="protocol_physical_boundary",
            submission_answer_contract={
                "status": "<string>",
                "consistency": "<string>",
                "goal_response": "<string>",
                "latest_observed_status": "<string>",
                "terminal_result": "<string>",
                "execution_started_observed": "<boolean>",
                "execution_started_at_ns": "<number-or-null>",
                "issue_codes": ["<string>"],
                "independent_physical_verification": "<boolean>",
            },
        )
        add(
            "action_execution_evidence",
            "separate_effect_observation_from_causal_success",
            {"assurance_id": assurance_id, "effect_count": len(projections["effects"])},
            "For every declared effect, report the selected truth, timing relative to observed execution, whether it counts as post-execution evidence, and whether causation is established.",
            "對每個宣告的 effect，回報選出的 truth、相對於已觀測 execution 的時間關係、是否算作 post-execution evidence，以及是否已建立因果關係。",
            {
                "effects": [
                    {
                        "effect_id": item["effect_id"],
                        "status": item["status"],
                        "truth": item["truth"],
                        "selected_record_ids": item["selected_record_ids"],
                        "temporal_relation_to_execution": item["temporal_relation_to_execution"],
                        "counts_as_post_execution_effect_evidence": item[
                            "counts_as_post_execution_effect_evidence"
                        ],
                        "caused_by_action": item["caused_by_action"],
                    }
                    for item in projections["effects"]
                ],
                "effect_summary_status": projections["effect_summary"]["status"],
                "causal_attribution": projections["effect_summary"]["causal_attribution"],
            },
            evidence_type="time_qualified_declared_effect_observation",
            evidence_locator="action-assurance.json#/projections/effects",
            evidence_sha256=assurance_sha256,
            exact=False,
            difficulty="temporal_causal_boundary",
            submission_answer_contract={
                "effects": [{
                    "effect_id": "<string>",
                    "status": "<string>",
                    "truth": "<string>",
                    "selected_record_ids": ["<string>"],
                    "temporal_relation_to_execution": "<string>",
                    "counts_as_post_execution_effect_evidence": "<boolean>",
                    "caused_by_action": "<string>",
                }],
                "effect_summary_status": "<string>",
                "causal_attribution": "<string>",
            },
        )
        outcome = projections["outcome"]
        add(
            "action_execution_evidence",
            "identify_action_evidence_discrepancies",
            {"assurance_id": assurance_id, "outcome": outcome["conclusion"]},
            "Report the action-evidence outcome and every discrepancy code while preserving the causal, physical-world, and safety boundaries.",
            "回報 action-evidence outcome 與每個 discrepancy code，同時保留 causal、physical-world 與 safety 邊界。",
            {
                "outcome_conclusion": outcome["conclusion"],
                "reported_terminal_result": outcome["reported_terminal_result"],
                "discrepancy_codes": [item["code"] for item in projections["discrepancies"]],
                "causal_success": outcome["causal_success"],
                "physical_world_truth": outcome["physical_world_truth"],
                "safety": outcome["safety"],
            },
            evidence_type="derived_action_evidence_discrepancy_projection",
            evidence_locator="action-assurance.json#/projections/outcome",
            evidence_sha256=assurance_sha256,
            exact=False,
            difficulty="cross_layer_discrepancy_reasoning",
            submission_answer_contract={
                "outcome_conclusion": "<string>",
                "reported_terminal_result": "<string>",
                "discrepancy_codes": ["<string>"],
                "causal_success": "<string>",
                "physical_world_truth": "<string>",
                "safety": "<string>",
            },
        )
        provenance = action_assurance["provenance_contract"]
        evidence_binding = action_assurance["action_evidence_binding"]
        add(
            "action_execution_evidence",
            "apply_action_evidence_provenance_boundary",
            {"assurance_id": assurance_id, "bundle_id": evidence_binding["bundle_id"]},
            "Report what artifact/provenance properties this assurance establishes and what remains outside its epistemic scope.",
            "回報此 assurance 建立了哪些 artifact/provenance 性質，以及哪些事項仍在其 epistemic scope 之外。",
            {
                "assurance_id": assurance_id,
                "bundle_id": evidence_binding["bundle_id"],
                "evidence_source_ids": [item["source_id"] for item in evidence_binding["evidence_sources"]],
                "evidence_sources_are_content_bound_entities": provenance[
                    "evidence_sources_are_content_bound_entities"
                ],
                "producers_are_responsible_agents_not_truth_oracles": provenance[
                    "producers_are_responsible_agents_not_truth_oracles"
                ],
                "goal_result_and_effect_observation_are_distinct": provenance[
                    "goal_result_and_effect_observation_are_distinct"
                ],
                "reported_success_is_physical_or_causal_proof": provenance[
                    "reported_success_is_physical_or_causal_proof"
                ],
                "readiness_is_dispatch_authorization": provenance["readiness_is_dispatch_authorization"],
                "epistemic_scope": action_assurance["epistemic_scope"],
            },
            evidence_type="digest_bound_action_provenance_contract",
            evidence_locator="action-assurance.json#/provenance_contract",
            evidence_sha256=assurance_sha256,
            exact=False,
            difficulty="provenance_epistemic_boundary",
            submission_answer_contract={
                "assurance_id": "<string>",
                "bundle_id": "<string>",
                "evidence_source_ids": ["<string>"],
                "evidence_sources_are_content_bound_entities": "<boolean>",
                "producers_are_responsible_agents_not_truth_oracles": "<boolean>",
                "goal_result_and_effect_observation_are_distinct": "<boolean>",
                "reported_success_is_physical_or_causal_proof": "<boolean>",
                "readiness_is_dispatch_authorization": "<boolean>",
                "epistemic_scope": "<string>",
            },
        )
    questions.sort(key=lambda record: (record["capability"], record["task"], record["question_id"]))
    keys.sort(key=lambda record: (record["capability"], record["task"], record["question_id"]))
    return questions, keys


def generate_evaluation(
    model: dict[str, Any],
    facts: list[dict[str, Any]],
    output_dir: Path,
    key_path: Path | None = None,
    concept_graph: dict[str, Any] | None = None,
    functional_model: dict[str, Any] | None = None,
    action_assurance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    questions, keys = generate_records(model, facts, concept_graph, functional_model, action_assurance)
    output_dir.mkdir(parents=True, exist_ok=True)
    question_path = output_dir / "questions.jsonl"
    key_path = key_path or output_dir.parent / f"{output_dir.name}-private" / "answer-key.jsonl"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / "answer-template.jsonl"
    manifest_path = output_dir / "manifest.json"
    question_path.write_text(jsonl_dump(questions), encoding="utf-8")
    key_path.write_text(jsonl_dump(keys), encoding="utf-8")
    template_path.write_text(jsonl_dump({"question_id": record["question_id"], "answer": None} for record in questions), encoding="utf-8")
    capability_counts: dict[str, int] = {}
    for question in questions:
        capability_counts[question["capability"]] = capability_counts.get(question["capability"], 0) + 1
    perfect_answers = [{"question_id": key["question_id"], "answer": key["answer"]} for key in keys]
    perfect_control = verify_answers(keys, perfect_answers)
    missing_control = verify_answers(keys, perfect_answers[:-1])
    manifest = {
        "schema_version": "robot-spatial-evaluation-manifest.v1",
        "robot": model["robot"]["name"],
        "pose": model["pose"]["name"],
        "source_urdf_sha256": model["source"]["sha256"],
        "spatial_truth_sha256": spatial_truth_sha256(model),
        "question_count": len(questions),
        "capability_counts": dict(sorted(capability_counts.items())),
        "grader_self_check": {
            "perfect_answers_control": {"status": perfect_control["status"], "accuracy": perfect_control["accuracy"]},
            "one_missing_answer_control": {"status": missing_control["status"], "accuracy": missing_control["accuracy"]},
        },
        "artifacts": {
            "questions": question_path.name,
            "answer_template": template_path.name,
        },
        "isolation_requirement": "The private answer key must be stored outside every filesystem and context surface available to the candidate. File separation alone is not a security boundary.",
        "evaluation_protocol": [
            "Move or generate the private answer key outside the candidate-readable workspace.",
            "Provide only the robot sources, skill, public manifest, questions.jsonl, and answer-template.jsonl to the candidate.",
            "Require one JSONL answer record per question using answer-template.jsonl shape.",
            "Grade with spatial_evaluation.py verify and retain the report as evidence.",
        ],
    }
    manifest_path.write_text(json_dump(manifest), encoding="utf-8")
    return {
        **manifest,
        "artifacts": {
            "questions": str(question_path.resolve()),
            "answer_template": str(template_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
        "private_artifacts": {"answer_key": str(key_path.resolve())},
    }


def compare_values(expected: Any, actual: Any, comparison: dict[str, Any], path: str = "answer") -> str | None:
    if isinstance(expected, bool):
        return None if isinstance(actual, bool) and actual == expected else f"{path}: expected boolean {expected!r}, got {actual!r}"
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isfinite(float(actual)):
            return f"{path}: expected finite number {expected!r}, got {actual!r}"
        absolute = float(comparison.get("absolute_tolerance", 1e-9))
        relative = float(comparison.get("relative_tolerance", 1e-9))
        if not math.isclose(float(expected), float(actual), rel_tol=relative, abs_tol=absolute):
            return f"{path}: expected {expected!r}, got {actual!r} (abs tolerance {absolute})"
        return None
    if isinstance(expected, str) or expected is None:
        return None if actual == expected else f"{path}: expected {expected!r}, got {actual!r}"
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return f"{path}: expected array, got {type(actual).__name__}"
        if comparison.get("list_order") == "unordered":
            expected_canonical = sorted(json.dumps(value, sort_keys=True, ensure_ascii=False) for value in expected)
            actual_canonical = sorted(json.dumps(value, sort_keys=True, ensure_ascii=False) for value in actual)
            return None if expected_canonical == actual_canonical else f"{path}: unordered array mismatch; expected {expected!r}, got {actual!r}"
        if path.endswith("quaternion_xyzw") and len(expected) == len(actual) == 4 and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in [*expected, *actual]):
            absolute = float(comparison.get("absolute_tolerance", 1e-6))
            relative = float(comparison.get("relative_tolerance", 1e-9))
            direct = all(math.isclose(float(left), float(right), rel_tol=relative, abs_tol=absolute) for left, right in zip(expected, actual))
            negated = all(math.isclose(float(left), -float(right), rel_tol=relative, abs_tol=absolute) for left, right in zip(expected, actual))
            return None if direct or negated else f"{path}: quaternion mismatch up to sign; expected {expected!r}, got {actual!r}"
        if len(expected) != len(actual):
            return f"{path}: expected array length {len(expected)}, got {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = compare_values(expected_item, actual_item, comparison, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{path}: expected object, got {type(actual).__name__}"
        if set(expected) != set(actual):
            return f"{path}: object keys mismatch; expected {sorted(expected)}, got {sorted(actual)}"
        for key in sorted(expected):
            difference = compare_values(expected[key], actual[key], comparison, f"{path}.{key}")
            if difference:
                return difference
        return None
    return None if expected == actual else f"{path}: expected {expected!r}, got {actual!r}"


def verify_answers(keys: list[dict[str, Any]], answers: list[dict[str, Any]], pass_threshold: float = 1.0) -> dict[str, Any]:
    if not 0.0 <= pass_threshold <= 1.0:
        raise EvaluationError("pass threshold must be between 0 and 1")
    key_by_id: dict[str, dict[str, Any]] = {}
    for key in keys:
        question_id = key.get("question_id")
        if not isinstance(question_id, str) or question_id in key_by_id:
            raise EvaluationError(f"answer key has missing or duplicate question_id: {question_id!r}")
        key_by_id[question_id] = key
    answer_by_id: dict[str, Any] = {}
    duplicates: list[str] = []
    malformed: list[dict[str, Any]] = []
    for index, record in enumerate(answers, 1):
        question_id = record.get("question_id")
        if not isinstance(question_id, str) or "answer" not in record:
            malformed.append({"record_number": index, "reason": "record must contain string question_id and answer"})
            continue
        if question_id in answer_by_id:
            duplicates.append(question_id)
            continue
        answer_by_id[question_id] = record["answer"]
    failures: list[dict[str, Any]] = []
    per_capability: dict[str, dict[str, int | float]] = {}
    correct = 0
    for question_id, key in key_by_id.items():
        capability = key["capability"]
        bucket = per_capability.setdefault(capability, {"total": 0, "correct": 0, "accuracy": 0.0})
        bucket["total"] = int(bucket["total"]) + 1
        if question_id not in answer_by_id:
            failures.append({"question_id": question_id, "capability": capability, "reason": "missing answer"})
            continue
        difference = compare_values(key["answer"], answer_by_id[question_id], key.get("comparison", {}))
        if difference:
            failures.append({
                "question_id": question_id,
                "capability": capability,
                "reason": difference,
                "expected": key["answer"],
                "actual": answer_by_id[question_id],
            })
            continue
        correct += 1
        bucket["correct"] = int(bucket["correct"]) + 1
    total = len(key_by_id)
    accuracy = correct / total if total else 0.0
    for bucket in per_capability.values():
        bucket["accuracy"] = int(bucket["correct"]) / int(bucket["total"])
    unexpected = sorted(set(answer_by_id) - set(key_by_id))
    passed = accuracy >= pass_threshold and not duplicates and not malformed and not unexpected
    return {
        "schema_version": "robot-spatial-evaluation-report.v1",
        "status": "passed" if passed else "failed",
        "pass_threshold": pass_threshold,
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": round(accuracy, 12),
        "per_capability": dict(sorted(per_capability.items())),
        "failures": failures,
        "duplicate_question_ids": sorted(set(duplicates)),
        "unexpected_question_ids": unexpected,
        "malformed_records": malformed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate blind questions, isolated answer key, and template")
    generate.add_argument("model", type=Path)
    generate.add_argument("facts", type=Path)
    generate.add_argument("--concept", type=Path, help="optional bound robot-spatial-concept-graph.v1 artifact")
    generate.add_argument("--functional", type=Path, help="optional bound robot-spatial-functional-model.v1 artifact")
    generate.add_argument("--action-assurance", type=Path, help="optional bound robot-spatial-action-assurance.v1 artifact")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--key-out", type=Path, help="private answer-key path outside the candidate-readable workspace")
    verify = subparsers.add_parser("verify", help="grade candidate JSONL answers against an isolated key")
    verify.add_argument("key", type=Path)
    verify.add_argument("answers", type=Path)
    verify.add_argument("--pass-threshold", type=float, default=1.0)
    verify.add_argument("--report", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == "generate":
        model = read_json(args.model)
        facts = read_jsonl(args.facts)
        concept_graph = read_json(args.concept) if args.concept is not None else None
        functional_model = read_json(args.functional) if args.functional is not None else None
        action_assurance = read_json(args.action_assurance) if args.action_assurance is not None else None
        print(json_dump(generate_evaluation(
            model,
            facts,
            args.out,
            args.key_out,
            concept_graph,
            functional_model,
            action_assurance,
        )), end="")
        return 0
    keys = read_jsonl(args.key)
    answers = read_jsonl(args.answers)
    report = verify_answers(keys, answers, args.pass_threshold)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json_dump(report), encoding="utf-8")
    print(json_dump(report), end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except EvaluationError as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)
