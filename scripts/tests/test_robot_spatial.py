#!/usr/bin/env python3

import importlib.util
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "robot_spatial.py"
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("robot_spatial", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
import mesh_geometry
import spatial_edit_evaluation
import spatial_edit_suite
import spatial_evaluation
import spatial_evaluation_suite
import spatial_graph_edit
import spatial_invariants
import triangle_geometry


class RobotSpatialTests(unittest.TestCase):
    def setUp(self):
        self.model = MODULE.RobotModel(FIXTURES / "two_dof.urdf")
        self.pose = {"shoulder": math.pi / 2.0, "slide": 0.5}

    def assertVectorAlmostEqual(self, actual, expected, places=9):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=places)

    def test_triangle_closest_point_and_surface_distance(self):
        triangle = ([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0])
        self.assertVectorAlmostEqual(
            triangle_geometry.closest_point_on_triangle([0.5, 0.5, 3.0], triangle),
            [0.5, 0.5, 0.0],
        )
        parallel = ([0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [0.0, 2.0, 2.0])
        distance, left, right = triangle_geometry.triangle_triangle_distance(triangle, parallel)
        self.assertAlmostEqual(distance, 2.0)
        self.assertAlmostEqual(left[2], 0.0)
        self.assertAlmostEqual(right[2], 2.0)

    def test_triangle_intersections_include_piercing_and_coplanar_crossing(self):
        base = ([-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0])
        piercing = ([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.5, 0.0, 0.0])
        coplanar = ([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, -2.0, 0.0])
        self.assertAlmostEqual(triangle_geometry.triangle_triangle_distance(base, piercing)[0], 0.0)
        self.assertAlmostEqual(triangle_geometry.triangle_triangle_distance(base, coplanar)[0], 0.0)

    def test_bvh_surface_distance_matches_brute_force(self):
        left = [
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [4.0, 1.0, 0.0]),
        ]
        right = [
            ([0.0, 0.0, 3.0], [1.0, 0.0, 3.0], [0.0, 1.0, 3.0]),
            ([4.0, 0.0, 0.25], [5.0, 0.0, 0.25], [4.0, 1.0, 0.25]),
        ]
        brute_force = min(
            triangle_geometry.triangle_triangle_distance(left_triangle, right_triangle)[0]
            for left_triangle in left
            for right_triangle in right
        )
        result = triangle_geometry.bvh_surface_distance(left, right, leaf_size=1)
        self.assertAlmostEqual(result["distance_m"], brute_force)
        self.assertEqual(result["distance_m"], 0.25)
        self.assertGreater(result["node_pairs_visited"], 0)
        self.assertLessEqual(result["triangle_pairs_tested"], len(left) * len(right))

    def test_box_surface_distance_touching_overlap_and_containment(self):
        unit = triangle_geometry.box_surface([2.0, 2.0, 2.0], MODULE.identity())
        separated_transform = MODULE.translation([3.0, 0.0, 0.0])
        touching_transform = MODULE.translation([2.0, 0.0, 0.0])
        overlap_transform = MODULE.translation([1.0, 0.0, 0.0])
        separated = triangle_geometry.box_surface([2.0, 2.0, 2.0], separated_transform)
        touching = triangle_geometry.box_surface([2.0, 2.0, 2.0], touching_transform)
        overlapping = triangle_geometry.box_surface([2.0, 2.0, 2.0], overlap_transform)
        self.assertAlmostEqual(triangle_geometry.bvh_surface_distance(unit, separated)["distance_m"], 1.0)
        self.assertAlmostEqual(triangle_geometry.bvh_surface_distance(unit, touching)["distance_m"], 0.0)
        self.assertAlmostEqual(triangle_geometry.bvh_surface_distance(unit, overlapping)["distance_m"], 0.0)

        outer = triangle_geometry.box_surface([4.0, 4.0, 4.0], MODULE.identity())
        inner = triangle_geometry.box_surface([1.0, 1.0, 1.0], MODULE.identity())
        self.assertGreater(triangle_geometry.bvh_surface_distance(outer, inner)["distance_m"], 0.0)
        self.assertTrue(triangle_geometry.point_inside_closed_surface(inner[0][0], outer))
        self.assertFalse(triangle_geometry.point_inside_closed_surface(outer[0][0], inner))

    def test_tree_and_root(self):
        self.assertEqual(self.model.root_link, "base_link")

    def test_xacro_detection_ignores_comments_but_rejects_real_namespace_elements(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            commented = Path(temp_dir) / "expanded.urdf"
            commented.write_text(
                '<robot name="expanded"><!-- <xacro:include filename="source.xacro"/> -->'
                '<link name="base"/></robot>'
            )
            self.assertEqual(MODULE.RobotModel(commented).root_link, "base")

            unexpanded = Path(temp_dir) / "unexpanded.urdf"
            unexpanded.write_text(
                '<robot name="source" xmlns:xacro="http://ros.org/wiki/xacro">'
                '<xacro:property name="prefix" value=""/><link name="base"/></robot>'
            )
            with self.assertRaisesRegex(MODULE.SpatialError, "unexpanded Xacro elements"):
                MODULE.RobotModel(unexpanded)
        self.assertEqual(len(self.model.links), 4)
        self.assertEqual(len(self.model.joints), 3)

    def test_chain_reports_ordered_structural_path(self):
        chain = self.model.chain("base_link", "tool0")
        self.assertEqual(chain["links"], ["base_link", "arm_link", "slider_link", "tool0"])
        self.assertEqual([step["joint"] for step in chain["steps"]], ["shoulder", "slide", "tool_mount"])
        self.assertEqual(chain["movable_joints"], ["shoulder", "slide"])

    def test_joint_effect_excludes_own_pre_motion_frame(self):
        affected = self.model.affected_by_joint("shoulder")
        self.assertEqual(affected["affected_links"], ["arm_link", "slider_link", "tool0"])
        self.assertNotIn("joint/shoulder", affected["affected_frames"])
        self.assertIn("joint/slide", affected["affected_frames"])
        self.assertIn("collision/slider_link/0", affected["affected_frames"])

    def test_forward_kinematics(self):
        transform = self.model.transform("base_link", "tool0", self.pose)
        self.assertVectorAlmostEqual([transform[index][3] for index in range(3)], [1.0, 1.5, 0.2])

    def test_transform_direction_is_explicit(self):
        transform = self.model.transform("arm_link", "tool0", self.pose)
        self.assertVectorAlmostEqual([transform[index][3] for index in range(3)], [1.5, 0.0, 0.2])

    def test_declared_mass_properties_aggregate_pose_and_inertia_with_parallel_axis_theorem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mass_tree.urdf"
            path.write_text('''<robot name="mass_tree">
              <link name="base"><inertial><mass value="2"/><inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.4"/></inertial></link>
              <link name="arm"><inertial><origin xyz="1 0 0" rpy="0 0 0"/><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.3"/></inertial></link>
              <link name="tool"/>
              <joint name="yaw" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/><limit lower="-3.2" upper="3.2" effort="1" velocity="1"/></joint>
              <joint name="mount" type="fixed"><parent link="arm"/><child link="tool"/></joint>
            </robot>''')
            model = MODULE.RobotModel(path)
            result = model.mass_properties({"yaw": math.pi / 2}, "base")
            self.assertEqual(result["status"], "computed")
            self.assertEqual(result["declared_mass_kg"], 3.0)
            self.assertVectorAlmostEqual(result["center_of_mass_in_expressed_frame_m"], [0.0, 1.0 / 3.0, 0.0])
            tensor = result["inertia_about_center_of_mass_in_expressed_frame_kg_m2"]["matrix_3x3_rowmajor"]
            self.assertVectorAlmostEqual([tensor[0][0], tensor[1][1], tensor[2][2]], [16.0 / 15.0, 0.4, 41.0 / 30.0])
            self.assertEqual(result["coverage"]["missing_inertial_links"], ["tool"])
            self.assertFalse(result["coverage"]["all_selected_links_declare_valid_inertial"])
            self.assertEqual(result["coverage"]["physical_world_completeness"], "not_established")

            subtree = model.mass_properties({"yaw": math.pi / 2}, "base", "arm")
            self.assertEqual(subtree["declared_mass_kg"], 1.0)
            self.assertVectorAlmostEqual(subtree["center_of_mass_in_expressed_frame_m"], [0.0, 1.0, 0.0])
            subtree_tensor = subtree["inertia_about_center_of_mass_in_expressed_frame_kg_m2"]["matrix_3x3_rowmajor"]
            self.assertVectorAlmostEqual([subtree_tensor[0][0], subtree_tensor[1][1], subtree_tensor[2][2]], [0.2, 0.1, 0.3])

            pose_path = Path(temp_dir) / "pose.json"
            pose_path.write_text(json.dumps({"pose_name": "quarter_turn", "joints": {"yaw": math.pi / 2}}))
            query = subprocess.run(
                [sys.executable, str(SCRIPT), "mass-properties", str(path), "--pose", str(pose_path), "--frame", "base", "--subtree-root", "arm"],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(query.stdout)
            self.assertEqual(payload["schema_version"], "robot-spatial-mass-properties.v1")
            self.assertEqual(payload["pose"]["name"], "quarter_turn")
            self.assertEqual(payload["query_evidence"]["method"], "urdf_declared_inertials_forward_kinematics_parallel_axis_theorem")
            self.assertTrue(payload["query_evidence"]["query_id"].startswith("query-"))

    def test_invalid_or_incomplete_inertial_declarations_never_produce_aggregate_truth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = Path(temp_dir) / "invalid_inertia.urdf"
            invalid_path.write_text('''<robot name="invalid"><link name="base"><inertial><mass value="1"/><inertia ixx="-1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link></robot>''')
            invalid = MODULE.RobotModel(invalid_path)
            result = invalid.mass_properties({})
            self.assertEqual(result["status"], "indeterminate")
            self.assertIsNone(result["declared_mass_kg"])
            self.assertIn("invalid", invalid.warnings()[0])

            incomplete_path = Path(temp_dir) / "incomplete_inertia.urdf"
            incomplete_path.write_text('''<robot name="incomplete"><link name="base"><inertial><mass value="1"/></inertial></link></robot>''')
            incomplete = MODULE.RobotModel(incomplete_path)
            result = incomplete.mass_properties({})
            self.assertEqual(result["status"], "indeterminate")
            self.assertIsNone(result["center_of_mass_in_expressed_frame_m"])
            self.assertIn("incomplete", incomplete.warnings()[0])

    def test_static_gravity_loads_match_analytic_revolute_and_prismatic_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gravity.urdf"
            path.write_text('''<robot name="gravity_cases">
              <link name="base"/>
              <link name="pendulum"><inertial><origin xyz="1 0 0"/><mass value="2"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
              <link name="slider"><inertial><mass value="3"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
              <joint name="hinge" type="revolute"><parent link="base"/><child link="pendulum"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="15" velocity="1"/></joint>
              <joint name="lift" type="prismatic"><parent link="base"/><child link="slider"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="25" velocity="1"/></joint>
            </robot>''')
            model = MODULE.RobotModel(path)
            result = model.static_gravity_loads({}, [0.0, -10.0, 0.0], "base")
            self.assertEqual(result["status"], "computed")
            self.assertEqual(result["gravity"]["vector_in_root_frame_xyz_m_s2"], [0.0, -10.0, 0.0])
            self.assertAlmostEqual(result["independent_driver_loads"]["hinge"]["generalized_gravity_force"], -20.0)
            self.assertAlmostEqual(result["independent_driver_loads"]["hinge"]["ideal_static_holding_effort"], 20.0)
            self.assertFalse(result["independent_driver_loads"]["hinge"]["modeled_load_within_declared_joint_effort_limit_magnitude"])
            self.assertAlmostEqual(result["independent_driver_loads"]["lift"]["generalized_gravity_force"], -30.0)
            self.assertAlmostEqual(result["independent_driver_loads"]["lift"]["ideal_static_holding_effort"], 30.0)
            self.assertFalse(result["independent_driver_loads"]["lift"]["modeled_load_within_declared_joint_effort_limit_magnitude"])
            self.assertEqual(result["coverage"]["missing_inertial_links"], ["base"])

            pose_path = Path(temp_dir) / "pose.json"
            pose_path.write_text(json.dumps({"pose_name": "analytic", "joints": {}}))
            query = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "gravity-loads",
                    str(path),
                    "--pose",
                    str(pose_path),
                    "--gravity-frame",
                    "base",
                    "--gravity",
                    "0",
                    "-10",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(query.stdout)
            self.assertEqual(payload["pose"]["name"], "analytic")
            self.assertEqual(
                payload["query_evidence"]["method"],
                "urdf_declared_inertials_forward_kinematics_gravity_projection_with_mimic_chain_rule",
            )

    def test_static_gravity_loads_apply_mimic_chain_rule_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mimic_gravity.urdf"
            path.write_text('''<robot name="mimic_gravity">
              <link name="base"/>
              <link name="left"><inertial><origin xyz="1 0 0"/><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
              <link name="right"><inertial><origin xyz="1 0 0"/><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
              <joint name="driver" type="revolute"><parent link="base"/><child link="left"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="100" velocity="1"/></joint>
              <joint name="follower" type="revolute"><parent link="base"/><child link="right"/><axis xyz="0 0 1"/><mimic joint="driver" multiplier="-2" offset="0"/><limit lower="-2" upper="2" effort="100" velocity="1"/></joint>
            </robot>''')
            model = MODULE.RobotModel(path)
            result = model.static_gravity_loads({}, [0.0, -10.0, 0.0])
            self.assertEqual(result["independent_driver_order"], ["driver"])
            self.assertAlmostEqual(result["independent_driver_loads"]["driver"]["generalized_gravity_force"], 10.0)
            contributions = result["independent_driver_loads"]["driver"]["physical_contributions"]
            follower = next(record for record in contributions if record["physical_joint"] == "follower")
            self.assertEqual(follower["derivative_of_physical_joint_from_driver"], -2.0)
            self.assertEqual(follower["contribution_to_independent_driver"], 20.0)

            invalid = Path(temp_dir) / "invalid.urdf"
            invalid.write_text(path.read_text().replace('ixx="0.1"', 'ixx="-0.1"', 1))
            invalid_result = MODULE.RobotModel(invalid).static_gravity_loads({}, [0.0, -10.0, 0.0])
            self.assertEqual(invalid_result["status"], "indeterminate")
            self.assertIsNone(invalid_result["independent_driver_loads"])

    def test_embedded_actuation_declarations_are_typed_and_do_not_claim_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "controlled.urdf"
            path.write_text('''<robot name="controlled">
              <link name="base"/><link name="arm"/>
              <joint name="axis" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="20" velocity="2"/><dynamics damping="0.2" custom_gain="7"/></joint>
              <transmission name="axis_trans"><type>transmission_interface/SimpleTransmission</type><joint name="axis"><hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface></joint><actuator name="axis_motor"><mechanicalReduction>50</mechanicalReduction></actuator></transmission>
              <ros2_control name="DemoSystem" type="system"><hardware><plugin>demo/Hardware</plugin><param name="mode">test</param></hardware><joint name="axis"><command_interface name="position"/><command_interface name="effort"><param name="min">-10</param></command_interface><state_interface name="position"/></joint><sensor name="ft"><state_interface name="force.x"/></sensor></ros2_control>
            </robot>''')
            model = MODULE.RobotModel(path)
            declaration = model.actuation
            self.assertEqual(declaration["coverage"]["ros2_control_system_count"], 1)
            self.assertEqual(declaration["coverage"]["legacy_transmission_count"], 1)
            binding = declaration["joint_bindings"]["axis"]
            self.assertEqual(binding["legacy_transmissions"], ["axis_trans"])
            self.assertEqual(binding["ros2_control"][0]["command_interfaces"], ["position", "effort"])
            self.assertEqual(binding["ros2_control"][0]["state_interfaces"], ["position"])
            self.assertEqual(declaration["legacy_transmissions"]["axis_trans"]["actuators"][0]["mechanical_reduction_declared"], 50.0)
            self.assertEqual(model.joints["axis"].dynamics["standard_urdf"]["damping"], 0.2)
            self.assertEqual(model.joints["axis"].dynamics["uninterpreted_extension_attributes"], {"custom_gain": "7"})
            self.assertIn("not proof", declaration["epistemic_scope"])

            query = json.loads(subprocess.run(
                [sys.executable, str(SCRIPT), "actuation", str(path), "--joint", "axis"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout)
            self.assertEqual(query["selection"], {"type": "joint", "name": "axis"})
            self.assertEqual(
                query["query_evidence"]["method"],
                "expanded_urdf_actuation_declaration_transcription_and_reference_validation",
            )
            canonical = model.canonical({}, "zero", workspace_samples=0)
            facts = MODULE.fact_records(model, canonical)
            predicates = {fact["predicate"] for fact in facts}
            self.assertIn("declares_control_system", predicates)
            self.assertIn("declares_legacy_transmission", predicates)
            self.assertIn("has_joint_dynamics_declaration", predicates)
            context_dir = Path(temp_dir) / "context"
            context_dir.mkdir()
            facts_path = context_dir / "facts.jsonl"
            facts_path.write_text(MODULE.jsonl_dump(facts))
            MODULE.write_agent_context(context_dir, canonical, facts, facts_path)
            manifest = json.loads((context_dir / "agent-context.json").read_text())
            self.assertEqual(manifest["statistics"]["entity_type_counts"]["ros2_control_system"], 1)
            self.assertEqual(manifest["statistics"]["entity_type_counts"]["transmission"], 1)
            self.assertEqual(manifest["statistics"]["entity_type_counts"]["actuator"], 1)
            questions, _ = spatial_evaluation.generate_records(canonical, facts)
            capabilities = {question["capability"] for question in questions}
            self.assertIn("actuation_declarations", capabilities)

    def test_cli_query_evidence_binds_source_method_and_parameters(self):
        command = [
            sys.executable,
            str(SCRIPT),
            "transform",
            str(FIXTURES / "two_dof.urdf"),
            "--pose",
            str(FIXTURES / "bent_pose.json"),
            "--from",
            "base_link",
            "--to",
            "tool0",
        ]
        first = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        second = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        evidence = first["query_evidence"]
        self.assertEqual(evidence["schema_version"], "robot-spatial-query-evidence.v1")
        self.assertEqual(evidence["query_id"], second["query_evidence"]["query_id"])
        self.assertEqual(evidence["source_urdf_sha256"], self.model.sha256)
        self.assertEqual(evidence["method"], "forward_kinematics")
        self.assertEqual(evidence["parameters"]["from_frame"], "base_link")
        zero_pose = list(command)
        del zero_pose[4:6]
        changed = json.loads(subprocess.run(zero_pose, check=True, capture_output=True, text=True).stdout)
        self.assertNotEqual(evidence["query_id"], changed["query_evidence"]["query_id"])

    def test_axes_are_expressed_in_requested_frame(self):
        self.assertVectorAlmostEqual(self.model.axis("shoulder", "base_link", self.pose), [0.0, 0.0, 1.0])
        self.assertVectorAlmostEqual(self.model.axis("slide", "base_link", self.pose), [0.0, 1.0, 0.0])

    def test_analytic_jacobian_matches_expected_structure(self):
        jacobian = self.model.geometric_jacobian("tool0", self.pose)
        self.assertEqual(jacobian["joint_order"], ["shoulder", "slide"])
        self.assertVectorAlmostEqual(jacobian["columns"][0]["linear_xyz_per_joint_unit"], [-1.5, 0.0, 0.0])
        self.assertVectorAlmostEqual(jacobian["columns"][0]["angular_xyz_per_joint_unit"], [0.0, 0.0, 1.0])
        self.assertVectorAlmostEqual(jacobian["columns"][1]["linear_xyz_per_joint_unit"], [0.0, 1.0, 0.0])
        self.assertVectorAlmostEqual(jacobian["columns"][1]["angular_xyz_per_joint_unit"], [0.0, 0.0, 0.0])

    def test_analytic_jacobian_matches_central_finite_difference(self):
        jacobian = self.model.geometric_jacobian("tool0", self.pose)
        step = 1e-6
        nominal = self.model.transform("base_link", "tool0", self.pose)
        nominal_rotation = [row[:3] for row in nominal[:3]]
        for column in jacobian["columns"]:
            joint = column["joint"]
            plus_pose, minus_pose = dict(self.pose), dict(self.pose)
            plus_pose[joint] += step
            minus_pose[joint] -= step
            plus = self.model.transform("base_link", "tool0", plus_pose)
            minus = self.model.transform("base_link", "tool0", minus_pose)
            linear = [(plus[index][3] - minus[index][3]) / (2.0 * step) for index in range(3)]
            rotation_derivative = [[(plus[row][column_index] - minus[row][column_index]) / (2.0 * step) for column_index in range(3)] for row in range(3)]
            omega_matrix = [[sum(rotation_derivative[row][index] * nominal_rotation[column_index][index] for index in range(3)) for column_index in range(3)] for row in range(3)]
            angular = [omega_matrix[2][1], omega_matrix[0][2], omega_matrix[1][0]]
            self.assertVectorAlmostEqual(linear, column["linear_xyz_per_joint_unit"], places=6)
            self.assertVectorAlmostEqual(angular, column["angular_xyz_per_joint_unit"], places=6)

    def test_sampled_workspace_is_deterministic_and_bounds_every_returned_sample(self):
        first = self.model.workspace_envelope("tool0", self.pose, 64, include_samples=True)
        second = self.model.workspace_envelope("tool0", self.pose, 64, include_samples=True)
        self.assertEqual(first, second)
        self.assertTrue(first["approximate"])
        self.assertEqual(first["independent_joint_order"], ["shoulder", "slide"])
        self.assertEqual(first["sampling"]["evaluated_sample_count"], 64)
        self.assertIn("does not prove", first["meaning"])
        bounds = first["observed_target_origin_aabb_in_root"]
        self.assertAlmostEqual(bounds["min_xyz_m"][0], -1.0, places=9)
        self.assertAlmostEqual(bounds["max_xyz_m"][0], 3.0, places=9)
        self.assertAlmostEqual(bounds["min_xyz_m"][2], 0.2, places=9)
        self.assertAlmostEqual(bounds["max_xyz_m"][2], 0.2, places=9)
        for sample in first["samples"]:
            joints = sample["independent_joint_positions"]
            x, y, z = sample["target_origin_in_root_xyz_m"]
            expected_radius = 1.0 + joints["slide"]
            self.assertAlmostEqual((x - 1.0) ** 2 + y ** 2, expected_radius ** 2, places=9)
            self.assertAlmostEqual(z, 0.2, places=9)
            for axis, value in enumerate(sample["target_origin_in_root_xyz_m"]):
                self.assertGreaterEqual(value, bounds["min_xyz_m"][axis] - 1e-12)
                self.assertLessEqual(value, bounds["max_xyz_m"][axis] + 1e-12)

    def test_workspace_rejects_missing_joint_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unbounded.urdf"
            path.write_text((FIXTURES / "two_dof.urdf").read_text().replace('lower="0" upper="1"', 'upper="1"'))
            model = MODULE.RobotModel(path)
            with self.assertRaisesRegex(MODULE.SpatialError, "incomplete position limits"):
                model.workspace_envelope("tool0", {}, 16)

    def test_evaluation_generator_is_blind_deterministic_and_covers_spatial_competencies(self):
        semantics = MODULE.read_semantics(FIXTURES / "semantics.json", self.model)
        canonical = self.model.canonical(self.pose, "bent", semantics, inspect_meshes=True, package_map_path=FIXTURES / "package_map.json", workspace_samples=64)
        facts = MODULE.fact_records(self.model, canonical)
        first_questions, first_keys = spatial_evaluation.generate_records(canonical, facts)
        second_questions, second_keys = spatial_evaluation.generate_records(canonical, facts)
        self.assertEqual(first_questions, second_questions)
        self.assertEqual(first_keys, second_keys)
        with_artifact_pointers = json.loads(json.dumps(canonical))
        with_artifact_pointers["artifacts"] = {"non_truth_path": "/different/machine/path"}
        artifact_questions, artifact_keys = spatial_evaluation.generate_records(with_artifact_pointers, facts)
        self.assertEqual(first_questions, artifact_questions)
        self.assertEqual(first_keys, artifact_keys)
        relocated = json.loads(json.dumps(canonical))
        relocated["source"]["urdf"] = "/another/machine/robot.urdf"
        relocated["semantics"]["source"]["path"] = "/another/machine/semantics.json"
        for geometry in relocated["geometry_analysis"].values():
            if geometry.get("source", {}).get("sha256"):
                geometry["source"]["path"] = "/another/machine/mesh.stl"
        relocated_questions, relocated_keys = spatial_evaluation.generate_records(relocated, facts)
        self.assertEqual(first_questions, relocated_questions)
        self.assertEqual(first_keys, relocated_keys)
        changed_truth = json.loads(json.dumps(canonical))
        changed_truth["frames"]["tool0"]["world_from_frame"]["translation_xyz_m"][2] += 0.001
        self.assertNotEqual(
            spatial_evaluation.spatial_truth_sha256(canonical),
            spatial_evaluation.spatial_truth_sha256(changed_truth),
        )
        self.assertEqual(len(first_questions), len(first_keys))
        self.assertGreaterEqual(len(first_questions), 30)
        self.assertTrue(all("answer" not in question and "evidence" not in question for question in first_questions))
        capabilities = {question["capability"] for question in first_questions}
        self.assertTrue({"topology", "frame_semantics", "pose_transform", "joint_axis", "kinematic_causality", "instantaneous_motion", "mass_properties", "geometry", "collision_epistemics", "semantic_grounding", "workspace_epistemics"}.issubset(capabilities))
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = spatial_evaluation.generate_evaluation(canonical, facts, Path(temp_dir) / "public", Path(temp_dir) / "private" / "answer-key.jsonl")
            self.assertEqual(manifest["grader_self_check"]["perfect_answers_control"]["status"], "passed")
            self.assertEqual(manifest["grader_self_check"]["one_missing_answer_control"]["status"], "failed")
            self.assertNotIn("answer_key", manifest["artifacts"])
            self.assertTrue((Path(temp_dir) / "private" / "answer-key.jsonl").exists())

    def test_evaluation_verifier_has_passing_and_failing_controls(self):
        semantics = MODULE.read_semantics(FIXTURES / "semantics.json", self.model)
        canonical = self.model.canonical(self.pose, "bent", semantics, workspace_samples=32)
        facts = MODULE.fact_records(self.model, canonical)
        _, keys = spatial_evaluation.generate_records(canonical, facts)
        perfect = [{"question_id": key["question_id"], "answer": key["answer"]} for key in keys]
        passed = spatial_evaluation.verify_answers(keys, perfect)
        self.assertEqual(passed["status"], "passed")
        self.assertEqual(passed["accuracy"], 1.0)
        wrong = [dict(record) for record in perfect]
        wrong[0]["answer"] = None
        failed = spatial_evaluation.verify_answers(keys, wrong)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(len(failed["failures"]), 1)
        missing = spatial_evaluation.verify_answers(keys, perfect[:-1])
        self.assertEqual(missing["status"], "failed")
        self.assertIn("missing answer", missing["failures"][0]["reason"])
        equivalent_quaternion = json.loads(json.dumps(perfect))
        quaternion_record = next(record for record in equivalent_quaternion if isinstance(record["answer"], dict) and "quaternion_xyzw" in record["answer"])
        quaternion_record["answer"]["quaternion_xyzw"] = [-value for value in quaternion_record["answer"]["quaternion_xyzw"]]
        self.assertEqual(spatial_evaluation.verify_answers(keys, equivalent_quaternion)["status"], "passed")

    def test_evaluation_cli_exit_code_is_a_ci_gate(self):
        canonical = self.model.canonical(self.pose, "bent", workspace_samples=0)
        facts = MODULE.fact_records(self.model, canonical)
        _, keys = spatial_evaluation.generate_records(canonical, facts)
        perfect = [{"question_id": key["question_id"], "answer": key["answer"]} for key in keys]
        with tempfile.TemporaryDirectory() as temp_dir:
            key_path = Path(temp_dir) / "key.jsonl"
            answers_path = Path(temp_dir) / "answers.jsonl"
            key_path.write_text(spatial_evaluation.jsonl_dump(keys))
            answers_path.write_text(spatial_evaluation.jsonl_dump(perfect))
            passed = subprocess.run([sys.executable, str(SCRIPT.parent / "spatial_evaluation.py"), "verify", str(key_path), str(answers_path)], capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0)
            self.assertEqual(json.loads(passed.stdout)["status"], "passed")
            answers_path.write_text(spatial_evaluation.jsonl_dump(perfect[:-1]))
            failed = subprocess.run([sys.executable, str(SCRIPT.parent / "spatial_evaluation.py"), "verify", str(key_path), str(answers_path)], capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            self.assertEqual(json.loads(failed.stdout)["status"], "failed")

    def test_readonly_evaluation_suite_binds_public_artifacts_private_keys_and_all_answers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            public_root = root / "public"
            private_root = root / "private"
            submissions_root = root / "submissions"
            evaluation_dir = public_root / "tasks" / "case" / "evaluation"
            key_dir = private_root / "keys"
            submission_dir = submissions_root / "case"
            for directory in (evaluation_dir, key_dir, submission_dir):
                directory.mkdir(parents=True)

            questions = [
                {
                    "schema_version": "robot-spatial-question.v1",
                    "question_id": "q-root",
                    "capability": "topology",
                    "task": "root",
                },
                {
                    "schema_version": "robot-spatial-question.v1",
                    "question_id": "q-axis",
                    "capability": "joint_axis",
                    "task": "axis",
                },
            ]
            templates = [{"question_id": question["question_id"], "answer": None} for question in questions]
            keys = [
                {
                    "schema_version": "robot-spatial-answer-key.v1",
                    "question_id": "q-root",
                    "capability": "topology",
                    "answer": "base",
                    "comparison": {},
                    "evidence": {"spatial_truth_sha256": "a" * 64},
                },
                {
                    "schema_version": "robot-spatial-answer-key.v1",
                    "question_id": "q-axis",
                    "capability": "joint_axis",
                    "answer": [0.0, 0.0, 1.0],
                    "comparison": {"absolute_tolerance": 1e-9},
                    "evidence": {"spatial_truth_sha256": "a" * 64},
                },
            ]
            questions_path = evaluation_dir / "questions.jsonl"
            template_path = evaluation_dir / "answer-template.jsonl"
            evaluation_manifest_path = evaluation_dir / "manifest.json"
            key_path = key_dir / "case.jsonl"
            questions_path.write_text(spatial_evaluation.jsonl_dump(questions))
            template_path.write_text(spatial_evaluation.jsonl_dump(templates))
            evaluation_manifest_path.write_text(json.dumps({
                "schema_version": "robot-spatial-evaluation-manifest.v1",
                "question_count": 2,
                "capability_counts": {"joint_axis": 1, "topology": 1},
                "spatial_truth_sha256": "a" * 64,
            }))
            key_path.write_text(spatial_evaluation.jsonl_dump(keys))

            artifact_paths = [evaluation_manifest_path, questions_path, template_path]
            public_manifest = {
                "schema_version": "robot-spatial-evaluation-suite.v1",
                "suite_id": "suite-test",
                "artifacts": {"INSTRUCTIONS.md": "placeholder"},
                "tasks": [{
                    "task_id": "case",
                    "robot_family": "fixture",
                    "evaluation_manifest": "tasks/case/evaluation/manifest.json",
                    "questions": "tasks/case/evaluation/questions.jsonl",
                    "answer_template": "tasks/case/evaluation/answer-template.jsonl",
                    "submission": "case/answers.jsonl",
                    "artifacts": {
                        str(path.relative_to(public_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in artifact_paths
                    },
                }],
            }
            instructions_path = public_root / "INSTRUCTIONS.md"
            instructions_path.write_text("public instructions\n")
            public_manifest["artifacts"]["INSTRUCTIONS.md"] = hashlib.sha256(instructions_path.read_bytes()).hexdigest()
            public_manifest_path = public_root / "manifest.json"
            public_manifest_path.write_text(json.dumps(public_manifest))
            private_manifest_path = private_root / "manifest.json"
            private_manifest_path.write_text(json.dumps({
                "schema_version": "robot-spatial-evaluation-suite-key.v1",
                "suite_id": "suite-test",
                "public_manifest_sha256": hashlib.sha256(public_manifest_path.read_bytes()).hexdigest(),
                "keys": {"case": {"answer_key": "keys/case.jsonl", "sha256": hashlib.sha256(key_path.read_bytes()).hexdigest()}},
            }))

            answers_path = submission_dir / "answers.jsonl"
            perfect = [{"question_id": key["question_id"], "answer": key["answer"]} for key in keys]
            answers_path.write_text(spatial_evaluation.jsonl_dump(perfect))
            passed = spatial_evaluation_suite.grade_suite(public_manifest_path, private_manifest_path, submissions_root)
            self.assertEqual(passed["status"], "passed")
            self.assertEqual(passed["overall_accuracy"], 1.0)

            answers_path.write_text(spatial_evaluation.jsonl_dump(perfect[:-1]))
            missing = spatial_evaluation_suite.grade_suite(public_manifest_path, private_manifest_path, submissions_root)
            self.assertEqual(missing["status"], "failed")
            self.assertEqual(missing["tasks"][0]["report"]["failures"][0]["reason"], "missing answer")
            public_summary = spatial_evaluation_suite.public_result_summary(missing)
            self.assertEqual(public_summary["tasks"][0]["failure_counts"]["missing"], 1)
            self.assertNotIn('"expected":', json.dumps(public_summary))
            self.assertNotIn("q-root", json.dumps(public_summary))
            self.assertNotIn("q-axis", json.dumps(public_summary))

            answers_path.write_text(spatial_evaluation.jsonl_dump([*perfect, perfect[0]]))
            duplicate = spatial_evaluation_suite.grade_suite(public_manifest_path, private_manifest_path, submissions_root)
            self.assertEqual(duplicate["status"], "failed")
            self.assertEqual(duplicate["tasks"][0]["report"]["duplicate_question_ids"], ["q-root"])

            source_context = root / "source-context"
            source_context.mkdir()
            digest = "b" * 64
            for filename in spatial_evaluation_suite.REQUIRED_CONTEXT_FILES:
                path = source_context / filename
                if filename == "agent-context.json":
                    path.write_text(json.dumps({"source": {"urdf": "/secret/source.urdf", "sha256": digest}}))
                elif filename == "model.json":
                    path.write_text(json.dumps({
                        "source": {"urdf": "/secret/source.urdf", "sha256": digest},
                        "geometry": {"source": {"path": "/secret/mesh.stl", "sha256": digest}},
                    }))
                elif filename.endswith(".json"):
                    path.write_text("{}\n")
                else:
                    path.write_text("public context\n")
            render_atlas = source_context / "render-atlas"
            (render_atlas / "views").mkdir(parents=True)
            (render_atlas / "manifest.json").write_text('{"schema_version":"robot-spatial-render-atlas.v1"}\n')
            (render_atlas / "views" / "front.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>\n')
            motion_atlas = source_context / "motion-atlas"
            (motion_atlas / "drivers" / "driver").mkdir(parents=True)
            (motion_atlas / "manifest.json").write_text(
                '{"schema_version":"robot-spatial-motion-atlas.v1"}\n'
            )
            (motion_atlas / "drivers" / "driver" / "front.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg"/>\n'
            )
            build_config_path = root / "build.json"
            build_config_path.write_text(json.dumps({
                "schema_version": "robot-spatial-evaluation-suite-build.v1",
                "suite_id": "suite-built",
                "tasks": [{
                    "task_id": "case",
                    "robot_family": "fixture",
                    "context_dir": str(source_context),
                    "evaluation_dir": str(evaluation_dir),
                    "answer_key": str(key_path),
                    "source": {"repository": "fixture", "commit": "pinned"},
                }],
            }))
            built_public = root / "built-public"
            built_private = root / "built-private"
            built = spatial_evaluation_suite.build_suite(build_config_path, built_public, built_private)
            self.assertEqual(built["public_leak_scan"], "passed")
            sanitized = json.loads((built_public / "tasks" / "case" / "context" / "model.json").read_text())
            self.assertEqual(sanitized["source"]["urdf"], f"sha256:{digest}")
            self.assertEqual(sanitized["geometry"]["source"]["path"], f"sha256:{digest}")
            self.assertTrue((built_public / "tasks" / "case" / "context" / "render-atlas" / "manifest.json").is_file())
            self.assertTrue((built_public / "tasks" / "case" / "context" / "render-atlas" / "views" / "front.svg").is_file())
            self.assertTrue((built_public / "tasks" / "case" / "context" / "motion-atlas" / "manifest.json").is_file())
            self.assertTrue((built_public / "tasks" / "case" / "context" / "motion-atlas" / "drivers" / "driver" / "front.svg").is_file())
            built_manifest = json.loads((built_public / "manifest.json").read_text())
            built_artifacts = built_manifest["tasks"][0]["artifacts"]
            self.assertIn("tasks/case/context/render-atlas/manifest.json", built_artifacts)
            self.assertIn("tasks/case/context/render-atlas/views/front.svg", built_artifacts)
            self.assertIn("tasks/case/context/motion-atlas/manifest.json", built_artifacts)
            self.assertIn("tasks/case/context/motion-atlas/drivers/driver/front.svg", built_artifacts)
            self.assertNotIn("robot-spatial-answer-key.v1", (built_public / "manifest.json").read_text())
            answers_path.write_text(spatial_evaluation.jsonl_dump(perfect))
            built_report = spatial_evaluation_suite.grade_suite(
                built_public / "manifest.json",
                built_private / "manifest.json",
                submissions_root,
            )
            self.assertEqual(built_report["status"], "passed")
            self_check = spatial_evaluation_suite.self_check_suite(
                built_public / "manifest.json",
                built_private / "manifest.json",
            )
            self.assertEqual(self_check["status"], "passed")
            self.assertEqual(self_check["perfect_submission_control"]["overall_accuracy"], 1.0)
            self.assertEqual(self_check["one_missing_answer_control"]["status"], "failed")

            questions_path.write_text(questions_path.read_text() + "\n")
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "public artifact digest mismatch"):
                spatial_evaluation_suite.grade_suite(public_manifest_path, private_manifest_path, submissions_root)

    def test_raw_source_task_can_require_ros_capture_normalization_without_public_observations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_root = Path(temp_dir) / "task"
            source = task_root / "source"
            source.mkdir(parents=True)
            shutil.copy2(FIXTURES / "two_dof.urdf", source / "robot.urdf")
            for filename in ("scene.json", "adapter-config.json", "capture.json", "query.json"):
                (source / filename).write_text("{}\n", encoding="utf-8")
            task = {
                "schema_version": "robot-spatial-raw-source-task.v1",
                "workflow": "direct",
                "input_format": "urdf",
                "entrypoint": "source/robot.urdf",
                "export_options": {"scene": "source/scene.json", "workspace_samples": 0},
                "ros_observation_adapter": {
                    "config": "source/adapter-config.json",
                    "capture": "source/capture.json",
                    "observation_query": "source/query.json",
                    "output_filename": "observations.json",
                    "report_filename": "normalization-report.json",
                },
            }
            spatial_evaluation_suite._validate_raw_task_spec(task, task_root, "task")

            ambiguous = json.loads(json.dumps(task))
            ambiguous["export_options"]["observations"] = "source/capture.json"
            ambiguous["export_options"]["observation_query"] = "source/query.json"
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "must not combine"):
                spatial_evaluation_suite._validate_raw_task_spec(ambiguous, task_root, "task")

            escaping = json.loads(json.dumps(task))
            escaping["ros_observation_adapter"]["capture"] = "../capture.json"
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "escapes"):
                spatial_evaluation_suite._validate_raw_task_spec(escaping, task_root, "task")

    def test_raw_source_task_can_declare_digest_bound_cross_representation_comparisons(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_root = Path(temp_dir) / "task"
            source = task_root / "source"
            source.mkdir(parents=True)
            shutil.copy2(FIXTURES / "two_dof.urdf", source / "robot.urdf")
            (source / "robot.sdf").write_text(
                '<sdf version="1.11"><model name="robot"><link name="root"/></model></sdf>\n'
            )
            (source / "mapping.json").write_text("{}\n")
            task = {
                "schema_version": "robot-spatial-raw-source-task.v1",
                "workflow": "direct",
                "input_format": "urdf",
                "entrypoint": "source/robot.urdf",
                "export_options": {"workspace_samples": 0},
                "articulation_sources": [
                    {"source_id": "reference", "format": "urdf", "path": "source/robot.urdf"},
                    {"source_id": "candidate", "format": "sdf", "path": "source/robot.sdf"},
                ],
                "articulation_comparisons": [{
                    "reference": "reference",
                    "candidate": "candidate",
                    "correspondence": "source/mapping.json",
                }],
            }
            spatial_evaluation_suite._validate_raw_task_spec(task, task_root, "task")

            duplicate = json.loads(json.dumps(task))
            duplicate["articulation_sources"][1]["source_id"] = "reference"
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "duplicate source_id"):
                spatial_evaluation_suite._validate_raw_task_spec(duplicate, task_root, "task")

            unknown = json.loads(json.dumps(task))
            unknown["articulation_comparisons"][0]["candidate"] = "missing"
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "distinct declared"):
                spatial_evaluation_suite._validate_raw_task_spec(unknown, task_root, "task")

            missing_comparisons = json.loads(json.dumps(task))
            missing_comparisons.pop("articulation_comparisons")
            spatial_evaluation_suite._validate_raw_task_spec(missing_comparisons, task_root, "task")

    def test_raw_source_task_can_declare_single_source_supplemental_constraint_graph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_root = Path(temp_dir) / "task"
            source = task_root / "source"
            source.mkdir(parents=True)
            shutil.copy2(FIXTURES / "two_dof.urdf", source / "robot.urdf")
            (source / "constraints.json").write_text("{}\n")
            task = {
                "schema_version": "robot-spatial-raw-source-task.v1",
                "workflow": "direct",
                "input_format": "urdf",
                "entrypoint": "source/robot.urdf",
                "export_options": {
                    "constraint_spec": "source/constraints.json",
                    "workspace_samples": 0,
                },
                "articulation_sources": [
                    {"source_id": "tree", "format": "urdf", "path": "source/robot.urdf"},
                ],
                "constraint_graphs": [{
                    "graph_id": "mechanism",
                    "articulation_source": "tree",
                    "spec": "source/constraints.json",
                }],
            }
            spatial_evaluation_suite._validate_raw_task_spec(task, task_root, "task")

            unknown = json.loads(json.dumps(task))
            unknown["constraint_graphs"][0]["articulation_source"] = "missing"
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "declared articulation source"):
                spatial_evaluation_suite._validate_raw_task_spec(unknown, task_root, "task")

            escaping = json.loads(json.dumps(task))
            escaping["constraint_graphs"][0]["spec"] = "../constraints.json"
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "escapes"):
                spatial_evaluation_suite._validate_raw_task_spec(escaping, task_root, "task")

            missing_sources = json.loads(json.dumps(task))
            missing_sources.pop("articulation_sources")
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "requires articulation_sources"):
                spatial_evaluation_suite._validate_raw_task_spec(missing_sources, task_root, "task")

            (source / "constraint-graph.json").write_text("{}\n")
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "generated context artifact"):
                spatial_evaluation_suite._copy_raw_source_tree(source, task_root / "copied-source")

    def test_raw_source_task_can_declare_finite_configuration_atlas(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_root = Path(temp_dir) / "task"
            source = task_root / "source"
            source.mkdir(parents=True)
            shutil.copy2(FIXTURES / "spherical_loop_tree.urdf", source / "robot.urdf")
            (source / "constraints.json").write_text("{}\n")
            (source / "configuration-atlas-spec.json").write_text("{}\n")
            task = {
                "schema_version": "robot-spatial-raw-source-task.v1",
                "workflow": "direct",
                "input_format": "urdf",
                "entrypoint": "source/robot.urdf",
                "export_options": {
                    "constraint_spec": "source/constraints.json",
                    "configuration_atlas_spec": "source/configuration-atlas-spec.json",
                    "workspace_samples": 0,
                },
                "articulation_sources": [
                    {"source_id": "tree", "format": "urdf", "path": "source/robot.urdf"},
                ],
                "constraint_graphs": [{
                    "graph_id": "mechanism",
                    "articulation_source": "tree",
                    "spec": "source/constraints.json",
                }],
                "configuration_atlases": [{
                    "atlas_id": "finite-witnesses",
                    "constraint_graph": "mechanism",
                    "spec": "source/configuration-atlas-spec.json",
                }],
            }
            spatial_evaluation_suite._validate_raw_task_spec(task, task_root, "task")

            unknown = json.loads(json.dumps(task))
            unknown["configuration_atlases"][0]["constraint_graph"] = "missing"
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "declared constraint graph"):
                spatial_evaluation_suite._validate_raw_task_spec(unknown, task_root, "task")

            missing_graphs = json.loads(json.dumps(task))
            missing_graphs.pop("constraint_graphs")
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "requires constraint_graphs"):
                spatial_evaluation_suite._validate_raw_task_spec(missing_graphs, task_root, "task")

            missing_export_constraint = json.loads(json.dumps(task))
            missing_export_constraint["export_options"].pop("constraint_spec")
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "requires constraint_spec"):
                spatial_evaluation_suite._validate_raw_task_spec(missing_export_constraint, task_root, "task")

            (source / "configuration-atlas.json").write_text("{}\n")
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "generated context artifact"):
                spatial_evaluation_suite._copy_raw_source_tree(source, task_root / "copied-atlas-source")

    def test_source_only_evaluation_suite_contains_raw_urdf_without_generated_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            evaluation_dir = root / "evaluation"
            key_dir = root / "evaluator-keys"
            submissions_root = root / "submissions"
            for directory in (source_dir, evaluation_dir, key_dir, submissions_root / "novel"):
                directory.mkdir(parents=True)
            (source_dir / "robot.urdf").write_text(
                '<robot name="novel"><link name="root"/><link name="tip"/>'
                '<joint name="reach" type="prismatic"><parent link="root"/><child link="tip"/>'
                '<axis xyz="1 0 0"/><limit lower="0" upper="1" effort="1" velocity="1"/>'
                '</joint></robot>\n'
            )
            (source_dir / "pose.json").write_text(
                json.dumps({"pose_name": "extended", "joints": {"reach": 0.25}}) + "\n"
            )
            (source_dir / "scene.json").write_text(json.dumps({
                "schema_version": "robot-spatial-world-scene.v1",
                "scene_id": "novel_scene",
                "snapshot": {
                    "id": "novel_snapshot",
                    "time_semantics": "static_snapshot",
                    "captured_at": None,
                    "valid_until": None,
                },
                "world_frame": "world",
                "frames": {},
                "robot": {
                    "instance_id": "novel_instance",
                    "robot_name": "novel",
                    "root_link": "root",
                    "parent_frame": "world",
                    "source": {"type": "synthetic", "reference": "test", "captured_at": None},
                },
                "objects": {},
            }) + "\n")
            questions = [{
                "schema_version": "robot-spatial-question.v1",
                "question_id": "q-raw-root",
                "capability": "topology",
                "task": "root",
            }]
            templates = [{"question_id": "q-raw-root", "answer": None}]
            keys = [{
                "schema_version": "robot-spatial-answer-key.v1",
                "question_id": "q-raw-root",
                "capability": "topology",
                "answer": "root",
                "comparison": {},
                "evidence": {"spatial_truth_sha256": "c" * 64},
            }]
            (evaluation_dir / "questions.jsonl").write_text(spatial_evaluation.jsonl_dump(questions))
            (evaluation_dir / "answer-template.jsonl").write_text(spatial_evaluation.jsonl_dump(templates))
            (evaluation_dir / "manifest.json").write_text(json.dumps({
                "schema_version": "robot-spatial-evaluation-manifest.v1",
                "question_count": 1,
                "capability_counts": {"topology": 1},
                "spatial_truth_sha256": "c" * 64,
            }))
            key_path = key_dir / "novel.jsonl"
            key_path.write_text(spatial_evaluation.jsonl_dump(keys))
            config = {
                "schema_version": "robot-spatial-evaluation-suite-build.v1",
                "suite_id": "raw-suite",
                "candidate_input": "raw_sources",
                "tasks": [{
                    "task_id": "novel",
                    "robot_family": "held_out_fixture",
                    "source_dir": str(source_dir),
                    "evaluation_dir": str(evaluation_dir),
                    "answer_key": str(key_path),
                    "candidate_task": {
                        "schema_version": "robot-spatial-raw-source-task.v1",
                        "input_format": "urdf",
                        "entrypoint": "source/robot.urdf",
                        "export_options": {
                            "pose": "source/pose.json",
                            "scene": "source/scene.json",
                            "motion_atlas": True,
                            "motion_angular_step_rad": 0.125,
                            "motion_linear_step_m": 0.02,
                            "render": True,
                            "workspace_samples": 0,
                        },
                    },
                }],
            }
            config_path = root / "build.json"
            config_path.write_text(json.dumps(config))
            public_root = root / "public"
            private_root = root / "private"
            built = spatial_evaluation_suite.build_suite(config_path, public_root, private_root)
            self.assertEqual(built["candidate_input"], "raw_sources")
            manifest = json.loads((public_root / "manifest.json").read_text())
            self.assertEqual(manifest["candidate_input"], "raw_sources")
            self.assertNotIn("context_entrypoint", manifest["tasks"][0])
            self.assertEqual(manifest["tasks"][0]["task_spec"], "tasks/novel/task.json")
            self.assertTrue((public_root / "tasks" / "novel" / "source" / "robot.urdf").is_file())
            self.assertTrue((public_root / "tasks" / "novel" / "source" / "scene.json").is_file())
            for filename in spatial_evaluation_suite.REQUIRED_CONTEXT_FILES:
                self.assertFalse((public_root / "tasks" / "novel" / "source" / filename).exists())
            (submissions_root / "novel" / "answers.jsonl").write_text(
                spatial_evaluation.jsonl_dump([{"question_id": "q-raw-root", "answer": "root"}])
            )
            report = spatial_evaluation_suite.grade_suite(
                public_root / "manifest.json", private_root / "manifest.json", submissions_root
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["candidate_input"], "raw_sources")
            control = spatial_evaluation_suite.self_check_suite(
                public_root / "manifest.json", private_root / "manifest.json"
            )
            self.assertEqual(control["candidate_input"], "raw_sources")

            public_source = public_root / "tasks" / "novel" / "source" / "robot.urdf"
            public_source.write_text(public_source.read_text() + "\n")
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "public artifact digest mismatch"):
                spatial_evaluation_suite.grade_suite(
                    public_root / "manifest.json", private_root / "manifest.json", submissions_root
                )

            leaking_source = root / "leaking-source"
            leaking_source.mkdir()
            (leaking_source / "robot.urdf").write_text('<robot name="leak"><link name="root"/></robot>\n')
            (leaking_source / "model.json").write_text("{}\n")
            leaking_config = json.loads(json.dumps(config))
            leaking_config["suite_id"] = "raw-leak"
            leaking_config["tasks"][0]["source_dir"] = str(leaking_source)
            leaking_path = root / "leaking-build.json"
            leaking_path.write_text(json.dumps(leaking_config))
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "generated context artifact"):
                spatial_evaluation_suite.build_suite(
                    leaking_path, root / "leaking-public", root / "leaking-private"
                )

            grammar_leak_source = root / "grammar-leaking-source"
            shutil.copytree(source_dir, grammar_leak_source)
            (grammar_leak_source / "articulation-grammar.json").write_text("{}\n")
            grammar_leak_config = json.loads(json.dumps(config))
            grammar_leak_config["suite_id"] = "raw-grammar-leak"
            grammar_leak_config["tasks"][0]["source_dir"] = str(grammar_leak_source)
            grammar_leak_path = root / "grammar-leaking-build.json"
            grammar_leak_path.write_text(json.dumps(grammar_leak_config))
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "generated context artifact"):
                spatial_evaluation_suite.build_suite(
                    grammar_leak_path,
                    root / "grammar-leaking-public",
                    root / "grammar-leaking-private",
                )

            comparison_leak_source = root / "comparison-leaking-source"
            shutil.copytree(source_dir, comparison_leak_source)
            (comparison_leak_source / "articulation-comparison.json").write_text("{}\n")
            comparison_leak_config = json.loads(json.dumps(config))
            comparison_leak_config["suite_id"] = "raw-comparison-leak"
            comparison_leak_config["tasks"][0]["source_dir"] = str(comparison_leak_source)
            comparison_leak_path = root / "comparison-leaking-build.json"
            comparison_leak_path.write_text(json.dumps(comparison_leak_config))
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "generated context artifact"):
                spatial_evaluation_suite.build_suite(
                    comparison_leak_path,
                    root / "comparison-leaking-public",
                    root / "comparison-leaking-private",
                )

            directory_leak_source = root / "directory-leaking-source"
            shutil.copytree(source_dir, directory_leak_source)
            (directory_leak_source / "motion-atlas").mkdir()
            (directory_leak_source / "motion-atlas" / "manifest.json").write_text("{}\n")
            directory_leak_config = json.loads(json.dumps(config))
            directory_leak_config["suite_id"] = "raw-directory-leak"
            directory_leak_config["tasks"][0]["source_dir"] = str(directory_leak_source)
            directory_leak_path = root / "directory-leaking-build.json"
            directory_leak_path.write_text(json.dumps(directory_leak_config))
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "generated context directory"):
                spatial_evaluation_suite.build_suite(
                    directory_leak_path,
                    root / "directory-leaking-public",
                    root / "directory-leaking-private",
                )

            escaping_config = json.loads(json.dumps(config))
            escaping_config["suite_id"] = "raw-escape"
            escaping_config["tasks"][0]["candidate_task"]["entrypoint"] = "../outside.urdf"
            escaping_path = root / "escaping-build.json"
            escaping_path.write_text(json.dumps(escaping_config))
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "escapes its declared root"):
                spatial_evaluation_suite.build_suite(
                    escaping_path, root / "escaping-public", root / "escaping-private"
                )

            xacro_source = root / "xacro-source"
            xacro_source.mkdir()
            (xacro_source / "robot.urdf.xacro").write_text(
                '<robot name="xacro_novel" xmlns:xacro="http://www.ros.org/wiki/xacro">'
                '<xacro:arg name="prefix" default=""/>'
                '<xacro:property name="prefix" value="$(arg prefix)"/>'
                '<link name="${prefix}root"/></robot>\n'
            )
            xacro_config = json.loads(json.dumps(config))
            xacro_config["suite_id"] = "raw-xacro"
            xacro_config["runtime_requirements"] = {
                "xacro": {"executable": "xacro", "version": "2.1.1", "provision": "evaluator"}
            }
            xacro_task = xacro_config["tasks"][0]
            xacro_task["source_dir"] = str(xacro_source)
            xacro_task["candidate_task"] = {
                "schema_version": "robot-spatial-raw-source-task.v1",
                "input_format": "xacro",
                "entrypoint": "source/robot.urdf.xacro",
                "expansion": {
                    "executable": "xacro",
                    "mappings": ["prefix:=held_"],
                    "output": "expanded.urdf",
                },
                "export_options": {"workspace_samples": 0},
            }
            xacro_config_path = root / "xacro-build.json"
            xacro_config_path.write_text(json.dumps(xacro_config))
            xacro_public = root / "xacro-public"
            xacro_private = root / "xacro-private"
            spatial_evaluation_suite.build_suite(xacro_config_path, xacro_public, xacro_private)
            xacro_manifest = json.loads((xacro_public / "manifest.json").read_text())
            self.assertEqual(xacro_manifest["runtime_requirements"]["xacro"]["version"], "2.1.1")
            xacro_spec = json.loads((xacro_public / "tasks" / "novel" / "task.json").read_text())
            self.assertEqual(xacro_spec["input_format"], "xacro")
            self.assertEqual(xacro_spec["expansion"]["mappings"], ["prefix:=held_"])
            xacro_report = spatial_evaluation_suite.grade_suite(
                xacro_public / "manifest.json", xacro_private / "manifest.json", submissions_root
            )
            self.assertEqual(xacro_report["status"], "passed")

            project_config = json.loads(json.dumps(xacro_config))
            project_config["suite_id"] = "raw-xacro-project"
            project_task = project_config["tasks"][0]["candidate_task"]
            project_task["workflow"] = "prepare"
            project_task["workspace_roots"] = ["source"]
            project_task["expansion"]["output"] = "resolved.urdf"
            project_config_path = root / "xacro-project-build.json"
            project_config_path.write_text(json.dumps(project_config))
            project_public = root / "xacro-project-public"
            project_private = root / "xacro-project-private"
            spatial_evaluation_suite.build_suite(
                project_config_path, project_public, project_private
            )
            project_spec = json.loads(
                (project_public / "tasks" / "novel" / "task.json").read_text()
            )
            self.assertEqual(project_spec["workflow"], "prepare")
            self.assertEqual(project_spec["workspace_roots"], ["source"])
            project_instructions = (project_public / "INSTRUCTIONS.md").read_text()
            self.assertIn("workflow` is `prepare", project_instructions)
            self.assertIn("`not_established` (semantic role or intent not explicitly declared)", project_instructions)

            escaping_workspace = json.loads(json.dumps(project_config))
            escaping_workspace["suite_id"] = "raw-project-workspace-escape"
            escaping_workspace["tasks"][0]["candidate_task"]["workspace_roots"] = ["../outside"]
            escaping_workspace_path = root / "escaping-workspace.json"
            escaping_workspace_path.write_text(json.dumps(escaping_workspace))
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "escapes its declared root"):
                spatial_evaluation_suite.build_suite(
                    escaping_workspace_path,
                    root / "escaping-workspace-public",
                    root / "escaping-workspace-private",
                )

            missing_runtime = json.loads(json.dumps(xacro_config))
            missing_runtime["suite_id"] = "raw-xacro-no-runtime"
            missing_runtime.pop("runtime_requirements")
            missing_runtime_path = root / "xacro-missing-runtime.json"
            missing_runtime_path.write_text(json.dumps(missing_runtime))
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "runtime_requirements.xacro"):
                spatial_evaluation_suite.build_suite(
                    missing_runtime_path, root / "xacro-no-runtime-public", root / "xacro-no-runtime-private"
                )

    def test_affects_includes_branches_driven_by_mimic_joint(self):
        model = MODULE.RobotModel(FIXTURES / "mimic_branch.urdf")
        affected = model.affected_by_joint("driver")
        self.assertEqual(affected["physical_joints_driven"], ["driver", "follower"])
        self.assertEqual(set(affected["affected_links"]), {"driver_link", "follower_link"})
        jacobian = model.geometric_jacobian("follower_link", {"driver": 0.2})
        self.assertEqual(jacobian["joint_order"], ["driver"])
        self.assertEqual(jacobian["columns"][0]["physical_contributions"][0]["derivative_multiplier"], -0.5)
        workspace = model.workspace_envelope("follower_link", {"driver": 0.2}, 8)
        self.assertEqual(workspace["joint_sampling_ranges"][0]["minimum"], -0.2)
        self.assertEqual(workspace["joint_sampling_ranges"][0]["maximum"], 0.6)
        self.assertIn("mimic_limits", workspace["joint_sampling_ranges"][0]["range_source"])

    def test_geometry_frames_remain_distinct(self):
        canonical = self.model.canonical(self.pose, "bent")
        self.assertEqual(canonical["frames"]["visual/base_link/0"]["type"], "visual")
        self.assertEqual(canonical["frames"]["collision/base_link/0"]["type"], "collision")
        self.assertEqual(canonical["geometry_analysis"]["collision/slider_link/0"]["status"], "not_inspected")

    def test_mesh_content_is_measured_in_geometry_and_root_frames(self):
        canonical = self.model.canonical(
            self.pose,
            "bent",
            inspect_meshes=True,
            package_map_path=FIXTURES / "package_map.json",
        )
        mesh = canonical["geometry_analysis"]["collision/slider_link/0"]
        self.assertEqual(mesh["status"], "measured")
        self.assertEqual(mesh["topology"]["unique_vertex_count"], 8)
        self.assertEqual(mesh["topology"]["triangle_count"], 12)
        self.assertTrue(mesh["topology"]["watertight"])
        self.assertAlmostEqual(mesh["volume_m3"], 0.001, places=9)
        self.assertVectorAlmostEqual(mesh["bounds_in_geometry_frame"]["min_xyz_m"], [0.0, -0.05, -0.025])
        self.assertVectorAlmostEqual(mesh["bounds_in_geometry_frame"]["max_xyz_m"], [0.2, 0.05, 0.025])
        self.assertVectorAlmostEqual(mesh["bounds_in_root_frame_at_pose"]["min_xyz_m"], [0.95, 1.6, -0.025])
        self.assertVectorAlmostEqual(mesh["bounds_in_root_frame_at_pose"]["max_xyz_m"], [1.05, 1.8, 0.025])
        self.assertTrue(canonical["collision_broadphase"]["complete_for_declared_collision_geometry"])

    def test_mesh_inspection_can_select_collision_without_claiming_visual_dae_support(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "unsupported.dae").write_text("<COLLADA/>")
            urdf = directory / "mixed_meshes.urdf"
            urdf.write_text(
                '<robot name="mixed_meshes"><link name="base">'
                '<visual><geometry><mesh filename="unsupported.dae"/></geometry></visual>'
                '<collision><geometry><mesh filename="package://demo/slider.stl"/></geometry></collision>'
                '</link></robot>'
            )
            model = MODULE.RobotModel(urdf)

            with self.assertRaisesRegex(mesh_geometry.GeometryError, "unsupported mesh format '.dae'"):
                model.geometry_analysis({}, inspect_meshes=True, package_map_path=FIXTURES / "package_map.json")

            canonical = model.canonical(
                {},
                "zero",
                package_map_path=FIXTURES / "package_map.json",
                inspect_mesh_kinds={"collision"},
            )
            self.assertEqual(canonical["geometry_analysis"]["visual/base/0"]["status"], "not_inspected")
            self.assertEqual(canonical["geometry_analysis"]["collision/base/0"]["status"], "measured")
            inspection = canonical["capabilities"]["mesh_content_inspection"]
            self.assertEqual(inspection["requested_kinds"], ["collision"])
            self.assertTrue(inspection["complete_for_requested_kinds"])
            self.assertFalse(inspection["complete_for_all_declared_meshes"])

            surface = model.canonical(
                {},
                "zero",
                package_map_path=FIXTURES / "package_map.json",
                surface_collisions=True,
            )
            self.assertEqual(surface["geometry_analysis"]["visual/base/0"]["status"], "not_inspected")
            self.assertEqual(surface["geometry_analysis"]["collision/base/0"]["status"], "measured")
            self.assertEqual(surface["capabilities"]["mesh_content_inspection"]["requested_kinds"], ["collision"])

            exported = directory / "exported"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "export",
                    str(urdf),
                    "--inspect-mesh-kind",
                    "collision",
                    "--package-map",
                    str(FIXTURES / "package_map.json"),
                    "--workspace-samples",
                    "0",
                    "--out",
                    str(exported),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads((exported / "model.json").read_text())["capabilities"]["mesh_content_inspection"]["requested_kinds"],
                ["collision"],
            )

    def test_obj_reader_measures_watertight_tetrahedron(self):
        mesh = mesh_geometry.load_obj(FIXTURES / "tetra.obj")
        measured, _ = mesh_geometry._mesh_analysis(mesh, [1.0, 1.0, 1.0], MODULE.identity(), FIXTURES / "tetra.obj")
        self.assertEqual(measured["topology"]["unique_vertex_count"], 4)
        self.assertEqual(measured["topology"]["triangle_count"], 4)
        self.assertTrue(measured["topology"]["watertight"])
        self.assertAlmostEqual(measured["volume_m3"], 1.0 / 6.0, places=9)

    def test_binary_stl_reader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "one_triangle.stl"
            header = b"binary fixture".ljust(80, b" ")
            triangle = struct.pack("<12fH", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0)
            path.write_bytes(header + struct.pack("<I", 1) + triangle)
            mesh = mesh_geometry.load_stl(path)
            self.assertEqual(mesh.source_format, "stl_binary")
            self.assertEqual(len(mesh.vertices), 3)
            self.assertEqual(len(mesh.faces), 1)

    def test_collision_broadphase_reports_candidate_without_calling_it_collision(self):
        def record(frame, link, minimum, maximum):
            return {
                "frame": frame,
                "kind": "collision",
                "link": link,
                "status": "measured",
                "bounds_in_root_frame_at_pose": {
                    "min_xyz_m": minimum,
                    "max_xyz_m": maximum,
                },
            }

        analysis = {
            "collision/a/0": record("collision/a/0", "a", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            "collision/b/0": record("collision/b/0", "b", [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]),
        }
        result = mesh_geometry.broadphase_overlaps(analysis, {frozenset(("a", "b"))})
        self.assertEqual(result["method"], "root_frame_axis_aligned_bounding_box_overlap")
        self.assertIn("does not prove", result["meaning"])
        self.assertEqual(len(result["overlap_pairs"]), 1)
        self.assertTrue(result["overlap_pairs"][0]["links_are_adjacent"])
        self.assertVectorAlmostEqual(result["overlap_pairs"][0]["intersection_extents_xyz_m"], [0.5, 0.5, 0.5])

    def test_surface_collision_distinguishes_intersection_containment_and_indeterminate(self):
        def robot_xml(first_geometry, second_geometry, offset="0 0 0"):
            return f"""<robot name="surface_cases">
              <link name="a"><collision><geometry>{first_geometry}</geometry></collision></link>
              <link name="b"><collision><geometry>{second_geometry}</geometry></collision></link>
              <joint name="mount" type="fixed"><parent link="a"/><child link="b"/><origin xyz="{offset}"/></joint>
            </robot>"""

        with tempfile.TemporaryDirectory() as temp_dir:
            urdf = Path(temp_dir) / "surface.urdf"

            urdf.write_text(robot_xml('<box size="2 2 2"/>', '<box size="2 2 2"/>', "1 0 0"))
            intersecting_model = MODULE.RobotModel(urdf)
            intersecting = intersecting_model.canonical({}, "zero", surface_collisions=True)["collision_surface"]
            self.assertEqual(intersecting["self_collision_status"], "collision")
            self.assertEqual(intersecting["candidate_results"][0]["surface_distance_m"], 0.0)
            self.assertTrue(intersecting["candidate_results"][0]["within_contact_tolerance"])

            urdf.write_text(robot_xml('<box size="2 2 2"/>', '<box size="2 2 2"/>', "2.0000000005 0 0"))
            near_contact_model = MODULE.RobotModel(urdf)
            near_contact = near_contact_model.canonical({}, "zero", surface_collisions=True, contact_tolerance_m=1e-9)["collision_surface"]
            self.assertEqual(near_contact["aabb_overlap_candidate_count"], 0)
            self.assertEqual(near_contact["aabb_within_contact_tolerance_candidate_count"], 1)
            self.assertEqual(near_contact["candidate_results"][0]["candidate_source"], "aabb_within_contact_tolerance")
            self.assertEqual(near_contact["self_collision_status"], "collision")
            self.assertAlmostEqual(near_contact["candidate_results"][0]["surface_distance_m"], 5e-10, places=12)

            urdf.write_text(robot_xml('<box size="4 4 4"/>', '<box size="1 1 1"/>'))
            contained_model = MODULE.RobotModel(urdf)
            canonical = contained_model.canonical({}, "zero", surface_collisions=True)
            contained = canonical["collision_surface"]
            self.assertEqual(contained["self_collision_status"], "collision")
            self.assertGreater(contained["candidate_results"][0]["surface_distance_m"], 0.0)
            self.assertFalse(contained["candidate_results"][0]["within_contact_tolerance"])
            self.assertTrue(contained["candidate_results"][0]["containment"])
            srdf_path = Path(temp_dir) / "surface.srdf"
            srdf_path.write_text('<robot name="surface_cases"><disable_collisions link1="a" link2="b" reason="Adjacent"/></robot>')
            srdf = MODULE.parse_srdf(srdf_path, contained_model)
            disabled_but_physical = contained_model.canonical({}, "zero", srdf=srdf, surface_collisions=True)["collision_surface"]["candidate_results"][0]
            self.assertEqual(disabled_but_physical["status"], "collision")
            self.assertTrue(disabled_but_physical["disabled_by_srdf"])
            self.assertEqual(disabled_but_physical["srdf_disable_reason"], "Adjacent")
            policy_report = contained_model.canonical({}, "zero", srdf=srdf, surface_collisions=True)["collision_surface"]
            self.assertEqual(policy_report["self_collision_status"], "collision")
            self.assertEqual(policy_report["srdf_policy_filtered_self_collision_status"], "collision_free")
            self.assertEqual(policy_report["srdf_policy_filtered_collision_pair_count"], 0)
            self.assertEqual(policy_report["srdf_disabled_physical_collision_pair_count"], 1)
            predicates = {fact["predicate"] for fact in MODULE.fact_records(contained_model, canonical)}
            self.assertIn("has_triangle_surface_distance_to", predicates)
            self.assertIn("has_verified_collision_with", predicates)
            questions, keys = spatial_evaluation.generate_records(canonical, MODULE.fact_records(contained_model, canonical))
            self.assertGreaterEqual(sum(question["capability"] == "collision_surface" for question in questions), 3)
            self.assertEqual(spatial_evaluation.verify_answers(keys, [{"question_id": key["question_id"], "answer": key["answer"]} for key in keys])["status"], "passed")
            context = MODULE.context_markdown(contained_model, canonical)
            self.assertIn("positive surface distance alone does not prove separation", context)

            urdf.write_text(robot_xml('<box size="4 4 4"/>', '<cylinder radius="0.2" length="1"/>'))
            unsupported_model = MODULE.RobotModel(urdf)
            unsupported = unsupported_model.canonical({}, "zero", surface_collisions=True)["collision_surface"]
            self.assertEqual(unsupported["self_collision_status"], "indeterminate")
            self.assertEqual(unsupported["indeterminate_candidate_count"], 1)
            self.assertIn("no tessellation", unsupported["geometry_surfaces"]["collision/b/0"]["reason"])

    def test_semantic_roles_are_explicit_and_validated(self):
        semantics = MODULE.read_semantics(FIXTURES / "semantics.json", self.model)
        canonical = self.model.canonical(self.pose, "bent", semantics)
        self.assertIn("tcp", canonical["semantics"]["frames"]["tool0"]["roles"])
        self.assertEqual(canonical["semantics"]["groups"]["manipulator"]["tip_frame"], "tool0")

    def test_project_spatial_invariant_contract_passes_and_catches_regression(self):
        contract = spatial_invariants.read_invariant_contract(FIXTURES / "invariants.json", self.model)
        report = spatial_invariants.verify_invariant_contract(
            self.model,
            contract,
            package_map_path=FIXTURES / "package_map.json",
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["passed_count"], 11)
        self.assertEqual(report["failed_count"], 0)
        self.assertEqual(
            {result["type"] for result in report["results"]},
            spatial_invariants.SUPPORTED_ASSERTIONS
            - {
                "scene_transform",
                "scene_gravity_loads",
                "robot_environment_collision",
                "observation_readiness",
                "observation_transform",
                "observation_collision",
            },
        )
        canonical = self.model.canonical(self.pose, "bent", workspace_samples=0)
        canonical["invariant_validation"] = report
        facts = MODULE.fact_records(self.model, canonical)
        questions, keys = spatial_evaluation.generate_records(canonical, facts)
        self.assertEqual(sum(question["capability"] == "project_intent" for question in questions), 12)
        perfect = [{"question_id": key["question_id"], "answer": key["answer"]} for key in keys]
        self.assertEqual(spatial_evaluation.verify_answers(keys, perfect)["status"], "passed")

        regressed = json.loads(json.dumps(contract))
        pose_assertion = next(assertion for assertion in regressed["assertions"] if assertion["id"] == "tool-offset-from-slider")
        pose_assertion["expected"]["translation_xyz_m"][2] = 0.25
        failed = spatial_invariants.verify_invariant_contract(
            self.model,
            regressed,
            package_map_path=FIXTURES / "package_map.json",
        )
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failed_count"], 1)
        failure = next(result for result in failed["results"] if result["status"] == "failed")
        self.assertEqual(failure["id"], "tool-offset-from-slider")
        self.assertAlmostEqual(failure["metrics"]["translation_error_m"], 0.05)

        with tempfile.TemporaryDirectory() as temp_dir:
            edited_urdf = Path(temp_dir) / "edited.urdf"
            edited_urdf.write_text((FIXTURES / "two_dof.urdf").read_text().replace('<origin xyz="0 0 0.2" rpy="0 0 0"/>', '<origin xyz="0 0 0.25" rpy="0 0 0"/>'))
            edited_model = MODULE.RobotModel(edited_urdf)
            edit_report = spatial_invariants.verify_invariant_contract(
                edited_model,
                contract,
                package_map_path=FIXTURES / "package_map.json",
            )
            self.assertEqual(edit_report["status"], "failed")
            failed_ids = {result["id"] for result in edit_report["results"] if result["status"] == "failed"}
            self.assertEqual(failed_ids, {"tool-offset-from-slider", "base-to-tool-distance"})

    def test_blind_edit_grader_accepts_only_the_intended_source_and_contract_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_contract = json.loads((FIXTURES / "invariants.json").read_text())
            baseline_contract["assertions"] = [
                assertion
                for assertion in baseline_contract["assertions"]
                if assertion["id"] != "base-to-tool-distance"
            ]
            baseline_invariants = root / "invariants.json"
            baseline_invariants.write_text(json.dumps(baseline_contract))
            task_path = root / "task.json"
            task = {
                "schema_version": "robot-spatial-edit-task.v1",
                "task_id": "tool-offset-020-to-025",
                "robot": "two_dof_demo",
                "prompt": {
                    "en": "Move tool0 from 0.20 m to 0.25 m along +Z of slider_link and preserve everything else.",
                    "zh_tw": "將 tool0 沿 slider_link 的 +Z 從 0.20 m 改為 0.25 m，其餘保持不變。",
                },
                "inputs": {
                    "urdf": str(FIXTURES / "two_dof.urdf"),
                    "invariants": str(baseline_invariants),
                    "package_map": str(FIXTURES / "package_map.json"),
                },
            }
            task_path.write_text(json.dumps(task))
            key = {
                "schema_version": "robot-spatial-edit-key.v1",
                "task_id": task["task_id"],
                "baseline": {
                    "urdf_sha256": spatial_edit_evaluation.sha256_path(FIXTURES / "two_dof.urdf"),
                    "invariants_sha256": spatial_edit_evaluation.sha256_path(baseline_invariants),
                    "robot": "two_dof_demo",
                    "root_link": "base_link",
                },
                "authorized_urdf_changes": [
                    {
                        "selector": {"joint": "tool_mount", "child_tag": "origin", "attribute": "xyz"},
                        "expected_numeric_vector": [0.0, 0.0, 0.25],
                        "absolute_tolerance": 1e-12,
                    }
                ],
                "authorized_invariant_changes": [
                    {
                        "assertion_id": "tool-offset-from-slider",
                        "field_path": ["expected", "translation_xyz_m"],
                        "expected_value": [0.0, 0.0, 0.25],
                        "absolute_tolerance": 1e-12,
                    }
                ],
                "required_spatial_outcomes": [
                    {
                        "type": "frame_pose",
                        "pose": "bent",
                        "joints": {"shoulder": math.pi / 2.0, "slide": 0.5},
                        "from": "slider_link",
                        "to": "tool0",
                        "expected": {
                            "translation_xyz_m": [0.0, 0.0, 0.25],
                            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                        "translation_tolerance_m": 1e-9,
                        "rotation_tolerance_deg": 1e-7,
                    }
                ],
            }
            key_path = root / "private-key.json"
            key_path.write_text(json.dumps(key))

            passing_urdf = root / "passing.urdf"
            passing_urdf.write_text(
                (FIXTURES / "two_dof.urdf").read_text().replace(
                    '<origin xyz="0 0 0.2" rpy="0 0 0"/>',
                    '<origin xyz="0 0 0.25" rpy="0 0 0"/>',
                )
            )
            passing_contract = json.loads(json.dumps(baseline_contract))
            next(
                assertion
                for assertion in passing_contract["assertions"]
                if assertion["id"] == "tool-offset-from-slider"
            )["expected"]["translation_xyz_m"][2] = 0.25
            passing_invariants = root / "passing-invariants.json"
            passing_invariants.write_text(json.dumps(passing_contract))

            passing = spatial_edit_evaluation.grade_edit(task_path, key_path, passing_urdf, passing_invariants)
            self.assertEqual(passing["status"], "passed", passing)
            self.assertEqual(passing["failed_count"], 0)

            suite_manifest = root / "suite.json"
            suite_manifest.write_text(json.dumps({
                "schema_version": "robot-spatial-edit-suite.v1",
                "suite_id": "single-task-suite",
                "tasks": [{
                    "task_id": task["task_id"],
                    "category": "joint_origin_translation",
                    "task": "task.json",
                    "candidate_urdf": "passing.urdf",
                    "candidate_invariants": "passing-invariants.json",
                }],
            }))
            suite_key = root / "suite-key.json"
            suite_key.write_text(json.dumps({
                "schema_version": "robot-spatial-edit-suite-key.v1",
                "suite_id": "single-task-suite",
                "keys": {task["task_id"]: "private-key.json"},
            }))
            suite = spatial_edit_suite.grade_suite(suite_manifest, suite_key, root)
            self.assertEqual(suite["status"], "passed")
            self.assertEqual(suite["category_summary"]["joint_origin_translation"]["pass_rate"], 1.0)

            unchanged = spatial_edit_evaluation.grade_edit(
                task_path,
                key_path,
                FIXTURES / "two_dof.urdf",
                baseline_invariants,
            )
            self.assertEqual(unchanged["status"], "failed")
            self.assertIn("authorized_urdf_values", unchanged["failed_checks"])
            self.assertIn("required_spatial_outcome_1", unchanged["failed_checks"])
            self.assertIn("authorized_invariant_values", unchanged["failed_checks"])

            collateral_urdf = root / "collateral.urdf"
            collateral_urdf.write_text(
                passing_urdf.read_text().replace(
                    '<origin xyz="1 0 0" rpy="0 0 0"/>',
                    '<origin xyz="1.1 0 0" rpy="0 0 0"/>',
                    1,
                )
            )
            collateral = spatial_edit_evaluation.grade_edit(
                task_path,
                key_path,
                collateral_urdf,
                passing_invariants,
            )
            self.assertEqual(collateral["status"], "failed")
            self.assertIn("urdf_change_allowlist", collateral["failed_checks"])

    def test_blind_edit_grader_supports_generic_joint_selector_and_axis_outcome(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_invariants = root / "invariants.json"
            baseline_invariants.write_text(json.dumps({
                "schema_version": "robot-spatial-invariants.v1",
                "robot": "two_dof_demo",
                "default_tolerances": {"axis_deg": 1e-7},
                "poses": {"bent": {"joints": {"shoulder": math.pi / 2.0, "slide": 0.5}}},
                "assertions": [{
                    "id": "slide-axis-in-base",
                    "type": "joint_axis",
                    "pose": "bent",
                    "joint": "slide",
                    "frame": "base_link",
                    "expected_unit_vector": [0.0, 1.0, 0.0],
                }],
            }))
            task_path = root / "task.json"
            task_path.write_text(json.dumps({
                "schema_version": "robot-spatial-edit-task.v1",
                "task_id": "slide-axis-x-to-y",
                "robot": "two_dof_demo",
                "inputs": {"urdf": str(FIXTURES / "two_dof.urdf"), "invariants": str(baseline_invariants)},
            }))
            key_path = root / "key.json"
            key_path.write_text(json.dumps({
                "schema_version": "robot-spatial-edit-key.v1",
                "task_id": "slide-axis-x-to-y",
                "baseline": {
                    "urdf_sha256": spatial_edit_evaluation.sha256_path(FIXTURES / "two_dof.urdf"),
                    "invariants_sha256": spatial_edit_evaluation.sha256_path(baseline_invariants),
                    "robot": "two_dof_demo",
                    "root_link": "base_link",
                },
                "authorized_urdf_changes": [{
                    "selector": {
                        "entity": {"tag": "joint", "name": "slide"},
                        "path": [{"tag": "axis", "index": 0}],
                        "attribute": "xyz",
                    },
                    "expected_numeric_vector": [0.0, 1.0, 0.0],
                }],
                "authorized_invariant_changes": [{
                    "assertion_id": "slide-axis-in-base",
                    "field_path": ["expected_unit_vector"],
                    "expected_value": [-1.0, 0.0, 0.0],
                }],
                "required_spatial_outcomes": [{
                    "type": "joint_axis",
                    "pose": "bent",
                    "joints": {"shoulder": math.pi / 2.0, "slide": 0.5},
                    "joint": "slide",
                    "frame": "base_link",
                    "expected_unit_vector": [-1.0, 0.0, 0.0],
                }],
            }))
            candidate_urdf = root / "candidate.urdf"
            candidate_urdf.write_text(
                (FIXTURES / "two_dof.urdf").read_text().replace(
                    '<axis xyz="1 0 0"/>',
                    '<axis xyz="0 1 0"/>',
                )
            )
            candidate_invariants = root / "candidate-invariants.json"
            candidate_contract = json.loads(baseline_invariants.read_text())
            candidate_contract["assertions"][0]["expected_unit_vector"] = [-1.0, 0.0, 0.0]
            candidate_invariants.write_text(json.dumps(candidate_contract))

            report = spatial_edit_evaluation.grade_edit(task_path, key_path, candidate_urdf, candidate_invariants)
            self.assertEqual(report["status"], "passed")
            outcome = next(check for check in report["checks"] if check["id"] == "required_spatial_outcome_1")
            self.assertEqual(outcome["outcome_type"], "joint_axis")
            self.assertVectorAlmostEqual(outcome["actual"]["unit_vector"], [-1.0, 0.0, 0.0])

    def test_blind_edit_grader_supports_nested_geometry_selector_and_aabb_outcome(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_urdf = root / "box.urdf"
            baseline_urdf.write_text("""<?xml version=\"1.0\"?>
<robot name=\"box_robot\">
  <link name=\"base\">
    <collision><origin xyz=\"0 0 0\" rpy=\"0 0 0\"/><geometry><box size=\"1 1 1\"/></geometry></collision>
  </link>
</robot>
""")
            baseline_invariants = root / "invariants.json"
            baseline_invariants.write_text(json.dumps({
                "schema_version": "robot-spatial-invariants.v1",
                "robot": "box_robot",
                "default_tolerances": {"aabb_m": 1e-9},
                "assertions": [{
                    "id": "box-bounds",
                    "type": "geometry_aabb",
                    "geometry_frame": "collision/base/0",
                    "expected": {"min_xyz_m": [-0.5, -0.5, -0.5], "max_xyz_m": [0.5, 0.5, 0.5]},
                }],
            }))
            protected_asset = root / "shape-source.bin"
            protected_asset.write_bytes(b"baseline geometry source")
            task_path = root / "task.json"
            task_path.write_text(json.dumps({
                "schema_version": "robot-spatial-edit-task.v1",
                "task_id": "resize-box",
                "robot": "box_robot",
                "inputs": {
                    "urdf": str(baseline_urdf),
                    "invariants": str(baseline_invariants),
                    "protected_files": [str(protected_asset)],
                },
            }))
            key_path = root / "key.json"
            key_path.write_text(json.dumps({
                "schema_version": "robot-spatial-edit-key.v1",
                "task_id": "resize-box",
                "baseline": {
                    "urdf_sha256": spatial_edit_evaluation.sha256_path(baseline_urdf),
                    "invariants_sha256": spatial_edit_evaluation.sha256_path(baseline_invariants),
                    "robot": "box_robot",
                    "root_link": "base",
                    "protected_file_sha256": {
                        str(protected_asset): spatial_edit_evaluation.sha256_path(protected_asset),
                    },
                },
                "authorized_urdf_changes": [{
                    "selector": {
                        "entity": {"tag": "link", "name": "base"},
                        "path": [{"tag": "collision"}, {"tag": "geometry"}, {"tag": "box"}],
                        "attribute": "size",
                    },
                    "expected_numeric_vector": [1.2, 0.8, 0.6],
                }],
                "authorized_invariant_changes": [
                    {"assertion_id": "box-bounds", "field_path": ["expected", "min_xyz_m"], "expected_value": [-0.6, -0.4, -0.3]},
                    {"assertion_id": "box-bounds", "field_path": ["expected", "max_xyz_m"], "expected_value": [0.6, 0.4, 0.3]},
                ],
                "required_spatial_outcomes": [{
                    "type": "geometry_aabb",
                    "pose": "zero",
                    "joints": {},
                    "geometry_frame": "collision/base/0",
                    "expected": {"min_xyz_m": [-0.6, -0.4, -0.3], "max_xyz_m": [0.6, 0.4, 0.3]},
                }],
            }))
            candidate_urdf = root / "candidate.urdf"
            candidate_urdf.write_text(baseline_urdf.read_text().replace('size="1 1 1"', 'size="1.2 0.8 0.6"'))
            candidate_invariants = root / "candidate-invariants.json"
            candidate_contract = json.loads(baseline_invariants.read_text())
            candidate_contract["assertions"][0]["expected"] = {
                "min_xyz_m": [-0.6, -0.4, -0.3],
                "max_xyz_m": [0.6, 0.4, 0.3],
            }
            candidate_invariants.write_text(json.dumps(candidate_contract))

            report = spatial_edit_evaluation.grade_edit(task_path, key_path, candidate_urdf, candidate_invariants)
            self.assertEqual(report["status"], "passed")
            outcome = next(check for check in report["checks"] if check["id"] == "required_spatial_outcome_1")
            self.assertEqual(outcome["outcome_type"], "geometry_aabb")
            self.assertEqual(outcome["actual"]["max_xyz_m"], [0.6, 0.4, 0.3])

            protected_asset.write_bytes(b"tampered geometry source")
            tampered = spatial_edit_evaluation.grade_edit(task_path, key_path, candidate_urdf, candidate_invariants)
            self.assertEqual(tampered["status"], "failed")
            self.assertEqual(tampered["failed_checks"], ["baseline_integrity"])

    def test_typed_graph_change_set_adds_and_removes_a_valid_leaf_branch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            added_urdf = root / "with-camera.urdf"
            add_change = root / "add-camera.json"
            add_change.write_text(json.dumps({
                "schema_version": "robot-spatial-graph-change-set.v1",
                "change_set_id": "add-camera-frame",
                "robot": "two_dof_demo",
                "baseline_urdf_sha256": spatial_edit_evaluation.sha256_path(FIXTURES / "two_dof.urdf"),
                "operations": [{
                    "operation_id": "add-camera-leaf",
                    "type": "add_leaf_link",
                    "parent_link": "tool0",
                    "new_link": "camera_link",
                    "new_joint": "camera_mount",
                    "joint_type": "fixed",
                    "origin": {
                        "xyz_m": [0.1, 0.0, 0.05],
                        "rpy_rad": [0.0, 0.0, math.pi / 2.0],
                    },
                }],
            }))
            added = spatial_graph_edit.apply_graph_change(FIXTURES / "two_dof.urdf", add_change, added_urdf)
            self.assertEqual(added["status"], "applied_and_validated")
            self.assertEqual(added["topology_delta"]["added_links"], ["camera_link"])
            self.assertEqual(added["topology_delta"]["added_joints"], ["camera_mount"])
            added_model = MODULE.RobotModel(added_urdf)
            camera_pose = MODULE.pose_record(added_model.transform("tool0", "camera_link", {}))
            self.assertVectorAlmostEqual(camera_pose["translation_xyz_m"], [0.1, 0.0, 0.05])
            self.assertVectorAlmostEqual(camera_pose["quaternion_xyzw"], [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])

            removed_urdf = root / "camera-removed.urdf"
            remove_change = root / "remove-camera.json"
            remove_change.write_text(json.dumps({
                "schema_version": "robot-spatial-graph-change-set.v1",
                "change_set_id": "remove-camera-frame",
                "robot": "two_dof_demo",
                "baseline_urdf_sha256": spatial_edit_evaluation.sha256_path(added_urdf),
                "operations": [{
                    "operation_id": "remove-camera-leaf",
                    "type": "remove_leaf_link",
                    "link": "camera_link",
                    "expected_parent_link": "tool0",
                    "expected_parent_joint": "camera_mount",
                }],
            }))
            removed = spatial_graph_edit.apply_graph_change(added_urdf, remove_change, removed_urdf)
            self.assertEqual(removed["topology_delta"]["removed_links"], ["camera_link"])
            self.assertEqual(removed["topology_delta"]["removed_joints"], ["camera_mount"])
            removed_model = MODULE.RobotModel(removed_urdf)
            self.assertEqual(set(removed_model.links), set(self.model.links))
            self.assertEqual(set(removed_model.joints), set(self.model.joints))

    def test_typed_graph_change_set_adds_and_removes_a_complete_articulated_subtree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            added_urdf = root / "with-gimbal.urdf"
            add_change = root / "add-gimbal.json"
            add_change.write_text(json.dumps({
                "schema_version": "robot-spatial-graph-change-set.v1",
                "change_set_id": "add-articulated-camera-subtree",
                "robot": "two_dof_demo",
                "baseline_urdf_sha256": spatial_edit_evaluation.sha256_path(FIXTURES / "two_dof.urdf"),
                "operations": [{
                    "operation_id": "add-gimbal-subtree",
                    "type": "add_subtree",
                    "root_link": "gimbal_base",
                    "expected_parent_link": "tool0",
                    "links": [
                        {"name": "gimbal_base"},
                        {
                            "name": "camera_link",
                            "element_xml": (
                                '<link name="camera_link"><collision name="camera_body">'
                                '<origin xyz="0.05 0 0" rpy="0 0 0" />'
                                '<geometry><box size="0.1 0.04 0.06" /></geometry>'
                                '</collision></link>'
                            ),
                        },
                    ],
                    "joints": [
                        {
                            "name": "gimbal_mount",
                            "joint_type": "fixed",
                            "parent_link": "tool0",
                            "child_link": "gimbal_base",
                            "origin": {"xyz_m": [0.1, 0.0, 0.05], "rpy_rad": [0.0, 0.0, 0.0]},
                        },
                        {
                            "name": "camera_yaw",
                            "joint_type": "revolute",
                            "parent_link": "gimbal_base",
                            "child_link": "camera_link",
                            "origin": {"xyz_m": [0.0, 0.0, 0.1], "rpy_rad": [0.0, 0.0, 0.0]},
                            "axis_xyz": [0.0, 0.0, 1.0],
                            "limit": {
                                "lower": -math.pi / 2.0,
                                "upper": math.pi / 2.0,
                                "effort": 2.0,
                                "velocity": 1.5,
                            },
                        },
                    ],
                }],
            }))
            added = spatial_graph_edit.apply_graph_change(FIXTURES / "two_dof.urdf", add_change, added_urdf)
            self.assertEqual(added["topology_delta"]["added_links"], ["camera_link", "gimbal_base"])
            self.assertEqual(added["topology_delta"]["added_joints"], ["camera_yaw", "gimbal_mount"])
            self.assertEqual(added["operations"][0]["added_subtree"], {
                "links": ["camera_link", "gimbal_base"],
                "joints": ["camera_yaw", "gimbal_mount"],
            })

            added_model = MODULE.RobotModel(added_urdf)
            chain = added_model.chain("tool0", "camera_link")
            self.assertEqual(chain["links"], ["tool0", "gimbal_base", "camera_link"])
            self.assertEqual([step["joint"] for step in chain["steps"]], ["gimbal_mount", "camera_yaw"])
            pose = {"shoulder": math.pi / 2.0, "slide": 0.5, "camera_yaw": math.pi / 2.0}
            camera_pose = MODULE.pose_record(added_model.transform("base_link", "camera_link", pose))
            self.assertVectorAlmostEqual(camera_pose["translation_xyz_m"], [1.0, 1.6, 0.35])
            self.assertVectorAlmostEqual(camera_pose["quaternion_xyzw"], [0.0, 0.0, 1.0, 0.0])
            self.assertVectorAlmostEqual(added_model.axis("camera_yaw", "base_link", pose), [0.0, 0.0, 1.0])
            geometry, _, _ = added_model.geometry_analysis(pose)
            camera_bounds = geometry["collision/camera_link/0"]["bounds_in_root_frame_at_pose"]
            self.assertVectorAlmostEqual(camera_bounds["min_xyz_m"], [0.9, 1.58, 0.32])
            self.assertVectorAlmostEqual(camera_bounds["max_xyz_m"], [1.0, 1.62, 0.38])

            remove_data = {
                "schema_version": "robot-spatial-graph-change-set.v1",
                "change_set_id": "remove-articulated-camera-subtree",
                "robot": "two_dof_demo",
                "baseline_urdf_sha256": spatial_edit_evaluation.sha256_path(added_urdf),
                "operations": [{
                    "operation_id": "remove-gimbal-subtree",
                    "type": "remove_subtree",
                    "root_link": "gimbal_base",
                    "expected_parent_link": "tool0",
                    "expected_parent_joint": "gimbal_mount",
                    "expected_subtree": {
                        "links": ["gimbal_base", "camera_link"],
                        "joints": ["gimbal_mount", "camera_yaw"],
                    },
                }],
            }

            incomplete_data = json.loads(json.dumps(remove_data))
            incomplete_data["change_set_id"] = "reject-incomplete-membership"
            incomplete_data["operations"][0]["expected_subtree"] = {
                "links": ["gimbal_base"],
                "joints": ["gimbal_mount"],
            }
            incomplete_change = root / "incomplete-remove.json"
            incomplete_change.write_text(json.dumps(incomplete_data))
            with self.assertRaisesRegex(spatial_graph_edit.GraphEditError, "expected_subtree does not match"):
                spatial_graph_edit.apply_graph_change(added_urdf, incomplete_change, root / "incomplete.urdf")

            externally_referenced = root / "externally-referenced.urdf"
            externally_referenced.write_text(
                added_urdf.read_text().replace(
                    "</robot>",
                    '<gazebo reference="camera_link"><sensor name="external_camera" /></gazebo>\n</robot>',
                )
            )
            external_data = json.loads(json.dumps(remove_data))
            external_data["change_set_id"] = "reject-external-reference"
            external_data["baseline_urdf_sha256"] = spatial_edit_evaluation.sha256_path(externally_referenced)
            external_change = root / "external-remove.json"
            external_change.write_text(json.dumps(external_data))
            with self.assertRaisesRegex(spatial_graph_edit.GraphEditError, "would leave external references"):
                spatial_graph_edit.apply_graph_change(
                    externally_referenced,
                    external_change,
                    root / "externally-broken.urdf",
                )

            remove_change = root / "remove-gimbal.json"
            remove_change.write_text(json.dumps(remove_data))
            removed_urdf = root / "gimbal-removed.urdf"
            removed = spatial_graph_edit.apply_graph_change(added_urdf, remove_change, removed_urdf)
            self.assertEqual(removed["topology_delta"]["removed_links"], ["camera_link", "gimbal_base"])
            self.assertEqual(removed["topology_delta"]["removed_joints"], ["camera_yaw", "gimbal_mount"])
            removed_model = MODULE.RobotModel(removed_urdf)
            self.assertEqual(set(removed_model.links), set(self.model.links))
            self.assertEqual(set(removed_model.joints), set(self.model.joints))

    def test_typed_graph_change_set_reparents_a_complete_non_leaf_subtree_and_rejects_cycles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            change_set = root / "reparent.json"
            change = {
                "schema_version": "robot-spatial-graph-change-set.v1",
                "change_set_id": "move-slider-subtree",
                "robot": "two_dof_demo",
                "baseline_urdf_sha256": spatial_edit_evaluation.sha256_path(FIXTURES / "two_dof.urdf"),
                "operations": [{
                    "operation_id": "reparent-slider",
                    "type": "reparent_subtree",
                    "joint": "slide",
                    "child_link": "slider_link",
                    "expected_parent_link": "arm_link",
                    "expected_joint_type": "prismatic",
                    "new_parent_link": "base_link",
                    "new_origin": {
                        "xyz_m": [0.0, 1.0, 0.0],
                        "rpy_rad": [0.0, 0.0, 0.0],
                    },
                    "expected_subtree": {
                        "links": ["slider_link", "tool0"],
                        "joints": ["slide", "tool_mount"],
                    },
                }],
            }
            change_set.write_text(json.dumps(change))
            output_urdf = root / "reparented.urdf"
            report = spatial_graph_edit.apply_graph_change(FIXTURES / "two_dof.urdf", change_set, output_urdf)
            self.assertEqual(report["operations"][0]["moved_subtree"], {
                "links": ["slider_link", "tool0"],
                "joints": ["slide", "tool_mount"],
            })
            self.assertEqual(report["topology_delta"]["changed_edges"], [{
                "before": {
                    "joint": "slide",
                    "type": "prismatic",
                    "parent_link": "arm_link",
                    "child_link": "slider_link",
                },
                "after": {
                    "joint": "slide",
                    "type": "prismatic",
                    "parent_link": "base_link",
                    "child_link": "slider_link",
                },
            }])
            model = MODULE.RobotModel(output_urdf)
            chain = model.chain("base_link", "tool0")
            self.assertEqual(chain["links"], ["base_link", "slider_link", "tool0"])
            self.assertEqual([step["joint"] for step in chain["steps"]], ["slide", "tool_mount"])
            self.assertEqual(model.affected_by_joint("shoulder")["affected_links"], ["arm_link"])
            pose = {"shoulder": math.pi / 2.0, "slide": 0.5}
            actual = MODULE.pose_record(model.transform("base_link", "tool0", pose))
            self.assertVectorAlmostEqual(actual["translation_xyz_m"], [0.5, 1.0, 0.2])

            cyclic = json.loads(json.dumps(change))
            cyclic["change_set_id"] = "invalid-cycle"
            cyclic["operations"][0]["new_parent_link"] = "tool0"
            cyclic_path = root / "cyclic.json"
            cyclic_path.write_text(json.dumps(cyclic))
            with self.assertRaisesRegex(spatial_graph_edit.GraphEditError, "inside the moved subtree"):
                spatial_graph_edit.apply_graph_change(
                    FIXTURES / "two_dof.urdf",
                    cyclic_path,
                    root / "must-not-exist.urdf",
                )

    def test_blind_edit_grader_requires_reproducible_graph_change_and_exact_topology(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_invariants = root / "baseline-invariants.json"
            baseline_contract = json.loads((FIXTURES / "invariants.json").read_text())
            baseline_contract["assertions"] = [
                assertion
                for assertion in baseline_contract["assertions"]
                if assertion["id"] != "shoulder-causal-subtree"
            ]
            baseline_invariants.write_text(json.dumps(baseline_contract))
            task_path = root / "task.json"
            task_path.write_text(json.dumps({
                "schema_version": "robot-spatial-edit-task.v1",
                "task_id": "add-camera-branch",
                "robot": "two_dof_demo",
                "inputs": {
                    "urdf": str(FIXTURES / "two_dof.urdf"),
                    "invariants": str(baseline_invariants),
                    "package_map": str(FIXTURES / "package_map.json"),
                },
            }))
            change_set = root / "change-set.json"
            change_set.write_text(json.dumps({
                "schema_version": "robot-spatial-graph-change-set.v1",
                "change_set_id": "add-camera-frame",
                "robot": "two_dof_demo",
                "baseline_urdf_sha256": spatial_edit_evaluation.sha256_path(FIXTURES / "two_dof.urdf"),
                "operations": [{
                    "operation_id": "add-camera-leaf",
                    "type": "add_leaf_link",
                    "parent_link": "tool0",
                    "new_link": "camera_link",
                    "new_joint": "camera_mount",
                    "joint_type": "fixed",
                    "origin": {
                        "xyz_m": [0.1, 0.0, 0.05],
                        "rpy_rad": [0.0, 0.0, math.pi / 2.0],
                    },
                }],
            }))
            candidate_urdf = root / "candidate.urdf"
            spatial_graph_edit.apply_graph_change(FIXTURES / "two_dof.urdf", change_set, candidate_urdf)
            camera_assertion = {
                "id": "camera-mount-pose",
                "type": "frame_pose",
                "pose": "bent",
                "from": "tool0",
                "to": "camera_link",
                "expected": {
                    "translation_xyz_m": [0.1, 0.0, 0.05],
                    "quaternion_xyzw": [0.0, 0.0, 0.707106781187, 0.707106781187],
                },
            }
            candidate_contract = json.loads(json.dumps(baseline_contract))
            candidate_contract["assertions"].append(camera_assertion)
            candidate_invariants = root / "candidate-invariants.json"
            candidate_invariants.write_text(json.dumps(candidate_contract))
            baseline_model = MODULE.RobotModel(FIXTURES / "two_dof.urdf")
            key_path = root / "key.json"
            key_path.write_text(json.dumps({
                "schema_version": "robot-spatial-edit-key.v1",
                "task_id": "add-camera-branch",
                "require_graph_change_set": True,
                "baseline": {
                    "urdf_sha256": spatial_edit_evaluation.sha256_path(FIXTURES / "two_dof.urdf"),
                    "invariants_sha256": spatial_edit_evaluation.sha256_path(baseline_invariants),
                    "robot": "two_dof_demo",
                    "root_link": "base_link",
                },
                "authorized_urdf_changes": [],
                "authorized_urdf_element_additions": [
                    {"tag": "link", "name": "camera_link", "element_xml": '<link name="camera_link"/>'},
                    {
                        "tag": "joint",
                        "name": "camera_mount",
                        "element_xml": (
                            '<joint name="camera_mount" type="fixed"><parent link="tool0"/>'
                            '<child link="camera_link"/><origin xyz="0.1 0 0.05" '
                            'rpy="0 0 1.5707963267948966"/></joint>'
                        ),
                    },
                ],
                "authorized_invariant_changes": [],
                "authorized_invariant_additions": [camera_assertion],
                "required_spatial_outcomes": [{
                    "type": "topology",
                    "expected": {
                        "root_link": "base_link",
                        "links": [*baseline_model.links, "camera_link"],
                        "joints": [*baseline_model.joints, "camera_mount"],
                    },
                }],
            }))

            passing = spatial_edit_evaluation.grade_edit(
                task_path,
                key_path,
                candidate_urdf,
                candidate_invariants,
                change_set,
            )
            self.assertEqual(passing["status"], "passed", passing)
            topology = next(check for check in passing["checks"] if check["id"] == "required_spatial_outcome_1")
            self.assertEqual(topology["outcome_type"], "topology")

            collateral_urdf = root / "collateral.urdf"
            collateral_urdf.write_text(candidate_urdf.read_text().replace(
                '<origin xyz="1 0 0" rpy="0 0 0" />',
                '<origin xyz="1.1 0 0" rpy="0 0 0" />',
                1,
            ))
            collateral = spatial_edit_evaluation.grade_edit(
                task_path,
                key_path,
                collateral_urdf,
                candidate_invariants,
                change_set,
            )
            self.assertEqual(collateral["status"], "failed")
            self.assertIn("graph_change_set_reproduces_candidate", collateral["failed_checks"])
            self.assertIn("urdf_change_allowlist", collateral["failed_checks"])

    def test_blind_edit_grader_verifies_reparented_edge_graph_and_joint_replacement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_contract = {
                "schema_version": "robot-spatial-invariants.v1",
                "robot": "two_dof_demo",
                "assertions": [
                    {
                        "id": "tool-chain",
                        "type": "chain",
                        "from_link": "base_link",
                        "to_link": "tool0",
                        "expected": {
                            "links": ["base_link", "arm_link", "slider_link", "tool0"],
                            "joints": ["shoulder", "slide", "tool_mount"],
                        },
                    },
                    {
                        "id": "shoulder-causal-subtree",
                        "type": "affected_links",
                        "joint": "shoulder",
                        "expected_links": ["arm_link", "slider_link", "tool0"],
                    },
                ],
            }
            baseline_invariants = root / "baseline-invariants.json"
            baseline_invariants.write_text(json.dumps(baseline_contract))
            task_path = root / "task.json"
            task_path.write_text(json.dumps({
                "schema_version": "robot-spatial-edit-task.v1",
                "task_id": "reparent-slider-subtree",
                "robot": "two_dof_demo",
                "inputs": {
                    "urdf": str(FIXTURES / "two_dof.urdf"),
                    "invariants": str(baseline_invariants),
                },
            }))
            change_set = root / "change-set.json"
            change_set.write_text(json.dumps({
                "schema_version": "robot-spatial-graph-change-set.v1",
                "change_set_id": "move-slider-subtree",
                "robot": "two_dof_demo",
                "baseline_urdf_sha256": spatial_edit_evaluation.sha256_path(FIXTURES / "two_dof.urdf"),
                "operations": [{
                    "operation_id": "reparent-slider",
                    "type": "reparent_subtree",
                    "joint": "slide",
                    "child_link": "slider_link",
                    "expected_parent_link": "arm_link",
                    "expected_joint_type": "prismatic",
                    "new_parent_link": "base_link",
                    "new_origin": {"xyz_m": [0.0, 1.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0]},
                    "expected_subtree": {
                        "links": ["slider_link", "tool0"],
                        "joints": ["slide", "tool_mount"],
                    },
                }],
            }))
            candidate_urdf = root / "candidate.urdf"
            spatial_graph_edit.apply_graph_change(FIXTURES / "two_dof.urdf", change_set, candidate_urdf)
            candidate_contract = json.loads(json.dumps(baseline_contract))
            candidate_contract["assertions"][0]["expected"] = {
                "links": ["base_link", "slider_link", "tool0"],
                "joints": ["slide", "tool_mount"],
            }
            candidate_contract["assertions"][1]["expected_links"] = ["arm_link"]
            candidate_invariants = root / "candidate-invariants.json"
            candidate_invariants.write_text(json.dumps(candidate_contract))
            key_path = root / "key.json"
            key_path.write_text(json.dumps({
                "schema_version": "robot-spatial-edit-key.v1",
                "task_id": "reparent-slider-subtree",
                "require_graph_change_set": True,
                "baseline": {
                    "urdf_sha256": spatial_edit_evaluation.sha256_path(FIXTURES / "two_dof.urdf"),
                    "invariants_sha256": spatial_edit_evaluation.sha256_path(baseline_invariants),
                    "robot": "two_dof_demo",
                    "root_link": "base_link",
                },
                "authorized_urdf_element_replacements": [{
                    "tag": "joint",
                    "name": "slide",
                    "element_xml": (
                        '<joint name="slide" type="prismatic"><parent link="base_link"/>'
                        '<child link="slider_link"/><origin xyz="0 1 0" rpy="0 0 0"/>'
                        '<axis xyz="1 0 0"/><limit lower="0" upper="1" effort="10" velocity="1"/></joint>'
                    ),
                }],
                "authorized_invariant_changes": [
                    {
                        "assertion_id": "tool-chain",
                        "field_path": ["expected", "links"],
                        "expected_value": ["base_link", "slider_link", "tool0"],
                    },
                    {
                        "assertion_id": "tool-chain",
                        "field_path": ["expected", "joints"],
                        "expected_value": ["slide", "tool_mount"],
                    },
                    {
                        "assertion_id": "shoulder-causal-subtree",
                        "field_path": ["expected_links"],
                        "expected_value": ["arm_link"],
                    },
                ],
                "required_spatial_outcomes": [
                    {
                        "type": "topology",
                        "expected": {
                            "root_link": "base_link",
                            "links": ["base_link", "arm_link", "slider_link", "tool0"],
                            "joints": ["shoulder", "slide", "tool_mount"],
                            "edges": [
                                {"joint": "shoulder", "type": "revolute", "parent_link": "base_link", "child_link": "arm_link"},
                                {"joint": "slide", "type": "prismatic", "parent_link": "base_link", "child_link": "slider_link"},
                                {"joint": "tool_mount", "type": "fixed", "parent_link": "slider_link", "child_link": "tool0"},
                            ],
                        },
                    },
                    {
                        "type": "frame_pose",
                        "pose": "bent",
                        "joints": {"shoulder": math.pi / 2.0, "slide": 0.5},
                        "from": "base_link",
                        "to": "tool0",
                        "expected": {
                            "translation_xyz_m": [0.5, 1.0, 0.2],
                            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                ],
            }))

            passing = spatial_edit_evaluation.grade_edit(
                task_path,
                key_path,
                candidate_urdf,
                candidate_invariants,
                change_set,
            )
            self.assertEqual(passing["status"], "passed", passing)
            topology = next(check for check in passing["checks"] if check["id"] == "required_spatial_outcome_1")
            self.assertEqual(topology["actual"]["edges"][1]["parent_link"], "base_link")

            unchanged = spatial_edit_evaluation.grade_edit(
                task_path,
                key_path,
                FIXTURES / "two_dof.urdf",
                baseline_invariants,
                change_set,
            )
            self.assertEqual(unchanged["status"], "failed")
            self.assertIn("graph_change_set_reproduces_candidate", unchanged["failed_checks"])
            self.assertIn("authorized_urdf_elements", unchanged["failed_checks"])
            self.assertIn("required_spatial_outcome_1", unchanged["failed_checks"])
            self.assertIn("authorized_invariant_values", unchanged["failed_checks"])

    def test_invariant_cli_is_a_ci_gate_and_export_grounding_layer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "invariants-report.json"
            passed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "check-invariants",
                    str(FIXTURES / "two_dof.urdf"),
                    "--contract",
                    str(FIXTURES / "invariants.json"),
                    "--package-map",
                    str(FIXTURES / "package_map.json"),
                    "--out",
                    str(report_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(passed.returncode, 0)
            self.assertEqual(json.loads(passed.stdout)["status"], "passed")
            self.assertEqual(json.loads(report_path.read_text())["passed_count"], 11)

            bad_contract = json.loads((FIXTURES / "invariants.json").read_text())
            bad_contract["assertions"][0]["expected"]["translation_xyz_m"][2] = 0.25
            bad_path = Path(temp_dir) / "bad-invariants.json"
            bad_path.write_text(json.dumps(bad_contract))
            failed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "check-invariants",
                    str(FIXTURES / "two_dof.urdf"),
                    "--contract",
                    str(bad_path),
                    "--package-map",
                    str(FIXTURES / "package_map.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(failed.returncode, 1)
            self.assertEqual(json.loads(failed.stdout)["failed_count"], 1)

            export_dir = Path(temp_dir) / "export"
            exported = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "export",
                    str(FIXTURES / "two_dof.urdf"),
                    "--pose",
                    str(FIXTURES / "bent_pose.json"),
                    "--invariants",
                    str(FIXTURES / "invariants.json"),
                    "--package-map",
                    str(FIXTURES / "package_map.json"),
                    "--out",
                    str(export_dir),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(exported.returncode, 0, exported.stderr)
            model = json.loads((export_dir / "model.json").read_text())
            self.assertEqual(model["invariant_validation"]["status"], "passed")
            self.assertEqual(model["artifacts"]["invariant_report"]["path"], "invariants-report.json")
            predicates = {json.loads(line)["predicate"] for line in (export_dir / "facts.jsonl").read_text().splitlines()}
            self.assertIn("has_spatial_invariant_contract_status", predicates)
            self.assertIn("has_validation_result", predicates)
            self.assertIn("## Project spatial invariants", (export_dir / "context.md").read_text())

            failed_export_dir = Path(temp_dir) / "failed-export"
            failed_export = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "export",
                    str(FIXTURES / "two_dof.urdf"),
                    "--invariants",
                    str(bad_path),
                    "--package-map",
                    str(FIXTURES / "package_map.json"),
                    "--out",
                    str(failed_export_dir),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(failed_export.returncode, 1)
            self.assertEqual(json.loads(failed_export.stdout)["status"], "exported_with_failed_invariants")
            self.assertEqual(json.loads((failed_export_dir / "invariants-report.json").read_text())["failed_count"], 1)

    def test_srdf_groups_named_pose_and_end_effector_are_validated(self):
        srdf = MODULE.parse_srdf(FIXTURES / "two_dof.srdf", self.model)
        self.assertIsNotNone(srdf)
        assert srdf is not None
        self.assertEqual(srdf["groups"]["manipulator"]["expanded_joints"], ["shoulder", "slide", "tool_mount"])
        self.assertEqual(srdf["groups"]["manipulator"]["expanded_links"], ["base_link", "arm_link", "slider_link", "tool0"])
        self.assertEqual(srdf["named_poses"]["manipulator/home"]["joints"], {"shoulder": 0.25, "slide": 0.3})
        self.assertEqual(srdf["end_effectors"]["demo_tool"]["component_group"], "tool")
        pose_name, pose = MODULE.resolve_named_pose(srdf, "home")
        self.assertEqual(pose_name, "manipulator/home")
        self.assertEqual(pose, {"shoulder": 0.25, "slide": 0.3})

    def test_srdf_group_scoped_passive_mimic_joint_is_membership_and_consistency_checked(self):
        model = MODULE.RobotModel(FIXTURES / "mimic_branch.urdf")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mimic.srdf"
            source = (
                '<robot name="mimic_branch">'
                '<group name="coupled"><joint name="driver"/><passive_joint name="follower"/></group>'
                '<group_state name="center" group="coupled">'
                '<joint name="driver" value="0.2"/><joint name="follower" value="0.0"/>'
                '</group_state></robot>'
            )
            path.write_text(source)
            srdf = MODULE.parse_srdf(path, model)
            assert srdf is not None
            self.assertEqual(srdf["groups"]["coupled"]["expanded_joints"], ["driver", "follower"])
            self.assertEqual(srdf["passive_joints"], ["follower"])
            self.assertEqual(srdf["named_poses"]["coupled/center"]["joints"]["follower"], 0.0)

            path.write_text(source.replace('name="follower" value="0.0"', 'name="follower" value="0.05"'))
            with self.assertRaisesRegex(MODULE.SRDFError, "declared mimic relation resolves"):
                MODULE.parse_srdf(path, model)

    def test_srdf_robot_identity_must_match_urdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid = Path(temp_dir) / "wrong.srdf"
            invalid.write_text((FIXTURES / "two_dof.srdf").read_text().replace('name="two_dof_demo"', 'name="different_robot"', 1))
            with self.assertRaisesRegex(MODULE.SRDFError, "does not match"):
                MODULE.parse_srdf(invalid, self.model)

    def test_joint_limit_is_enforced(self):
        with self.assertRaisesRegex(MODULE.SpatialError, "above upper limit"):
            self.model.world_frames({"slide": 1.1})

    def test_invalid_zero_mesh_scale_is_rejected_during_urdf_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid = Path(temp_dir) / "invalid.urdf"
            invalid.write_text((FIXTURES / "two_dof.urdf").read_text().replace('scale="1 1 1"', 'scale="0 1 1"'))
            with self.assertRaisesRegex(MODULE.SpatialError, "scale components must be non-zero"):
                MODULE.RobotModel(invalid)

    def test_misplaced_non_geometry_child_is_tolerated_with_explicit_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "misplaced.urdf"
            path.write_text('''<robot name="misplaced"><link name="base"><visual><geometry><box size="1 1 1"/><material name="grey"/></geometry></visual></link></robot>''')
            model = MODULE.RobotModel(path)
            self.assertEqual(model.links["base"]["visuals"][0]["geometry"]["type"], "box")
            self.assertIn("misplaced non-geometry", model.warnings()[0])

    def test_multiple_recognized_geometry_primitives_remain_an_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ambiguous.urdf"
            path.write_text('''<robot name="ambiguous"><link name="base"><visual><geometry><box size="1 1 1"/><sphere radius="1"/></geometry></visual></link></robot>''')
            with self.assertRaisesRegex(MODULE.SpatialError, "exactly one recognized"):
                MODULE.RobotModel(path)

    def test_cli_exports_ai_context_and_machine_truth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "export", str(FIXTURES / "two_dof.urdf"), "--pose", str(FIXTURES / "bent_pose.json"), "--out", temp_dir],
                check=True,
                capture_output=True,
                text=True,
            )
            response = json.loads(result.stdout)
            self.assertEqual(response["status"], "exported")
            model = json.loads((Path(temp_dir) / "model.json").read_text())
            context = (Path(temp_dir) / "context.md").read_text()
            facts = [json.loads(line) for line in (Path(temp_dir) / "facts.jsonl").read_text().splitlines()]
            manifest = json.loads((Path(temp_dir) / "agent-context.json").read_text())
            entity_index = json.loads((Path(temp_dir) / "entity-index.json").read_text())
            cards = [json.loads(line) for line in (Path(temp_dir) / "entity-cards.jsonl").read_text().splitlines()]
            self.assertEqual(model["pose"]["name"], "bent")
            self.assertIn("## Trust boundary", context)
            self.assertIn("`joint/shoulder`", context)
            self.assertIn("connects_links", {fact["predicate"] for fact in facts})
            self.assertIn("can_change_pose_of_link", {fact["predicate"] for fact in facts})
            self.assertIn("has_declared_mass_properties", {fact["predicate"] for fact in facts})
            self.assertEqual(model["physical_analysis"]["declared_mass_properties"]["declared_mass_kg"], 2.0)
            self.assertIn("## Declared mass properties", context)
            self.assertEqual(model["artifacts"]["facts_jsonl"]["path"], "facts.jsonl")
            self.assertEqual(manifest["schema_version"], "robot-spatial-agent-context.v1")
            self.assertEqual(manifest["load_order"][1], "query-concepts#task_relevant_proof_closure")
            self.assertEqual(manifest["load_order"][2], "entity-cards.jsonl#one_exact_entity")
            self.assertIn("link/tool0", entity_index["by_entity_id"])
            self.assertIn("frame/tool0", entity_index["by_entity_id"])
            self.assertIn("semantic_roles", {record["topic"] for record in manifest["unresolved_claims"]})
            slide_card = next(card for card in cards if card["entity_id"] == "joint/slide")
            self.assertIn("axis_in_pre_motion", slide_card["summary_cnl"])
            self.assertIn("fact-", " ".join(slide_card["fact_ids"]))
            arm_card = next(card for card in cards if card["entity_id"] == "link/arm_link")
            self.assertEqual(arm_card["data"]["declared_inertial"]["mass_kg"], 2.0)
            self.assertIn("mass-properties", {query["command"] for query in arm_card["tool_queries"]})
            self.assertEqual(model["artifacts"]["agent_context"]["manifest"], "agent-context.json")

            retrieved = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "retrieve",
                    temp_dir,
                    "--entity",
                    "joint/slide",
                    "--predicate",
                    "has_axis",
                    "--evidence",
                    "exact",
                    "--compact",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            retrieval = json.loads(retrieved.stdout)
            self.assertEqual(retrieval["schema_version"], "robot-spatial-context-retrieval.v1")
            self.assertEqual(retrieval["entity_card"]["entity_id"], "joint/slide")
            self.assertEqual(retrieval["count"], 1)
            self.assertEqual(retrieval["facts"][0]["predicate"], "has_axis")

            ambiguous = subprocess.run(
                [sys.executable, str(SCRIPT), "retrieve", temp_dir, "--entity", "tool0"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(ambiguous.returncode, 2)
            self.assertIn("ambiguous", json.loads(ambiguous.stderr)["error"])

            cards_path = Path(temp_dir) / "entity-cards.jsonl"
            cards_path.write_text(cards_path.read_text() + "\n")
            tampered = subprocess.run(
                [sys.executable, str(SCRIPT), "retrieve", temp_dir, "--entity", "joint/slide"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(tampered.returncode, 2)
            self.assertIn("digest mismatch", json.loads(tampered.stderr)["error"])

    def test_cli_exports_measured_mesh_and_svg_views(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "export",
                    str(FIXTURES / "two_dof.urdf"),
                    "--pose",
                    str(FIXTURES / "bent_pose.json"),
                    "--semantics",
                    str(FIXTURES / "semantics.json"),
                    "--inspect-meshes",
                    "--package-map",
                    str(FIXTURES / "package_map.json"),
                    "--render",
                    "--generate-evaluation",
                    "--evaluation-key-out",
                    str(Path(temp_dir) / "private-answer-key.jsonl"),
                    "--out",
                    temp_dir,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            response = json.loads(result.stdout)
            self.assertEqual(response["status"], "exported")
            svg = (Path(temp_dir) / "scene.svg").read_text()
            self.assertIn("Front (X–Z)", svg)
            self.assertIn("Isometric", svg)
            model = json.loads((Path(temp_dir) / "model.json").read_text())
            self.assertEqual(model["capabilities"]["mesh_content_inspection"]["measured_mesh_count"], 1)
            self.assertEqual(model["artifacts"]["scene_svg"]["geometry_count"], 4)
            self.assertIn("tool0", model["kinematic_analysis"]["targets"])
            self.assertTrue(model["capabilities"]["analytic_geometric_jacobian"])
            self.assertIn("## Instantaneous motion effects", (Path(temp_dir) / "context.md").read_text())
            facts = [json.loads(line) for line in (Path(temp_dir) / "facts.jsonl").read_text().splitlines()]
            mesh_bounds = [fact for fact in facts if fact["subject"] == "frame/collision/slider_link/0" and fact["predicate"] == "has_root_frame_aabb"]
            self.assertEqual(len(mesh_bounds), 1)
            self.assertTrue(mesh_bounds[0]["evidence"]["exact"])
            self.assertEqual(mesh_bounds[0]["evidence"]["source_type"], "measured_mesh")
            predicates = {fact["predicate"] for fact in facts}
            self.assertIn("has_instantaneous_linear_effect_on", predicates)
            self.assertIn("has_sampled_workspace_observation", predicates)
            evaluation = model["artifacts"]["evaluation"]
            self.assertGreaterEqual(evaluation["question_count"], 30)
            self.assertTrue((Path(temp_dir) / "evaluation" / "questions.jsonl").exists())
            self.assertNotIn("answer_key", evaluation)
            self.assertEqual(evaluation["questions"], "evaluation/questions.jsonl")
            public_manifest = json.loads((Path(temp_dir) / "evaluation" / "manifest.json").read_text())
            self.assertEqual(public_manifest["artifacts"]["questions"], "questions.jsonl")
            self.assertTrue((Path(temp_dir) / "private-answer-key.jsonl").exists())
            self.assertIn("## Spatial understanding evaluation", (Path(temp_dir) / "context.md").read_text())

    def test_agent_context_cards_preserve_branched_mimic_causality(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "export",
                    str(FIXTURES / "mimic_branch.urdf"),
                    "--workspace-samples",
                    "16",
                    "--out",
                    temp_dir,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            cards = {
                card["entity_id"]: card
                for card in (
                    json.loads(line)
                    for line in (Path(temp_dir) / "entity-cards.jsonl").read_text().splitlines()
                )
            }
            base = cards["link/base"]
            self.assertEqual(
                {record["joint"] for record in base["data"]["outgoing_joints"]},
                {"driver", "follower"},
            )
            driver = cards["joint/driver"]
            self.assertEqual(driver["data"]["affected_links"], ["driver_link", "follower_link"])
            follower = cards["joint/follower"]
            self.assertEqual(follower["data"]["mimic"], {
                "joint": "driver",
                "multiplier": -0.5,
                "offset": 0.1,
            })
            predicates = {
                fact["predicate"]
                for fact in json.loads(subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "retrieve",
                        temp_dir,
                        "--entity",
                        "joint/follower",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout)["facts"]
            }
            self.assertIn("is_driven_by_mimic_relation", predicates)

    def test_cli_uses_srdf_named_pose_and_exports_srdf_facts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "export",
                    str(FIXTURES / "two_dof.urdf"),
                    "--srdf",
                    str(FIXTURES / "two_dof.srdf"),
                    "--pose-name",
                    "manipulator/home",
                    "--out",
                    temp_dir,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(result.stdout)["status"], "exported")
            model = json.loads((Path(temp_dir) / "model.json").read_text())
            self.assertEqual(model["pose"]["name"], "manipulator/home")
            self.assertEqual(model["pose"]["joint_positions"]["shoulder"], 0.25)
            self.assertTrue(model["capabilities"]["srdf_semantics"])
            facts = [json.loads(line) for line in (Path(temp_dir) / "facts.jsonl").read_text().splitlines()]
            self.assertIn("has_named_pose", {fact["predicate"] for fact in facts})

    def test_expand_xacro_records_provenance_and_validates_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "expanded.urdf"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "expand-xacro",
                    str(FIXTURES / "simple.urdf.xacro"),
                    "--xacro-bin",
                    str(FIXTURES / "fake_xacro.py"),
                    "--arg",
                    "base_height:=0.2",
                    "--out",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            response = json.loads(result.stdout)
            self.assertEqual(response["status"], "expanded_and_validated")
            self.assertEqual(response["mappings"], ["base_height:=0.2"])
            self.assertEqual(response["validation"]["root_link"], "base_link")
            metadata = json.loads(output.with_suffix(".urdf.meta.json").read_text())
            self.assertEqual(metadata["output"]["sha256"], response["output"]["sha256"])
            expanded = MODULE.RobotModel(output)
            self.assertEqual(expanded.links["base_link"]["visuals"][0]["geometry"]["size_xyz_m"], [1.0, 1.0, 0.2])

    def test_prepare_discovers_workspace_packages_and_binds_source_closure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entry_package = root / "entry_config"
            included_package = root / "included_description"
            entry_package.mkdir()
            included_package.mkdir()
            (entry_package / "package.xml").write_text(
                '<package format="3"><name>entry_config</name><version>1.0.0</version>'
                '<description>test</description><maintainer email="test@example.com">Test</maintainer>'
                '<license>CC0-1.0</license></package>'
            )
            (included_package / "package.xml").write_text(
                '<package format="3"><name>included_description</name><version>1.0.0</version>'
                '<description>test</description><maintainer email="test@example.com">Test</maintainer>'
                '<license>CC0-1.0</license></package>'
            )
            (included_package / "dimensions.json").write_text('{"x": 0.2, "y": 0.3, "z": 0.4}')
            shutil.copy2(FIXTURES / "demo" / "slider.stl", included_package / "shape.stl")
            source = entry_package / "robot.urdf.xacro"
            source.write_text(
                '<?xml version="1.0"?><robot name="source" xmlns:xacro="http://www.ros.org/wiki/xacro">'
                '<xacro:include filename="$(find included_description)/fragment.xacro"/>'
                '</robot>'
            )
            first_output = root / "first-output"
            command = [
                sys.executable,
                str(SCRIPT),
                "prepare",
                str(source),
                "--workspace-root",
                str(root),
                "--xacro-bin",
                str(FIXTURES / "fake_workspace_xacro.py"),
                "--arg",
                "prefix:=held_",
                "--inspect-mesh-kind",
                "collision",
                "--out",
                str(first_output),
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            response = json.loads(result.stdout)
            self.assertEqual(response["status"], "prepared")
            self.assertEqual(response["input_format"], "xacro")
            self.assertEqual(response["xacro_package_lookups"], ["included_description"])
            self.assertEqual(response["used_packages"], ["entry_config", "included_description"])
            self.assertEqual(response["robot"]["root_link"], "held_base")
            resolved_text = (first_output / "resolved.urdf").read_text()
            self.assertIn("package://included_description/shape.stl", resolved_text)
            self.assertNotIn(str(included_package), resolved_text)
            resolution_metadata = json.loads((first_output / "resolved.urdf.meta.json").read_text())
            self.assertEqual(resolution_metadata["normalization"]["rewritten_package_path_count"], 1)
            self.assertEqual(resolution_metadata["normalization"]["removed_xml_comment_count"], 1)
            source_manifest = json.loads((first_output / "source-manifest.json").read_text())
            used = {record["package_name"]: record for record in source_manifest["used_packages"]}
            self.assertEqual(set(used), {"entry_config", "included_description"})
            included_paths = {record["path"] for record in used["included_description"]["files"]}
            self.assertTrue({"dimensions.json", "shape.stl", "package.xml"}.issubset(included_paths))
            model = json.loads((first_output / "context" / "model.json").read_text())
            self.assertEqual(model["source_compilation"]["package_lookups"], ["included_description"])
            self.assertTrue(model["capabilities"]["mesh_content_inspection"]["by_kind"]["collision"]["complete"])
            collision = model["geometry_analysis"]["collision/held_base/0"]
            self.assertEqual(collision["status"], "measured")

            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("must not already exist", repeated.stderr)

            first_tree_digest = used["included_description"]["tree_sha256"]
            (included_package / "dimensions.json").write_text('{"x": 0.2, "y": 0.3, "z": 0.5}')
            second_output = root / "second-output"
            second_command = list(command)
            second_command[-1] = str(second_output)
            subprocess.run(second_command, check=True, capture_output=True, text=True)
            second_manifest = json.loads((second_output / "source-manifest.json").read_text())
            second_used = {record["package_name"]: record for record in second_manifest["used_packages"]}
            self.assertNotEqual(first_tree_digest, second_used["included_description"]["tree_sha256"])
            self.assertNotEqual(
                response["resolved_urdf"]["sha256"],
                json.loads((second_output / "prepare.json").read_text())["resolved_urdf"]["sha256"],
            )

    def test_prepare_rejects_missing_and_ambiguous_packages_without_partial_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entry = root / "entry"
            entry.mkdir()
            (entry / "package.xml").write_text(
                '<package format="3"><name>entry_config</name><version>1.0.0</version>'
                '<description>test</description><maintainer email="test@example.com">Test</maintainer>'
                '<license>CC0-1.0</license></package>'
            )
            source = entry / "robot.urdf.xacro"
            source.write_text(
                '<robot name="source" xmlns:xacro="http://www.ros.org/wiki/xacro">'
                '<xacro:include filename="$(find included_description)/fragment.xacro"/>'
                '</robot>'
            )
            output = root / "partial-must-not-remain"
            missing = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "prepare",
                    str(source),
                    "--workspace-root",
                    str(root),
                    "--xacro-bin",
                    str(FIXTURES / "fake_workspace_xacro.py"),
                    "--out",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(missing.returncode, 2)
            self.assertFalse(output.exists())

            duplicate_a = root / "duplicate-a"
            duplicate_b = root / "duplicate-b"
            duplicate_a.mkdir()
            duplicate_b.mkdir()
            manifest = (
                '<package format="3"><name>same_name</name><version>1.0.0</version>'
                '<description>test</description><maintainer email="test@example.com">Test</maintainer>'
                '<license>CC0-1.0</license></package>'
            )
            (duplicate_a / "package.xml").write_text(manifest)
            (duplicate_b / "package.xml").write_text(manifest)
            with self.assertRaisesRegex(MODULE.WorkspaceError, "ambiguous ROS package"):
                MODULE.discover_packages([root])

    def test_compare_detects_pose_conditioned_frame_changes(self):
        before = self.model.canonical({}, "zero")
        after = self.model.canonical(self.pose, "bent")
        with tempfile.TemporaryDirectory() as temp_dir:
            before_path, after_path = Path(temp_dir) / "before.json", Path(temp_dir) / "after.json"
            before_path.write_text(MODULE.json_dump(before))
            after_path.write_text(MODULE.json_dump(after))
            comparison = MODULE.compare_artifacts(before_path, after_path, 1e-9, 1e-7)
            self.assertTrue(comparison["changed"])
            self.assertTrue(comparison["pose_changed"])
            self.assertFalse(comparison["source_urdf_changed"])
            changed_names = {change["frame"] for change in comparison["changed_frames"]}
            self.assertIn("tool0", changed_names)
            self.assertNotIn("base_link", changed_names)

    def test_compare_detects_mesh_scale_without_frame_change(self):
        before = self.model.canonical({}, "zero", inspect_meshes=True, package_map_path=FIXTURES / "package_map.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            modified_urdf = Path(temp_dir) / "scaled.urdf"
            source = (FIXTURES / "two_dof.urdf").read_text()
            modified_urdf.write_text(source.replace('scale="1 1 1"', 'scale="2 1 1"'))
            modified_model = MODULE.RobotModel(modified_urdf)
            after = modified_model.canonical({}, "zero", inspect_meshes=True, package_map_path=FIXTURES / "package_map.json")
            before_path, after_path = Path(temp_dir) / "before.json", Path(temp_dir) / "after.json"
            before_path.write_text(MODULE.json_dump(before))
            after_path.write_text(MODULE.json_dump(after))
            comparison = MODULE.compare_artifacts(before_path, after_path, 1e-9, 1e-7)
            self.assertIn("collision/slider_link/0", comparison["changed_geometry_intrinsics"])
            self.assertIn("collision/slider_link/0", {change["frame"] for change in comparison["changed_geometry_world_bounds"]})
            self.assertEqual(comparison["changed_frames"], [])

    def test_inertial_regression_is_visible_to_compare_and_project_invariants(self):
        before = self.model.canonical(self.pose, "bent")
        contract = spatial_invariants.read_invariant_contract(FIXTURES / "invariants.json", self.model)
        with tempfile.TemporaryDirectory() as temp_dir:
            modified_urdf = Path(temp_dir) / "changed-mass.urdf"
            modified_urdf.write_text((FIXTURES / "two_dof.urdf").read_text().replace('<mass value="2.0"/>', '<mass value="3.0"/>'))
            modified_model = MODULE.RobotModel(modified_urdf)
            after = modified_model.canonical(self.pose, "bent")
            before_path, after_path = Path(temp_dir) / "before.json", Path(temp_dir) / "after.json"
            before_path.write_text(MODULE.json_dump(before))
            after_path.write_text(MODULE.json_dump(after))

            comparison = MODULE.compare_artifacts(before_path, after_path, 1e-9, 1e-7)
            self.assertEqual([record["link"] for record in comparison["changed_declared_inertials"]], ["arm_link"])
            self.assertEqual(comparison["declared_mass_properties_change"]["declared_mass_delta_kg"], 1.0)
            self.assertTrue(comparison["declared_mass_properties_change"]["changed"])
            self.assertTrue(comparison["declared_static_gravity_loads_change"]["changed"])
            self.assertEqual(
                comparison["declared_static_gravity_loads_change"]["independent_driver_deltas"]["shoulder"]["delta"],
                0.0,
            )
            self.assertFalse(comparison["actuation_declarations_changed"])

            report = spatial_invariants.verify_invariant_contract(
                modified_model,
                contract,
                package_map_path=FIXTURES / "package_map.json",
            )
            self.assertEqual(report["status"], "failed")
            failures = [record for record in report["results"] if record["status"] == "failed"]
            self.assertEqual([record["id"] for record in failures], ["declared-mass-properties-at-bent-pose"])
            self.assertEqual(failures[0]["metrics"]["absolute_mass_error_kg"], 1.0)
            self.assertEqual(failures[0]["physical_world_completeness"], "not_established")


if __name__ == "__main__":
    unittest.main()
