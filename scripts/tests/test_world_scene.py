#!/usr/bin/env python3

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(SCRIPTS))

import robot_spatial
import spatial_context
import spatial_evaluation
import spatial_invariants
from world_scene import SceneError, WorldScene


class WorldSceneTests(unittest.TestCase):
    def setUp(self):
        self.model = robot_spatial.RobotModel(FIXTURES / "two_dof.urdf")
        self.scene = WorldScene(
            FIXTURES / "world_scene.json",
            expected_robot_name=self.model.name,
            expected_root_link=self.model.root_link,
        )

    def assertVectorAlmostEqual(self, actual, expected, places=9):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=places)

    def _simple_scene(self, object_geometry, object_xyz=(3.0, 0.0, 0.0), robot_rpy=(0.0, 0.0, 0.0)):
        return {
            "schema_version": "robot-spatial-world-scene.v1",
            "scene_id": "simple_scene",
            "snapshot": {
                "id": "simple_snapshot",
                "time_semantics": "static_snapshot",
                "captured_at": None,
                "valid_until": None,
            },
            "world_frame": "world",
            "gravity": {
                "vector_xyz_m_s2": [0.0, 0.0, -9.80665],
                "expressed_in_frame": "world",
                "source": {"type": "declared", "reference": "test", "captured_at": None},
            },
            "frames": {},
            "robot": {
                "instance_id": "box_robot",
                "robot_name": "box_robot",
                "root_link": "base",
                "parent_frame": "world",
                "pose": {"xyz_m": [0.0, 0.0, 0.0], "rpy_rad": list(robot_rpy)},
                "source": {"type": "synthetic", "reference": "test", "captured_at": None},
            },
            "objects": {
                "obstacle": {
                    "parent_frame": "world",
                    "pose": {"xyz_m": list(object_xyz), "rpy_rad": [0.0, 0.0, 0.0]},
                    "semantics": {"category": "obstacle", "roles": ["collision"], "meaning": "test object"},
                    "source": {"type": "synthetic", "reference": "test", "captured_at": None},
                    "collision_geometries": [{"id": "body", "geometry": object_geometry}],
                }
            },
        }

    def _write_simple_robot_and_scene(self, directory, robot_geometry, scene_data):
        urdf = directory / "robot.urdf"
        if robot_geometry["type"] == "box":
            geometry_xml = '<box size="{}"/>'.format(" ".join(str(value) for value in robot_geometry["size_xyz_m"]))
        elif robot_geometry["type"] == "cylinder":
            geometry_xml = f'<cylinder radius="{robot_geometry["radius_m"]}" length="{robot_geometry["length_m"]}"/>'
        else:
            raise AssertionError("unsupported test geometry")
        urdf.write_text(
            '<robot name="box_robot"><link name="base">'
            '<collision><geometry>' + geometry_xml + '</geometry></collision>'
            '</link></robot>',
            encoding="utf-8",
        )
        scene_path = directory / "scene.json"
        scene_path.write_text(json.dumps(scene_data), encoding="utf-8")
        model = robot_spatial.RobotModel(urdf)
        scene = WorldScene(scene_path, expected_robot_name=model.name, expected_root_link=model.root_link)
        return model, scene

    def test_scene_frame_graph_root_mount_and_gravity_conversion(self):
        self.assertEqual(self.scene.scene_id, "two_dof_workcell")
        self.assertEqual(self.scene.snapshot["time_semantics"], "static_snapshot")
        transform = self.scene.transform(
            "scene_frame/world",
            "robot_frame/base_link",
            self.model,
            {},
        )
        self.assertVectorAlmostEqual([transform[index][3] for index in range(3)], [1.0, 2.0, 0.0])
        self.assertVectorAlmostEqual(robot_spatial.quaternion_xyzw(transform), [0.5, 0.5, 0.5, 0.5])
        gravity = self.scene.gravity_in_robot_root()
        self.assertEqual(gravity["status"], "computed")
        self.assertVectorAlmostEqual(gravity["vector_in_robot_root_xyz_m_s2"], [0.0, -10.0, 0.0])
        load = robot_spatial.scene_gravity_load_analysis(self.model, self.scene, {}, "zero")
        self.assertEqual(load["status"], "computed")
        self.assertAlmostEqual(load["loads"]["independent_driver_loads"]["shoulder"]["generalized_gravity_force"], -10.0)

    def test_scene_rejects_cycles_identity_mismatch_and_nonunit_quaternion(self):
        base = json.loads((FIXTURES / "world_scene.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            cycle = json.loads(json.dumps(base))
            cycle["frames"]["cell"]["parent"] = "camera_mount"
            path = directory / "cycle.json"
            path.write_text(json.dumps(cycle), encoding="utf-8")
            with self.assertRaisesRegex(SceneError, "cycle"):
                WorldScene(path, expected_robot_name=self.model.name, expected_root_link=self.model.root_link)

            mismatch = json.loads(json.dumps(base))
            mismatch["robot"]["root_link"] = "wrong_root"
            path = directory / "mismatch.json"
            path.write_text(json.dumps(mismatch), encoding="utf-8")
            with self.assertRaisesRegex(SceneError, "does not match URDF root"):
                WorldScene(path, expected_robot_name=self.model.name, expected_root_link=self.model.root_link)

            quaternion = json.loads(json.dumps(base))
            quaternion["frames"]["camera_mount"]["pose"]["quaternion_xyzw"] = [0.0, 0.0, 0.0, 2.0]
            path = directory / "quaternion.json"
            path.write_text(json.dumps(quaternion), encoding="utf-8")
            with self.assertRaisesRegex(SceneError, "unit length"):
                WorldScene(path, expected_robot_name=self.model.name, expected_root_link=self.model.root_link)

    def test_robot_environment_collision_preserves_pair_coverage(self):
        result = self.scene.robot_environment_collisions(self.model, {})
        self.assertEqual(result["status"], "collision")
        self.assertEqual(result["minimum_separation"]["distance_m"], 0.0)
        self.assertEqual(result["coverage"]["collision_pair_count"], 1)
        self.assertEqual(result["coverage"]["indeterminate_pair_count"], 2)
        collision = next(record for record in result["pair_results"] if record["status"] == "collision")
        self.assertEqual(collision["robot_geometry"], "robot_geometry/collision/base_link/0")
        self.assertEqual(collision["environment_geometry"], "scene_geometry/near_obstacle/body")
        self.assertTrue(collision["containment_or_solid_overlap"] is False)

    def test_exact_box_clearance_and_closed_solid_containment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            separated_scene = self._simple_scene(
                {"type": "box", "size_xyz_m": [1.0, 1.0, 1.0]},
                object_xyz=(3.0, 0.0, 0.0),
            )
            model, scene = self._write_simple_robot_and_scene(
                directory,
                {"type": "box", "size_xyz_m": [1.0, 1.0, 1.0]},
                separated_scene,
            )
            result = scene.robot_environment_collisions(model, {})
            self.assertEqual(result["status"], "collision_free")
            self.assertEqual(result["minimum_separation"]["status"], "computed")
            self.assertAlmostEqual(result["minimum_separation"]["distance_m"], 2.0)

            contained_scene = self._simple_scene(
                {"type": "box", "size_xyz_m": [4.0, 4.0, 4.0]},
                object_xyz=(0.0, 0.0, 0.0),
            )
            contained_path = directory / "contained.json"
            contained_path.write_text(json.dumps(contained_scene), encoding="utf-8")
            contained = WorldScene(contained_path, expected_robot_name=model.name, expected_root_link=model.root_link)
            result = contained.robot_environment_collisions(model, {})
            self.assertEqual(result["status"], "collision")
            pair = result["pair_results"][0]
            self.assertTrue(pair["containment_or_solid_overlap"])
            self.assertGreater(pair["surface_distance_m"], 0.0)
            self.assertEqual(pair["separation_m"], 0.0)

    def test_unsupported_overlapping_cylinder_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            data = self._simple_scene(
                {"type": "box", "size_xyz_m": [2.0, 2.0, 2.0]},
                object_xyz=(0.0, 0.0, 0.0),
            )
            model, scene = self._write_simple_robot_and_scene(
                directory,
                {"type": "cylinder", "radius_m": 0.5, "length_m": 1.0},
                data,
            )
            result = scene.robot_environment_collisions(model, {})
            self.assertEqual(result["status"], "indeterminate")
            self.assertEqual(result["minimum_separation"]["status"], "indeterminate")
            self.assertIn("exact solid classification is unavailable", result["pair_results"][0]["reason"])

    def test_missing_mesh_reason_is_relocation_stable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            data = self._simple_scene(
                {"type": "mesh", "uri": "missing-obstacle.stl", "scale_xyz": [1.0, 1.0, 1.0]},
                object_xyz=(3.0, 0.0, 0.0),
            )
            model, scene = self._write_simple_robot_and_scene(
                directory,
                {"type": "box", "size_xyz_m": [1.0, 1.0, 1.0]},
                data,
            )
            result = scene.robot_environment_collisions(model, {})
            reason = result["pair_results"][0]["reason"]
            self.assertIn("mesh source 'missing-obstacle.stl' does not resolve to an existing file", reason)
            self.assertNotIn(str(directory), reason)

    def test_canonical_context_facts_and_evaluation_expose_typed_scene_entities(self):
        canonical = self.model.canonical({}, "zero", workspace_samples=0, world_scene=self.scene)
        self.assertEqual(canonical["world_scene"]["status"], "parsed_validated_and_bound")
        self.assertEqual(canonical["world_scene"]["robot_environment_collision"]["status"], "collision")
        facts = robot_spatial.fact_records(self.model, canonical)
        self.assertTrue(any(fact["predicate"] == "mounts_robot_root_in_scene" for fact in facts))
        self.assertTrue(any(fact["predicate"] == "has_scene_bound_static_gravity_load" for fact in facts))
        questions, _ = spatial_evaluation.generate_records(canonical, facts)
        capabilities = {question["capability"] for question in questions}
        self.assertTrue({"world_scene", "world_scene_gravity", "robot_environment_collision"}.issubset(capabilities))

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            facts_path = directory / "facts.jsonl"
            facts_path.write_text(robot_spatial.jsonl_dump(facts), encoding="utf-8")
            spatial_context.write_agent_context(directory, canonical, facts, facts_path)
            object_result = spatial_context.retrieve_context(directory, entity="scene_object/near_obstacle")
            self.assertEqual(object_result["entity_card"]["entity_type"], "scene_object")
            geometry_result = spatial_context.retrieve_context(directory, entity="scene_geometry/near_obstacle/body")
            self.assertTrue(any(fact["predicate"] == "has_robot_environment_pair_result" for fact in geometry_result["facts"]))
            robot_geometry_result = spatial_context.retrieve_context(
                directory,
                entity="robot_geometry/collision/base_link/0",
            )
            self.assertEqual(robot_geometry_result["entity_card"]["entity_type"], "robot_geometry")
            self.assertTrue(any(
                fact["predicate"] == "has_robot_environment_pair_result"
                for fact in robot_geometry_result["facts"]
            ))
            instance_result = spatial_context.retrieve_context(directory, entity="robot_instance/demo_arm")
            self.assertTrue(any(fact["predicate"] == "has_robot_environment_collision_status" for fact in instance_result["facts"]))

    def test_scene_cli_queries_bind_urdf_scene_snapshot_and_parameters(self):
        command_base = [sys.executable, str(SCRIPTS / "robot_spatial.py")]
        transform = subprocess.run(
            command_base + [
                "scene-transform",
                str(FIXTURES / "two_dof.urdf"),
                "--scene",
                str(FIXTURES / "world_scene.json"),
                "--from",
                "scene_frame/world",
                "--to",
                "robot_frame/base_link",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        transform_payload = json.loads(transform.stdout)
        self.assertEqual(transform_payload["translation_xyz_m"], [1.0, 2.0, 0.0])
        self.assertEqual(transform_payload["query_evidence"]["source_world_scene_sha256"], self.scene.sha256)
        self.assertEqual(transform_payload["query_evidence"]["snapshot_id"], "fixture_snapshot_001")

        collision = subprocess.run(
            command_base + [
                "scene-collisions",
                str(FIXTURES / "two_dof.urdf"),
                "--scene",
                str(FIXTURES / "world_scene.json"),
                "--contact-tolerance-m",
                "1e-8",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        collision_payload = json.loads(collision.stdout)
        self.assertEqual(collision_payload["status"], "collision")
        self.assertEqual(collision_payload["query_evidence"]["parameters"]["contact_tolerance_m"], 1e-8)

    def test_scene_invariant_contract_passes_and_is_digest_bound(self):
        contract = spatial_invariants.read_invariant_contract(
            FIXTURES / "world_invariants.json",
            self.model,
            self.scene,
        )
        report = spatial_invariants.verify_invariant_contract(
            self.model,
            contract,
            package_map_path=None,
            world_scene=self.scene,
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual({result["type"] for result in report["results"]}, {
            "scene_transform",
            "scene_gravity_loads",
            "robot_environment_collision",
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_contract = json.loads((FIXTURES / "world_invariants.json").read_text(encoding="utf-8"))
            bad_contract["world_scene"]["sha256"] = "0" * 64
            path = Path(temp_dir) / "bad.json"
            path.write_text(json.dumps(bad_contract), encoding="utf-8")
            with self.assertRaisesRegex(spatial_invariants.InvariantError, "sha256"):
                spatial_invariants.read_invariant_contract(path, self.model, self.scene)

    def test_comparison_detects_root_mount_scene_and_scene_gravity_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            changed_data = json.loads((FIXTURES / "world_scene.json").read_text(encoding="utf-8"))
            changed_data["robot"]["pose"]["rpy_rad"] = [0.0, 0.0, 0.0]
            changed_path = directory / "changed_scene.json"
            changed_path.write_text(json.dumps(changed_data), encoding="utf-8")
            changed_scene = WorldScene(changed_path, expected_robot_name=self.model.name, expected_root_link=self.model.root_link)
            before = self.model.canonical({}, "zero", workspace_samples=0, world_scene=self.scene)
            after = self.model.canonical({}, "zero", workspace_samples=0, world_scene=changed_scene)
            before_path = directory / "before.json"
            after_path = directory / "after.json"
            before_path.write_text(robot_spatial.json_dump(before), encoding="utf-8")
            after_path.write_text(robot_spatial.json_dump(after), encoding="utf-8")
            report = robot_spatial.compare_artifacts(before_path, after_path, 1e-9, 1e-7)
            self.assertTrue(report["changed"])
            self.assertTrue(report["world_scene_changed"])
            self.assertTrue(report["scene_gravity_loads_change"]["changed"])
            self.assertNotEqual(
                report["scene_gravity_loads_change"]["independent_driver_deltas"]["shoulder"]["delta"],
                0.0,
            )

    def test_prepare_propagates_world_scene_into_progressive_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            source_model, source_scene = self._write_simple_robot_and_scene(
                directory,
                {"type": "box", "size_xyz_m": [1.0, 1.0, 1.0]},
                self._simple_scene(
                    {"type": "box", "size_xyz_m": [1.0, 1.0, 1.0]},
                    object_xyz=(3.0, 0.0, 0.0),
                ),
            )
            output = directory / "prepared"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "robot_spatial.py"),
                    "prepare",
                    str(source_model.path),
                    "--scene",
                    str(source_scene.path),
                    "--out",
                    str(output),
                    "--workspace-samples",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "prepared")
            self.assertEqual(payload["world_scene"]["sha256"], source_scene.sha256)
            model = json.loads((output / "context" / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(model["world_scene"]["scene_id"], "simple_scene")
            manifest = json.loads((output / "context" / "agent-context.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["statistics"]["entity_type_counts"]["scene_object"], 1)
            self.assertEqual(manifest["statistics"]["entity_type_counts"]["robot_geometry"], 1)


if __name__ == "__main__":
    unittest.main()
