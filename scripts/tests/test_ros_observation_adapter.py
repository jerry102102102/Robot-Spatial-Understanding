#!/usr/bin/env python3

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(SCRIPTS))

import robot_spatial
from ros_observation_adapter import RosAdapterError, _publisher_id, normalize, read_capture, read_config
from temporal_observation import TemporalObservationLog, read_observation_query
from world_scene import WorldScene


class RosObservationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.model = robot_spatial.RobotModel(FIXTURES / "two_dof.urdf")
        self.scene = WorldScene(
            FIXTURES / "world_scene.json",
            expected_robot_name=self.model.name,
            expected_root_link=self.model.root_link,
        )

    @staticmethod
    def _write(directory, name, value):
        path = directory / name
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _config(self, *, timestamp_source="message_header", max_joint_age=50, max_tf_age=50):
        return {
            "schema_version": "robot-spatial-ros-adapter-config.v1",
            "adapter_id": "fixture_adapter",
            "clock": {"domain": "fixture_ros_time", "unit": "nanoseconds", "epoch": "fixture_epoch"},
            "binding": {
                "robot_name": self.model.name,
                "root_link": self.model.root_link,
                "source_urdf_semantic_sha256": self.model.semantic_sha256,
                "scene_id": self.scene.scene_id,
                "scene_sha256": self.scene.sha256,
            },
            "topics": {
                "joint_states": ["/joint_states"],
                "tf_dynamic": ["/tf"],
                "tf_static": ["/tf_static"],
            },
            "frames": {
                "ros_reference_frame": "world",
                "scene_parent_frame": "world",
                "robot_root_frame": "base_mount",
                "objects": {"near_obstacle": "near_obstacle_tf"},
            },
            "joint_mapping": {"shoulder": "shoulder_encoder", "slide": "slide_encoder"},
            "policies": {
                "timestamp_source": timestamp_source,
                "joint_snapshot": {
                    "maximum_component_age_ns": max_joint_age,
                    "reject_multiple_publishers_per_joint": True,
                },
                "tf_snapshot": {
                    "maximum_dynamic_edge_age_ns": max_tf_age,
                    "reject_multiple_publishers_per_child": True,
                    "reject_parent_switches": True,
                    "matrix_component_tolerance": 1e-9,
                },
            },
        }

    @staticmethod
    def _joint(record_id, timestamp, receipt, publisher, names, positions):
        return {
            "record_id": record_id,
            "kind": "joint_state",
            "topic": "/joint_states",
            "publisher_id": publisher,
            "receipt_timestamp_ns": receipt,
            "message_timestamp_ns": timestamp,
            "names": names,
            "positions": positions,
        }

    @staticmethod
    def _tf(record_id, timestamp, receipt, publisher, parent, child, xyz, *, static=False, transform_id=None):
        return {
            "record_id": record_id,
            "kind": "tf",
            "topic": "/tf_static" if static else "/tf",
            "publisher_id": publisher,
            "receipt_timestamp_ns": receipt,
            "static": static,
            "transforms": [
                {
                    "transform_id": transform_id or f"{record_id}_transform",
                    "message_timestamp_ns": timestamp,
                    "parent_frame": parent,
                    "child_frame": child,
                    "pose": {"xyz_m": xyz, "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
                }
            ],
        }

    def _capture(self, config_path, records, *, clock=None):
        return {
            "schema_version": "robot-spatial-ros-capture.v1",
            "capture_id": "fixture_capture",
            "adapter_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "clock": clock or {"domain": "fixture_ros_time", "unit": "nanoseconds", "epoch": "fixture_epoch"},
            "capture": {"started_timestamp_ns": 50, "ended_timestamp_ns": 300, "node_use_sim_time": True},
            "source": {
                "transport": "synthetic_fixture",
                "reference": "unit test capture",
                "ros_distro": None,
                "authority_visibility": "publisher IDs are explicit fixture values",
            },
            "records": records,
        }

    def _happy_records(self):
        return [
            self._joint("j090", 90, 91, "joint_pub", ["shoulder_encoder", "ignored_joint"], [0.0, 99.0]),
            self._joint("j100", 100, 101, "joint_pub", ["slide_encoder"], [0.5]),
            self._joint("j200", 200, 201, "joint_pub", ["shoulder_encoder"], [0.5]),
            self._joint("j220", 220, 221, "joint_pub", ["slide_encoder"], [0.25]),
            self._tf("tf_static", 0, 60, "static_pub", "world", "cell", [1.0, 0.0, 0.0], static=True),
            self._tf("tf_root_100", 100, 101, "root_pub", "cell", "base_mount", [2.0, 0.0, 0.0]),
            self._tf("tf_root_200", 200, 201, "root_pub", "cell", "base_mount", [3.0, 0.0, 0.0]),
            self._tf("tf_object_100", 100, 102, "object_pub", "world", "near_obstacle_tf", [11.0, 0.0, 0.0]),
            self._tf("tf_object_200", 200, 202, "object_pub", "world", "near_obstacle_tf", [12.0, 0.0, 0.0]),
        ]

    def test_current_rclpy_message_info_publisher_gid_shape_is_preserved(self):
        message_info = {
            "publisher_gid": {
                "implementation_identifier": "rmw_test",
                "data": bytes([0, 1, 2, 255]),
            }
        }
        self.assertEqual(_publisher_id(message_info), "rmw_test:000102ff")
        self.assertIsNone(_publisher_id({"publisher_gid": None}))

    def test_partial_joint_messages_and_tf_paths_normalize_into_resolvable_observations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            config_path = self._write(directory, "config.json", self._config())
            capture_path = self._write(directory, "capture.json", self._capture(config_path, self._happy_records()))
            config = read_config(config_path)
            capture = read_capture(capture_path, config)
            log, report = normalize(self.model, self.scene, config, capture)

            self.assertEqual([sample["timestamp_ns"] for sample in log["streams"]["joint_states"]], [100, 220])
            self.assertEqual(log["streams"]["joint_states"][1]["positions"], {"shoulder": 0.5, "slide": 0.25})
            self.assertEqual(report["joint_normalization"]["ignored_unmapped_ros_joint_names"], ["ignored_joint"])
            self.assertEqual(len(report["joint_normalization"]["stale_component_events"]), 1)
            root = log["streams"]["robot_root_poses"]
            self.assertEqual([sample["timestamp_ns"] for sample in root], [100, 200])
            self.assertEqual(root[-1]["pose"]["xyz_m"], [4.0, 0.0, 0.0])
            objects = log["streams"]["object_poses"]["near_obstacle"]
            self.assertEqual(objects[-1]["pose"]["xyz_m"], [12.0, 0.0, 0.0])
            self.assertFalse(report["timestamp_policy"]["future_samples_consumed"])

            log_path = self._write(directory, "observations.json", log)
            query_path = self._write(directory, "query.json", {
                "schema_version": "robot-spatial-observation-query.v1",
                "query_id": "after_capture_sample",
                "time_ns": 230,
                "maximum_age_ns": {"joint_states": 20, "robot_root_pose": 30, "object_pose": 30},
                "fallbacks": {"robot_root": "require_observed", "objects": "require_observed"},
                "required_object_ids": ["near_obstacle"],
            })
            observation_log = TemporalObservationLog(log_path)
            query, _ = read_observation_query(query_path, self.scene)
            resolved = observation_log.resolve(self.model, self.scene, query)
            self.assertTrue(resolved["all_required_current"])
            self.assertEqual(resolved["report"]["status"], "current")
            self.assertEqual(resolved["report"]["effective_state"]["sources"]["robot_root"]["layer"], "observation")

    def test_clock_config_digest_and_future_header_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            config_path = self._write(directory, "config.json", self._config())
            mismatch = self._capture(config_path, [], clock={"domain": "system_time", "unit": "nanoseconds", "epoch": "unix"})
            mismatch_path = self._write(directory, "mismatch.json", mismatch)
            with self.assertRaisesRegex(RosAdapterError, "clock mismatch"):
                read_capture(mismatch_path, read_config(config_path))

            changed_config = self._config()
            changed_config["adapter_id"] = "changed_after_capture"
            changed_path = self._write(directory, "changed.json", changed_config)
            capture_path = self._write(directory, "capture.json", self._capture(config_path, []))
            with self.assertRaisesRegex(RosAdapterError, "adapter_config_sha256"):
                read_capture(capture_path, read_config(changed_path))

            future_record = self._joint("future", 200, 190, "joint_pub", ["shoulder_encoder"], [0.0])
            future_path = self._write(directory, "future.json", self._capture(config_path, [future_record]))
            with self.assertRaisesRegex(RosAdapterError, "later than receipt"):
                read_capture(future_path, read_config(config_path))

    def test_zero_header_requires_explicit_receipt_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            strict_path = self._write(directory, "strict-config.json", self._config())
            record = self._joint("zero", 0, 100, "joint_pub", ["shoulder_encoder"], [0.0])
            capture_path = self._write(directory, "strict-capture.json", self._capture(strict_path, [record]))
            with self.assertRaisesRegex(RosAdapterError, "zero or missing header"):
                read_capture(capture_path, read_config(strict_path))

            fallback_path = self._write(directory, "fallback-config.json", self._config(timestamp_source="message_header_or_receipt"))
            fallback_capture = self._write(directory, "fallback-capture.json", self._capture(fallback_path, [record]))
            parsed = read_capture(fallback_capture, read_config(fallback_path))
            self.assertEqual(parsed["records"][0]["timestamp_ns"], 100)
            self.assertEqual(parsed["records"][0]["timestamp_origin"], "receipt_fallback_for_zero_or_missing_header")

    def test_joint_and_tf_authority_conflicts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            config_path = self._write(directory, "config.json", self._config())
            joint_records = [
                self._joint("j1", 100, 101, "publisher_a", ["shoulder_encoder", "slide_encoder"], [0.0, 0.1]),
                self._joint("j2", 110, 111, "publisher_b", ["shoulder_encoder"], [0.2]),
            ]
            capture_path = self._write(directory, "joint-conflict.json", self._capture(config_path, joint_records))
            config = read_config(config_path)
            capture = read_capture(capture_path, config)
            with self.assertRaisesRegex(RosAdapterError, "multiple ROS authorities observed for joint"):
                normalize(self.model, self.scene, config, capture)

            tf_records = [
                self._tf("tf1", 100, 101, "publisher_a", "world", "base_mount", [0.0, 0.0, 0.0]),
                self._tf("tf2", 110, 111, "publisher_b", "world", "base_mount", [0.1, 0.0, 0.0]),
            ]
            tf_capture_path = self._write(directory, "tf-conflict.json", self._capture(config_path, tf_records))
            tf_capture = read_capture(tf_capture_path, config)
            with self.assertRaisesRegex(RosAdapterError, "multiple ROS authorities observed for TF child"):
                normalize(self.model, self.scene, config, tf_capture)

    def test_conflicting_same_time_joint_values_and_limit_violations_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            config_path = self._write(directory, "config.json", self._config())
            config = read_config(config_path)
            conflict = [
                self._joint("j1", 100, 101, "publisher", ["shoulder_encoder", "slide_encoder"], [0.0, 0.1]),
                self._joint("j2", 100, 102, "publisher", ["shoulder_encoder"], [0.2]),
            ]
            conflict_path = self._write(directory, "conflict.json", self._capture(config_path, conflict))
            with self.assertRaisesRegex(RosAdapterError, "conflicting positions"):
                normalize(self.model, self.scene, config, read_capture(conflict_path, config))

            outside_limit = [
                self._joint("j1", 100, 101, "publisher", ["shoulder_encoder", "slide_encoder"], [0.0, 50.0]),
            ]
            outside_path = self._write(directory, "outside.json", self._capture(config_path, outside_limit))
            with self.assertRaisesRegex(RosAdapterError, "violates the bound URDF model"):
                normalize(self.model, self.scene, config, read_capture(outside_path, config))

    def test_parent_switch_cycles_and_static_dynamic_collision_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            config_path = self._write(directory, "config.json", self._config())
            config = read_config(config_path)
            parent_switch = [
                self._tf("tf1", 100, 101, "publisher", "world", "base_mount", [0.0, 0.0, 0.0]),
                self._tf("tf2", 110, 111, "publisher", "other_parent", "base_mount", [0.0, 0.0, 0.0]),
            ]
            switch_path = self._write(directory, "switch.json", self._capture(config_path, parent_switch))
            with self.assertRaisesRegex(RosAdapterError, "switches parents"):
                normalize(self.model, self.scene, config, read_capture(switch_path, config))

            cycle = [
                self._tf("tf1", 100, 101, "publisher", "base_mount", "a", [0.0, 0.0, 0.0]),
                self._tf("tf2", 100, 102, "publisher", "a", "base_mount", [0.0, 0.0, 0.0]),
            ]
            cycle_path = self._write(directory, "cycle.json", self._capture(config_path, cycle))
            with self.assertRaisesRegex(RosAdapterError, "cycle"):
                normalize(self.model, self.scene, config, read_capture(cycle_path, config))

            static_dynamic = [
                self._tf("static", 0, 60, "publisher", "world", "base_mount", [0.0, 0.0, 0.0], static=True),
                self._tf("dynamic", 100, 101, "publisher", "world", "base_mount", [0.0, 0.0, 0.0]),
            ]
            mixed_path = self._write(directory, "mixed.json", self._capture(config_path, static_dynamic))
            with self.assertRaisesRegex(RosAdapterError, "both static and dynamic"):
                normalize(self.model, self.scene, config, read_capture(mixed_path, config))

    def test_stale_tf_path_does_not_emit_a_fresh_composite_pose(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            config_path = self._write(directory, "config.json", self._config(max_tf_age=50))
            records = [
                self._tf("upstream", 100, 101, "publisher", "world", "cell", [1.0, 0.0, 0.0]),
                self._tf("root100", 100, 102, "publisher", "cell", "base_mount", [2.0, 0.0, 0.0]),
                self._tf("root200", 200, 201, "publisher", "cell", "base_mount", [3.0, 0.0, 0.0]),
            ]
            capture_path = self._write(directory, "capture.json", self._capture(config_path, records))
            config = read_config(config_path)
            capture = read_capture(capture_path, config)
            log, report = normalize(self.model, self.scene, config, capture)
            self.assertEqual([sample["timestamp_ns"] for sample in log["streams"]["robot_root_poses"]], [100])
            skipped = report["tf_normalization"]["targets"]["robot_root"]["skipped_events"]
            self.assertEqual(skipped, [{"event_timestamp_ns": 200, "reason": "stale_tf_edge:cell:age_ns=100"}])

    def test_cli_writes_digest_bound_log_and_report_and_probe_is_honest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            config_path = self._write(directory, "config.json", self._config())
            capture_path = self._write(directory, "capture.json", self._capture(config_path, self._happy_records()))
            log_path = directory / "observations.json"
            report_path = directory / "normalization-report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "ros_observation_adapter.py"),
                    "normalize",
                    str(self.model.path),
                    "--scene",
                    str(self.scene.path),
                    "--config",
                    str(config_path),
                    "--capture",
                    str(capture_path),
                    "--out",
                    str(log_path),
                    "--report",
                    str(report_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            response = json.loads(result.stdout)
            self.assertEqual(response["status"], "normalized")
            self.assertEqual(response["observation_log_sha256"], hashlib.sha256(log_path.read_bytes()).hexdigest())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["capture"]["sha256"], hashlib.sha256(capture_path.read_bytes()).hexdigest())
            self.assertEqual(report["output"]["observation_log_sha256"], response["observation_log_sha256"])
            resolved = TemporalObservationLog(log_path).resolve(
                self.model,
                self.scene,
                {
                    "query_id": "cli_check",
                    "time_ns": 230,
                    "maximum_age_ns": {"joint_states": 20, "robot_root_pose": 30, "object_pose": 30},
                    "fallbacks": {"robot_root": "require_observed", "objects": "require_observed"},
                    "required_object_ids": ["near_obstacle"],
                },
            )
            normalization = resolved["report"]["observation_log"]["normalization"]
            self.assertEqual(normalization["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())
            self.assertEqual(normalization["capture_sha256"], hashlib.sha256(capture_path.read_bytes()).hexdigest())
            self.assertFalse(normalization["clock_policy"]["synchronization_verified"])

            query_path = self._write(directory, "query.json", {
                "schema_version": "robot-spatial-observation-query.v1",
                "query_id": "adapter_context",
                "time_ns": 230,
                "maximum_age_ns": {"joint_states": 20, "robot_root_pose": 30, "object_pose": 30},
                "fallbacks": {"robot_root": "require_observed", "objects": "require_observed"},
                "required_object_ids": ["near_obstacle"],
            })
            context_dir = directory / "context"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "robot_spatial.py"),
                    "export",
                    str(self.model.path),
                    "--scene",
                    str(self.scene.path),
                    "--observations",
                    str(log_path),
                    "--observation-query",
                    str(query_path),
                    "--workspace-samples",
                    "0",
                    "--out",
                    str(context_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            entity_index = json.loads((context_dir / "entity-index.json").read_text(encoding="utf-8"))
            self.assertIn("ros_capture/fixture_capture", entity_index["by_entity_id"])
            fact_index = json.loads((context_dir / "fact-index.json").read_text(encoding="utf-8"))
            self.assertIn("ros_capture/fixture_capture", fact_index["by_subject"])

            probe = subprocess.run(
                [sys.executable, str(SCRIPTS / "ros_observation_adapter.py"), "probe"],
                check=True,
                capture_output=True,
                text=True,
            )
            probe_payload = json.loads(probe.stdout)
            self.assertEqual(probe_payload["deterministic_normalize"], "available")
            self.assertIn(probe_payload["live_capture"], {"available", "unavailable"})


if __name__ == "__main__":
    unittest.main()
