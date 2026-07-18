#!/usr/bin/env python3

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
import spatial_invariants
from temporal_observation import ObservationError, TemporalObservationLog, read_observation_query, resolve_observation
from world_scene import WorldScene


class TemporalObservationTests(unittest.TestCase):
    def setUp(self):
        self.model = robot_spatial.RobotModel(FIXTURES / "two_dof.urdf")
        self.scene = WorldScene(
            FIXTURES / "world_scene.json",
            expected_robot_name=self.model.name,
            expected_root_link=self.model.root_link,
        )

    @staticmethod
    def _source(reference="test sensor"):
        return {
            "type": "synthetic",
            "reference": reference,
            "sensor_id": "fixture_sensor",
            "topic": None,
        }

    def _log(self):
        source = self._source()
        return {
            "schema_version": "robot-spatial-observation-log.v1",
            "observation_log_id": "fixture_log",
            "clock": {"domain": "dataset", "unit": "nanoseconds", "epoch": "fixture_epoch"},
            "binding": {
                "robot_name": self.model.name,
                "root_link": self.model.root_link,
                "source_urdf_semantic_sha256": self.model.semantic_sha256,
                "scene_id": self.scene.scene_id,
                "scene_sha256": self.scene.sha256,
            },
            "source": source,
            "streams": {
                "joint_states": [
                    {
                        "sample_id": "j100",
                        "timestamp_ns": 100,
                        "positions": {"shoulder": 0.0, "slide": 0.5},
                        "position_standard_deviation": {"shoulder": 0.01},
                        "source": source,
                    },
                    {
                        "sample_id": "j300_future",
                        "timestamp_ns": 300,
                        "positions": {"shoulder": 1.0, "slide": 0.1},
                        "source": source,
                    },
                ],
                "robot_root_poses": [
                    {
                        "sample_id": "r200",
                        "timestamp_ns": 200,
                        "parent_scene_frame": "world",
                        "pose": {"xyz_m": [10.0, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0]},
                        "source": source,
                    }
                ],
                "object_poses": {
                    "near_obstacle": [
                        {
                            "sample_id": "o_near_200",
                            "timestamp_ns": 200,
                            "parent_scene_frame": "world",
                            "pose": {"xyz_m": [11.0, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0]},
                            "source": source,
                        }
                    ],
                    "far_sphere": [
                        {
                            "sample_id": "o_far_200",
                            "timestamp_ns": 200,
                            "parent_scene_frame": "world",
                            "pose": {"xyz_m": [20.0, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0]},
                            "source": source,
                        }
                    ],
                },
            },
        }

    def _query(self, *, maximum_age=200, fallback="require_observed"):
        return {
            "schema_version": "robot-spatial-observation-query.v1",
            "query_id": "at_250",
            "time_ns": 250,
            "maximum_age_ns": {
                "joint_states": maximum_age,
                "robot_root_pose": maximum_age,
                "object_pose": maximum_age,
            },
            "fallbacks": {"robot_root": fallback, "objects": fallback},
            "required_object_ids": ["near_obstacle", "far_sphere"],
        }

    def _write(self, directory, name, data):
        path = directory / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_latest_past_sample_is_selected_and_future_state_is_never_consumed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            log_path = self._write(directory, "observations.json", self._log())
            query_path = self._write(directory, "query.json", self._query())
            log = TemporalObservationLog(log_path)
            query, _ = read_observation_query(query_path, self.scene)
            resolved = log.resolve(self.model, self.scene, query)
            report = resolved["report"]
            self.assertEqual(report["status"], "current")
            self.assertTrue(report["readiness"]["all_required_observations_current"])
            selected = report["selections"]["joint_states"]
            self.assertEqual(selected["selected_sample"]["sample_id"], "j100")
            self.assertEqual(selected["future_samples_ignored_count"], 1)
            self.assertEqual(resolved["joint_pose"], {"shoulder": 0.0, "slide": 0.5})
            transform = self.scene.transform(
                "scene_frame/world",
                "robot_frame/tool0",
                self.model,
                resolved["joint_pose"],
                world_from_robot_root=resolved["world_from_robot_root"],
                world_from_objects=resolved["world_from_objects"],
            )
            self.assertAlmostEqual(transform[0][3], 12.5)
            self.assertAlmostEqual(transform[2][3], 0.2)

    def test_stale_required_stream_fails_closed_and_explicit_fallback_is_labeled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            log_path = self._write(directory, "observations.json", self._log())
            strict_path = self._write(directory, "strict.json", self._query(maximum_age=20))
            log = TemporalObservationLog(log_path)
            strict_query, _ = read_observation_query(strict_path, self.scene)
            strict = log.resolve(self.model, self.scene, strict_query)
            self.assertFalse(strict["nominal_computable"])
            self.assertEqual(strict["report"]["selections"]["joint_states"]["status"], "stale")
            self.assertEqual(strict["report"]["status"], "not_current_or_incomplete")

            fallback_query = self._query(maximum_age=200, fallback="allow_static_declaration")
            fallback_query["maximum_age_ns"]["robot_root_pose"] = 20
            fallback_query["maximum_age_ns"]["object_pose"] = 20
            fallback_path = self._write(directory, "fallback.json", fallback_query)
            parsed, _ = read_observation_query(fallback_path, self.scene)
            fallback = log.resolve(self.model, self.scene, parsed)
            self.assertTrue(fallback["nominal_computable"])
            self.assertFalse(fallback["all_required_current"])
            self.assertEqual(fallback["report"]["status"], "nominal_with_declaration_fallback")
            self.assertIn("robot_root", fallback["report"]["readiness"]["declaration_fallback_entities"])
            self.assertEqual(
                fallback["report"]["effective_state"]["sources"]["robot_root"]["layer"],
                "static_scene_declaration",
            )

    def test_binding_missing_driver_and_duplicate_timestamp_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            query_path = self._write(directory, "query.json", self._query())
            query, _ = read_observation_query(query_path, self.scene)

            mismatch = self._log()
            mismatch["binding"]["scene_sha256"] = "wrong"
            mismatch_path = self._write(directory, "mismatch.json", mismatch)
            with self.assertRaisesRegex(ObservationError, "binding mismatch"):
                TemporalObservationLog(mismatch_path).resolve(self.model, self.scene, query)

            missing = self._log()
            del missing["streams"]["joint_states"][0]["positions"]["slide"]
            missing_path = self._write(directory, "missing.json", missing)
            with self.assertRaisesRegex(ObservationError, "missing independent drivers"):
                TemporalObservationLog(missing_path).resolve(self.model, self.scene, query)

            duplicate = self._log()
            duplicate["streams"]["joint_states"][1]["timestamp_ns"] = 100
            duplicate_path = self._write(directory, "duplicate.json", duplicate)
            with self.assertRaisesRegex(ObservationError, "timestamps must be unique"):
                TemporalObservationLog(duplicate_path)

    def test_cli_exposes_observation_policy_and_never_promotes_nominal_collision_to_safety(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            log_path = self._write(directory, "observations.json", self._log())
            query_path = self._write(directory, "query.json", self._query())
            command = [
                sys.executable,
                str(SCRIPTS / "robot_spatial.py"),
                "observe-collisions",
                str(self.model.path),
                "--scene",
                str(self.scene.path),
                "--observations",
                str(log_path),
                "--observation-query",
                str(query_path),
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["status"].endswith("under_current_selected_observations"))
            self.assertEqual(payload["physical_collision_status"], "not_established")
            self.assertEqual(payload["safety_conclusion"], "not_established")
            self.assertFalse(payload["observation"]["selection_method"]["future_samples_consumed"])
            evidence = payload["query_evidence"]
            self.assertEqual(evidence["source_observation_log_sha256"], TemporalObservationLog(log_path).sha256)
            self.assertEqual(evidence["query_time_ns"], 250)

    def test_export_teaches_agent_context_three_distinct_epistemic_layers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            log_path = self._write(directory, "observations.json", self._log())
            query_path = self._write(directory, "query.json", self._query())
            output = directory / "context"
            private_key = directory / "private" / "answer-key.jsonl"
            result = subprocess.run(
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
                    "--generate-evaluation",
                    "--evaluation-key-out",
                    str(private_key),
                    "--out",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(result.stdout)["status"], "exported")
            model = json.loads((output / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(model["pose"]["name"], "observed/at_250")
            self.assertEqual(model["observed_world"]["observation"]["status"], "current")
            self.assertEqual(
                model["observed_world"]["observation"]["epistemic_layers"]["model"],
                "URDF mechanism declarations and deterministic kinematic consequences",
            )
            facts = [json.loads(line) for line in (output / "facts.jsonl").read_text(encoding="utf-8").splitlines()]
            predicates = {fact["predicate"] for fact in facts}
            self.assertIn("has_time_selection_result", predicates)
            self.assertIn("has_nominal_observation_conditioned_collision_status", predicates)
            manifest = json.loads((output / "agent-context.json").read_text(encoding="utf-8"))
            self.assertIn("observation_log", manifest["identity_grammar"])
            self.assertEqual(manifest["statistics"]["entity_type_counts"]["observation_log"], 1)
            guide = (output / "agent-guide.md").read_text(encoding="utf-8")
            self.assertIn("three separate epistemic layers", guide)
            context = (output / "context.md").read_text(encoding="utf-8")
            self.assertIn("## Timestamped observed world", context)
            self.assertIn("future samples", context)
            evaluation_manifest = json.loads((output / "evaluation" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(evaluation_manifest["capability_counts"]["temporal_observation"], 4)
            self.assertEqual(evaluation_manifest["capability_counts"]["temporal_observation_collision"], 1)
            self.assertTrue(private_key.is_file())

    def test_temporal_invariants_bind_log_query_readiness_transform_and_nominal_collision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            log_path = self._write(directory, "observations.json", self._log())
            query_path = self._write(directory, "query.json", self._query())
            resolved = resolve_observation(log_path, query_path, self.model, self.scene)
            contract_data = {
                "schema_version": "robot-spatial-invariants.v1",
                "robot": self.model.name,
                "world_scene": {
                    "scene_id": self.scene.scene_id,
                    "snapshot_id": self.scene.snapshot["id"],
                    "sha256": self.scene.sha256,
                },
                "observation": {
                    "log_id": resolved["report"]["observation_log"]["id"],
                    "log_sha256": resolved["report"]["observation_log"]["sha256"],
                    "query_id": resolved["report"]["query"]["query_id"],
                    "query_sha256": resolved["query_sha256"],
                },
                "assertions": [
                    {
                        "id": "time_current",
                        "type": "observation_readiness",
                        "expected": {
                            "status": "current",
                            "all_required_observations_current": True,
                            "nominal_world_state_computable": True,
                            "declaration_fallback_used": False,
                            "declaration_fallback_entities": [],
                        },
                    },
                    {
                        "id": "observed_tool_pose",
                        "type": "observation_transform",
                        "from": "scene_frame/world",
                        "to": "robot_frame/tool0",
                        "expected": {
                            "translation_xyz_m": [12.5, 0.0, 0.2],
                            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                    {
                        "id": "nominal_collision_boundary",
                        "type": "observation_collision",
                        "expected": {
                            "nominal_status": "indeterminate",
                            "analysis_status": "computed_from_current_observations",
                            "all_required_observations_current": True,
                        },
                    },
                ],
            }
            contract_path = self._write(directory, "invariants.json", contract_data)
            contract = spatial_invariants.read_invariant_contract(
                contract_path,
                self.model,
                self.scene,
                resolved,
            )
            report = spatial_invariants.verify_invariant_contract(
                self.model,
                contract,
                package_map_path=None,
                world_scene=self.scene,
                observation_resolved=resolved,
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["observation"]["query_id"], "at_250")
            collision = next(item for item in report["results"] if item["id"] == "nominal_collision_boundary")
            self.assertEqual(collision["physical_collision_status"], "not_established")
            self.assertEqual(collision["safety_conclusion"], "not_established")

            contract_data["observation"]["query_sha256"] = "wrong"
            contract_path.write_text(json.dumps(contract_data), encoding="utf-8")
            with self.assertRaisesRegex(spatial_invariants.InvariantError, "query_sha256"):
                spatial_invariants.read_invariant_contract(contract_path, self.model, self.scene, resolved)

    def test_prepare_propagates_digest_bound_temporal_sources_into_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            log_path = self._write(directory, "observations.json", self._log())
            query_path = self._write(directory, "query.json", self._query())
            output = directory / "prepared"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "robot_spatial.py"),
                    "prepare",
                    str(self.model.path),
                    "--scene",
                    str(self.scene.path),
                    "--observations",
                    str(log_path),
                    "--observation-query",
                    str(query_path),
                    "--workspace-samples",
                    "0",
                    "--workspace-root",
                    str(FIXTURES),
                    "--out",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "prepared")
            self.assertEqual(payload["temporal_observation"]["log"]["sha256"], TemporalObservationLog(log_path).sha256)
            model = json.loads((output / "context" / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(model["observed_world"]["observation"]["query"]["query_id"], "at_250")
            self.assertEqual(model["source_compilation"]["temporal_observation"]["query"]["sha256"], payload["temporal_observation"]["query"]["sha256"])
            source_manifest = json.loads((output / "source-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(source_manifest["temporal_observation"]["log"]["sha256"], payload["temporal_observation"]["log"]["sha256"])


if __name__ == "__main__":
    unittest.main()
