from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent
EXAMPLE = REPO_ROOT / "examples" / "pickcube"
LIVE_EXAMPLE = REPO_ROOT / "examples" / "pickcube-live"
LIVE_FIXTURES = REPO_ROOT / "benchmarks" / "fixtures" / "maniskill-pickcube-v1"
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(SCRIPT_DIR))

from robot_spatial_understanding.adapters import GazeboRos2Adapter, ManiSkillAdapter  # noqa: E402
from robot_spatial_understanding.action_bridge import build_action_evidence_source  # noqa: E402
from robot_spatial_understanding.benchmark import (  # noqa: E402
    BENCHMARK_SCHEMA,
    REFERENCE_SCHEMA,
    BenchmarkSuite,
)
from robot_spatial_understanding.corruption import corrupt_run  # noqa: E402
from robot_spatial_understanding.counterfactual import CounterfactualAssurance  # noqa: E402
from robot_spatial_understanding.deformable import DeformableStateSummary  # noqa: E402
from robot_spatial_understanding.errors import (  # noqa: E402
    AdapterError,
    IntegrityError,
    OracleIsolationError,
    SchemaError,
)
from robot_spatial_understanding.report import AssuranceReport  # noqa: E402
from robot_spatial_understanding.simulation import GENERIC_TRACE_SCHEMA, SimulationRun  # noqa: E402
from robot_spatial_understanding.task import TASK_SCHEMA, TaskSpec  # noqa: E402
from robot_spatial_understanding.util import sha256_json  # noqa: E402


class SimulationEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.trace = json.loads((EXAMPLE / "trace.json").read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def write_json(path: Path, value: object) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def import_trace(self, trace: dict | None = None, name: str = "run") -> SimulationRun:
        path = self.write_json(self.root / f"{name}.json", self.trace if trace is None else trace)
        return SimulationRun.import_generic_trace(path, self.root / name)

    def task(self) -> TaskSpec:
        return TaskSpec.load(EXAMPLE / "task.yaml")

    def test_import_is_deterministic_and_complete_for_available_channels(self) -> None:
        first = self.import_trace(name="first")
        second = self.import_trace(name="second")
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.completeness["status"], "complete_for_available_channels")
        self.assertEqual(
            first.manifest["channels"]["pose"]["sha256"],
            second.manifest["channels"]["pose"]["sha256"],
        )
        self.assertFalse(first.manifest["boundaries"]["official_reward_or_success_imported"])

    def test_pickcube_composite_grasp_requires_four_independent_predicates(self) -> None:
        report = AssuranceReport.evaluate(self.import_trace(), self.task())
        predicates = {item["predicate_id"]: item for item in report.data["predicates"]}
        self.assertEqual(predicates["grasp_contact"]["status"], "supported")
        self.assertEqual(predicates["follows_tool"]["status"], "supported")
        self.assertEqual(predicates["lifted"]["status"], "supported")
        self.assertEqual(predicates["grasped"]["status"], "supported")
        self.assertTrue(report.data["verdict"]["confirmed_success"])
        self.assertEqual(report.data["layers"]["controller_action_protocol"]["status"], "reported")
        self.assertEqual(report.data["layers"]["causation"]["status"], "unknown")
        self.assertEqual(report.data["layers"]["safety"]["status"], "unknown")

    def test_controller_succeeded_does_not_override_failed_world_effect(self) -> None:
        trace = copy.deepcopy(self.trace)
        trace["samples"]["pose"][-1]["entities"]["cube"]["position_m"] = [0.9, 0.0, 0.02]
        report = AssuranceReport.evaluate(self.import_trace(trace), self.task())
        self.assertEqual(report.data["layers"]["controller_action_protocol"]["latest_report"]["status"], "succeeded")
        self.assertEqual(report.data["verdict"]["simulation_bounded_physical_success"], "refuted")
        self.assertFalse(report.data["verdict"]["confirmed_success"])

    def test_contact_without_following_or_lift_is_not_a_grasp(self) -> None:
        trace = copy.deepcopy(self.trace)
        for sample in trace["samples"]["pose"][2:]:
            sample["entities"]["cube"]["position_m"] = [0.0, 0.0, 0.02]
        report = AssuranceReport.evaluate(self.import_trace(trace), self.task())
        predicates = {item["predicate_id"]: item["status"] for item in report.data["predicates"]}
        self.assertEqual(predicates["grasp_contact"], "supported")
        self.assertEqual(predicates["follows_tool"], "refuted")
        self.assertEqual(predicates["lifted"], "refuted")
        self.assertEqual(predicates["grasped"], "refuted")

    def test_missing_contact_channel_abstains_instead_of_guessing(self) -> None:
        trace = copy.deepcopy(self.trace)
        del trace["samples"]["contact"]
        report = AssuranceReport.evaluate(self.import_trace(trace), self.task())
        predicates = {item["predicate_id"]: item["status"] for item in report.data["predicates"]}
        self.assertEqual(predicates["grasp_contact"], "unknown")
        self.assertEqual(predicates["grasped"], "unknown")
        self.assertEqual(report.data["verdict"]["simulation_bounded_physical_success"], "unknown")

    def test_reward_and_official_success_fields_are_rejected(self) -> None:
        for key in ("reward", "success", "oracle_result"):
            with self.subTest(key=key):
                trace = copy.deepcopy(self.trace)
                trace[key] = True
                path = self.write_json(self.root / f"leak-{key}.json", trace)
                with self.assertRaises(IntegrityError):
                    SimulationRun.import_generic_trace(path, self.root / f"leak-{key}")

    def test_dropped_frame_marks_continuous_pose_evidence_unknown(self) -> None:
        run = self.import_trace()
        corrupted = corrupt_run(run.root, self.root / "dropped", kind="dropped-frame", channel="pose")
        dropped = SimulationRun.load(corrupted)
        self.assertEqual(dropped.channel_completeness("pose")["status"], "incomplete")
        report = AssuranceReport.evaluate(dropped, self.task())
        predicates = {item["predicate_id"]: item["status"] for item in report.data["predicates"]}
        self.assertEqual(predicates["follows_tool"], "unknown")
        self.assertEqual(predicates["grasped"], "unknown")

    def test_out_of_order_stream_is_conflicting(self) -> None:
        run = self.import_trace()
        corrupted = corrupt_run(run.root, self.root / "out-of-order", kind="out-of-order", channel="pose")
        conflicting = SimulationRun.load(corrupted)
        self.assertEqual(conflicting.channel_completeness("pose")["status"], "invalid")
        report = AssuranceReport.evaluate(conflicting, self.task())
        predicates = {item["predicate_id"]: item["status"] for item in report.data["predicates"]}
        self.assertEqual(predicates["follows_tool"], "conflicting")

    def test_conflicting_duplicate_stream_is_conflicting(self) -> None:
        run = self.import_trace()
        corrupted = corrupt_run(run.root, self.root / "conflicting-duplicate", kind="conflicting-duplicate", channel="pose")
        conflicting = SimulationRun.load(corrupted)
        issue_types = {item["type"] for item in conflicting.channel_completeness("pose")["issues"]}
        self.assertIn("conflicting_duplicate", issue_types)
        report = AssuranceReport.evaluate(conflicting, self.task())
        predicates = {item["predicate_id"]: item["status"] for item in report.data["predicates"]}
        self.assertEqual(predicates["follows_tool"], "conflicting")

    def test_missing_channel_control_marks_dependent_predicates_unknown(self) -> None:
        run = self.import_trace()
        corrupted = corrupt_run(run.root, self.root / "missing-contact", kind="missing-channel", channel="contact")
        missing = SimulationRun.load(corrupted)
        self.assertEqual(missing.channel_completeness("contact")["status"], "unavailable")
        report = AssuranceReport.evaluate(missing, self.task())
        predicates = {item["predicate_id"]: item["status"] for item in report.data["predicates"]}
        self.assertEqual(predicates["grasp_contact"], "unknown")
        self.assertEqual(predicates["grasped"], "unknown")

    def test_digest_tamper_is_rejected_before_predicate_evaluation(self) -> None:
        run = self.import_trace()
        corrupted = corrupt_run(run.root, self.root / "tampered", kind="digest-tamper", channel="pose")
        with self.assertRaises(IntegrityError):
            SimulationRun.load(corrupted)

    def test_adapter_checks_declared_simulator_family(self) -> None:
        path = self.write_json(self.root / "trace.json", self.trace)
        ManiSkillAdapter().import_source(path, self.root / "maniskill")
        with self.assertRaises(AdapterError):
            GazeboRos2Adapter().import_source(path, self.root / "gazebo")

    def test_live_pickcube_fixtures_verify_and_reproduce_reports(self) -> None:
        task = TaskSpec.load(LIVE_EXAMPLE / "task.yaml")
        expected = {
            "success-seed-002": "supported",
            "failure-seed-002-no-op": "refuted",
        }
        for fixture, verdict in expected.items():
            with self.subTest(fixture=fixture):
                root = LIVE_FIXTURES / fixture
                run = SimulationRun.load(root / "run")
                recorded = json.loads((root / "result" / "report.json").read_text(encoding="utf-8"))
                reproduced = AssuranceReport.evaluate(run, task)
                self.assertEqual(reproduced.digest, recorded["report_sha256"])
                self.assertEqual(reproduced.data["verdict"]["simulation_bounded_physical_success"], verdict)

    def test_committed_live_pickcube_records_are_digest_bound(self) -> None:
        records = REPO_ROOT / "benchmarks" / "records"
        expected = {
            "maniskill-pickcube-v1-100.json": ("benchmark_sha256", 100, 100),
            "maniskill-pickcube-v1-negatives.json": ("matrix_sha256", 8, 8),
            "maniskill-pickcube-v1-corruption.json": ("matrix_sha256", 9, 9),
            "maniskill-pickcube-v1-cuda-smoke.json": ("record_sha256", 16, 16),
        }
        for name, (digest_key, case_count, passed_count) in expected.items():
            with self.subTest(record=name):
                record = json.loads((records / name).read_text(encoding="utf-8"))
                recorded_digest = record.pop(digest_key)
                self.assertEqual(sha256_json(record), recorded_digest)
                self.assertEqual(record["case_count"], case_count)
                if "passed_count" in record:
                    self.assertEqual(record["passed_count"], passed_count)
                else:
                    self.assertEqual(sum(bool(case["agreement"]) for case in record["cases"]), passed_count)
                if name == "maniskill-pickcube-v1-100.json":
                    self.assertEqual(record["fixed_horizon"], 100)
                    self.assertTrue(record["acceptance"]["passed"])
                    self.assertEqual(record["acceptance"]["minimum_supported_references"], 25)
                    self.assertEqual(record["acceptance"]["minimum_refuted_references"], 25)
                    matrix = record["episode_metrics"]["confusion_matrix"]
                    self.assertEqual(matrix["supported"]["supported"], 51)
                    self.assertEqual(matrix["refuted"]["refuted"], 49)

    def test_pickcube_position_velocity_range_and_relative_lift_predicates(self) -> None:
        trace = copy.deepcopy(self.trace)
        trace["task_id"] = "PickCube-v1"
        for sample in trace["samples"]["joint_state"]:
            sample["positions"].update({"arm_a": 0.1, "arm_b": -0.2, "finger_b": 0.01})
            sample["velocities"] = {"arm_a": 0.1, "arm_b": -0.19}
        for sample in trace["samples"]["pose"]:
            sample["entities"]["goal"] = {
                "position_m": [0.5, 0.0, 0.2],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        task_data = {
            "schema_version": TASK_SCHEMA,
            "task_id": "PickCube-v1",
            "entities": {"cube": "cube", "goal": "goal", "arm_a": "arm_a", "arm_b": "arm_b", "finger": "finger_joint", "finger_b": "finger_b"},
            "requirements": {"channels": ["joint_state", "pose"]},
            "predicates": [
                {"predicate_id": "placed", "type": "frame_position_within_tolerance", "parameters": {"entity": "cube", "target": {"entity": "goal"}, "position_tolerance_m": 0.025}},
                {"predicate_id": "static", "type": "joint_velocity_below_threshold", "parameters": {"joints": ["arm_a", "arm_b"], "maximum_abs_velocity": 0.2}},
                {"predicate_id": "closed", "type": "joint_position_in_range", "parameters": {"ranges": {"finger": [0.0, 0.02], "finger_b": {"minimum": 0.0, "maximum": 0.02}}}},
                {"predicate_id": "lifted_delta", "type": "object_above_height", "parameters": {"entity": "cube", "minimum_delta_m": 0.02}},
            ],
            "goal": {"all": ["placed", "static"]},
        }
        report = AssuranceReport.evaluate(
            self.import_trace(trace, "official-predicates"),
            TaskSpec.load(self.write_json(self.root / "official-task.json", task_data)),
        )
        predicates = {item["predicate_id"]: item for item in report.data["predicates"]}
        self.assertEqual({item["status"] for item in predicates.values()}, {"supported"})
        self.assertAlmostEqual(predicates["placed"]["evidence"][0]["position_error_m"], 0.0)
        self.assertAlmostEqual(predicates["static"]["evidence"][0]["maximum_observed"], 0.19)
        self.assertAlmostEqual(predicates["lifted_delta"]["evidence"][0]["initial_height_m"], 0.02)

    def test_dual_contact_rows_are_not_conflicting_and_both_are_required_for_grasp(self) -> None:
        trace = copy.deepcopy(self.trace)
        contact_rows = []
        for row in trace["samples"]["contact"]:
            for finger in ("left_finger", "right_finger"):
                contact_rows.append({**row, "body_a": finger})
        trace["samples"]["contact"] = contact_rows
        task_data = {
            "schema_version": TASK_SCHEMA,
            "task_id": trace["task_id"],
            "entities": {"left": "left_finger", "right": "right_finger", "cube": "cube", "gripper": "finger_joint", "tcp": "tcp"},
            "requirements": {"channels": ["joint_state", "pose", "contact"]},
            "predicates": [
                {"predicate_id": "left", "type": "contact_sustained", "parameters": {"pair": ["left", "cube"], "minimum_duration_s": 0.6, "minimum_normal_force_n": 0.5}},
                {"predicate_id": "right", "type": "contact_sustained", "parameters": {"pair": ["right", "cube"], "minimum_duration_s": 0.6, "minimum_normal_force_n": 0.5}},
                {"predicate_id": "closed", "type": "joint_position_in_range", "parameters": {"ranges": {"gripper": [0.0, 0.01]}}},
                {"predicate_id": "follows", "type": "object_follows_frame_for_duration", "parameters": {"reference": "tcp", "target": "cube", "minimum_duration_s": 0.6, "position_tolerance_m": 0.01, "orientation_tolerance_rad": 0.01}},
                {"predicate_id": "lifted", "type": "object_above_height", "parameters": {"entity": "cube", "minimum_delta_m": 0.02}},
                {"predicate_id": "grasped", "type": "object_grasped", "parameters": {"contact_predicates": ["left", "right"], "gripper_predicate": "closed", "follows_predicate": "follows", "lift_predicate": "lifted"}},
            ],
            "goal": {"predicate": "grasped"},
        }
        run = self.import_trace(trace, "dual-contact")
        self.assertEqual(run.channel_completeness("contact")["status"], "complete")
        report = AssuranceReport.evaluate(run, TaskSpec.load(self.write_json(self.root / "dual-contact-task.json", task_data)))
        self.assertEqual(report.data["verdict"]["goal_status"], "supported")
        missing_right = copy.deepcopy(trace)
        missing_right["samples"]["contact"] = [row for row in contact_rows if row["body_a"] != "right_finger"]
        missing_report = AssuranceReport.evaluate(
            self.import_trace(missing_right, "missing-right"),
            TaskSpec.load(self.root / "dual-contact-task.json"),
        )
        statuses = {item["predicate_id"]: item["status"] for item in missing_report.data["predicates"]}
        self.assertEqual(statuses["right"], "refuted")
        self.assertEqual(statuses["grasped"], "refuted")

    def test_collision_allowed_pairs_do_not_hide_unlisted_collision(self) -> None:
        trace = copy.deepcopy(self.trace)
        trace["samples"]["collision"] = []
        for time_s in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            trace["samples"]["collision"].extend(
                [
                    {"time_s": time_s, "body_a": "cube", "body_b": "table", "active": True},
                    {"time_s": time_s, "body_a": "robot", "body_b": "wall", "active": time_s == 0.6},
                ]
            )
        task_data = {
            "schema_version": TASK_SCHEMA,
            "task_id": trace["task_id"],
            "entities": {"cube": "cube", "table": "table"},
            "requirements": {"channels": ["collision"]},
            "predicates": [{"predicate_id": "clear", "type": "collision_free_over_interval", "parameters": {"allowed_pairs": [["cube", "table"]]}}],
            "goal": {"predicate": "clear"},
        }
        task = TaskSpec.load(self.write_json(self.root / "allowed-collision-task.json", task_data))
        report = AssuranceReport.evaluate(self.import_trace(trace, "allowed-collision"), task)
        self.assertEqual(report.data["predicates"][0]["status"], "refuted")
        for row in trace["samples"]["collision"]:
            if {row["body_a"], row["body_b"]} == {"robot", "wall"}:
                row["active"] = False
        clear_report = AssuranceReport.evaluate(self.import_trace(trace, "only-allowed"), task)
        self.assertEqual(clear_report.data["predicates"][0]["status"], "supported")

    def test_pose_target_can_reference_an_observed_entity_without_embedding_goal_coordinates(self) -> None:
        trace = copy.deepcopy(self.trace)
        trace["task_id"] = "observed-goal"
        for sample in trace["samples"]["pose"]:
            sample["entities"]["goal"] = copy.deepcopy(sample["entities"]["cube"])
        task_data = {
            "schema_version": TASK_SCHEMA,
            "task_id": "observed-goal",
            "entities": {"end_effector": "cube", "goal": "goal"},
            "requirements": {"channels": ["pose"]},
            "predicates": [
                {
                    "predicate_id": "reached",
                    "type": "frame_within_pose_tolerance",
                    "parameters": {
                        "entity": "end_effector",
                        "target": {"entity": "goal"},
                        "position_tolerance_m": 1e-6,
                        "orientation_tolerance_rad": 1e-6,
                    },
                }
            ],
            "goal": {"predicate": "reached"},
        }
        report = AssuranceReport.evaluate(
            self.import_trace(trace, "observed-goal"),
            TaskSpec.load(self.write_json(self.root / "observed-goal-task.json", task_data)),
        )
        self.assertEqual(report.data["verdict"]["goal_status"], "supported")
        self.assertEqual(report.data["predicates"][0]["evidence"][0]["target"]["entity"], "goal")

    def test_malformed_channel_policy_is_a_schema_error(self) -> None:
        trace = copy.deepcopy(self.trace)
        trace["channel_policies"]["pose"] = "not-an-object"
        path = self.write_json(self.root / "bad-policy.json", trace)
        with self.assertRaises(SchemaError):
            SimulationRun.import_generic_trace(path, self.root / "bad-policy")

    def test_oracle_is_loaded_only_after_predictions_and_scores_exact_agreement(self) -> None:
        run = self.import_trace()
        task_path = self.root / "task.yaml"
        task_path.write_text((EXAMPLE / "task.yaml").read_text(encoding="utf-8"), encoding="utf-8")
        task = TaskSpec.load(task_path)
        report = AssuranceReport.evaluate(run, task)
        reference = {
            "schema_version": REFERENCE_SCHEMA,
            "case_id": "pickcube",
            "run_manifest_sha256": run.digest,
            "task_spec_sha256": task.digest,
            "predicates": {item["predicate_id"]: item["status"] for item in report.data["predicates"]},
            "verdict": report.data["verdict"]["simulation_bounded_physical_success"],
        }
        self.write_json(self.root / "oracle" / "pickcube.json", reference)
        suite = {
            "schema_version": BENCHMARK_SCHEMA,
            "suite_id": "test/pickcube",
            "cases": [
                {
                    "case_id": "pickcube",
                    "run": "run",
                    "task": "task.yaml",
                    "reference": "oracle/pickcube.json",
                }
            ],
        }
        suite_path = self.write_json(self.root / "suite.json", suite)
        result = BenchmarkSuite.load(suite_path).run(self.root / "benchmark")
        self.assertTrue(result["oracle_isolation"]["prediction_phase_completed_before_reference_load"])
        self.assertEqual(result["predicate_metrics"]["macro_f1"], 1.0)
        self.assertEqual(result["episode_metrics"]["accuracy"]["estimate"], 1.0)

    def test_simulation_predicates_bridge_into_existing_action_evidence_contract(self) -> None:
        report = AssuranceReport.evaluate(self.import_trace(), self.task())
        source = build_action_evidence_source(report, EXAMPLE / "action-map.json")
        self.assertEqual(source["schema_version"], "robot-spatial-action-evidence-source.v1")
        self.assertEqual(source["clock"]["unit"], "nanoseconds")
        self.assertEqual(source["records"][0]["value"], "true")
        self.assertEqual(source["records"][0]["evidence_type"], "effect_observation")
        self.assertIn("simulation_report_sha256=", " ".join(source["records"][0]["limitations"]))

    def test_reference_result_inside_candidate_run_is_rejected(self) -> None:
        self.import_trace()
        self.write_json(self.root / "run" / "reference.json", {})
        self.write_json(self.root / "task.json", {"schema_version": TASK_SCHEMA})
        suite = {
            "schema_version": BENCHMARK_SCHEMA,
            "suite_id": "test/leak",
            "cases": [
                {"case_id": "leak", "run": "run", "task": "task.json", "reference": "run/reference.json"}
            ],
        }
        with self.assertRaises(OracleIsolationError):
            BenchmarkSuite.load(self.write_json(self.root / "suite.json", suite))

    def test_agv_goal_and_polyline_corridor_predicates(self) -> None:
        trace = copy.deepcopy(self.trace)
        trace["run_id"] = "example/gazebo-agv/seed-3"
        trace["task_id"] = "agv/reach-goal"
        trace["simulator"] = {"name": "Gazebo Harmonic", "version": "8"}
        trace["samples"]["odometry"] = [
            {
                "time_s": index * 0.2,
                "entity": "base",
                "position_m": [index * 0.2, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "linear_velocity_mps": [1.0, 0.0, 0.0],
                "angular_velocity_radps": [0.0, 0.0, 0.0],
            }
            for index in range(6)
        ]
        task_data = {
            "schema_version": TASK_SCHEMA,
            "task_id": "agv/reach-goal",
            "entities": {"base": "base"},
            "requirements": {"channels": ["odometry", "collision"]},
            "predicates": [
                {
                    "predicate_id": "reached",
                    "type": "base_reached_goal",
                    "parameters": {
                        "channel": "odometry",
                        "entity": "base",
                        "target": {"position_m": [1.0, 0.0, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
                        "position_tolerance_m": 0.01,
                        "orientation_tolerance_rad": 0.01,
                    },
                },
                {
                    "predicate_id": "corridor",
                    "type": "path_stayed_within_corridor",
                    "parameters": {
                        "channel": "odometry",
                        "entity": "base",
                        "corridor": {"type": "polyline_xy", "points_m": [[0.0, 0.0], [1.0, 0.0]], "half_width_m": 0.05},
                    },
                },
                {"predicate_id": "collision_free", "type": "collision_free_over_interval", "parameters": {}},
            ],
            "goal": {"all": ["reached", "corridor", "collision_free"]},
        }
        task_path = self.write_json(self.root / "agv-task.json", task_data)
        report = AssuranceReport.evaluate(self.import_trace(trace, "agv"), TaskSpec.load(task_path))
        self.assertEqual(report.data["verdict"]["goal_status"], "supported")

    def test_deformable_keypoint_summary_does_not_claim_full_surface(self) -> None:
        trace = copy.deepcopy(self.trace)
        trace["samples"]["deformable"] = [
            {
                "time_s": time_s,
                "entity": "cloth",
                "keypoints_m": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.1]],
            }
            for time_s in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        ]
        run = self.import_trace(trace)
        summary = DeformableStateSummary.from_run(run, "cloth")
        self.assertEqual(summary.keypoint_count, 3)
        self.assertIn("does not prove topology", summary.to_dict()["limitations"][0])
        task_data = {
            "schema_version": TASK_SCHEMA,
            "task_id": "maniskill/PickCube-v1",
            "entities": {"cloth": "cloth"},
            "requirements": {"channels": ["deformable"]},
            "predicates": [
                {
                    "predicate_id": "contained_keypoints",
                    "type": "deformable_keypoints_in_region",
                    "parameters": {
                        "entity": "cloth",
                        "minimum_fraction": 1.0,
                        "region": {"type": "aabb", "min_m": [-0.1, -0.1, -0.1], "max_m": [1.1, 1.1, 0.2]},
                    },
                },
                {
                    "predicate_id": "shape",
                    "type": "deformable_shape_within_tolerance",
                    "parameters": {
                        "entity": "cloth",
                        "expected_keypoints_m": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.1]],
                        "rmse_tolerance_m": 0.001,
                        "maximum_point_error_m": 0.001,
                    },
                },
            ],
            "goal": {"all": ["contained_keypoints", "shape"]},
        }
        task = TaskSpec.load(self.write_json(self.root / "deformable-task.json", task_data))
        report = AssuranceReport.evaluate(run, task)
        self.assertEqual(report.data["verdict"]["goal_status"], "supported")
        self.assertIn("unobserved surface regions", report.data["predicates"][0]["limitations"][0])

    def test_inserted_to_depth_uses_declared_axis_and_lateral_tolerance(self) -> None:
        trace = copy.deepcopy(self.trace)
        trace["run_id"] = "example/scara-insertion/seed-2"
        trace["task_id"] = "scara/insert"
        for sample in trace["samples"]["pose"]:
            sample["entities"]["socket"] = {
                "position_m": [0.0, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
            sample["entities"]["peg"] = {
                "position_m": [0.001, 0.0, 0.08],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        task_data = {
            "schema_version": TASK_SCHEMA,
            "task_id": "scara/insert",
            "entities": {"peg": "peg", "socket": "socket"},
            "requirements": {"channels": ["pose"]},
            "predicates": [
                {
                    "predicate_id": "inserted",
                    "type": "inserted_to_depth",
                    "parameters": {
                        "entity": "peg",
                        "reference": "socket",
                        "axis": [0.0, 0.0, 1.0],
                        "minimum_depth_m": 0.05,
                        "maximum_lateral_error_m": 0.002,
                    },
                }
            ],
            "goal": {"predicate": "inserted"},
        }
        task = TaskSpec.load(self.write_json(self.root / "insertion-task.json", task_data))
        report = AssuranceReport.evaluate(self.import_trace(trace, "insertion"), task)
        self.assertEqual(report.data["verdict"]["goal_status"], "supported")

    def test_release_requires_region_and_absence_of_sustained_contact(self) -> None:
        trace = copy.deepcopy(self.trace)
        trace["run_id"] = "example/release/seed-4"
        trace["task_id"] = "manipulation/release"
        for sample in trace["samples"]["contact"]:
            sample["active"] = False
            sample["normal_force_n"] = 0.0
        trace["samples"]["joint_state"][-1]["positions"]["finger_joint"] = 0.04
        task_data = {
            "schema_version": TASK_SCHEMA,
            "task_id": "manipulation/release",
            "entities": {"tool": "tcp", "target": "cube", "gripper_joint": "finger_joint"},
            "requirements": {"channels": ["joint_state", "pose", "contact"]},
            "predicates": [
                {
                    "predicate_id": "inside",
                    "type": "object_inside_region",
                    "parameters": {
                        "entity": "target",
                        "region": {"type": "aabb", "min_m": [0.45, -0.05, 0.15], "max_m": [0.55, 0.05, 0.25]},
                    },
                },
                {
                    "predicate_id": "contact",
                    "type": "contact_sustained",
                    "parameters": {"pair": ["tool", "target"], "minimum_duration_s": 0.0},
                },
                {
                    "predicate_id": "gripper_open",
                    "type": "joint_within_tolerance",
                    "parameters": {"targets": {"gripper_joint": 0.04}, "tolerance": 0.005},
                },
                {
                    "predicate_id": "released",
                    "type": "object_released_in_region",
                    "parameters": {
                        "inside_predicate": "inside",
                        "contact_predicate": "contact",
                        "gripper_predicate": "gripper_open",
                    },
                },
            ],
            "goal": {"predicate": "released"},
        }
        task = TaskSpec.load(self.write_json(self.root / "release-task.json", task_data))
        report = AssuranceReport.evaluate(self.import_trace(trace, "release"), task)
        predicates = {item["predicate_id"]: item["status"] for item in report.data["predicates"]}
        self.assertEqual(predicates["contact"], "refuted")
        self.assertEqual(predicates["released"], "supported")

    def test_matched_no_op_replay_supports_only_simulation_bounded_contribution(self) -> None:
        action_trace = copy.deepcopy(self.trace)
        action_run = self.import_trace(action_trace, "action")
        control_trace = copy.deepcopy(self.trace)
        control_trace["run_id"] = "example/maniskill-pickcube/seed-7/no-op"
        control_trace["intervention"] = {"type": "no_op"}
        for index, sample in enumerate(control_trace["samples"]["joint_state"]):
            sample["positions"]["finger_joint"] = 0.04
        initial_tool = [0.0, 0.0, 0.1]
        initial_cube = [0.0, 0.0, 0.02]
        for index, sample in enumerate(control_trace["samples"]["pose"]):
            sample["entities"]["tcp"]["position_m"] = initial_tool
            sample["entities"]["cube"]["position_m"] = initial_cube
        for sample in control_trace["samples"]["contact"]:
            sample["active"] = False
            sample["normal_force_n"] = 0.0
        control_run = self.import_trace(control_trace, "control")
        comparison = CounterfactualAssurance.compare(action_run, control_run, self.task())
        self.assertEqual(
            comparison.data["causal_contribution"]["status"],
            "supported_under_controlled_simulation",
        )
        self.assertIn("does not prove real-world causation", comparison.data["limitations"][0])

    def test_installed_cli_forwards_legacy_validate(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "robot_spatial_understanding", "validate", str(FIXTURES / "two_dof.urdf")],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(json.loads(completed.stdout)["status"], "valid")


if __name__ == "__main__":
    unittest.main()
