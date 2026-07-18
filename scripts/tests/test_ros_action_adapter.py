from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
FIXTURES = TEST_DIR / "fixtures"
ROBOT_SPATIAL = SCRIPT_DIR / "robot_spatial.py"
ADAPTER = SCRIPT_DIR / "ros_action_adapter.py"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(TEST_DIR))

from ros_action_adapter import (  # noqa: E402
    CAPTURE_SCHEMA,
    CONFIG_SCHEMA,
    RosActionAdapterError,
    build_bundle_and_report,
    execute_capture,
    make_config,
    normalize,
    _publisher_id,
    read_capture,
    read_config,
)
from spatial_action_assurance import SOURCE_SCHEMA, build_action_assurance  # noqa: E402
from spatial_functional import read_functional_model, write_functional_model_from_context  # noqa: E402
import test_spatial_functional as functional_test_helpers  # noqa: E402


class RosActionAdapterTests(unittest.TestCase):
    clock = {"domain": "test_monotonic", "unit": "nanoseconds", "epoch": "capture-1"}
    goal_uuid = "0123456789abcdef0123456789abcdef"

    @staticmethod
    def write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def canonical_sha(value: object) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def run_cli(self, *arguments: object, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ADAPTER), *(str(value) for value in arguments)],
            check=check,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def prepare_functional_model(self, root: Path) -> Path:
        context = root / "context"
        subprocess.run(
            [
                sys.executable,
                str(ROBOT_SPATIAL),
                "export",
                str(FIXTURES / "mimic_branch.urdf"),
                "--workspace-samples",
                "0",
                "--out",
                str(context),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        spec = functional_test_helpers.SpatialFunctionalModelTests.spec(context)
        spec_path = root / "functional-spec.json"
        model_path = root / "functional-model.json"
        self.write_json(spec_path, spec)
        write_functional_model_from_context(context, spec_path, model_path)
        return model_path

    def config_value(self, functional_model_path: Path) -> dict:
        functional = read_functional_model(functional_model_path)
        return {
            "schema_version": CONFIG_SCHEMA,
            "adapter_id": "test_grasp",
            "clock": self.clock,
            "functional_model_binding": {
                "functional_model_id": functional["functional_model_id"],
                "functional_model_sha256": functional["functional_model_sha256"],
                "functional_model_artifact_sha256": self.sha256(functional_model_path),
            },
            "ros_action": {
                "name": "/gripper/grasp",
                "type": "example_interfaces/action/Fibonacci",
                "status_topic": "/gripper/grasp/_action/status",
            },
            "action_instance": {
                "action_instance_id": "action_instance/test-action",
                "affordance_id": "affordance/grasp",
                "offered_by": "component/gripper",
                "action_verb": "grasp",
                "target_object_type": "object_type/graspable",
                "target_instance_id": "object_instance/test-object",
                "argument_bindings": {
                    "actor": "component/gripper",
                    "target": "object_instance/test-object",
                },
            },
            "evidence_policy": {
                "maximum_age_ns": {
                    "operator_confirmation": 50,
                    "planner_verification": 50,
                    "project_assumption": 1000,
                    "runtime_observation": 50,
                },
                "require_goal_acceptance_before_status": True,
                "require_terminal_result_status_match": True,
            },
            "policies": {
                "reject_multiple_status_publishers": True,
                "reject_conflicting_same_time_status": True,
            },
        }

    def write_config(self, root: Path, functional_model_path: Path) -> Path:
        path = root / "action-config.json"
        self.write_json(path, self.config_value(functional_model_path))
        return path

    def record(
        self,
        sequence: int,
        kind: str,
        timestamp: int,
        payload: dict,
        publisher_id: str | None = None,
    ) -> dict:
        return {
            "record_id": f"ros_action_record/test/{sequence:06d}",
            "sequence": sequence,
            "kind": kind,
            "event_timestamp_ns": timestamp,
            "publisher_id": publisher_id,
            "payload": payload,
        }

    def capture_value(self, config_path: Path, *, records: list[dict] | None = None) -> dict:
        goal_payload = {"order": 5}
        if records is None:
            records = [
                self.record(1, "send_goal_request", 145, {}),
                self.record(
                    2,
                    "goal_response",
                    150,
                    {"accepted": True, "server_acceptance_timestamp_ns": 148},
                ),
                self.record(
                    3,
                    "status_array",
                    160,
                    {"statuses": [{
                        "goal_uuid": self.goal_uuid,
                        "accepted_at_ns": 148,
                        "status_code": 2,
                    }]},
                    "rmw:test-server",
                ),
                self.record(
                    4,
                    "feedback",
                    165,
                    {"goal_uuid": self.goal_uuid, "feedback": {"sequence": [0, 1, 1]}},
                ),
                self.record(
                    5,
                    "status_array",
                    180,
                    {"statuses": [{
                        "goal_uuid": self.goal_uuid,
                        "accepted_at_ns": 148,
                        "status_code": 4,
                    }]},
                    "rmw:test-server",
                ),
                self.record(6, "get_result_request", 181, {}),
                self.record(
                    7,
                    "result_response",
                    185,
                    {"goal_uuid": self.goal_uuid, "status_code": 4, "result": {"sequence": [0, 1, 1, 2]}},
                ),
            ]
        return {
            "schema_version": CAPTURE_SCHEMA,
            "capture_id": "test-capture",
            "adapter_config_sha256": self.sha256(config_path),
            "clock": self.clock,
            "interval": {
                "started_at_ns": 100,
                "requested_at_ns": 100,
                "decision_time_ns": 140,
                "evaluation_time_ns": 200,
                "ended_at_ns": 200,
                "termination_reason": "result_received",
            },
            "source": {
                "transport": "synthetic_fixture",
                "reference": "unit test",
                "ros_distro": None,
                "status_publisher_identity_visibility": "synthetic fixture IDs",
                "service_server_identity_visibility": "unavailable",
                "feedback_publisher_identity_visibility": "unavailable",
            },
            "action": {
                "name": "/gripper/grasp",
                "type": "example_interfaces/action/Fibonacci",
                "goal_uuid": self.goal_uuid,
                "goal_payload": goal_payload,
                "goal_payload_sha256": self.canonical_sha(goal_payload),
            },
            "client": {"node_name": "test_client", "use_sim_time": False},
            "records": records,
        }

    def write_capture(self, root: Path, config_path: Path, *, records: list[dict] | None = None) -> Path:
        path = root / "action-capture.json"
        self.write_json(path, self.capture_value(config_path, records=records))
        return path

    def supplemental_source(self, root: Path) -> Path:
        path = root / "evidence" / "conditions-effects.json"
        source = {
            "schema_version": SOURCE_SCHEMA,
            "source_id": "evidence_source/runtime",
            "clock": self.clock,
            "producer": {"producer_id": "test/observer", "producer_type": "synthetic_test_harness"},
            "records": [
                {
                    "record_id": "evidence/between",
                    "evidence_type": "runtime_observation",
                    "subject_ref": "condition/target_between_fingers",
                    "predicate": "target_between_fingers",
                    "bindings": {"actor": "component/gripper", "target": "object_instance/test-object"},
                    "value": "true",
                    "observed_at_ns": 130,
                    "valid_until_ns": None,
                    "claim_scope": "Synthetic condition observation.",
                    "limitations": ["Does not establish producer truthfulness."],
                },
                {
                    "record_id": "evidence/plan",
                    "evidence_type": "planner_verification",
                    "subject_ref": "condition/plan_approved",
                    "predicate": "plan_approved",
                    "bindings": {"actor": "component/gripper", "target": "object_instance/test-object"},
                    "value": "true",
                    "observed_at_ns": 135,
                    "valid_until_ns": None,
                    "claim_scope": "Synthetic planner report.",
                    "limitations": ["Does not authorize dispatch."],
                },
                {
                    "record_id": "evidence/retained",
                    "evidence_type": "effect_observation",
                    "subject_ref": "effect/target_retained",
                    "predicate": "target_retained_by",
                    "bindings": {"actor": "component/gripper", "target": "object_instance/test-object"},
                    "value": "true",
                    "observed_at_ns": 190,
                    "valid_until_ns": None,
                    "claim_scope": "Synthetic post-action observation.",
                    "limitations": ["Does not establish that the action caused the observation."],
                },
            ],
        }
        self.write_json(path, source)
        return path

    def loaded_inputs(self, root: Path):
        functional_path = self.prepare_functional_model(root)
        config_path = self.write_config(root, functional_path)
        capture_path = self.write_capture(root, config_path)
        return (
            functional_path,
            read_functional_model(functional_path),
            read_config(config_path),
            read_capture(capture_path, read_config(config_path)),
        )

    def test_happy_path_feeds_action_assurance_without_promoting_result_to_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path, functional, config, capture = self.loaded_inputs(root)
            source, report = normalize(functional_path, functional, config, capture)
            supplemental = self.supplemental_source(root)
            source_out = root / "evidence" / "ros-action.json"
            bundle_out = root / "action-bundle.json"
            bundle, report = build_bundle_and_report(
                functional_path,
                functional,
                config,
                capture,
                source,
                report,
                source_out,
                bundle_out,
                [supplemental],
            )
            self.write_json(source_out, source)
            self.write_json(bundle_out, bundle)
            assurance = build_action_assurance(functional_path, bundle_out)

            self.assertEqual([item["value"] for item in source["records"]], [
                "accepted", "executing", "succeeded", "succeeded"
            ])
            self.assertFalse(report["feedback"]["promoted_to_condition_or_effect_evidence"])
            self.assertFalse(report["result"]["promoted_to_effect_observation"])
            self.assertEqual(assurance["projections"]["lifecycle"]["status"], "result_succeeded")
            self.assertEqual(
                assurance["projections"]["readiness"]["conclusion"],
                "ready_under_declared_model_and_evidence",
            )
            self.assertEqual(
                assurance["projections"]["effect_summary"]["status"],
                "all_declared_effects_observed_true_after_execution_started",
            )
            self.assertEqual(assurance["projections"]["outcome"]["causal_success"], "not_established")

    def test_rejected_goal_normalizes_without_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path = self.prepare_functional_model(root)
            config_path = self.write_config(root, functional_path)
            records = [
                self.record(1, "send_goal_request", 145, {}),
                self.record(2, "goal_response", 150, {
                    "accepted": False,
                    "server_acceptance_timestamp_ns": 0,
                }),
            ]
            capture_path = self.write_capture(root, config_path, records=records)
            config = read_config(config_path)
            capture = read_capture(capture_path, config)
            source, report = normalize(
                functional_path,
                read_functional_model(functional_path),
                config,
                capture,
            )
            self.assertEqual(len(source["records"]), 1)
            self.assertEqual(source["records"][0]["value"], "rejected")
            self.assertIsNone(report["result"])

    def test_unknown_status_and_other_goal_are_audited_not_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path = self.prepare_functional_model(root)
            config_path = self.write_config(root, functional_path)
            other_uuid = "f" * 32
            records = [
                self.record(1, "send_goal_request", 145, {}),
                self.record(2, "goal_response", 150, {
                    "accepted": True,
                    "server_acceptance_timestamp_ns": 148,
                }),
                self.record(3, "status_array", 160, {"statuses": [
                    {"goal_uuid": self.goal_uuid, "accepted_at_ns": 148, "status_code": 0},
                    {"goal_uuid": other_uuid, "accepted_at_ns": 120, "status_code": 2},
                ]}, "rmw:one"),
            ]
            capture_path = self.write_capture(root, config_path, records=records)
            config = read_config(config_path)
            source, report = normalize(
                functional_path,
                read_functional_model(functional_path),
                config,
                read_capture(capture_path, config),
            )
            self.assertEqual([item["evidence_type"] for item in source["records"]], ["goal_response"])
            self.assertEqual(report["status_normalization"]["unknown_target_status_count"], 1)
            self.assertEqual(report["status_normalization"]["ignored_other_goal_status_count"], 1)

    def test_multiple_visible_status_publishers_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path = self.prepare_functional_model(root)
            config_path = self.write_config(root, functional_path)
            capture = self.capture_value(config_path)
            capture["records"][4]["publisher_id"] = "rmw:second-server"
            capture_path = root / "capture.json"
            self.write_json(capture_path, capture)
            config = read_config(config_path)
            with self.assertRaisesRegex(RosActionAdapterError, "multiple visible publishers"):
                normalize(
                    functional_path,
                    read_functional_model(functional_path),
                    config,
                    read_capture(capture_path, config),
                )

    def test_conflicting_same_time_target_status_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path = self.prepare_functional_model(root)
            config_path = self.write_config(root, functional_path)
            capture = self.capture_value(config_path)
            capture["records"][2]["payload"]["statuses"].append({
                "goal_uuid": self.goal_uuid,
                "accepted_at_ns": 148,
                "status_code": 4,
            })
            capture_path = root / "capture.json"
            self.write_json(capture_path, capture)
            config = read_config(config_path)
            with self.assertRaisesRegex(RosActionAdapterError, "conflicting same-receipt-time"):
                normalize(
                    functional_path,
                    read_functional_model(functional_path),
                    config,
                    read_capture(capture_path, config),
                )

    def test_nonterminal_result_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path = self.prepare_functional_model(root)
            config_path = self.write_config(root, functional_path)
            capture = self.capture_value(config_path)
            capture["records"][-1]["payload"]["status_code"] = 2
            capture_path = root / "capture.json"
            self.write_json(capture_path, capture)
            config = read_config(config_path)
            with self.assertRaisesRegex(RosActionAdapterError, "must be terminal"):
                normalize(
                    functional_path,
                    read_functional_model(functional_path),
                    config,
                    read_capture(capture_path, config),
                )

    def test_missing_goal_response_is_not_invented(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path = self.prepare_functional_model(root)
            config_path = self.write_config(root, functional_path)
            capture_path = self.write_capture(
                root,
                config_path,
                records=[self.record(1, "send_goal_request", 145, {})],
            )
            config = read_config(config_path)
            with self.assertRaisesRegex(RosActionAdapterError, "do not invent acceptance"):
                normalize(
                    functional_path,
                    read_functional_model(functional_path),
                    config,
                    read_capture(capture_path, config),
                )

    def test_result_requires_get_result_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path = self.prepare_functional_model(root)
            config_path = self.write_config(root, functional_path)
            capture = self.capture_value(config_path)
            capture["records"].pop(-2)
            for index, record in enumerate(capture["records"], start=1):
                record["sequence"] = index
                record["record_id"] = f"ros_action_record/test/{index:06d}"
            capture_path = root / "capture.json"
            self.write_json(capture_path, capture)
            with self.assertRaisesRegex(RosActionAdapterError, "requires an observed get_result_request"):
                read_capture(capture_path, read_config(config_path))

    def test_capture_and_config_digests_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path = self.prepare_functional_model(root)
            config_path = self.write_config(root, functional_path)
            capture = self.capture_value(config_path)
            capture["adapter_config_sha256"] = "0" * 64
            capture_path = root / "capture.json"
            self.write_json(capture_path, capture)
            with self.assertRaisesRegex(RosActionAdapterError, "config digest mismatch"):
                read_capture(capture_path, read_config(config_path))

    def test_goal_payload_digest_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path = self.prepare_functional_model(root)
            config_path = self.write_config(root, functional_path)
            capture = self.capture_value(config_path)
            capture["action"]["goal_payload_sha256"] = "0" * 64
            capture_path = root / "capture.json"
            self.write_json(capture_path, capture)
            with self.assertRaisesRegex(RosActionAdapterError, "goal payload digest mismatch"):
                read_capture(capture_path, read_config(config_path))

    def test_event_after_evaluation_time_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path = self.prepare_functional_model(root)
            config_path = self.write_config(root, functional_path)
            capture = self.capture_value(config_path)
            capture["interval"]["evaluation_time_ns"] = 184
            capture_path = root / "capture.json"
            self.write_json(capture_path, capture)
            with self.assertRaisesRegex(RosActionAdapterError, "after capture evaluation_time_ns"):
                read_capture(capture_path, read_config(config_path))

    def test_supplemental_source_must_be_inside_bundle_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path, functional, config, capture = self.loaded_inputs(root)
            source, report = normalize(functional_path, functional, config, capture)
            outside = self.supplemental_source(root)
            bundle_directory = root / "bundle-dir"
            with self.assertRaisesRegex(RosActionAdapterError, "inside the evidence bundle directory"):
                build_bundle_and_report(
                    functional_path,
                    functional,
                    config,
                    capture,
                    source,
                    report,
                    bundle_directory / "ros-action.json",
                    bundle_directory / "bundle.json",
                    [outside],
                )

    def test_malformed_supplemental_records_fail_before_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path, functional, config, capture = self.loaded_inputs(root)
            source, report = normalize(functional_path, functional, config, capture)
            malformed = root / "malformed.json"
            self.write_json(malformed, {
                "schema_version": SOURCE_SCHEMA,
                "source_id": "evidence_source/malformed",
                "clock": self.clock,
                "producer": {"producer_id": "test/bad", "producer_type": "malformed_fixture"},
                "records": [{"record_id": "evidence/incomplete"}],
            })
            source_out = root / "ros-action.json"
            bundle_out = root / "bundle.json"
            with self.assertRaisesRegex(RosActionAdapterError, "full assurance-compiler prevalidation"):
                build_bundle_and_report(
                    functional_path,
                    functional,
                    config,
                    capture,
                    source,
                    report,
                    source_out,
                    bundle_out,
                    [malformed],
                )
            self.assertFalse(source_out.exists())
            self.assertFalse(bundle_out.exists())

    def test_dispatch_gate_fails_before_ros_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path = self.prepare_functional_model(root)
            config_path = self.write_config(root, functional_path)
            config = read_config(config_path)
            goal = root / "goal.json"
            self.write_json(goal, {"order": 5})
            args = argparse.Namespace(
                authorize_dispatch="action_instance/wrong",
                server_timeout_sec=1.0,
                goal_response_timeout_sec=1.0,
                result_timeout_sec=1.0,
                settle_sec=0.0,
                goal=goal,
            )
            with self.assertRaisesRegex(RosActionAdapterError, "may move physical hardware"):
                execute_capture(config, args)

    def test_typed_dict_publisher_gid_is_preserved_when_callback_provides_it(self):
        gid = _publisher_id({
            "publisher_gid": {
                "implementation_identifier": "rmw_cyclonedds_cpp",
                "data": bytes(range(16)),
            }
        })
        self.assertEqual(
            gid,
            "rmw_cyclonedds_cpp:000102030405060708090a0b0c0d0e0f",
        )

    def test_make_config_derives_action_verb_and_exact_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path = self.prepare_functional_model(root)
            args = argparse.Namespace(
                affordance_id="affordance/grasp",
                offered_by="component/gripper",
                target_instance_id="object_instance/test-object",
                argument_binding=[],
                action_instance_id="action_instance/test-action",
                target_object_type="object_type/graspable",
                action_name="/gripper/grasp",
                action_type="example_interfaces/action/Fibonacci",
                adapter_id="test_grasp",
                clock_domain="test_monotonic",
                clock_epoch="capture-1",
                maximum_operator_confirmation_age_ns=50,
                maximum_planner_verification_age_ns=50,
                maximum_project_assumption_age_ns=1000,
                maximum_runtime_observation_age_ns=50,
            )
            config = make_config(functional_path, args)
            self.assertEqual(config["action_instance"]["action_verb"], "grasp")
            self.assertEqual(config["ros_action"]["status_topic"], "/gripper/grasp/_action/status")
            self.assertEqual(
                config["functional_model_binding"]["functional_model_artifact_sha256"],
                self.sha256(functional_path),
            )

    def test_cli_normalize_writes_bundle_that_compiles_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_path = self.prepare_functional_model(root)
            config_path = self.write_config(root, functional_path)
            capture_path = self.write_capture(root, config_path)
            supplemental = self.supplemental_source(root)
            source_out = root / "evidence" / "ros-action.json"
            bundle_out = root / "action-bundle.json"
            report_out = root / "normalization-report.json"
            result = self.run_cli(
                "normalize",
                functional_path,
                "--config",
                config_path,
                "--capture",
                capture_path,
                "--evidence-source",
                source_out,
                "--bundle",
                bundle_out,
                "--report",
                report_out,
                "--supplemental-source",
                supplemental,
            )
            output = json.loads(result.stdout)
            self.assertEqual(output["status"], "normalized")
            self.assertEqual(build_action_assurance(functional_path, bundle_out)["schema_version"], "robot-spatial-action-assurance.v1")
            refused = self.run_cli(
                "normalize",
                functional_path,
                "--config",
                config_path,
                "--capture",
                capture_path,
                "--evidence-source",
                source_out,
                "--bundle",
                bundle_out,
                "--report",
                report_out,
                check=False,
            )
            self.assertEqual(refused.returncode, 2)
            self.assertIn("output path already exists", refused.stderr)


if __name__ == "__main__":
    unittest.main()
