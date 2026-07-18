from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
SCRIPT = SCRIPT_DIR / "robot_spatial.py"
ORACLE = SCRIPT_DIR / "crosscheck_action_assurance.py"
FIXTURES = TEST_DIR / "fixtures"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(TEST_DIR))

from spatial_action_assurance import (  # noqa: E402
    ActionAssuranceError,
    BUNDLE_SCHEMA,
    MODEL_SCHEMA,
    QUERY_SCHEMA,
    SOURCE_SCHEMA,
    build_action_assurance,
    query_action_assurance,
    read_action_assurance,
    verify_action_assurance,
    write_action_assurance,
)
from spatial_functional import write_functional_model_from_context  # noqa: E402
import test_spatial_functional as functional_test_helpers  # noqa: E402
import spatial_evaluation  # noqa: E402
import spatial_evaluation_suite  # noqa: E402


class SpatialActionAssuranceTests(unittest.TestCase):
    clock = {"domain": "test_monotonic", "unit": "nanoseconds", "epoch": "test-start"}

    def run_cli(self, *arguments: object, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(value) for value in arguments)],
            check=check,
            capture_output=True,
            text=True,
            timeout=60,
        )

    @staticmethod
    def write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def rehash(model: dict) -> dict:
        body = {key: value for key, value in model.items() if key != "assurance_sha256"}
        model["assurance_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return model

    def prepare_functional_model(self, root: Path, *, ungrounded: bool = False) -> Path:
        context = root / "context"
        self.run_cli(
            "export",
            FIXTURES / "mimic_branch.urdf",
            "--workspace-samples",
            "0",
            "--out",
            context,
        )
        spec = functional_test_helpers.SpatialFunctionalModelTests.spec(context)
        if ungrounded:
            spec["capabilities"][0]["enabling_requirements"].append({
                "requirement_id": "requirement/missing_required_link",
                "type": "entity_exists",
                "parameters": {"entity": "link/not_present"},
            })
        spec_path = root / "functional-spec.json"
        model_path = root / "functional-model.json"
        self.write_json(spec_path, spec)
        write_functional_model_from_context(context, spec_path, model_path)
        return model_path

    @staticmethod
    def condition_record(
        name: str,
        condition: str,
        predicate: str,
        evidence_type: str,
        value: str,
        observed_at_ns: int,
        *,
        valid_until_ns: int | None = None,
        bindings: dict[str, str] | None = None,
    ) -> dict:
        return {
            "record_id": f"evidence/{name}",
            "evidence_type": evidence_type,
            "subject_ref": f"condition/{condition}",
            "predicate": predicate,
            "bindings": bindings or {"actor": "component/gripper", "target": "object_instance/test-object"},
            "value": value,
            "observed_at_ns": observed_at_ns,
            "valid_until_ns": valid_until_ns,
            "claim_scope": "Synthetic test observation for the named predicate only.",
            "limitations": ["Does not establish physical truth beyond the report."],
        }

    @staticmethod
    def lifecycle_record(name: str, evidence_type: str, value: str, observed_at_ns: int) -> dict:
        subject, predicate = {
            "goal_response": ("lifecycle/goal_response", "goal_response"),
            "action_status": ("lifecycle/action_status", "action_status"),
            "action_result": ("lifecycle/action_result", "action_result"),
        }[evidence_type]
        return {
            "record_id": f"evidence/{name}",
            "evidence_type": evidence_type,
            "subject_ref": subject,
            "predicate": predicate,
            "bindings": {"action_instance": "action_instance/test-action"},
            "value": value,
            "observed_at_ns": observed_at_ns,
            "valid_until_ns": None,
            "claim_scope": "Observed action-server protocol report only.",
            "limitations": ["Does not establish physical execution or success."],
        }

    @staticmethod
    def effect_record(name: str, value: str, observed_at_ns: int) -> dict:
        return {
            "record_id": f"evidence/{name}",
            "evidence_type": "effect_observation",
            "subject_ref": "effect/target_retained",
            "predicate": "target_retained_by",
            "bindings": {"actor": "component/gripper", "target": "object_instance/test-object"},
            "value": value,
            "observed_at_ns": observed_at_ns,
            "valid_until_ns": None,
            "claim_scope": "Observed declared effect predicate only.",
            "limitations": ["Does not establish that the action caused the observation."],
        }

    def ready_records(self, *, effect: str = "true") -> list[dict]:
        return [
            self.condition_record(
                "between", "target_between_fingers", "target_between_fingers", "runtime_observation", "true", 130
            ),
            self.condition_record("plan", "plan_approved", "plan_approved", "planner_verification", "true", 135),
            self.lifecycle_record("goal", "goal_response", "accepted", 150),
            self.lifecycle_record("executing", "action_status", "executing", 160),
            self.lifecycle_record("succeeded-status", "action_status", "succeeded", 180),
            self.lifecycle_record("succeeded-result", "action_result", "succeeded", 185),
            self.effect_record("retained", effect, 190),
        ]

    def write_bundle(self, root: Path, functional_model: Path, records: list[dict]) -> Path:
        source_path = root / "evidence-source.json"
        source = {
            "schema_version": SOURCE_SCHEMA,
            "source_id": "evidence_source/test",
            "clock": self.clock,
            "producer": {"producer_id": "test/fixture", "producer_type": "synthetic_test_harness"},
            "records": records,
        }
        self.write_json(source_path, source)
        functional = json.loads(functional_model.read_text(encoding="utf-8"))
        bundle = {
            "schema_version": BUNDLE_SCHEMA,
            "bundle_id": "action_evidence_bundle/test-action",
            "functional_model_binding": {
                "functional_model_id": functional["functional_model_id"],
                "functional_model_sha256": functional["functional_model_sha256"],
                "functional_model_artifact_sha256": self.sha256(functional_model),
            },
            "clock": self.clock,
            "action_instance": {
                "action_instance_id": "action_instance/test-action",
                "affordance_id": "affordance/grasp",
                "offered_by": "component/gripper",
                "action_verb": "grasp",
                "target_object_type": "object_type/graspable",
                "target_instance_id": "object_instance/test-object",
                "argument_bindings": {"actor": "component/gripper", "target": "object_instance/test-object"},
                "requested_at_ns": 100,
                "decision_time_ns": 140,
                "evaluation_time_ns": 200,
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
            "evidence_sources": [{
                "source_id": source["source_id"],
                "path": source_path.name,
                "sha256": self.sha256(source_path),
            }],
        }
        bundle_path = root / "evidence-bundle.json"
        self.write_json(bundle_path, bundle)
        return bundle_path

    @staticmethod
    def query(model: dict, intent: str, parameters: dict | None = None) -> dict:
        return query_action_assurance(model, {
            "schema_version": QUERY_SCHEMA,
            "query_id": f"query/{intent}",
            "intent": intent,
            "parameters": parameters or {},
        })

    def test_ready_success_and_post_execution_effect_remain_epistemically_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_model = self.prepare_functional_model(root)
            bundle = self.write_bundle(root, functional_model, self.ready_records())
            model = build_action_assurance(functional_model, bundle)

            self.assertEqual(model["schema_version"], MODEL_SCHEMA)
            self.assertEqual(model["projections"]["readiness"]["conclusion"], "ready_under_declared_model_and_evidence")
            self.assertEqual(model["projections"]["lifecycle"]["status"], "result_succeeded")
            self.assertEqual(
                model["projections"]["effect_summary"]["status"],
                "all_declared_effects_observed_true_after_execution_started",
            )
            self.assertEqual(
                model["projections"]["outcome"]["conclusion"],
                "action_server_reported_success_and_all_declared_effects_observed_after_execution_started",
            )
            self.assertEqual(model["projections"]["outcome"]["causal_success"], "not_established")
            self.assertEqual(model["projections"]["outcome"]["safety"], "not_established")
            self.assertEqual(model["projections"]["readiness"]["authorization_to_dispatch"], "not_provided")

            summary = self.query(model, "summarize_action")
            self.assertEqual(summary["status"], "answered")
            self.assertIn("producer truthfulness", summary["unknowns"][0])
            effect = self.query(model, "explain_effect", {"effect": "target_retained"})
            self.assertEqual(effect["answer"]["temporal_relation_to_execution"], "at_or_after_observed_execution_start")
            self.assertEqual(effect["answer"]["caused_by_action"], "not_established")

    def test_time_selection_false_stale_conflicting_future_type_and_binding_cases(self):
        cases = [
            ("false", [self.condition_record("between-false", "target_between_fingers", "target_between_fingers", "runtime_observation", "false", 130)], "not_satisfied", "not_ready_declared_precondition_false"),
            ("stale", [self.condition_record("between-stale", "target_between_fingers", "target_between_fingers", "runtime_observation", "true", 80)], "unknown_stale_evidence", "not_ready_missing_stale_conflicting_or_invalid_evidence"),
            ("conflict", [
                self.condition_record("between-yes", "target_between_fingers", "target_between_fingers", "runtime_observation", "true", 130),
                self.condition_record("between-no", "target_between_fingers", "target_between_fingers", "runtime_observation", "false", 130),
            ], "unknown_conflicting_latest_evidence", "not_ready_missing_stale_conflicting_or_invalid_evidence"),
            ("future", [self.condition_record("between-future", "target_between_fingers", "target_between_fingers", "runtime_observation", "true", 145)], "unknown_future_only", "not_ready_missing_stale_conflicting_or_invalid_evidence"),
            ("wrong-type", [self.condition_record("between-wrong-type", "target_between_fingers", "target_between_fingers", "planner_verification", "true", 130)], "unknown_wrong_evidence_type", "not_ready_missing_stale_conflicting_or_invalid_evidence"),
            ("wrong-binding", [self.condition_record(
                "between-wrong-binding", "target_between_fingers", "target_between_fingers", "runtime_observation", "true", 130,
                bindings={"actor": "component/gripper", "target": "object_instance/other"},
            )], "unknown_binding_mismatch", "not_ready_missing_stale_conflicting_or_invalid_evidence"),
        ]
        for name, special_records, expected_status, expected_readiness in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                functional_model = self.prepare_functional_model(root)
                records = [
                    *special_records,
                    self.condition_record("plan", "plan_approved", "plan_approved", "planner_verification", "true", 135),
                ]
                bundle = self.write_bundle(root, functional_model, records)
                model = build_action_assurance(functional_model, bundle)
                between = next(
                    item for item in model["projections"]["preconditions"]
                    if item["condition_id"] == "condition/target_between_fingers"
                )
                self.assertEqual(between["status"], expected_status)
                self.assertEqual(model["projections"]["readiness"]["conclusion"], expected_readiness)

    def test_ungrounded_capability_overrides_positive_condition_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_model = self.prepare_functional_model(root, ungrounded=True)
            bundle = self.write_bundle(root, functional_model, self.ready_records())
            model = build_action_assurance(functional_model, bundle)
            self.assertFalse(model["projections"]["declared_action"]["structurally_grounded"])
            self.assertEqual(model["projections"]["readiness"]["conclusion"], "not_ready_ungrounded_capability_requirements")
            self.assertIn(
                "goal_accepted_without_complete_declared_readiness_evidence",
                {item["code"] for item in model["projections"]["discrepancies"]},
            )

    def test_lifecycle_inconsistency_and_reported_success_effect_mismatch_are_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_model = self.prepare_functional_model(root)
            false_effect_bundle = self.write_bundle(root, functional_model, self.ready_records(effect="false"))
            false_effect = build_action_assurance(functional_model, false_effect_bundle)
            self.assertEqual(
                false_effect["projections"]["outcome"]["conclusion"],
                "action_server_reported_success_but_declared_effect_observation_false",
            )
            self.assertIn(
                "reported_success_without_complete_positive_declared_effect_evidence",
                {item["code"] for item in false_effect["projections"]["discrepancies"]},
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_model = self.prepare_functional_model(root)
            records = self.ready_records()
            next(item for item in records if item["record_id"] == "evidence/succeeded-status")["value"] = "aborted"
            bundle = self.write_bundle(root, functional_model, records)
            inconsistent = build_action_assurance(functional_model, bundle)
            self.assertEqual(inconsistent["projections"]["lifecycle"]["consistency"], "failed")
            self.assertEqual(inconsistent["projections"]["outcome"]["conclusion"], "inconsistent_lifecycle_evidence")
            self.assertIn(
                "terminal_status_result_mismatch",
                {item["code"] for item in inconsistent["projections"]["lifecycle"]["issues"]},
            )

    def test_digest_path_symlink_and_action_binding_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_model = self.prepare_functional_model(root)
            bundle_path = self.write_bundle(root, functional_model, self.ready_records())
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

            bad_digest = copy.deepcopy(bundle)
            bad_digest["evidence_sources"][0]["sha256"] = "0" * 64
            self.write_json(bundle_path, bad_digest)
            with self.assertRaisesRegex(ActionAssuranceError, "digest mismatch"):
                build_action_assurance(functional_model, bundle_path)

            outside = root.parent / f"outside-{root.name}.json"
            outside.write_text("{}\n", encoding="utf-8")
            try:
                escaping = copy.deepcopy(bundle)
                escaping["evidence_sources"][0]["path"] = f"../{outside.name}"
                escaping["evidence_sources"][0]["sha256"] = self.sha256(outside)
                self.write_json(bundle_path, escaping)
                with self.assertRaisesRegex(ActionAssuranceError, "escapes"):
                    build_action_assurance(functional_model, bundle_path)
            finally:
                outside.unlink(missing_ok=True)

            source = root / "evidence-source.json"
            symlink = root / "source-link.json"
            os.symlink(source.name, symlink)
            linked = copy.deepcopy(bundle)
            linked["evidence_sources"][0]["path"] = symlink.name
            self.write_json(bundle_path, linked)
            with self.assertRaisesRegex(ActionAssuranceError, "non-symlink"):
                build_action_assurance(functional_model, bundle_path)

            mismatch = copy.deepcopy(bundle)
            mismatch["functional_model_binding"]["functional_model_sha256"] = "0" * 64
            self.write_json(bundle_path, mismatch)
            with self.assertRaisesRegex(ActionAssuranceError, "functional model binding mismatch"):
                build_action_assurance(functional_model, bundle_path)

    def test_tamper_detection_and_exact_regeneration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_model = self.prepare_functional_model(root)
            bundle = self.write_bundle(root, functional_model, self.ready_records())
            model_path = root / "action-assurance.json"
            write_action_assurance(functional_model, bundle, model_path)
            self.assertEqual(verify_action_assurance(functional_model, bundle, model_path)["status"], "passed")

            simple = json.loads(model_path.read_text(encoding="utf-8"))
            simple["projections"]["outcome"]["causal_success"] = "established"
            self.write_json(model_path, simple)
            with self.assertRaisesRegex(ActionAssuranceError, "semantic digest mismatch"):
                read_action_assurance(model_path)

            coherent_projection = copy.deepcopy(simple)
            self.write_json(model_path, self.rehash(coherent_projection))
            with self.assertRaisesRegex(ActionAssuranceError, "projections do not match"):
                read_action_assurance(model_path)

            coherent_scope = build_action_assurance(functional_model, bundle)
            coherent_scope["epistemic_scope"] += " altered but internally rehashed"
            self.write_json(model_path, self.rehash(coherent_scope))
            self.assertEqual(read_action_assurance(model_path)["schema_version"], MODEL_SCHEMA)
            verification = verify_action_assurance(functional_model, bundle, model_path)
            self.assertEqual(verification["status"], "failed")
            self.assertFalse(verification["exact_regeneration_match"])

    def test_cli_compile_query_verify_and_nonzero_failed_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_model = self.prepare_functional_model(root)
            bundle = self.write_bundle(root, functional_model, self.ready_records())
            model_path = root / "action-assurance.json"
            generated = json.loads(self.run_cli(
                "action-assurance", functional_model, bundle, "--out", model_path
            ).stdout)
            self.assertEqual(generated["status"], "generated")
            self.assertEqual(generated["readiness_conclusion"], "ready_under_declared_model_and_evidence")

            query_path = root / "query.json"
            self.write_json(query_path, {
                "schema_version": QUERY_SCHEMA,
                "query_id": "query/lifecycle",
                "intent": "explain_lifecycle",
                "parameters": {},
            })
            answer = json.loads(self.run_cli(
                "query-action-assurance", model_path, query_path, "--compact"
            ).stdout)
            self.assertEqual(answer["answer"]["terminal_result"], "succeeded")
            passed = json.loads(self.run_cli(
                "verify-action-assurance", functional_model, bundle, "--model", model_path
            ).stdout)
            self.assertTrue(passed["exact_regeneration_match"])

            tampered = json.loads(model_path.read_text(encoding="utf-8"))
            tampered["epistemic_scope"] += " changed"
            self.write_json(model_path, self.rehash(tampered))
            failed = self.run_cli(
                "verify-action-assurance", functional_model, bundle, "--model", model_path, check=False
            )
            self.assertEqual(failed.returncode, 1)
            self.assertEqual(json.loads(failed.stdout)["status"], "failed")

    def test_dependency_free_public_cli_oracle(self):
        process = subprocess.run(
            [sys.executable, str(ORACLE), "--cases", "8", "--seed", "20260718"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        result = json.loads(process.stdout)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["case_count"], 8)
        self.assertEqual(result["public_query_count"], 8)
        self.assertEqual(result["failures"], [])

    def test_action_assurance_evaluation_questions_have_exact_public_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            functional_model_path = self.prepare_functional_model(root)
            bundle = self.write_bundle(root, functional_model_path, self.ready_records(effect="false"))
            assurance = build_action_assurance(functional_model_path, bundle)
            functional = json.loads(functional_model_path.read_text(encoding="utf-8"))
            canonical = json.loads((root / "context" / "model.json").read_text(encoding="utf-8"))
            canonical.setdefault("artifacts", {})["functional_model"] = {
                "functional_model_id": functional["functional_model_id"],
                "functional_model_sha256": functional["functional_model_sha256"],
                "sha256": self.sha256(functional_model_path),
            }
            facts = [
                json.loads(line)
                for line in (root / "context" / "facts.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            questions, keys = spatial_evaluation.generate_records(
                canonical,
                facts,
                None,
                functional,
                assurance,
            )
            action_questions = {
                item["task"]: item for item in questions if item["capability"] == "action_execution_evidence"
            }
            self.assertEqual(set(action_questions), {
                "apply_action_evidence_provenance_boundary",
                "evaluate_action_readiness_at_decision_time",
                "identify_action_evidence_discrepancies",
                "separate_action_server_lifecycle_from_physical_execution",
                "separate_effect_observation_from_causal_success",
            })
            lifecycle_contract = action_questions[
                "separate_action_server_lifecycle_from_physical_execution"
            ]["submission_contract"]["record"]["answer"]
            self.assertEqual(set(lifecycle_contract), {
                "status",
                "consistency",
                "goal_response",
                "latest_observed_status",
                "terminal_result",
                "execution_started_observed",
                "execution_started_at_ns",
                "issue_codes",
                "independent_physical_verification",
            })
            action_keys = [item for item in keys if item["capability"] == "action_execution_evidence"]
            self.assertEqual(len(action_keys), 5)
            readiness_key = next(
                item for item in action_keys if item["task"] == "evaluate_action_readiness_at_decision_time"
            )
            self.assertEqual(readiness_key["comparison"]["list_order"], "unordered")
            discrepancy = next(
                item for item in action_keys if item["task"] == "identify_action_evidence_discrepancies"
            )
            self.assertEqual(discrepancy["answer"]["causal_success"], "not_established")
            self.assertEqual(discrepancy["answer"]["physical_world_truth"], "not_established")

    def test_raw_task_action_assurance_contract_is_source_only_and_function_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary)
            source = task_root / "source"
            source.mkdir()
            (source / "robot.urdf").write_text("<robot name='test'><link name='base'/></robot>\n", encoding="utf-8")
            self.write_json(source / "functional-spec.json", {})
            self.write_json(source / "evidence-bundle.json", {})
            task = {
                "schema_version": "robot-spatial-raw-source-task.v1",
                "workflow": "direct",
                "input_format": "urdf",
                "entrypoint": "source/robot.urdf",
                "export_options": {
                    "functional_spec": "source/functional-spec.json",
                    "workspace_samples": 0,
                },
                "action_assurances": [{
                    "assurance_id": "grasp-run",
                    "functional_model_source": "exported_functional_model",
                    "evidence_bundle": "source/evidence-bundle.json",
                    "output": "action-assurance.json",
                }],
            }
            spatial_evaluation_suite._validate_raw_task_spec(task, task_root, "task")

            no_function = copy.deepcopy(task)
            no_function["export_options"].pop("functional_spec")
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "requires export_options.functional_spec"):
                spatial_evaluation_suite._validate_raw_task_spec(no_function, task_root, "task")

            escaping = copy.deepcopy(task)
            escaping["action_assurances"][0]["evidence_bundle"] = "../outside.json"
            with self.assertRaises(spatial_evaluation_suite.EvaluationSuiteError):
                spatial_evaluation_suite._validate_raw_task_spec(escaping, task_root, "task")

    def test_raw_task_ros_action_adapter_contract_is_source_only_function_bound_and_output_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary)
            source = task_root / "source"
            source.mkdir()
            (source / "robot.urdf").write_text(
                "<robot name='test'><link name='base'/></robot>\n", encoding="utf-8"
            )
            for filename in (
                "functional-spec.json",
                "ros-action-config.json",
                "ros-action-capture.json",
                "condition-evidence.json",
            ):
                self.write_json(source / filename, {})
            task = {
                "schema_version": "robot-spatial-raw-source-task.v1",
                "workflow": "direct",
                "input_format": "urdf",
                "entrypoint": "source/robot.urdf",
                "export_options": {
                    "functional_spec": "source/functional-spec.json",
                    "workspace_samples": 0,
                },
                "ros_action_adapters": [{
                    "adapter_id": "grasp-run",
                    "functional_model_source": "exported_functional_model",
                    "config": "source/ros-action-config.json",
                    "capture": "source/ros-action-capture.json",
                    "supplemental_sources": [{
                        "source": "source/condition-evidence.json",
                        "output": "condition-evidence.json",
                    }],
                    "evidence_source_output": "ros-action-evidence.json",
                    "bundle_output": "generated-action-bundle.json",
                    "report_output": "ros-action-report.json",
                    "assurance_output": "ros-action-assurance.json",
                }],
            }
            spatial_evaluation_suite._validate_raw_task_spec(task, task_root, "task")

            no_function = copy.deepcopy(task)
            no_function["export_options"].pop("functional_spec")
            with self.assertRaisesRegex(
                spatial_evaluation_suite.EvaluationSuiteError,
                "requires export_options.functional_spec",
            ):
                spatial_evaluation_suite._validate_raw_task_spec(no_function, task_root, "task")

            escaping = copy.deepcopy(task)
            escaping["ros_action_adapters"][0]["supplemental_sources"][0]["source"] = "../outside.json"
            with self.assertRaises(spatial_evaluation_suite.EvaluationSuiteError):
                spatial_evaluation_suite._validate_raw_task_spec(escaping, task_root, "task")

            duplicate = copy.deepcopy(task)
            duplicate["ros_action_adapters"][0]["bundle_output"] = "ros-action-evidence.json"
            with self.assertRaisesRegex(
                spatial_evaluation_suite.EvaluationSuiteError,
                "output filenames must differ",
            ):
                spatial_evaluation_suite._validate_raw_task_spec(duplicate, task_root, "task")

            collision = copy.deepcopy(task)
            collision["action_assurances"] = [{
                "assurance_id": "prebuilt-run",
                "functional_model_source": "exported_functional_model",
                "evidence_bundle": "source/condition-evidence.json",
                "output": "ros-action-assurance.json",
            }]
            with self.assertRaisesRegex(
                spatial_evaluation_suite.EvaluationSuiteError,
                "output filenames collide",
            ):
                spatial_evaluation_suite._validate_raw_task_spec(collision, task_root, "task")


if __name__ == "__main__":
    unittest.main()
