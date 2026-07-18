# Action execution evidence and assurance contract

## Contents

1. Purpose
2. Layer boundary
3. Inputs and binding
4. Clock contract
5. Action instance contract
6. Evidence policy
7. Evidence source contract
8. Condition evidence selection
9. Readiness derivation
10. Action-server lifecycle projection
11. Effect evidence and causal boundary
12. Outcome and discrepancy projection
13. Assurance artifact
14. Query contract
15. Verification contract
16. Failure behavior
17. Evaluation and independent oracle
18. Interpretation rules
19. Explicit exclusions

## Purpose

URDF and the functional model describe structure and project-declared action meaning. They do not say whether one action instance was ready at a particular decision time, what an action server later reported, or whether a declared effect was observed.

The action-assurance layer binds one declared affordance instance to digest-bound, time-qualified evidence. It produces a replayable distinction among:

- the declared actor, action, target, preconditions, and effects;
- structural grounding inherited from the functional model;
- evidence-qualified readiness at the declared decision time;
- observed goal, status, and result protocol reports;
- observed values of declared effect predicates;
- unresolved physical execution, causal attribution, producer truthfulness, safety, and authorization.

The artifact is evidence accounting, not a controller and not a safety certificate.

## Layer boundary

The complete reasoning stack is:

1. URDF or another supported articulation source declares represented kinematic structure.
2. The concept graph exposes deterministic structural relations and proof closure.
3. The project-owned functional model declares components, functions, capabilities, conditions, effects, and affordances.
4. The action-assurance bundle names one action instance and supplies evidence reports.
5. The assurance compiler selects evidence by declared time and policy, then derives bounded readiness, lifecycle, effect, and discrepancy projections.

Each layer is necessary for its own claims. No later layer retroactively makes an earlier project declaration physically true.

`ready_under_declared_model_and_evidence` means only that the selected affordance is structurally grounded and every declared precondition has eligible positive evidence at the decision time.

It does not mean:

- dispatch is authorized;
- a controller can execute the action;
- the mechanism can generate required forces or motion;
- the environment is collision-free;
- the action is safe;
- the action occurred;
- a later observation was caused by the action.

## Inputs and binding

The compiler consumes:

- one validated `robot-spatial-functional-model.v1` artifact;
- one `robot-spatial-action-evidence-bundle.v1` artifact;
- one or more referenced `robot-spatial-action-evidence-source.v1` artifacts.

The bundle binds the functional artifact by all three fields:

```json
{
  "functional_model_id": "functional_model/gripper/0123456789abcdef",
  "functional_model_sha256": "<semantic-sha256>",
  "functional_model_artifact_sha256": "<file-sha256>"
}
```

The semantic digest binds model meaning. The artifact digest binds the exact serialized input. Both must match.

Every evidence source reference contains exactly:

```json
{
  "source_id": "evidence_source/runtime",
  "path": "evidence/runtime.json",
  "sha256": "<file-sha256>"
}
```

Paths are relative to the bundle directory. Absolute paths, escapes, duplicate paths, duplicate IDs, missing files, and final symlinks are rejected. The referenced file must be a regular file and its SHA-256 must match before any record is used.

## Clock contract

The bundle and every evidence source carry the same exact clock object:

```json
{
  "domain": "ros_time",
  "unit": "nanoseconds",
  "epoch": "bag/capture-17"
}
```

`unit` is exactly `nanoseconds`. `domain` and `epoch` are non-empty project declarations. Equality verifies consistent labeling, not actual clock synchronization or accuracy.

All action and evidence timestamps are non-negative integers in that declared clock.

The action times must satisfy:

```text
requested_at_ns <= decision_time_ns <= evaluation_time_ns
```

Condition evidence is evaluated at `decision_time_ns`.

Lifecycle and effect evidence is evaluated at `evaluation_time_ns`.

## Action instance contract

`action_instance` contains exactly:

```json
{
  "action_instance_id": "action_instance/grasp-17",
  "affordance_id": "affordance/grasp",
  "offered_by": "component/gripper",
  "action_verb": "grasp",
  "target_object_type": "object_type/graspable",
  "target_instance_id": "object_instance/part-42",
  "argument_bindings": {
    "actor": "component/gripper",
    "target": "object_instance/part-42"
  },
  "requested_at_ns": 100,
  "decision_time_ns": 140,
  "evaluation_time_ns": 220
}
```

The affordance, provider, action verb, and target type must match one exact affordance in the bound functional model.

`argument_bindings` contains exactly `actor`, `target`, and every symbolic argument named by the selected preconditions and effects. Extra and missing bindings are rejected.

The `actor` binding equals `offered_by`. The `target` binding equals `target_instance_id`.

The compiler runs the functional `can_perform_action` query and retains its functional and structural proof clause IDs. This binds action evidence to the same declared affordance reasoning used elsewhere.

## Evidence policy

The policy contains maximum ages for exactly these condition evidence types:

- `runtime_observation`
- `planner_verification`
- `operator_confirmation`
- `project_assumption`

Each maximum age is a non-negative integer in nanoseconds.

The policy also declares:

```json
{
  "require_goal_acceptance_before_status": true,
  "require_terminal_result_status_match": true
}
```

These booleans control protocol consistency checks. They do not relax the evidence-source digest or clock requirements.

## Evidence source contract

Each source contains:

```json
{
  "schema_version": "robot-spatial-action-evidence-source.v1",
  "source_id": "evidence_source/runtime",
  "clock": {"domain": "ros_time", "unit": "nanoseconds", "epoch": "bag/capture-17"},
  "producer": {
    "producer_id": "node/perception",
    "producer_type": "runtime_observer"
  },
  "records": []
}
```

The source must contain at least one record. Producer identity establishes responsibility and provenance only. It is not a truth oracle.

Every record contains exactly:

- `record_id`
- `evidence_type`
- `subject_ref`
- `predicate`
- `bindings`
- `value`
- `observed_at_ns`
- `valid_until_ns`
- `claim_scope`
- `limitations`

Record IDs are unique across all sources.

Condition and effect record values are exactly `true`, `false`, or `unknown`.

Condition `subject_ref` uses `condition/`. Effect `subject_ref` uses `effect/`.

Lifecycle records use fixed contracts:

| Evidence type | Subject | Predicate | Values |
|---|---|---|---|
| `goal_response` | `lifecycle/goal_response` | `goal_response` | `accepted`, `rejected` |
| `action_status` | `lifecycle/action_status` | `action_status` | `accepted`, `executing`, `canceling`, `succeeded`, `aborted`, `canceled` |
| `action_result` | `lifecycle/action_result` | `action_result` | `succeeded`, `aborted`, `canceled` |

Lifecycle bindings contain exactly `action_instance`, and that value must equal the bundle action instance.

`valid_until_ns`, when present, must not precede `observed_at_ns`.

## Condition evidence selection

The functional condition `truth_source` maps to one required evidence type:

| Functional truth source | Required evidence type |
|---|---|
| `runtime_observation_required` | `runtime_observation` |
| `planner_verification_required` | `planner_verification` |
| `operator_confirmation_required` | `operator_confirmation` |
| `project_assumption` | `project_assumption` |

For one condition, the compiler first matches exact subject, predicate, symbolic bindings, and required evidence type.

It then rejects records that are:

- in the future relative to decision time;
- expired before decision time;
- older than the policy maximum age;
- of the wrong evidence type;
- bound to different action arguments.

Among eligible records, the latest timestamp wins.

If multiple latest records have different truth values, the result is `unknown_conflicting_latest_evidence`. No source priority or silent arbitration is allowed.

Condition statuses are:

- `satisfied`
- `not_satisfied`
- `unknown_reported`
- `unknown_conflicting_latest_evidence`
- `unknown_wrong_evidence_type`
- `unknown_binding_mismatch`
- `unknown_expired_evidence`
- `unknown_stale_evidence`
- `unknown_future_only`
- `unknown_missing_evidence`

The projection retains selected records and every ignored record ID by reason.

Artifact integrity is verified. Producer truthfulness is always unverified by this layer.

## Readiness derivation

Readiness is derived in this order:

1. If any selected capability requirement is not structurally grounded, return `not_ready_ungrounded_capability_requirements`.
2. Otherwise, if any precondition is `not_satisfied`, return `not_ready_declared_precondition_false`.
3. Otherwise, if any precondition is not `satisfied`, return `not_ready_missing_stale_conflicting_or_invalid_evidence`.
4. Otherwise, return `ready_under_declared_model_and_evidence`.

A false condition is kept distinct from missing, stale, future, conflicting, wrong-type, or wrong-binding evidence.

Every readiness projection also states:

```json
{
  "authorization_to_dispatch": "not_provided",
  "physical_executability": "not_established",
  "safety": "not_established"
}
```

## Action-server lifecycle projection

The lifecycle projection follows the separation used by ROS 2 actions: goal response, status stream, and terminal result are different evidence records. See the [ROS 2 Actions design article](https://design.ros2.org/articles/actions.html).

The compiler does not synthesize missing states.

It detects:

- conflicting goal responses;
- conflicting terminal results;
- status or result after a rejected goal;
- status or result without a prior accepted goal when policy requires acceptance;
- status or result timestamped before acceptance;
- invalid observed status transitions;
- mismatch between terminal status and result when policy requires agreement.

Observed transitions are checked only among recorded statuses. Direct `accepted` to a terminal status is allowed because an evidence stream may be partial.

Lifecycle status can be:

- `inconsistent_lifecycle_evidence`
- `goal_rejected`
- `goal_response_not_observed`
- `goal_accepted_no_execution_status`
- `status_<state>`
- `result_<terminal>`

Execution start is considered observed only when a status record reports `executing`, `canceling`, `succeeded`, `aborted`, or `canceled`.

Goal acceptance alone is not execution start.

Every lifecycle projection states `action_server_reports_are_independent_physical_verification: false`.

## Effect evidence and causal boundary

Effect records are selected by exact effect subject, predicate, bindings, and `effect_observation` type at `evaluation_time_ns`.

Effect evidence has no maximum age in version 1. `valid_until_ns` can still expire a state-valued observation. A record without `valid_until_ns` remains an eligible historical report.

The selected effect timestamp is compared with the first observed execution-start status:

- `at_or_after_observed_execution_start`
- `before_observed_execution_start`
- `execution_start_not_observed`
- `no_selected_effect_record`

Only the first relation counts as post-execution effect evidence.

Post-execution succession is not causal attribution. Every effect states `caused_by_action: not_established`.

The summary states one of:

- `all_declared_effects_observed_true_after_execution_started`
- `one_or_more_declared_effects_observed_false_after_execution_started`
- `incomplete_or_temporally_unlinked_effect_evidence`

It always states `causal_attribution: not_established`.

## Outcome and discrepancy projection

Outcome combines protocol and effect reports without collapsing them.

Possible conclusions include:

- `inconsistent_lifecycle_evidence`
- `action_server_reported_success_and_all_declared_effects_observed_after_execution_started`
- `action_server_reported_success_but_declared_effect_observation_false`
- `action_server_reported_success_effect_evidence_incomplete_or_temporally_unlinked`
- `action_server_reported_aborted`
- `action_server_reported_canceled`
- `no_terminal_action_result_observed`

Every outcome retains:

- `reported_terminal_result`
- `causal_success: not_established`
- `physical_world_truth: not_established`
- `safety: not_established`

The compiler adds `goal_accepted_without_complete_declared_readiness_evidence` when a server accepted a goal even though the declared readiness model was not satisfied.

It adds `reported_success_without_complete_positive_declared_effect_evidence` when a succeeded result lacks complete positive post-execution effect evidence.

Lifecycle consistency issues remain visible in the same discrepancy list.

## Assurance artifact

The compiler writes `robot-spatial-action-assurance.v1`.

It embeds normalized derivation inputs, not just conclusions:

- exact action instance;
- exact evidence policy;
- selected functional basis and clause IDs;
- normalized evidence records with source/producer provenance.

It also contains:

- functional-model binding;
- evidence-bundle and evidence-source bindings;
- clock;
- projections;
- coverage;
- query contract;
- provenance contract;
- epistemic scope;
- semantic digest.

The model ID is content-derived from the functional semantic digest, exact bundle artifact digest, and action instance ID.

Reading a model recomputes projections and coverage from its embedded derivation input. Rehashing a changed conclusion does not make it valid.

## Query contract

Queries use `robot-spatial-action-assurance-query.v1`:

```json
{
  "schema_version": "robot-spatial-action-assurance-query.v1",
  "query_id": "question/grasp-readiness",
  "intent": "summarize_action",
  "parameters": {}
}
```

Supported intents are:

- `summarize_action`
- `explain_precondition` with one typed `condition`
- `explain_effect` with one typed `effect`
- `explain_lifecycle`
- `explain_evidence` with one typed evidence record
- `why_not_ready`

Bare suffixes are accepted only when they resolve uniquely.

Answers include:

- the assurance ID and digest;
- the bounded answer object;
- controlled-language summary;
- selected evidence records;
- functional support when relevant;
- unresolved physical, causal, producer, calibration, and clock boundaries.

`why_not_ready` can return no declared blockers while still stating `authorization_to_dispatch: not_provided`.

## Verification contract

`verify-action-assurance` performs exact regeneration from:

- the supplied functional model artifact;
- the supplied evidence bundle artifact;
- every digest-bound evidence source.

Verification passes only when the stored assurance is byte-equivalent under canonical serialization to exact regeneration.

It verifies:

- functional ID, semantic digest, and artifact digest;
- bundle digest;
- evidence-source digests;
- strict schemas and clocks;
- evidence selection;
- readiness, lifecycle, effect, outcome, discrepancy, and coverage derivation;
- deterministic serialization.

It does not verify:

- producer truthfulness;
- clock synchronization;
- sensor calibration;
- physical causation;
- hardware state;
- execution safety.

## Failure behavior

Invalid inputs fail closed with a non-zero CLI exit for compilation and query errors.

The verifier returns a structured failed report and exits non-zero when the stored model is invalid or differs from regeneration.

Semantic outcomes such as not-ready, rejected, aborted, canceled, or effect-false are valid compiled evidence states. They do not make compilation fail.

This distinction lets an agent reason about failures without confusing a negative action outcome with artifact corruption.

## Evaluation and independent oracle

Action-evidence evaluations test five capabilities:

1. decision-time condition evidence selection and readiness;
2. lifecycle protocol versus physical execution;
3. effect observation versus causal success;
4. cross-layer discrepancies and outcome boundaries;
5. content binding, producer provenance, and epistemic scope.

Raw-source tasks declare `action_assurances` only alongside `export_options.functional_spec`.

Each declaration binds:

- an evaluator-assigned assurance ID;
- `functional_model_source: exported_functional_model`;
- a raw evidence-bundle path under `source/`;
- a JSON output basename under candidate work.

Candidates must generate the exact functional model first, compile the assurance against it, verify regeneration, and query the assurance. Generated assurance artifacts are forbidden from public raw source.

Use `crosscheck_action_assurance.py` as the dependency-free oracle. It must call only public CLI commands, generate raw URDF and function specs, create randomized time-qualified evidence, and independently compute condition selection, readiness, lifecycle, effect, and outcome expectations.

The oracle must not import production action-assurance, functional, concept, or URDF modules.

Preserve seed, case count, scenario families, public query count, assertion count, failures, and exclusions.

Agreement validates deterministic evidence accounting and public query behavior only.

## Interpretation rules

When answering “can the robot do this now?” report at least:

1. the declared affordance and actor/target bindings;
2. structural grounding;
3. decision time and each precondition status;
4. readiness conclusion;
5. dispatch authorization boundary;
6. physical executability and safety boundary.

When answering “did the action succeed?” report at least:

1. goal response;
2. latest observed status;
3. terminal result;
4. lifecycle consistency;
5. effect evidence and its time relative to observed execution start;
6. causal, physical-world, and safety boundary;
7. every discrepancy.

Never paraphrase `result_succeeded` as “the robot physically succeeded.”

Never paraphrase a post-execution effect observation as “the action caused the effect.”

Never paraphrase `ready_under_declared_model_and_evidence` as “safe to run.”

## Explicit exclusions

Version 1 does not provide:

- live ROS subscriptions or dispatch;
- action cancellation commands;
- controller or hardware introspection;
- trajectory validation;
- collision checking beyond other explicitly bound layers;
- force, torque, payload, or grasp-quality proof;
- sensor calibration proof;
- Byzantine or faulty producer arbitration;
- distributed clock synchronization;
- probabilistic belief fusion;
- causal intervention or counterfactual proof;
- safety certification;
- authorization policy.

Those concerns may supply future evidence sources or independent gates. They must not be silently inferred from this artifact.
