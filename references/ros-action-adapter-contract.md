# ROS 2 action capture and normalization contract

## Contents

1. Purpose
2. Why an action is not one topic
3. Artifact separation
4. Workflow
5. Adapter config
6. Active dispatch boundary
7. Immutable capture
8. Clock and event-time contract
9. Goal identity and payload binding
10. Status and authority normalization
11. Result and feedback treatment
12. Lifecycle evidence output
13. Assurance bundle integration
14. Failure behavior
15. Live ROS requirements
16. Verification
17. Version 1 boundary

## Purpose

`ros_action_adapter.py` turns one explicitly bound ROS 2 action-client exchange into digest-bound lifecycle evidence for `spatial_action_assurance.py`.

It answers a narrow, important question:

> What goal response, status, and result protocol reports did this bound client observe for this exact goal UUID, at these client-local times?

It does not answer:

- whether dispatch was authorized;
- whether the server is authentic or truthful;
- whether the mechanism moved;
- whether a reported `SUCCEEDED` means physical success;
- whether a result payload proves a declared effect;
- whether a later observation was caused by the action;
- whether the action was safe.

The adapter is a transport-to-evidence boundary, not a controller policy, physical verifier, or safety system.

## Why an action is not one topic

A ROS 2 action client observes several protocol surfaces:

- send-goal request and response;
- feedback messages;
- status-array messages;
- get-result request and response;
- optionally, cancel request and response.

Goal acknowledgement and result are service exchanges. Status and feedback are topic traffic. A passive status subscriber therefore cannot reconstruct a complete goal exchange and must never be presented as if it had observed goal acceptance or result delivery.

Version 1 supports two valid input paths:

1. `execute-capture` acts as the client, sends exactly one goal, and records its service futures plus feedback and a separate status subscription.
2. An external system writes a conforming immutable capture, which the dependency-free normalizer validates offline.

## Artifact separation

Keep these artifacts distinct:

1. `robot-spatial-functional-model.v1` declares the project action relation, preconditions, and intended effects.
2. `robot-spatial-ros-action-adapter-config.v1` binds one ROS action interface to one functional action instance and evidence policy.
3. `robot-spatial-ros-action-capture.v1` preserves one client exchange, exact goal payload, UUID, event times, and visible transport identities.
4. `robot-spatial-action-evidence-source.v1` contains only normalized lifecycle reports.
5. `robot-spatial-action-evidence-bundle.v1` combines that lifecycle source with separately authored condition/effect sources.
6. `robot-spatial-action-assurance.v1` deterministically derives readiness, lifecycle, effects, discrepancies, and unresolved boundaries.
7. `robot-spatial-ros-action-normalization-report.v1` audits ignored goals, unknown/duplicate statuses, feedback, result payload, authorities, digests, and limitations.

Changing the functional model, config bytes, capture bytes, goal payload, or supplemental evidence bytes changes a bound digest. The report is an audit artifact and never substitutes for the evidence source, bundle, or assurance model.

## Workflow

Probe the environment:

```bash
python3 scripts/ros_action_adapter.py probe
```

Create a functional-model-bound config:

```bash
python3 scripts/ros_action_adapter.py make-config functional-model.json \
  --adapter-id gripper_a \
  --clock-domain ros_time \
  --clock-epoch commissioning-run-42 \
  --action-name /gripper_controller/gripper_cmd \
  --action-type control_msgs/action/GripperCommand \
  --action-instance-id action_instance/grasp-42 \
  --affordance-id affordance/grasp \
  --offered-by component/gripper \
  --target-object-type object_type/graspable \
  --target-instance-id object_instance/part-42 \
  --out action-adapter-config.json
```

Capture by acting as the client only after explicit dispatch authorization:

```bash
python3 scripts/ros_action_adapter.py execute-capture \
  --config action-adapter-config.json \
  --goal gripper-goal.json \
  --capture-id run-42 \
  --authorize-dispatch action_instance/grasp-42 \
  --result-timeout-sec 60 \
  --out action-capture.json
```

Normalize offline and combine separately supplied condition/effect evidence:

```bash
python3 scripts/ros_action_adapter.py normalize functional-model.json \
  --config action-adapter-config.json \
  --capture action-capture.json \
  --evidence-source evidence/ros-action.json \
  --supplemental-source evidence/conditions-effects.json \
  --bundle action-evidence-bundle.json \
  --report action-normalization-report.json
```

Compile and verify assurance using the normal action-assurance commands:

```bash
python3 scripts/robot_spatial.py action-assurance functional-model.json \
  action-evidence-bundle.json --out action-assurance.json
python3 scripts/robot_spatial.py verify-action-assurance functional-model.json \
  action-evidence-bundle.json --model action-assurance.json \
  --out action-assurance-verification.json
```

The lifecycle-only source is enough to describe protocol evidence, but readiness and effect conclusions remain unknown unless the bundle includes eligible, separately typed condition and effect evidence.

## Adapter config

The config contains exactly:

- `adapter_id`;
- a non-empty nanosecond clock domain and epoch;
- functional model semantic and file digests;
- canonical absolute action name;
- `package/action/Type` interface identifier;
- derived `/<action>/_action/status` topic;
- one exact functional action instance without runtime timestamps;
- the four condition-evidence maximum ages used by action assurance;
- mandatory multiple-publisher and same-time-conflict rejection policies.

`make-config` resolves the selected affordance from the functional model. It derives the action verb, checks provider and target type membership, and requires bindings for exactly `actor`, `target`, and every symbolic precondition/effect argument. Names are never matched approximately.

The v1 ambiguity policies must both remain true. A future schema may define explicit arbitration, but v1 never chooses among competing visible status publishers or conflicting simultaneous reports.

## Active dispatch boundary

`execute-capture` can cause real robot motion. It therefore requires:

```text
--authorize-dispatch <exact configured action_instance_id>
```

A generic boolean is insufficient. The token must equal the exact action instance that the config binds, and this check occurs before ROS imports or server contact.

This gate establishes only explicit CLI intent to send that configured goal. It does not establish operator competence, organizational authorization, precondition satisfaction, collision safety, protective stops, workspace clearance, or hardware safety. Those remain external responsibilities.

The offline `normalize`, `probe`, and `make-config` commands never dispatch.

## Immutable capture

The capture binds the exact config file SHA-256 and contains:

- one capture ID and exact clock;
- ordered start, request, decision, evaluation, and end times;
- termination reason;
- transport and ROS distribution declarations;
- separate identity-visibility statements for status, service responses, and feedback;
- canonical action name and type;
- one 128-bit lowercase hexadecimal goal UUID;
- canonical JSON goal payload and its semantic SHA-256;
- client node name and observed `use_sim_time` value;
- a non-empty, contiguous, nondecreasing record sequence.

Supported record kinds are:

| Kind | Meaning | Required payload |
| --- | --- | --- |
| `send_goal_request` | client initiated the one bound goal request | empty object |
| `goal_response` | client future completed | `accepted`, server acceptance stamp or null |
| `feedback` | bound feedback callback fired | goal UUID and payload |
| `status_array` | status subscriber received an array | zero or more goal status records |
| `get_result_request` | client requested result for an accepted goal | empty object |
| `result_response` | result future completed | goal UUID, status code, result payload |

Every record has one client event timestamp. A status array may contain other goals because ROS status messages are arrays; those entries are counted and ignored rather than relabeled.

An incomplete capture is still useful raw evidence. For example, `execute-capture` can preserve a goal-response timeout. Normalization refuses to create a lifecycle evidence source when no goal response was observed, because inventing `accepted`, `rejected`, or `unknown` as a server report would be false provenance.

## Clock and event-time contract

The interval must satisfy:

```text
started_at_ns <= requested_at_ns <= decision_time_ns
                <= evaluation_time_ns <= ended_at_ns
```

Every record time lies between start and evaluation time, is nondecreasing in record sequence, and lies inside the capture interval.

The generated assurance bundle uses:

- `requested_at_ns` from the capture interval;
- `decision_time_ns` from the last pre-dispatch decision boundary;
- `evaluation_time_ns` from the declared capture evaluation boundary.

Lifecycle evidence `observed_at_ns` always uses the client event or receipt time. `GoalInfo.stamp` and status `accepted_at_ns` are preserved separately as server-side reports. The adapter never substitutes them for receipt time and never claims a measured clock offset or transport latency.

Exact clock-object equality verifies only consistent labels. It does not measure ROS `/clock` alignment, PTP/NTP synchronization, jumps, queuing, callback delay, or server/client offset.

## Goal identity and payload binding

The capture has one exact 16-byte goal UUID encoded as 32 lowercase hexadecimal characters. Feedback and result records must use that UUID. Target status extraction uses exact UUID equality.

The goal payload is JSON converted into the configured ROS Goal message by `rosidl_runtime_py.set_message_fields`. The capture stores the input JSON object and hashes its canonical JSON representation. The normalizer recomputes this digest before using any record.

The digest binds the declared payload representation. It does not prove that a remote server received identical serialized bytes, interpreted units correctly, or executed the requested behavior.

## Status and authority normalization

ROS `action_msgs/msg/GoalStatus` numeric values map as follows:

| Code | Lifecycle value |
| ---: | --- |
| 0 | unknown; audited but not promoted |
| 1 | accepted |
| 2 | executing |
| 3 | canceling |
| 4 | succeeded |
| 5 | canceled |
| 6 | aborted |

For the exact goal UUID, the normalizer:

1. preserves every status report in the audit report;
2. ignores code `0` as unset/unknown rather than manufacturing an assurance lifecycle value;
3. de-duplicates exact same-code reports at the same client receipt time;
4. rejects two different nonzero status codes at one receipt time;
5. rejects more than one visible status publisher identity;
6. preserves missing publisher identity as an explicit visibility limitation;
7. emits one lifecycle evidence record for every remaining report.

Visible uniqueness is not authentication. A publisher GID may be absent, replaced by replay tooling, or associated with a misconfigured/malicious producer. The report therefore distinguishes `visible_publisher_ids`, missing identity records, uniqueness verification, and unverified truthfulness.

Status transition consistency is evaluated later by action assurance. The adapter preserves ordered reports instead of silently repairing an invalid transition.

## Result and feedback treatment

A result response must follow a recorded get-result request for an accepted goal and must carry one terminal status code: succeeded, canceled, or aborted. Its lifecycle value becomes `action_result` evidence.

The result payload is preserved and canonically hashed in the normalization report. It is never converted into `effect_observation`. A project that wants to interpret a typed result field as condition/effect evidence needs a separate, explicit field-to-predicate adapter with its own contract and tests.

Feedback payloads and hashes are also preserved in the report. They are never promoted to condition, effect, progress, or physical-state truth by v1.

This separation prevents two common errors:

```text
server result SUCCEEDED != independently observed physical success
result/feedback payload != declared effect evidence
```

## Lifecycle evidence output

The generated source uses `robot-spatial-action-evidence-source.v1`. It contains only:

- one `goal_response` record;
- zero or more `action_status` records;
- zero or one `action_result` record.

Each record binds exactly the configured `action_instance_id`, uses client observation time, and states three limitations:

- server identity is unauthenticated;
- client timing does not establish server time or latency;
- the report does not establish physical execution, success, causation, safety, or authorization.

The producer type means “server report observed through this bound client capture.” It is provenance, not a truth oracle.

## Assurance bundle integration

The normalizer creates a `robot-spatial-action-evidence-bundle.v1` next to the declared source paths. Every reference is relative to the bundle directory and contains the exact file digest.

Supplemental sources must:

- be existing regular non-symlink files;
- lie inside the bundle directory;
- use `robot-spatial-action-evidence-source.v1`;
- have the exact same clock object;
- contain at least one record;
- have unique source IDs and paths.

The later assurance compiler performs full record validation and exact digest checks. The action adapter does not merge records or reinterpret a supplemental producer.

Conditions are selected at decision time. Lifecycle and effects are selected at evaluation time. Thus an action may have a valid server result while readiness is unknown, or have every precondition satisfied while no dispatch/result was observed.

## Failure behavior

All output paths must be new and distinct. On validation rejection, normalization writes no output.

Typical failures and required responses are:

| Failure | Meaning | Correct response |
| --- | --- | --- |
| config/capture digest mismatch | capture belongs to different config bytes | restore exact config or recapture |
| functional semantic/file digest mismatch | declared action source changed | regenerate config and capture binding |
| goal payload digest mismatch | payload bytes/meaning changed | restore exact payload or recapture |
| wrong UUID on feedback/result | record belongs to another goal | repair capture producer; never relabel |
| missing goal response | acceptance/rejection was not observed | retain incomplete capture; do not normalize lifecycle evidence |
| result without accepted goal/get-result request | trace is incomplete or contradictory | fix capture producer or retain rejection evidence |
| nonterminal result status | result protocol record is malformed | inspect server/capture implementation |
| multiple visible status publishers | server authority is ambiguous | remove competing server or define a future arbitration contract |
| conflicting same-time target status | ordering cannot resolve the conflict | retain rejection; investigate producers/timestamps |
| event after evaluation time | assurance selection boundary excludes captured event | choose an honest later evaluation time or split the trace |
| supplemental source outside bundle | artifact is not safely content-addressable | place it under the bundle directory |

Do not weaken the v1 rejection rules merely to make a capture pass.

## Live ROS requirements

`execute-capture` requires a sourced ROS 2 environment containing:

- `rclpy`;
- `action_msgs`;
- `unique_identifier_msgs`;
- `rosidl_runtime_py` conversion, field-setting, and dynamic interface utilities;
- the package that defines the configured action type.

The command dynamically loads `package/action/Type`, converts the goal JSON, supplies an explicit UUID to `ActionClient.send_goal_async`, subscribes to the derived status topic using the action status QoS profile, waits for goal and result futures, and records feedback callbacks.

`probe` verifies only Python imports. It does not verify discovery, action availability, QoS compatibility, DDS security, message delivery, goal safety, result timing, or physical hardware.

The host development environment may report `live_execute_capture: unavailable`. That is an honest host limitation; offline normalization and its tests remain usable.

The v1 reference integration was exercised on 2026-07-18 in the official `ros:jazzy-ros-base` container with `example_interfaces/action/Fibonacci`, a known rclpy `ActionServer`, and the adapter's rclpy `ActionClient`. The trace contained one send-goal request, an accepted response, five feedback callbacks, executing and succeeded status reports, one get-result request, and a succeeded result carrying the expected Fibonacci sequence. Offline normalization, action-assurance compilation, and exact assurance verification all passed. Jazzy's callback metadata in that run exposed receipt/source sequence metadata but no publisher GID; the report therefore correctly retained status authority as not verified. This proves the tested Jazzy client/server interface path, not hardware motion, every ROS distribution/RMW, QoS behavior under loss, authenticated server identity, or safety.

## Verification

Run the focused unit/system tests:

```bash
python3 -m unittest scripts/tests/test_ros_action_adapter.py -v
```

Run the independent generated-case oracle:

```bash
python3 scripts/crosscheck_ros_action_adapter.py \
  --cases 32 --seed 20260718 --out work/ros-action-crosscheck.json
```

The oracle does not import the adapter. It derives expected lifecycle records directly from raw synthetic captures and invokes normalization through the CLI. It covers success, abort, cancellation, acceptance-only, rejection, unknown/other goals, duplicates, identity visibility, publisher conflict, simultaneous status conflict, incomplete service exchange, nonterminal result, missing get-result request, digest tampering, and time-boundary rejection.

These checks validate deterministic offline semantics. A separate sourced ROS test is required for middleware/API integration.

## Version 1 boundary

Version 1 handles one client, one goal UUID, goal response, feedback, status arrays, and result response. It does not implement cancel request/response capture, service event introspection, DDS authentication, SROS2 identity verification, original rosbag service reconstruction, serialized-wire digesting, action type semantic interpretation, feedback/result-to-predicate mappings, clock synchronization measurement, transport latency estimation, multi-goal client sessions, status authority arbitration, server failover, physical telemetry validation, causal inference, controller semantics, motion planning, collision checking, dispatch policy, or safety certification.

The adapter makes action protocol evidence legible to the AI while preserving exactly what that evidence cannot mean.
