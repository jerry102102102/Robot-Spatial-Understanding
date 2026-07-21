# Simulation evidence contract

## Contents

1. Purpose and truth boundary
2. `simulation-run.v1`
3. Generic trace import
4. `task-spec.v1`
5. Predicate results
6. Assurance report
7. Action-assurance bridge
8. Counterfactual replay
9. Oracle isolation
10. Failure handling

## 1. Purpose and truth boundary

The simulation evidence layer turns immutable simulator state, events, contacts, and collisions
into typed claims. It does not import a benchmark reward or success label. It establishes results
only for the declared simulator, versions, assets, seed, clock, streams, interval, task policy,
and thresholds.

Keep these claims separate:

- a controller or ROS Action reported success;
- the commanded trajectory reached a declared state;
- a task effect was observed in simulator state;
- the full task goal is supported;
- a matched counterfactual supports contribution inside the simulator;
- hardware truth, authorization, and safety remain unestablished.

## 2. `simulation-run.v1`

A run is a directory containing:

```text
run/
├── run.json
├── completeness.json
├── events.jsonl
└── streams/
    ├── joint_state.npz
    ├── pose.npz
    ├── contact.npz
    └── collision.npz
```

`run.json` uses schema `robot-spatial-simulation-run.v1`. It binds:

- `run_id`, simulator name/version, adapter name/version, seed, timestep, and intervention;
- one clock ID and domain;
- exact episode interval and task ID;
- robot/world declarations and model or asset digests supplied by the producer;
- meter/radian/`xyzw`/`world_from_entity` conventions;
- every standard channel as `available` with a relative path and digest, or `unavailable` with a
  reason;
- the immutable event log and source-trace digest;
- explicit simulation, oracle, hardware, causation, and safety boundaries.

The manifest contains a canonical `manifest_sha256`. Each NPZ is written deterministically with
fixed ZIP metadata so the same normalized arrays have the same digest.

Standard channels are:

| Channel | Core arrays | Meaning |
| --- | --- | --- |
| `joint_state` | time, joint IDs, position/velocity/effort plus presence masks | normalized joint reports |
| `pose` | time, entity IDs, position, quaternion, presence | world-from-entity rigid poses |
| `odometry` | pose plus linear/angular velocity | mobile-base state |
| `contact` | time, body pair, active, optional normal force | simulator contact reports |
| `collision` | time, body pair, active | simulator collision reports |
| `force_torque` | time, sensor IDs, force and torque | declared wrench reports |
| `deformable` | time, entity IDs, keypoints and presence | partial deformable state, not a full surface proof |

`completeness.json` detects out-of-order samples, conflicting duplicates, gaps, missing interval
coverage, missing values, and invalid quaternions. Unavailable optional channels do not invalidate
available channels. A predicate still declares its own required channels and continuity policy.

## 3. Generic trace import

`robot-spatial-generic-trace.v1` is the stable adapter input. It contains run metadata, conventions,
channel policies, sample arrays expressed as JSON records, events, and optional asset declarations.

The importer rejects any field named `reward`, `success`, `is_success`, `official_success`,
`oracle`, `evaluator`, or equivalent. Adapters map simulator state fields and entity identities;
they must never implement task-success logic.

Import with:

```bash
robot-spatial import --adapter generic-json trace.json --out run/
robot-spatial capture --adapter maniskill --source state-export.json --out run/
robot-spatial capture --adapter maniskill --env-id PickCube-v1 --seed 2 \
  --trajectory actions.h5 --entity-map pickcube-entities.yaml \
  --sim-backend physx_cpu --fixed-horizon 100 --out run/
robot-spatial capture --adapter gymnasium-robotics --env-id FetchReach-v3 --seed 2 --out run/
```

The core package remains offline by default. The optional `mujoco` extra adds a bounded live
Gymnasium Robotics GoalEnv capture. The optional `maniskill` extra adds fixed-horizon replay of an
action-only trajectory. These adapters record raw state, actions, simulator/model digests, and
versions while deliberately discarding outcome-bearing return values. Live Gazebo and deformable
capture remain adapter/plugin work.

## 4. `task-spec.v1`

`robot-spatial-task-spec.v1` declares:

- `task_id` and explicit role-to-entity bindings;
- required evidence channels;
- generic predicates with parameters and optional time windows;
- goal and failure expressions using `all`, `any`, `not`, or `predicate`;
- termination and claim boundaries.

Task-specific Python evaluators are prohibited. A new task may supply a new declarative spec, but a
held-out benchmark may not add new success code.

Core predicate types:

- `joint_within_tolerance`
- `joint_position_in_range`
- `joint_velocity_below_threshold`
- `frame_within_pose_tolerance`
- `frame_position_within_tolerance`
- `frame_position_in_bounds`
- `frame_velocity_below_threshold`
- `base_reached_goal`
- `collision_free_over_interval`
- `path_stayed_within_corridor`
- `contact_sustained`
- `contact_force_at_terminal`
- `object_above_height`
- `object_follows_frame_for_duration`
- `object_inside_region`
- `object_grasped`
- `object_released_in_region`
- `inserted_to_depth`
- `evidence_conjunction`
- `deformable_keypoints_in_region`
- `deformable_shape_within_tolerance`

`object_grasped` is a composite. It requires every referenced contact predicate plus gripper-state,
relative-following, and lift predicates. Contact alone cannot support a grasp.

`object_above_height` accepts either an absolute `minimum_m` or a `minimum_delta_m` relative to the
first observed pose in the evaluation window. `collision_free_over_interval` can exempt explicitly
declared `allowed_pairs` and `ignored_pairs`; an unlisted active pair still refutes the predicate.

`frame_within_pose_tolerance.target` may be either a fixed pose or `{entity: role}`. The latter
compares two observed poses at the exact evaluated sample and lets a simulator-supplied goal remain
state evidence instead of copying episode-specific coordinates into the task spec.

`frame_position_within_tolerance.axes` can project a comparison onto a declared subset of `x`,
`y`, and `z`. `frame_position_in_bounds` can check world or reference-local component bounds;
each bound can be a literal or a digest-bound per-episode measurement from `run.world.measurements`.
`frame_velocity_below_threshold` evaluates captured rigid-body linear/angular velocity and must
abstain when the terminal velocity is stale. `contact_force_at_terminal` can additionally compare
a contact vector to a declared local body axis. `evidence_conjunction` preserves the evidence
digests of every required predicate instead of introducing task-specific Python success code.

## 5. Predicate results

Every predicate returns exactly one status:

- `supported`: declared evidence supports the predicate;
- `refuted`: complete relevant evidence contradicts it;
- `unknown`: required evidence is unavailable, stale, incomplete, or not observed;
- `conflicting`: required evidence is internally inconsistent.

Every result stores an evidence digest, source stream digest, sample indices or time interval,
measured values, thresholds, missing evidence, and limitations. Results do not use a confidence
score.

Positive interval-wide claims require complete interval evidence. A directly observed collision
can refute collision freedom even if another part of the interval is missing; the inverse claim
cannot be supported without complete coverage.

## 6. Assurance report

`robot-spatial-simulation-assurance-report.v1` binds the run, completeness report, task spec, and
all predicate evidence. It separately reports:

1. model/geometry validation;
2. controller or Action protocol reports;
3. trajectory execution;
4. observed task effects;
5. simulation-bounded physical success;
6. causation;
7. authorization;
8. safety;
9. unknown or conflicting evidence.

Use:

```bash
robot-spatial evaluate run/ --task task.yaml --out result/
robot-spatial explain result/report.json --out result/report.md
```

The Markdown explanation cites predicate evidence digests. Natural-language answers must not
replace these artifacts with an unsupported interpretation.

## 7. Action-assurance bridge

`robot-spatial-simulation-action-map.v1` explicitly maps a simulation predicate to one declared
functional-model `effect/` ID, predicate, bindings, producer, and action instance. The bridge emits
an exact `robot-spatial-action-evidence-source.v1` file:

```bash
robot-spatial action-evidence result/report.json \
  --mapping action-map.json \
  --out evidence/simulation-effects.json
```

`supported`, `refuted`, and insufficient predicate results become `true`, `false`, and `unknown`
effect observations. The record retains the run, report, predicate, and mapping digests. Add the
result as a digest-bound evidence source in the existing action-evidence bundle; do not bypass the
existing action-assurance lifecycle compiler.

## 8. Counterfactual replay

Two runs can support a stronger, simulator-bounded contribution claim only when they share exact:

- simulator/version, seed, timestep, clock, robot, world, and conventions;
- normalized initial-state fingerprint;
- task spec;
- different declared interventions, where the control is `no_op` or `controlled_perturbation`.

Use:

```bash
robot-spatial counterfactual \
  --action-run action-run/ \
  --control-run no-op-run/ \
  --task task.yaml \
  --out counterfactual.json
```

Only action-supported plus control-refuted yields
`supported_under_controlled_simulation`. This remains weaker than real-world causal proof.

## 9. Oracle isolation

A `robot-spatial-benchmark-suite.v1` lists run, task, and reference-result paths. Reference files
must be outside candidate run directories. The runner completes and writes every prediction before
opening any `robot-spatial-reference-result.v1`.

```bash
robot-spatial benchmark --suite suite.yaml --out benchmark-result/
```

The report includes confusion matrices, per-label precision/recall/F1, macro-F1, Wilson 95%
intervals, confirmed-success precision, false-positive rate, per-case results, and binding digests.
References may score an exact subset of the candidate predicate inventory only when they also list
the complementary IDs as `unscored_predicates`. Scored and unscored IDs must be disjoint and cover
the candidate inventory exactly.

## 10. Failure handling

- Reject digest mismatches before evaluation.
- Return `conflicting` for out-of-order, conflicting duplicate, or invalid quaternion evidence.
- Return `unknown` for unavailable, stale, gapped, or incomplete evidence required by a claim.
- Reject task/run ID mismatch.
- Reject reference results inside a candidate run.
- Never promote controller status, reward, feedback, result payload, or benchmark label into effect
  evidence.
- Never upgrade simulation output into hardware truth or a safety certificate.
