# Project spatial invariant contract

## Contents

- Purpose and trust model
- Contract schema
- Assertion types
- Tolerances and pose rules
- Edit acceptance workflow

## Purpose and trust model

`robot-spatial-invariants.v1` records spatial, declared mass/static-load, and embedded actuation-declaration relationships that the project owner intends to preserve. URDF describes what the current model says; the invariant contract declares what must remain true after an edit.

Never infer invariants from frame names. A TCP offset, collision-free home pose, required chain, or protected frame identity becomes an invariant only when the project or user asserts it. Contract evaluation is deterministic, and a failed assertion is an edit rejection signal rather than a warning.

## Contract schema

```json
{
  "schema_version": "robot-spatial-invariants.v1",
  "robot": "two_dof_demo",
  "world_scene": {
    "scene_id": "cell_a",
    "snapshot_id": "cell_a_snapshot_001",
    "sha256": "<exact-scene-file-sha256>"
  },
  "observation": {
    "log_id": "cell_a_run_42",
    "log_sha256": "<exact-observation-log-sha256>",
    "query_id": "control_cycle_1700000020",
    "query_sha256": "<exact-observation-query-sha256>"
  },
  "default_tolerances": {
    "translation_m": 1e-6,
    "rotation_deg": 1e-5,
    "axis_deg": 1e-5,
    "distance_m": 1e-6,
    "aabb_m": 1e-6,
    "contact_m": 1e-9,
    "mass_kg": 1e-9,
    "center_of_mass_m": 1e-9,
    "inertia_kg_m2": 1e-9,
    "generalized_effort": 1e-9,
    "gravity_m_s2": 1e-9
  },
  "poses": {
    "home": {"joints": {"shoulder": 0.0, "slide": 0.2}}
  },
  "assertions": [
    {
      "id": "declared-arm-mass-model",
      "type": "declared_mass_properties",
      "pose": "home",
      "subtree_root": "arm_link",
      "frame": "base_link",
      "expected": {
        "status": "computed",
        "declared_mass_kg": 12.5,
        "center_of_mass_xyz_m": [0.31, 0.0, 0.42],
        "inertia_about_center_of_mass_matrix_3x3_kg_m2": [
          [0.2, 0.0, 0.0],
          [0.0, 0.8, 0.0],
          [0.0, 0.0, 0.9]
        ],
        "missing_inertial_links": [],
        "invalid_or_incomplete_inertial_links": []
      }
    },
    {
      "id": "home-gravity-hold-model",
      "type": "static_gravity_loads",
      "pose": "home",
      "subtree_root": "arm_link",
      "gravity_frame": "base_link",
      "gravity_vector_xyz_m_s2": [0.0, 0.0, -9.80665],
      "expected": {
        "status": "computed",
        "generalized_gravity_forces": {"shoulder": -4.2, "slide": -19.6},
        "ideal_static_holding_efforts": {"shoulder": 4.2, "slide": 19.6},
        "missing_inertial_links": [],
        "invalid_or_incomplete_inertial_links": []
      }
    },
    {
      "id": "embedded-control-contract",
      "type": "actuation_declarations",
      "expected": {
        "ros2_control_systems": ["ArmSystem"],
        "legacy_transmissions": [],
        "joint_command_interfaces": {"shoulder": ["effort", "position"]},
        "joint_state_interfaces": {"shoulder": ["effort", "position", "velocity"]}
      }
    },
    {
      "id": "tool-offset",
      "type": "frame_pose",
      "pose": "home",
      "from": "flange",
      "to": "tool0",
      "expected": {
        "translation_xyz_m": [0.0, 0.0, 0.12],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
      }
    },
    {
      "id": "home-is-clear",
      "type": "self_collision_status",
      "pose": "home",
      "expected": "collision_free"
    },
    {
      "id": "mounted-root",
      "type": "scene_transform",
      "pose": "home",
      "from": "scene_frame/world",
      "to": "robot_frame/base_link",
      "expected": {
        "translation_xyz_m": [1.0, 2.0, 0.3],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
      }
    },
    {
      "id": "world-gravity-hold",
      "type": "scene_gravity_loads",
      "pose": "home",
      "expected": {
        "status": "computed",
        "gravity_in_robot_root_xyz_m_s2": [0.0, 0.0, -9.80665],
        "generalized_gravity_forces": {"shoulder": -4.2}
      }
    },
    {
      "id": "fixture-collision-state",
      "type": "robot_environment_collision",
      "pose": "home",
      "expected": {
        "status": "collision_free",
        "minimum_separation_status": "computed",
        "minimum_separation_m": 0.08,
        "collision_pairs": [],
        "indeterminate_pair_count": 0
      }
    }
  ]
}
```

Pose name `zero` is reserved for the default all-zero independent joint state. Every other referenced pose must be declared in `poses`; joint values use radians for revolute/continuous joints and meters for prismatic joints.

The optional `world_scene` block is mandatory when any scene assertion is present. `scene_id` and `snapshot_id` must exactly match the supplied `--scene`; include `sha256` to reject a different scene file even when its IDs are reused.

The optional `observation` block is mandatory for temporal assertions. Its four fields bind the exact normalized log and query-time policy. Supply `--observations`, `--observation-query`, and `--scene` together. See [temporal-observation-contract.md](temporal-observation-contract.md) for sample selection and epistemic rules.

## Assertion types

| Type | Required fields | Meaning |
|---|---|---|
| `frame_pose` | `pose`, `from`, `to`, expected translation and `quaternion_xyzw` | Preserve a complete directed relative transform. |
| `scene_transform` | `pose`, typed scene-bound `from` and `to`, expected translation and `quaternion_xyzw` | Preserve a directed transform across scene and mounted-robot namespaces. |
| `frame_distance` | `pose`, `from`, `to`, `expected_m` | Preserve Euclidean distance between frame origins. |
| `declared_mass_properties` | `pose`, `subtree_root`, `frame`, non-empty `expected` | Preserve selected URDF-declared mass, center of mass, aggregate inertia, declaration status, and/or coverage. |
| `static_gravity_loads` | `pose`, `subtree_root`, `gravity_frame`, gravity vector, non-empty `expected` | Preserve declared-model generalized gravity forces, opposite ideal static holding efforts, status, and/or inertial coverage under one explicit convention. |
| `scene_gravity_loads` | `pose`, non-empty `expected` | Preserve scene-derived root gravity, declared-model generalized gravity forces, holding efforts, and/or status for the bound snapshot. |
| `actuation_declarations` | non-empty `expected` | Preserve exact embedded ros2_control system names, legacy transmission names, and selected joints' command/state interface sets without claiming runtime capability. |
| `joint_axis` | `pose`, `joint`, `frame`, `expected_unit_vector` | Preserve the signed joint motion direction expressed in an explicit frame. |
| `chain` | `from_link`, `to_link`, expected `links` and/or `joints` | Preserve ordered kinematic connectivity. |
| `affected_links` | `joint`, `expected_links` | Preserve the exact unordered causal subtree of an independent driver. |
| `frame_semantics` | `frame`, expected `semantic_type` and/or `parent_frame` | Prevent link/joint/visual/collision/inertial frame identity from being collapsed or reparented. |
| `geometry_aabb` | `pose`, `geometry_frame`, expected root-frame min/max | Preserve pose-conditioned declared geometry placement and scale. Meshes require inspection and package resolution. |
| `self_collision_status` | `pose`, expected `collision`, `collision_free`, or `indeterminate` | Preserve physical self-collision status under an explicit contact tolerance. SRDF policy remains annotation only. |
| `robot_environment_collision` | `pose`, non-empty `expected` | Preserve aggregate status, global minimum status/value, exact collision-pair set, and/or indeterminate-pair count for the bound scene and contact tolerance. |
| `observation_readiness` | non-empty `expected` | Preserve query status, current/computable flags, and/or exact declaration-fallback entities. |
| `observation_transform` | typed `from` and `to`, expected translation and quaternion | Preserve a directed transform under the effective time-selected joint/root/object state. |
| `observation_collision` | non-empty `expected` | Preserve nominal declared-geometry status, current-versus-fallback analysis status, and/or required-observation readiness without claiming physical safety. |

Structural assertions use `pose: "pose_independent"` in reports. Pose-conditioned assertions name the evaluated contract pose.

## Tolerances and pose rules

Default tolerance fields are unit-bearing and apply by assertion family. An assertion may override one with:

- `translation_tolerance_m`
- `rotation_tolerance_deg`
- `axis_tolerance_deg`
- `distance_tolerance_m`
- `aabb_tolerance_m`
- `contact_tolerance_m`
- `mass_tolerance_kg`
- `center_of_mass_tolerance_m`
- `inertia_tolerance_kg_m2`
- `generalized_effort_tolerance`
- `gravity_tolerance_m_s2`

Quaternion comparison accepts `q` and `-q` as the same rotation. Joint-axis comparison is signed: opposite axes differ by 180 degrees because they encode opposite positive motion. AABB comparison uses the maximum absolute root-frame coordinate error. Center-of-mass comparison uses Euclidean distance; inertia comparison uses the maximum absolute 3x3 tensor-component error in the explicitly named frame.

`declared_mass_properties` applies forward kinematics and the parallel-axis theorem to valid URDF inertials in the selected subtree. Its numeric result is exact for that declared model and pose, subject to floating-point tolerance. It does not prove payload, cabling, tooling, calibration, or physical-hardware mass properties. Missing inertials remain a reported coverage gap and never mean zero physical mass.

`static_gravity_loads` applies the explicit gravity vector/frame at the contract pose and compares only the named independent-driver values. Revolute/continuous values are N·m and prismatic values are N, so each expected joint's type supplies its unit. The tolerance is a magnitude in the corresponding joint effort unit. Sign is not discarded: generalized gravity force and ideal holding effort are opposites. This assertion protects a declared gravity-only model, not actual mounting orientation, payload, contact, dynamics, controller behavior, or actuator feasibility.

`scene_gravity_loads` obtains gravity from the digest-bound scene and compares the rotated root-frame vector with `gravity_m_s2`, then compares named loads with `generalized_effort`. `scene_transform` uses the normal translation and rotation tolerances. Neither assertion establishes that the scene calibration or snapshot is physically correct or current.

`robot_environment_collision` compares the exact sorted collision-pair set when requested. A protected `collision_free` status is conditional on complete declared pair coverage in that exact scene; it is not a physical safety certificate. An expected `indeterminate_pair_count` makes unresolved coverage visible instead of allowing a future engine or source change to silently alter the epistemic state.

Temporal assertions use the exact bound log/query artifacts and never consult future samples. `observation_transform` uses normal translation/rotation tolerances. `observation_collision` always reports `physical_collision_status` and `safety_conclusion` as `not_established`; its invariant protects only the nominal declared-model result and temporal readiness. Do not loosen maximum ages, allow declaration fallback, or edit the expected status merely to recover a passing gate.

`actuation_declarations` compares system/transmission arrays and interface arrays as unordered exact sets. It is pose-independent and intentionally ignores external controller YAML, launch/runtime state, plugin installation, interface claiming, hardware connectivity, and behavior. Protecting an embedded declaration does not certify that it works; it prevents an edit from silently changing what the expanded URDF says.

Do not use a loose tolerance to hide a design change. If a relationship is intentionally changed, update the authoring source, regenerate evidence, review the numeric delta, and then update the contract in the same reviewed change.

`self_collision_status: "indeterminate"` or a scene collision expectation of `indeterminate` may document a known unsupported case, but neither may be treated as evidence of collision freedom in a safety gate.

## Edit acceptance workflow

1. Run `check-invariants` before an edit and retain the passing report as the baseline.
2. Make the smallest authoring-source change; do not patch generated URDF unless it is the declared source of truth.
3. Rerun `validate`, `export`, collision analysis, rendering, and `check-invariants`.
4. Reject the edit when the command exits non-zero. Inspect each result's expected value, actual value, numeric error, tolerance, pose, and source digest.
5. Use `compare` on before/after `model.json` artifacts to identify spatial changes not covered by the contract.
6. Update an invariant only when the design intent itself changed and that change has explicit project approval.

`export --invariants contract.json` writes `invariants-report.json`, embeds the report in `model.json`, adds grounded facts and evaluation questions, and exits non-zero when any assertion fails. This makes the same intent visible to Codex, reviewers, retrieval systems, and CI.
