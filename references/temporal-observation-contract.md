# Timestamped observation contract

## Purpose and epistemic separation

`robot-spatial-observation-log.v1` records what named sources reported at particular timestamps. `robot-spatial-observation-log.v2` preserves the same stream semantics and adds digest-bound ROS normalization provenance when produced from JointState/TF capture. Neither replaces URDF or `robot-spatial-world-scene.v1`:

1. URDF is the robot mechanism model and its deterministic robot-local consequences.
2. The world scene is a static declaration of frames, root mounting, gravity, objects, and geometry.
3. The observation log is a timestamped report layer selected under an explicit query-time policy.

For ROS 2 input, do not hand-author v1 samples from topic output. Use `ros_observation_adapter.py` to preserve an immutable capture, reject ambiguous authority/clock/TF conditions, and produce v2. Read [ros-observation-adapter-contract.md](ros-observation-adapter-contract.md) before interpreting an assembled joint snapshot or composed TF pose.

Never call a static-scene fallback an observation. Never call a sample `current` without naming its clock domain, query time, age, and maximum age. `current` establishes only that the selected timestamp passes that policy; it does not establish source truth, calibration, physical completeness, covariance-bounded geometry, or safety.

## Observation log schema

```json
{
  "schema_version": "robot-spatial-observation-log.v1",
  "observation_log_id": "cell_a_run_42",
  "clock": {
    "domain": "ros",
    "unit": "nanoseconds",
    "epoch": "ROS_TIME"
  },
  "binding": {
    "robot_name": "my_robot",
    "root_link": "base_link",
    "source_urdf_semantic_sha256": "<exact semantic digest>",
    "scene_id": "cell_a",
    "scene_sha256": "<exact scene-file digest>"
  },
  "source": {
    "type": "measured",
    "reference": "rosbag2/run_42",
    "sensor_id": null,
    "topic": null
  },
  "streams": {
    "joint_states": [
      {
        "sample_id": "joint_1700000000",
        "timestamp_ns": 1700000000,
        "positions": {"shoulder": 0.2, "slide": 0.15},
        "position_standard_deviation": {"shoulder": 0.001},
        "source": {
          "type": "measured",
          "reference": "message 431",
          "sensor_id": "joint_encoders",
          "topic": "/joint_states"
        }
      }
    ],
    "robot_root_poses": [
      {
        "sample_id": "base_1700000005",
        "timestamp_ns": 1700000005,
        "parent_scene_frame": "world",
        "pose": {
          "xyz_m": [1.0, 2.0, 0.3],
          "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
        },
        "covariance": {
          "order": "xyz_m_then_rotation_vector_rad",
          "matrix_6x6_rowmajor": [
            0.0001, 0, 0, 0, 0, 0,
            0, 0.0001, 0, 0, 0, 0,
            0, 0, 0.0001, 0, 0, 0,
            0, 0, 0, 0.0004, 0, 0,
            0, 0, 0, 0, 0.0004, 0,
            0, 0, 0, 0, 0, 0.0004
          ]
        },
        "source": {
          "type": "measured",
          "reference": "tf message 992",
          "sensor_id": "localization",
          "topic": "/tf"
        }
      }
    ],
    "object_poses": {
      "fixture": [
        {
          "sample_id": "fixture_1700000010",
          "timestamp_ns": 1700000010,
          "parent_scene_frame": "world",
          "pose": {"xyz_m": [1.4, 2.0, 0.7], "rpy_rad": [0.0, 0.0, 0.2]},
          "source": {
            "type": "measured",
            "reference": "tracker detection 17",
            "sensor_id": "camera_1",
            "topic": "/tracked_objects"
          }
        }
      ]
    }
  }
}
```

Every sample ID and stream timestamp must be unique inside its stream; duplicate timestamps are rejected rather than resolved by input order. Timestamps are non-negative integer nanoseconds in exactly one declared clock domain. The log binds the exact URDF semantic digest and scene-file digest so a valid sample cannot silently migrate to a different mechanism or scene.

Every joint sample must include all independent movable drivers. Known movable mimic followers may also be present, but their reported values must agree with the complete mimic relation within `1e-9`. Fixed and unknown joints are rejected. Joint standard deviations and pose covariance are preserved as reported probabilistic metadata; version 1 does not turn them into hard geometric bounds.

Root and object samples may use any declared static scene frame as their parent. Object stream IDs must name objects declared by the bound world scene. Version 1 observes poses of declared objects; it does not dynamically create previously unknown object geometry.

## Query policy schema and time selection

```json
{
  "schema_version": "robot-spatial-observation-query.v1",
  "query_id": "control_cycle_1700000020",
  "time_ns": 1700000020,
  "maximum_age_ns": {
    "joint_states": 20000000,
    "robot_root_pose": 100000000,
    "object_pose": 250000000
  },
  "fallbacks": {
    "robot_root": "require_observed",
    "objects": "allow_static_declaration"
  },
  "required_object_ids": ["fixture"]
}
```

For each stream, the resolver selects the sample with the greatest timestamp satisfying `sample.timestamp_ns <= query.time_ns`. This is zero-order hold. It does not interpolate, extrapolate, average, or consume future samples. Age is `query.time_ns - selected.timestamp_ns`:

- `current`: age is no greater than that stream's maximum age;
- `stale`: a past sample exists but its age exceeds the maximum;
- `missing`: no sample exists at or before the query time.

Joint state never falls back to a declaration. A root or required object can use the static scene only when its fallback is explicitly `allow_static_declaration`; the effective source is then labeled `static_scene_declaration`. An unrequired object may remain static context, but it does not become current observed evidence.

`all_required_observations_current` is true only when joint state, root pose, and every required object pose are current. `nominal_world_state_computable` may also be true with an explicit static-declaration fallback. These flags must not be conflated.

## Observation-conditioned analysis

The effective root and object poses overlay the static scene while retaining the scene's coordinate-frame graph and geometry declarations. The effective joint driver values feed URDF forward kinematics. `observe-transform`, `observe-collisions`, and `observe-gravity-loads` therefore compute deterministic nominal consequences of:

- exact model and scene digests;
- the explicit query-time policy;
- the selected past sample IDs and ages;
- any visible declaration fallback;
- declared geometry, inertials, gravity, and tolerance.

`observe-collisions` always separates `nominal_declared_geometry_result` from `physical_collision_status` and `safety_conclusion`. The latter two remain `not_established`. Even all-current nominal collision-free geometry does not establish that sensors are truthful, calibration is correct, covariance cannot reach collision, every physical object was observed, or no collision occurred between samples.

## Commands

```bash
python3 scripts/robot_spatial.py observe-summary robot.urdf --scene world-scene.json --observations observations.json --observation-query query.json --package-map package-map.json
python3 scripts/robot_spatial.py observe-transform robot.urdf --scene world-scene.json --observations observations.json --observation-query query.json --from scene_frame/world --to robot_frame/tool0
python3 scripts/robot_spatial.py observe-collisions robot.urdf --scene world-scene.json --observations observations.json --observation-query query.json --package-map package-map.json --contact-tolerance-m 1e-9
python3 scripts/robot_spatial.py observe-gravity-loads robot.urdf --scene world-scene.json --observations observations.json --observation-query query.json
python3 scripts/robot_spatial.py export robot.urdf --scene world-scene.json --observations observations.json --observation-query query.json --package-map package-map.json --out work/observed-context
python3 scripts/robot_spatial.py check-invariants robot.urdf --scene world-scene.json --observations observations.json --observation-query query.json --contract invariants.json --package-map package-map.json
```

An observation query result binds the raw URDF and semantic digests, scene digest, log digest, query-policy digest, log/query IDs, query time, parameters, method, and epistemic scope. `export` adds the resolved observations, nominal analysis, grounded facts, an `observation_log/<id>` entity card, unresolved physical claims, and blind temporal competency questions. `prepare` preserves both input digests in its source manifest and compilation record.

## Temporal invariants

An invariant contract may bind the exact observation artifacts:

```json
{
  "observation": {
    "log_id": "cell_a_run_42",
    "log_sha256": "<exact log digest>",
    "query_id": "control_cycle_1700000020",
    "query_sha256": "<exact query-policy digest>"
  }
}
```

Supported temporal assertions are:

- `observation_readiness`: protect status, current/computable flags, and exact fallback entities;
- `observation_transform`: protect a directed typed transform under the effective selected state;
- `observation_collision`: protect nominal status, analysis-current/fallback status, and current-readiness flag while leaving physical truth and safety explicitly unestablished.

These assertions gate a data/model pipeline. They are not safety certificates and do not authorize changing age limits merely to restore a passing result.

## Version 1 boundary

The resolver is offline and discrete. It accepts normalized v1/v2 JSON, not ROS messages directly. The separate adapter can capture live JointState/TF into an immutable transport artifact and normalize it, but the resolver still supports only joint positions, one robot-root pose stream, and poses for objects already declared in one static single-robot scene. It does not implement interpolation, velocity/effort observations, verified clock synchronization, message latency compensation, transform-buffer topology changes, sensor calibration, data association, dynamic object creation, occupancy maps, probabilistic collision, continuous collision, contact state, trajectory history, multi-robot worlds, or online safety decisions.

Those are deliberate unknowns, not implied future behavior. Extend the schema and evaluation suite before claiming any of them.
