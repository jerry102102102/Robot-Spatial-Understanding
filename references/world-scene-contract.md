# Static world scene contract

## Purpose

URDF describes a robot-local mechanism. It does not establish where that robot is mounted, which direction physical gravity has relative to its root, or which external objects exist. Supply a separate `robot-spatial-world-scene.v1` JSON document when those questions matter.

The scene is one immutable, digest-bound static snapshot containing one mounted URDF robot instance. It is not live TF, a perception stream, a planning-scene monitor, or proof of the physical workcell. A separate timestamped observation layer may overlay root/object poses without changing what this static declaration means; see [temporal-observation-contract.md](temporal-observation-contract.md).

## Input schema

```json
{
  "schema_version": "robot-spatial-world-scene.v1",
  "scene_id": "cell_a",
  "snapshot": {
    "id": "cell_a_2026_07_18_001",
    "time_semantics": "static_snapshot",
    "captured_at": "2026-07-18T06:00:00Z",
    "valid_until": null
  },
  "world_frame": "world",
  "source": {
    "type": "measured",
    "reference": "survey export 42",
    "captured_at": "2026-07-18T06:00:00Z"
  },
  "gravity": {
    "vector_xyz_m_s2": [0.0, 0.0, -9.80665],
    "expressed_in_frame": "world",
    "source": {
      "type": "declared",
      "reference": "cell convention",
      "captured_at": null
    }
  },
  "frames": {
    "pedestal": {
      "parent": "world",
      "pose": {
        "xyz_m": [1.0, 2.0, 0.3],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
      },
      "semantics": {
        "category": "mount",
        "roles": ["planning_frame"],
        "meaning": "surveyed pedestal origin"
      },
      "source": {
        "type": "calibrated",
        "reference": "base survey",
        "captured_at": "2026-07-18T06:00:00Z"
      }
    }
  },
  "robot": {
    "instance_id": "arm_1",
    "robot_name": "my_robot",
    "root_link": "base_link",
    "parent_frame": "pedestal",
    "pose": {
      "xyz_m": [0.0, 0.0, 0.0],
      "rpy_rad": [0.0, 0.0, 0.0]
    },
    "source": {
      "type": "calibrated",
      "reference": "base survey",
      "captured_at": "2026-07-18T06:00:00Z"
    }
  },
  "objects": {
    "fixture": {
      "parent_frame": "world",
      "pose": {
        "xyz_m": [1.4, 2.0, 0.7],
        "rpy_rad": [0.0, 0.0, 0.2]
      },
      "semantics": {
        "category": "fixture",
        "roles": ["collision", "keep_out"],
        "meaning": "declared workholding fixture"
      },
      "source": {
        "type": "measured",
        "reference": "fixture scan",
        "captured_at": "2026-07-18T06:00:00Z"
      },
      "collision_geometries": [
        {
          "id": "body",
          "pose": {
            "xyz_m": [0.0, 0.0, 0.0],
            "rpy_rad": [0.0, 0.0, 0.0]
          },
          "geometry": {
            "type": "box",
            "size_xyz_m": [0.4, 0.3, 0.5]
          }
        }
      ]
    }
  }
}
```

Every ID is non-empty and cannot contain `/`, which is reserved for typed entity IDs. The scene-frame graph must be a single acyclic parent graph rooted at `world_frame`. The robot's `robot_name` and `root_link` must exactly match the parsed URDF. Object parents and gravity expression frames must name declared scene frames or `world_frame`.

A pose contains `xyz_m` and either `rpy_rad` or unit `quaternion_xyzw`, never both. Omitted pose components are identity. RPY is fixed-axis roll, pitch, yaw under the same transform convention used by URDF. Quaternions are `xyzw` and must be unit length within `1e-6`.

Supported provenance labels are `declared`, `measured`, `calibrated`, `synthetic`, `imported`, and `unknown`. They are author-supplied labels, not independently verified truth. Supported collision shapes are positive-dimension boxes, cylinders, spheres, and STL/OBJ meshes with optional nonzero `scale_xyz`. A geometry inherits its object's source when it does not declare one.

## Typed identities and transforms

Keep robot-local and snapshot-bound identities separate:

- `frame/X`: URDF-local frame whose exported `world_from_frame` uses the URDF root as a mathematical origin;
- `robot_frame/X`: the same exact URDF frame mounted in this scene snapshot;
- `robot_geometry/collision/link/index`: mounted robot collision geometry;
- `scene_frame/X`: a scene-coordinate frame;
- `scene_object/X`: an external object;
- `scene_geometry/object/geometry`: one external collision geometry.

`scene-transform --from A --to B` returns `A_from_B`: the pose of B expressed in A. Use typed snapshot identities for every cross-boundary transform. Do not relabel URDF-local `world_from_frame` as a physical world transform.

## World gravity and static loads

When `gravity` is present, the scene rotates its declared vector from `expressed_in_frame` into the world frame and then into the mounted robot root. `scene-gravity-loads` applies that root-frame vector to the URDF-declared inertials at the stated joint pose.

The result is exact for the declared inertial model, scene transform, gravity vector, and pose. It remains a gravity-only static model. It does not establish payload, actual mass calibration, contact forces, friction, acceleration, transmission loss, controller behavior, hardware feasibility, or that the supplied mount and gravity match the physical cell.

If no gravity record exists, world-conditioned gravity loads are `not_provided`. Do not silently substitute the canonical root-frame gravity convention.

## Robot/environment collision

`scene-collisions` evaluates every declared robot collision geometry against every declared environment collision geometry. It never stops after the first collision, because complete pair coverage is part of the evidence.

The exact solid representations are:

- URDF or scene STL/OBJ triangles after declared scale and world pose;
- analytic boxes represented by exact boundary triangles;
- analytic sphere/sphere solid distance;
- analytic sphere against a supported complete triangle solid.

For supported triangle solids, deterministic BVH surface distance is followed by closed-solid containment when surfaces are watertight and consistently wound. Positive boundary distance does not by itself exclude containment.

If exact solid classification is unavailable but exact world AABBs are disjoint, the pair is exactly `collision_free`; its reported distance is only a conservative lower bound. If AABBs overlap or touch and a participating solid is unsupported or unmeasured, the pair is `indeterminate`. In particular, overlapping cylinder pairs currently fail closed. A global minimum separation is promoted only when every inexact pair's lower bound cannot be smaller than the best exact candidate.

Aggregate statuses are:

- `collision` when any declared pair collides;
- `indeterminate` when no collision is established but at least one pair remains unresolved;
- `collision_free` only when every declared pair is resolved collision-free;
- `not_applicable` when the robot or scene declares no collision geometry.

These statuses concern only the objects actually declared in this static snapshot at the stated robot pose and contact tolerance. Even `collision_free` does not prove that the physical world is complete, current, calibrated, or safe.

## Commands and evidence binding

```bash
python3 scripts/robot_spatial.py validate robot.urdf --scene world-scene.json
python3 scripts/robot_spatial.py scene-summary robot.urdf --scene world-scene.json --package-map package-map.json
python3 scripts/robot_spatial.py scene-transform robot.urdf --scene world-scene.json --from scene_frame/world --to robot_frame/tool0 --pose pose.json
python3 scripts/robot_spatial.py scene-gravity-loads robot.urdf --scene world-scene.json --pose pose.json
python3 scripts/robot_spatial.py scene-collisions robot.urdf --scene world-scene.json --pose pose.json --package-map package-map.json --contact-tolerance-m 1e-9
python3 scripts/robot_spatial.py export robot.urdf --scene world-scene.json --package-map package-map.json --out work/context
python3 scripts/robot_spatial.py check-invariants robot.urdf --scene world-scene.json --contract invariants.json --package-map package-map.json
```

Every scene query binds the URDF semantic digest, scene file SHA-256, `scene_id`, `snapshot.id`, pose, parameters, method, and epistemic scope. Preserve that `query_evidence` with any claim. `prepare --scene` copies the same binding into source compilation and the generated progressive context.

## Explicit boundary

The scene schema itself supports one mounted robot and static external objects. It does not ingest live TF, sensors, occupancy maps, temporal trajectories, moving obstacles, uncertainty distributions, contact state, multi-robot interaction, continuous collision, penetration depth, or dynamics. `captured_at` and `valid_until` are preserved strings; the scene parser does not consult a clock or independently establish currency. The separate observation schema can select discrete timestamped joint/root/object poses for declared objects, but it does not turn this scene into a live or complete physical-world model. Omitted objects are unknown, not absent from reality.
