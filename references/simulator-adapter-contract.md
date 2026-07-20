# Simulator adapter contract

## Purpose

A `SimulatorAdapter` converts an immutable simulator export into
`robot-spatial-simulation-run.v1`. It is a field and identity mapper, not an evaluator.

## Required behavior

An adapter must:

- preserve simulator name/version, seed, timestep, task ID, clock, robot/world IDs, and asset
  digests;
- convert length to meters, angles to radians, quaternions to `xyzw`, and poses to
  `world_from_entity`;
- retain original sample times without sorting away out-of-order evidence;
- declare unavailable channels rather than fabricating zero values;
- map stable entity IDs explicitly;
- reject reward, success, official evaluator, or oracle fields;
- emit no task-specific success code;
- record adapter name/version in the run manifest.

## Built-in offline adapters

| Adapter | Accepted export family | Core purpose | Runtime boundary |
| --- | --- | --- | --- |
| `generic-json` | Any producer following the generic trace contract | Developer-owned simulator logs | Dependency-free apart from NumPy/PyYAML |
| `maniskill` | ManiSkill or SAPIEN state exports | Pick, push, insertion, contact, transport | Does not call `env.evaluate()` during prediction |
| `mujoco` | MuJoCo, robosuite, or Meta-World exports | Cross-engine manipulation | Does not import reward/success |
| `gymnasium-robotics` | Live Gymnasium Robotics GoalEnv or matching offline MuJoCo export | Direct three-dimensional achieved/desired-goal capture | Optional `mujoco` extra; live capture discards reward and `info` |
| `gazebo-ros2` | Gazebo/Gz state exports | AGV, UR5e, SCARA, ROS 2 integration | Core does not dispatch commands or start Gazebo |
| `deformable-json` | OmniGibson/BEHAVIOR, Isaac, or SAPIEN keypoint exports | Partial deformable state | Keypoints do not prove complete topology/surface state |

All built-ins normalize immutable JSON exports. `gymnasium-robotics` additionally provides one
optional live GoalEnv capture surface; its heavy runtime is isolated in the `mujoco` package extra.
`maniskill` provides an optional action-only live replay surface isolated in the pinned
`maniskill` package extra. Live Gazebo, ROS 2, and deformable capture remains adapter/plugin work so
the core package stays lightweight and offline-safe.

Both live paths are fixed-horizon. They refuse outcome-dependent early termination because episode
length would otherwise become a hidden success channel.

The ManiSkill live path:

- opens only `traj_N/actions` from its input HDF5 and rejects invalid action shapes;
- maps every active joint and the TCP, fingers, cube, and goal through an explicit entity map;
- captures qpos/qvel, world poses, pairwise finger contact vectors, actions, and lifecycle events;
- enumerates complete scene collision pairs only for a single `physx_cpu` environment;
- marks collision unavailable for `physx_cuda` instead of copying data from another replay;
- creates an independently digest-bound run for every requested sub-environment;
- discards all values returned by `step()` and never calls `evaluate()`.

## Plugin interface

Implement:

```python
from robot_spatial_understanding import SimulatorAdapter

class MyAdapter(SimulatorAdapter):
    name = "my-simulator"
    version = "1.0"

    def import_source(self, source, out):
        ...
```

Register it with the `robot_spatial.adapters` Python entry-point group. The returned artifact must
pass `SimulationRun.load()` with digest verification.

## External benchmark mapping rules

- ManiSkill: capture qpos, link/object poses, contacts, collisions, and action lifecycle separately;
  keep `evaluate()`, reward, termination success, and official labels in the reference scorer only.
  Complete every candidate report before starting fresh same-seed/action official replays. An
  official reference may score only the official placement/static predicates and must list every
  contact/grasp/follow/lift/collision diagnostic under `unscored_predicates`.
- Gazebo/ROS 2: bind `/joint_states`, TF, odometry, contact/collision plugins, controller/action
  lifecycle, ROS clock label, world/model digests, and publisher identity. Missing contact plugins
  make grasp or collision claims unknown.
- MuJoCo/robosuite: bind `qpos`, `qvel`, body/site poses, contacts, forces when used, model XML
  digest, and control timestep. Keep environment `_check_success()` outside prediction.
- Gymnasium Robotics live smoke: capture every candidate run and write every Robot Spatial report
  before starting a same-seed official replay. Never persist `reward` or `info["is_success"]` under
  the candidate run. `benchmarks/gymnasium_fetch_reach_smoke.py` demonstrates this two-phase path.
- BARN: preserve world geometry, start/goal, robot footprint, path, collision radius, and scoring
  definition. A Gazebo Harmonic conversion is not leaderboard-equivalent until those quantities are
  independently checked.
- SCARA: identify revolute/prismatic axes from the model, not names. Project-owned move/insert/pick
  tasks must be labeled regression cases rather than external benchmark evidence.
- Deformables: preserve topology/material digest plus particles, keypoints, or mesh state when
  available. If the export is insufficient for a requested shape or coverage predicate, return
  unknown rather than reducing it to one rigid pose.
