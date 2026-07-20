# Changelog

All notable changes to Robot Spatial Understanding are recorded here.

## Unreleased

### Added

- Live fixed-horizon ManiSkill 3.0.1 capture from action-only trajectories, including raw Panda
  joint state, TCP/finger/object/goal poses, pairwise contact forces, CPU collision pairs, lifecycle
  events, entity bindings, simulator versions, and model/task/config digests.
- Generic terminal frame-position, joint-velocity, and joint-range predicates; relative lift,
  multi-contact grasp, and allowed/ignored collision-pair semantics.
- Oracle-isolated PickCube benchmark, semantic-negative generator, expanded corruption matrix,
  two small normalized live fixtures, and CPU/GPU capture checks.

### Validation status

- At a fixed 100-step horizon, 100/100 primary PickCube cases agree with the isolated official
  evaluator: 51 supported and 49 refuted references, including 50/50 official-planner successes.
- Semantic negatives pass 8/8, corruption cases pass 9/9, and the 16-environment CUDA capture smoke
  passes 16/16 with collision correctly unavailable. No `v0.3` release is claimed because the
  Meta-World and unfamiliar-developer onboarding gates remain open.

## 0.2.0 - 2026-07-20

### Added

- Installable Python package and `robot-spatial` console command.
- `simulation-run.v1` and `task-spec.v1` contracts.
- Generic JSON simulation importer and deterministic NPZ stream storage.
- Evidence-grounded predicate evaluation, completeness reporting, Markdown explanation, oracle-isolated benchmark scoring, and trace corruption controls.
- Simulator adapter interfaces for generic, ManiSkill/SAPIEN, MuJoCo, Gazebo/ROS 2, and deformable-state exports without importing heavy simulator runtimes into the core package.
- Optional live Gymnasium Robotics/MuJoCo GoalEnv capture plus a two-phase, same-seed official-oracle smoke replay.
- Relative observed-frame targets for task specs, counterfactual replay comparison, action-evidence bridging, deformable keypoint predicates, and benchmark metrics.
- A PickCube reference episode with positive, negative, missing-evidence, and oracle-isolation coverage.

### Compatibility

- Existing `python3 scripts/robot_spatial.py ...` commands remain supported.
- The installed `robot-spatial` command forwards legacy model commands to the existing CLI.

## 0.1.0 - 2026-07-18

- Initial evidence-grounded Codex Skill release.
