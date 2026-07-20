# Changelog

All notable changes to Robot Spatial Understanding are recorded here.

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
