# Delivery status and remaining gates

This file separates code that exists from benchmark work that has not been run. A registry entry,
adapter class, schema, or unit test is not counted as simulator/benchmark evidence.

| Milestone | Current status | Implemented evidence | Remaining release gate |
| --- | --- | --- | --- |
| M0 product baseline | implemented, pending tagged GitHub workflow run | installable wheel/sdist, legacy CLI forwarding, CI/release workflows, changelog, license manifest, 187 passing tests | observe remote CI and tagged release jobs |
| M1 evidence protocol | protocol implemented; intended ManiSkill vertical slice not yet live-run | versioned run/task/report contracts, 14 predicate types, corruption controls, action bridge, synthetic PickCube chain | run pinned ManiSkill PickCube with raw qpos/link/object/contact/collision state and isolated `evaluate()` reference; add per-predicate stale/missing/conflict matrix |
| M2 developer preview | local preview implemented; external usability gate open | generic importer, SDK, CLI, two quickstarts, adapter contract, core and ManiSkill Dockerfiles | five unfamiliar-developer trials, Meta-World commit suite, publish v0.3 only after 4/5 onboarding gate |
| M3 robot-family validation | adapter surfaces and semantic predicates only | AGV corridor/goal unit case, SCARA insertion unit case, grasp/push negative semantics | Gazebo Harmonic/Jazzy adapters and runs for BARN, UR5e, SCARA; 400 ManiSkill episodes; equivalence audit for BARN conversion |
| M4 cross-engine/deformable | partial schema surface only | deformable keypoint/shape predicates with explicit partial-state limits | robosuite 300, BEHAVIOR 200, LIBERO held-out translation, full `deformable-state.v1` topology/material/particle or mesh snapshots |
| M5 bounded causation/v1.0 | report-level comparison implemented; replay orchestration open | exact matched-run counterfactual checker and no-op negative control | deterministic snapshot/restore adapters, external beta, acceptance metrics, PyPI/Docker/reproducibility release |

## Live evidence currently committed

`benchmarks/records/gymnasium-fetchreach-v3-smoke.json` records two real Gymnasium
Robotics/MuJoCo `FetchReach-v3` episodes. The Robot Spatial path predicts from pose streams only;
separate same-seed replays reveal the official terminal result after all predictions are written.
One seed is supported, one is refuted, both agree, and exact artifact digests repeat.

This is a functional live integration check. Two cases cannot estimate F1, confidence intervals,
cross-task generalization, collision/grasp quality, or hardware reliability.

## Next execution order

1. Run the pinned ManiSkill PickCube vertical slice in the Linux GPU image.
2. Complete the predicate-by-status corruption matrix and make it a release gate.
3. Add the Meta-World commit subset without exposing `_check_success()` during prediction.
4. Run the five-developer onboarding study and turn every failure into a regression test.
5. Begin Gazebo M3 with one AGV episode before UR5e and SCARA so clock/frame/collision mapping is
   stabilized once rather than separately per robot.
