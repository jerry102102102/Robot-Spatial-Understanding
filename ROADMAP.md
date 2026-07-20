# Delivery status and remaining gates

This file separates code that exists from benchmark work that has not been run. A registry entry,
adapter class, schema, or unit test is not counted as simulator/benchmark evidence.

| Milestone | Current status | Implemented evidence | Remaining release gate |
| --- | --- | --- | --- |
| M0 product baseline | implemented, pending tagged GitHub workflow run | installable wheel/sdist, legacy CLI forwarding, CI/release workflows, changelog, license manifest, 194 passing tests | observe remote CI and tagged release jobs |
| M1 evidence protocol | live PickCube gate passed locally; pending remote CI | versioned run/task/report contracts, 17 predicate types, live raw qpos/qvel/pose/contact/collision capture, oracle-isolated 100-case record with 51 supported/49 refuted references, semantic negatives, corruption controls, CUDA parity smoke | observe the pushed GitHub workflow; no simulator or hardware scope expansion is implied |
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

The ManiSkill `PickCube-v1` records add a larger live chain on ManiSkill 3.0.1/SAPIEN 3.0.3.
The primary 100 cases (seeds 0–49 plus matched no-op) agree 100/100 with the independently replayed
official evaluator at a fixed 100-step horizon: 51 supported and 49 refuted, with an accuracy Wilson 95% interval of
`[0.9630, 1.0]`. The semantic-negative matrix passes 8/8 and the corruption/abstention matrix
passes 9/9. A 16-environment `physx_cuda` smoke completed; collision is `unknown` there because
the GPU backend did not expose complete scene-contact enumeration.

All 50 official-planner cases are supported. Forty-nine no-op controls are refuted; seed 8 is
correctly supported because the cube starts within the official goal tolerance. The local M1 gate
therefore meets the required minimum of 25 supported and 25 refuted references without reading an
outcome during prediction.

## Next execution order

1. Observe the remote CI result for the live PickCube delivery.
2. Add the Meta-World commit subset without exposing `_check_success()` during prediction.
3. Run the five-developer onboarding study and turn every failure into a regression test.
4. Begin Gazebo M3 with one AGV episode before UR5e and SCARA so clock/frame/collision mapping is
   stabilized once rather than separately per robot.
