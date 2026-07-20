# Developer quickstart

## 15 minutes: evaluate a bundled episode

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/robot-spatial import --adapter maniskill \
  examples/pickcube/trace.json --out work/pickcube-run
.venv/bin/robot-spatial inspect-run work/pickcube-run
.venv/bin/robot-spatial evaluate work/pickcube-run \
  --task examples/pickcube/task.yaml --out work/pickcube-result
.venv/bin/robot-spatial explain work/pickcube-result/report.json \
  --out work/pickcube-result/report.md
```

The bundled PickCube-shaped trace is synthetic contract data. It is useful for learning the
workflow, not for claiming upstream ManiSkill performance.

## Optional live MuJoCo check

```bash
.venv/bin/python -m pip install -e '.[mujoco]'
.venv/bin/robot-spatial capture --adapter gymnasium-robotics \
  --env-id FetchReach-v3 --seed 2 --max-steps 50 \
  --out work/fetch-run
.venv/bin/robot-spatial evaluate work/fetch-run \
  --task examples/fetch-reach/task.yaml --out work/fetch-result
```

This optional adapter supports fixed-horizon three-dimensional Gymnasium GoalEnv observations. It
discards rewards and `info`, refuses outcome-dependent early termination, and does not treat the
demo controller as evaluation evidence.

## 45 minutes: import your own episode

1. Export one immutable JSON file using schema `robot-spatial-generic-trace.v1`. Start from
   `examples/pickcube/trace.json`.
2. Give the run a stable simulator/version, seed, clock, timestep, task ID, robot/world IDs, and
   model/asset digests.
3. Declare meters, radians, `xyzw`, and `world_from_entity`. Convert before import if the simulator
   uses another convention.
4. Populate only channels the simulator actually exposes. Omit unavailable channels; never fill
   them with fabricated zeros.
5. Keep reward, success, evaluator, and oracle outputs in a separate reference directory. The
   importer rejects those keys anywhere in a candidate trace.
6. Write a declarative `robot-spatial-task-spec.v1`. Bind roles to exact stream entity IDs and use
   generic predicates, time windows, thresholds, goal, and failure expressions.
7. Import, inspect completeness, evaluate, and explain:

```bash
robot-spatial import --adapter generic-json my-trace.json --out work/my-run
robot-spatial inspect-run work/my-run
robot-spatial evaluate work/my-run --task my-task.yaml --out work/my-result
robot-spatial explain work/my-result/report.json --out work/my-result/report.md
```

Do not edit a trace to hide gaps, reorder samples, or repair identity conflicts. Those conditions
are evidence and should produce `unknown` or `conflicting` where appropriate.

## Read the result

Use `report.json` as the source of truth. Each predicate contains:

- exactly one of `supported`, `refuted`, `unknown`, or `conflicting`;
- sample indices/times, measured values, thresholds, and a source-stream digest;
- an evidence digest and explicit missing evidence;
- limitations that bound the claim to the declared simulator episode.

The report deliberately separates controller/Action protocol, trajectory, observed effects,
simulation-bounded physical success, causation, authorization, and safety. A controller's
`succeeded` event does not override a refuted world-state predicate.

## Add an adapter

Implement `SimulatorAdapter.import_source()` and register the class in the
`robot_spatial.adapters` entry-point group. An adapter may map fields, units, frames, clocks, and
entity IDs. It may not implement task success, call the benchmark evaluator during prediction, or
silently invent unavailable observations. See `references/simulator-adapter-contract.md`.
