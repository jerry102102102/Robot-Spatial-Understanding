# PickCube evidence example

`trace.json` is a small, synthetic ManiSkill/SAPIEN-shaped state export. It intentionally contains
no reward, `success`, or official evaluator output. It demonstrates the public import contract;
it is not claimed to be an upstream ManiSkill episode.

```bash
robot-spatial import --adapter maniskill trace.json --out work/pickcube-run
robot-spatial evaluate work/pickcube-run --task task.yaml --out work/pickcube-result
robot-spatial explain work/pickcube-result/report.json --out work/pickcube-result/report.md
```

The `grasped` result is supported only when contact, closed-gripper joint state, relative object
following, and lift evidence all pass. Removing any required channel must produce `unknown` rather
than a guessed success.
