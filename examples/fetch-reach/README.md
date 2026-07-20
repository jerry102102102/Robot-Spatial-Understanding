# Live MuJoCo FetchReach smoke case

This task evaluates only the final raw `achieved_goal` pose against the raw
`desired_goal` pose. The Gymnasium Robotics adapter never imports `reward` or
`info["is_success"]` into `simulation-run.v1`.

```bash
python -m pip install '.[mujoco]'
robot-spatial capture --adapter gymnasium-robotics \
  --env-id FetchReach-v3 --seed 2 --max-steps 50 --out /tmp/fetch-run
robot-spatial evaluate /tmp/fetch-run \
  --task examples/fetch-reach/task.yaml --out /tmp/fetch-result
robot-spatial explain /tmp/fetch-result/report.json \
  --out /tmp/fetch-report.md
```

For an oracle-isolated positive and negative replay, run:

```bash
python benchmarks/gymnasium_fetch_reach_smoke.py --out /tmp/fetch-smoke
```

The script completes both Robot Spatial predictions before starting separate,
same-seed official-oracle replays.
