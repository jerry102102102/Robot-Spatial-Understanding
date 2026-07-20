# Live ManiSkill PickCube evidence

This directory is distinct from the synthetic contract example in `examples/pickcube`.
`pickcube-entities.yaml` binds the pinned ManiSkill `PickCube-v1` Panda objects and all active
joints to simulator-neutral evidence IDs. `task.yaml` reproduces the official placement and
robot-static success conditions; contact, grasp, following, lift, and collision predicates are
unscored diagnostics.

The live adapter reads only `traj_N/actions` from the supplied HDF5 file. It never reads recorded
reward, success, observation, info, or environment-state datasets. CPU capture enumerates scene
contacts as the collision channel. GPU capture keeps pairwise finger contact evidence but marks
complete collision enumeration unavailable.

```bash
robot-spatial capture --adapter maniskill --env-id PickCube-v1 --seed 0 \
  --trajectory actions.h5 --entity-map pickcube-entities.yaml \
  --sim-backend physx_cpu --num-envs 1 --fixed-horizon 100 \
  --out work/pickcube-live/run
robot-spatial evaluate work/pickcube-live/run --task task.yaml \
  --out work/pickcube-live/result
```

The benchmark tools keep candidate prediction and official replay in separate phases:

```bash
PYTHONPATH=scripts python benchmarks/maniskill_pickcube_evidence.py \
  --out work/pickcube-100 --seeds $(seq 0 49) --include-no-op \
  --fixed-horizon 100 --minimum-supported 25 --minimum-refuted 25
PYTHONPATH=scripts python benchmarks/maniskill_pickcube_cuda_smoke.py \
  --out work/pickcube-cuda --trajectory actions.h5
```

On hosts where complete Vulkan rendering is unavailable, a supported CPU Vulkan ICD may be selected
with `--render-backend cpu`; this does not change the requested PhysX CPU/CUDA simulation backend.
