# Third-party models and benchmarks

This repository does not redistribute the following datasets or simulator assets. Adapters and
manifests refer to user-installed copies so that each upstream license remains visible and can be
accepted independently.

| Source | Intended use | Code license | Asset/data boundary |
| --- | --- | --- | --- |
| ManiSkill | Core manipulation benchmark and official task oracle | Apache-2.0 | Some assets are non-commercial; download from upstream and record the exact asset manifest. |
| Gymnasium Robotics / MuJoCo | Live FetchReach smoke and optional cross-engine capture | Install from upstream under their published package terms | No upstream model XML or assets are redistributed; the run stores only their digest. |
| Meta-World | Lightweight MuJoCo smoke suite | MIT | Install from upstream; no task assets are vendored here. |
| BARN | AGV navigation environments | Upstream terms must be reviewed before each release | Do not redistribute BARN worlds until an explicit redistribution grant is recorded. |
| Universal Robots Gazebo simulation | UR5e trajectory scenarios | BSD-3-Clause | Robot packages remain external ROS dependencies. |
| ICube SCARA tutorial | Project-owned SCARA regression scenarios | Apache-2.0 | Tutorial dynamics are not treated as an industrial-dynamics oracle. |
| robosuite | Independent MuJoCo manipulation oracle | MIT | Install environments and assets from upstream. |
| BEHAVIOR / OmniGibson | Rigid and deformable semantic release suite | Upstream terms | Assets require a separate download and may not be redistributed. |
| LIBERO | Language-to-goal and long-horizon evaluation | MIT code; dataset terms apply | Install dataset independently and retain its manifest. |

Every benchmark result must record upstream version, simulator version, task ID, robot, scene,
seed, adapter version, and asset/model digests. A missing or ambiguous license blocks bundling but
does not block a user from pointing an adapter at their own legally obtained installation.
