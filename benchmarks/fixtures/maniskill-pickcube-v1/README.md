# ManiSkill PickCube live fixtures

These are small normalized numeric fixtures captured from ManiSkill `PickCube-v1` seed 2 with
ManiSkill 3.0.1, SAPIEN 3.0.3, `physx_cpu`, `pd_joint_pos`, and a fixed 100-step horizon. They do not
contain upstream meshes, textures, URDFs, videos, observations, rewards, `info`, success fields, or
official evaluator output.

| Fixture | Robot Spatial verdict | Run manifest | Report |
| --- | --- | --- | --- |
| `success-seed-002` | `supported` | `c298aa53634785ae0230dd15e6b0fd4c650cc511337416e5f7b455c411e9fc09` | `73dda74d1e76f4510ff14013ef14b0b18aa14acc95ec99e0ecefcec7315a521c` |
| `failure-seed-002-no-op` | `refuted` | `608bcf9576be2719941c26c8f5eed826a8e47707872aff7aa6ef4a3dbaf3a258` | `02612f54ae111b442e8e87d598ab1820f51656670cf5172a837d63bd89f5063c` |

The official reference is intentionally absent from each candidate directory. The external
100-case benchmark record binds these run/report digests to independently replayed official
references.
