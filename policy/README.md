# policy/ — the G23's ONNX policies (not committed)

The walking sim (`locomotion: policy`, `syncai_sim/policy.py`) loads two files
from here, the ones `g2_deploy_rs`'s `main.rs` loads by default:

| file | set_policy_mode | sha256 |
|---|---|---|
| `ppo/policy7.onnx` | 0 (PPO) | `30e52f4e5eb824fead128894ca86b67446cf250deb8ce8e81aaff93c938c5b4b` |
| `isaacsim/policy_0812.onnx` | 1 (HimLoco) | `26e760f14396b99d779ab3e5b808b7f2b5a32e77f50abb60448f7af907c284e8` |

The `.onnx` files are gitignored on purpose: they are the robot's trained
weights, and in `g2_deploy_rs` (internal GitLab) they are not under version
control either, so this repository (GitHub) does not publish them. Copy them
from a g2_deploy_rs checkout (taken from one at `bca8b73`, 2026-10-23):

```bash
cp ../../g2_deploy_rs/policy/ppo/policy7.onnx          policy/ppo/
cp ../../g2_deploy_rs/policy/isaacsim/policy_0812.onnx policy/isaacsim/
shasum -a 256 policy/*/*.onnx                           # compare with the table
```

The repository is mounted at `/home/syncrobotic/sim`, so the container sees
them as `/home/syncrobotic/sim/policy/...` (`ppo_policy` / `himloco_policy` in
`config/params/sim_params.yaml`). A different policy is a different path
there; one with another observation layout needs `policy.py` changed too.
