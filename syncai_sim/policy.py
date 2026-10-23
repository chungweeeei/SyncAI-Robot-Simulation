"""The G23's locomotion policies, as the real controller runs them.

The robot's low-level controller is `g2_deploy_rs` (Rust): in its RlControl
state it reads the IMU and the twelve joints, builds an observation, runs an
ONNX policy at 50 Hz and hands the motors a joint target with fixed PD gains.
This module is a line-by-line port of the two runners the navigation stack
switches between with `set_policy_mode`:

    policy 0  PPO      src/policy_runner.rs           45-d obs, ppo/policy7.onnx
    policy 1  HimLoco  src/himloco_policy_runner.rs  6 x 45-d history, isaacsim/policy_0812.onnx

(the files main.rs loads by default). Any other policy number falls back to
PPO, as `MultiPolicyRunnerOnnx::select_policy` does — policy 2 is MPC there,
another state, and 3 is a third IsaacSim runner the stack never asks for.

Every constant below is copied from those files; change them together. The
joint order is the controller's (LF, RF, LR, RR) x (hip, thigh, calf), which
is the URDF's (FL, FR, HL, HR) x (HipX, HipY, Knee) with the same signs: on
flat ground PPO walks the URDF as is and falls within a second with the hip
signs flipped (measured 2026-10-23, scratch harness, both policies at 1, 2 and
5 ms physics steps: vx 0.5 -> 0.40 / 0.45 m/s, wz 1.0 -> 1.00 / 0.97 rad/s).
"""

from __future__ import annotations

import numpy as np

# Shared by both runners (dof_pos_default in each file).
Q_DEFAULT = np.array([0.0, -0.65, 1.3] * 4, dtype=np.float32)
OBS_DIM = 45


def projected_gravity(R_world_base: np.ndarray) -> np.ndarray:
    """Gravity's direction in the base frame (base_rot_mat^T * (0, 0, -1))."""
    return (R_world_base.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)


class OnnxPolicy:
    """onnxruntime with the controller's settings (one intra-op thread).

    Imported lazily so that the kinematic sim and the tests need no onnxruntime."""

    def __init__(self, path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name  # "obs" in every bundled policy
        self.dim = int(self.session.get_inputs()[0].shape[1])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input: obs[None].astype(np.float32)})[0][0]


class PpoRunner:
    """src/policy_runner.rs, Lite3PpoPolicyRunnerOnnx."""

    name = "PPO"
    obs_dim = OBS_DIM
    kp, kd = 30.0, 1.0
    omega_scale, dof_vel_scale = 0.25, 0.05
    action_scale = np.array([0.125, 0.25, 0.25] * 4, dtype=np.float32)
    q_min = np.array([-0.53, -3.50, 0.349] * 4, dtype=np.float32)
    q_max = np.array([0.53, 0.32, 2.80] * 4, dtype=np.float32)

    def __init__(self, net):
        self.net = net
        self.last_action = np.zeros(12, np.float32)

    def on_enter(self) -> None:
        self.last_action[:] = 0.0

    def observation(self, omega, gravity, cmd, q, dq) -> np.ndarray:
        return np.concatenate([
            np.asarray(omega, np.float32) * self.omega_scale,
            gravity,
            np.clip(np.asarray(cmd, np.float32), -1.0, 1.0),
            np.asarray(q, np.float32) - Q_DEFAULT,
            np.asarray(dq, np.float32) * self.dof_vel_scale,
            self.last_action,
        ]).astype(np.float32)

    def act(self, omega, gravity, cmd, q, dq) -> np.ndarray:
        self.last_action = np.asarray(self.net(self.observation(omega, gravity, cmd, q, dq)),
                                      np.float32)[:12].copy()
        return np.clip(self.last_action * self.action_scale + Q_DEFAULT, self.q_min, self.q_max)


class HimlocoRunner:
    """src/himloco_policy_runner.rs, Lite3HimlocoPolicyRunnerOnnx.

    Differences from PPO that matter: the command comes first and is not
    clipped, the joints are reordered by type (all hips, all thighs, all calves),
    the input is the last six observations newest first (the first call fills
    all six with the same one), the hip actions are halved and nothing is
    clamped."""

    name = "HimLoco"
    history = 6
    obs_dim = OBS_DIM * 6
    kp, kd = 25.0, 0.6
    hip_scale = 0.5
    robot2policy = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])
    policy2robot = np.array([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])

    def __init__(self, net):
        self.net = net
        self.last_action = np.zeros(12, np.float32)  # policy order
        self.obs = np.zeros(self.obs_dim, np.float32)
        self.run_cnt = 0

    def on_enter(self) -> None:
        self.last_action[:] = 0.0
        self.obs[:] = 0.0
        self.run_cnt = 0

    def observation(self, omega, gravity, cmd, q, dq) -> np.ndarray:
        r2p = self.robot2policy
        one = np.concatenate([
            np.asarray(cmd, np.float32),
            np.asarray(omega, np.float32) * 0.25,
            gravity,
            (np.asarray(q, np.float32) - Q_DEFAULT)[r2p],
            (np.asarray(dq, np.float32) * 0.05)[r2p],
            self.last_action,
        ]).astype(np.float32)
        if self.run_cnt == 0:
            self.obs[:] = np.tile(one, self.history)
        else:
            self.obs[OBS_DIM:] = self.obs[:-OBS_DIM].copy()
            self.obs[:OBS_DIM] = one
        return self.obs

    def act(self, omega, gravity, cmd, q, dq) -> np.ndarray:
        self.last_action = np.asarray(self.net(self.observation(omega, gravity, cmd, q, dq)),
                                      np.float32)[:12].copy()
        self.run_cnt += 1
        a = self.last_action[self.policy2robot] * 0.25
        a[[0, 3, 6, 9]] *= self.hip_scale
        return a + Q_DEFAULT


class MultiPolicy:
    """MultiPolicyRunnerOnnx: the runner for a policy number, reset on a switch."""

    def __init__(self, runners: dict[int, PpoRunner | HimlocoRunner], fallback: int = 0, log=print):
        self.runners = runners
        self.fallback = fallback
        self.active: int | None = None
        self.log = log

    def select(self, requested: int):
        effective = requested if requested in self.runners else self.fallback
        if effective != self.active:
            self.active = effective
            self.runners[effective].on_enter()
            self.log(f"[policy] active runner: {self.runners[effective].name} "
                     f"(policy_mode {requested})")
        return self.runners[effective]

    def reset(self) -> None:
        """RlControlState::on_enter — the next select() starts the runner fresh."""
        self.active = None


def load_policies(ppo_path: str, himloco_path: str, log=print) -> MultiPolicy:
    runners: dict[int, PpoRunner | HimlocoRunner] = {}
    for num, cls, path in ((0, PpoRunner, ppo_path), (1, HimlocoRunner, himloco_path)):
        net = OnnxPolicy(path)
        if net.dim != cls.obs_dim:
            raise ValueError(f"{path}: input is {net.dim}-d, {cls.name} expects {cls.obs_dim}")
        runners[num] = cls(net)
        log(f"[policy] {num} = {cls.name}: {path}")
    return MultiPolicy(runners, log=log)
