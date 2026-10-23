"""The twelve joints under the controller's state machine, in MuJoCo.

`g2_deploy_rs` (the robot's Rust controller) runs one state at a time and
every state ends in the same motor command: a joint target, a target velocity,
PD gains. The motors close that loop themselves at 1 kHz; here it is closed on
every physics step, `tau = kp (q* - q) + kd (dq* - dq)`, clipped to the URDF's
joint effort, and applied as generalized force on the joint.

Which state, from the fake driver's `motion_state` (fake_driver.py):

    LOCOMOTION, MPC   the policy for `policy` (policy.py) at 50 Hz, held between
                      inferences; MPC has no simulation here and walks the policy
    STAND, UNKNOWN    PD to the policy's default pose (src/states/standup.rs gains)
    LIE_DOWN          PD to the pose at `down_height` (src/states/liedown.rs)
    DAMPING, ESTOP    kp 0, kd only (src/states/joint_damping.rs)

A change of PD target is eased over `transition_s` from where the joints are,
as standup.rs / liedown.rs do with their splines. The posture check is
RlControlState::posture_unsafe_check: |roll| > 30 deg or |pitch| > 45 deg is a
fall, and the real controller then locks into JointDamping.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from .fake_driver import DAMPING, ESTOP, LIE_DOWN, LOCOMOTION, MPC
from .policy import Q_DEFAULT, MultiPolicy, projected_gravity

# The controller's joint order, matched by name to the URDF's joints.
LEGS = ("FL", "FR", "HL", "HR")
JOINTS = ("HipX", "HipY", "Knee")
# URDF <limit effort>; the motors' own clamp.
TORQUE_LIMIT = np.array([24.0, 24.0, 36.0] * 4)
# src/parameters.rs ControlParameters::lite3
THIGH_LEN, SHANK_LEN = 0.20, 0.21
DOWN_HEIGHT = 0.12
STAND_KP, STAND_KD = 70.0, 2.0  # swing_leg_kp / swing_leg_kd
DAMPING_KD = 2.0
POLICY_HZ = 50.0  # config/g2_control.env G2_POLICY_RATE_HZ


def pose_at_height(h: float) -> np.ndarray:
    """math.rs get_hip_y_pos_by_height / get_knee_pos_by_height, all four legs."""
    l1, l2 = THIGH_LEN, SHANK_LEN
    hip = -math.acos(max(-1.0, min(1.0, (l1 * l1 + h * h - l2 * l2) / (2 * h * l1))))
    knee = math.pi - math.acos(max(-1.0, min(1.0, (l1 * l1 + l2 * l2 - h * h) / (2 * l1 * l2))))
    return np.array([0.0, hip, knee] * 4)


@dataclass
class _Ease:
    start: np.ndarray
    goal: np.ndarray
    t0: float
    duration: float

    def at(self, t: float) -> np.ndarray:
        s = min(1.0, max(0.0, (t - self.t0) / self.duration))
        s = s * s * (3.0 - 2.0 * s)
        return self.start + (self.goal - self.start) * s


class LeggedController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, base_body: int,
                 policies: MultiPolicy, transition_s: float = 1.0, log=print):
        self.model, self.data = model, data
        self.base_body = base_body
        self.policies = policies
        self.transition_s = transition_s
        self.log = log
        jids = []
        for leg in LEGS:
            for j in JOINTS:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{j}_joint")
                if jid < 0:
                    raise ValueError(f"robot has no joint {leg}_{j}_joint")
                jids.append(jid)
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.vadr = np.array([model.jnt_dofadr[j] for j in jids])
        # The free joint's angular velocity is in the base frame: the IMU's gyro.
        # Not mj_objectVelocity(..., flg_local=1) after mj_step: fed that, PPO
        # veered 0.6 rad in 2 s of straight walking on dp where qvel walked
        # straight (same scene, same loop, 2026-10-23; cause not pinned down).
        self.omega_adr = int(model.jnt_dofadr[model.body_jntadr[base_body]]) + 3
        dt = model.opt.timestep
        self.decimation = max(1, int(round(1.0 / (POLICY_HZ * dt))))
        if abs(self.decimation * dt * POLICY_HZ - 1.0) > 1e-6:
            raise ValueError(f"physics_hz {1 / dt:g} is not a multiple of the policy's {POLICY_HZ:g} Hz")
        self.t = 0.0
        self.tick = 0
        self.mode: str | None = None
        self.target = Q_DEFAULT.astype(np.float64).copy()
        self.kp = np.zeros(12)
        self.kd = np.zeros(12)
        self._ease: _Ease | None = None
        self.policy_ms = 0.0

    # -- state -------------------------------------------------------------------
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def dq(self) -> np.ndarray:
        return self.data.qvel[self.vadr].copy()

    def set_joints(self, q: np.ndarray) -> None:
        self.data.qpos[self.qadr] = q
        self.data.qvel[self.vadr] = 0.0

    def reset(self, q: np.ndarray = Q_DEFAULT) -> None:
        """After a spawn / teleport: joints at q, at rest, the policy fresh."""
        self.set_joints(q)
        self.target = np.asarray(q, float).copy()
        self.mode = None
        self._ease = None
        self.tick = 0
        self.policies.reset()

    def roll_pitch(self) -> tuple[float, float]:
        R = self.data.xmat[self.base_body].reshape(3, 3)
        return math.atan2(R[2, 1], R[2, 2]), math.asin(max(-1.0, min(1.0, -R[2, 0])))

    def upset(self) -> bool:
        roll, pitch = self.roll_pitch()
        return abs(roll) > math.radians(30.0) or abs(pitch) > math.radians(45.0)

    # -- one physics step -----------------------------------------------------------
    def step(self, cmd: np.ndarray, policy: int, motion_state: int) -> None:
        dt = self.model.opt.timestep
        mode = self._mode_of(motion_state)
        if mode != self.mode:
            self._enter(mode)
        if mode == "policy":
            if self.tick % self.decimation == 0:
                self._infer(cmd, policy)
        elif self._ease is not None:
            self.target = self._ease.at(self.t)
        q, dq = self.q(), self.dq()
        tau = self.kp * (self.target - q) - self.kd * dq
        self.data.qfrc_applied[self.vadr] = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
        self.tick += 1
        self.t += dt

    @staticmethod
    def _mode_of(motion_state: int) -> str:
        if motion_state in (LOCOMOTION, MPC):
            return "policy"
        if motion_state == LIE_DOWN:
            return "lie"
        if motion_state in (DAMPING, ESTOP):
            return "damping"
        return "stand"

    def _enter(self, mode: str) -> None:
        self.mode = mode
        self.tick = 0
        self._ease = None
        if mode == "policy":
            self.policies.reset()
        elif mode == "damping":
            self.kp[:] = 0.0
            self.kd[:] = DAMPING_KD
        else:
            goal = Q_DEFAULT.astype(float) if mode == "stand" else pose_at_height(DOWN_HEIGHT)
            self.kp[:] = STAND_KP
            self.kd[:] = STAND_KD
            self._ease = _Ease(self.q(), goal, self.t, self.transition_s)

    def _infer(self, cmd: np.ndarray, policy: int) -> None:
        t0 = time.perf_counter()
        runner = self.policies.select(policy)
        R = self.data.xmat[self.base_body].reshape(3, 3)
        omega = self.data.qvel[self.omega_adr:self.omega_adr + 3]
        self.target = runner.act(omega, projected_gravity(R), cmd, self.q(),
                                 self.dq()).astype(np.float64)
        self.kp[:] = runner.kp
        self.kd[:] = runner.kd
        self.policy_ms = (time.perf_counter() - t0) * 1000.0
