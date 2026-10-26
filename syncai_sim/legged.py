"""The twelve joints under the controller's state machine, in MuJoCo.

`g2_deploy_rs` (the robot's Rust controller) runs one state at a time and
every state ends in the same motor command: a joint target, a target velocity,
PD gains. The motors close that loop themselves at 1 kHz; here it is closed on
every physics step, `tau = kp (q* - q) + kd (dq* - dq)`, either computed here
and applied as a generalized force on the joint (`actuator: qfrc`) or by a
MuJoCo position actuator whose gains are written here (`actuator: position`,
robot.py; with the implicitfast integrator the -kd dq part is integrated
implicitly, which is what keeps a stiff PD from ringing at a coarse step).
The torque is clamped to the motor model's limits (robot.py: the URDF's
effort, or the J60's torque-speed envelope).

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

Two of the knobs g2_deploy_rs's own MuJoCo simulation offers, off by default
like there (mujoco_simulation_G23.py G2_SIM_CMD_DELAY_MS / G2_SIM_TORQUE_BW_HZ):
`cmd_delay_s` holds every motor command back that long (the link from the
controller to the motor driver), `torque_bw_hz` filters the torque first
order (the driver's current loop; qfrc only, a MuJoCo actuator's force
cannot be filtered from outside).
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from .fake_driver import DAMPING, ESTOP, LIE_DOWN, LOCOMOTION, MPC
from .policy import Q_DEFAULT, MultiPolicy, projected_gravity
from .robot import J60_MAX_SPEED, J60_PEAK_TORQUE, joint_kind

# The controller's joint order, matched by name to the URDF's joints.
LEGS = ("FL", "FR", "HL", "HR")
JOINTS = ("HipX", "HipY", "Knee")
# URDF <limit effort>; the motors' own clamp (motor_model ideal).
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


def j60_torque_limits(dq: np.ndarray, peak: np.ndarray, wmax: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """mujoco_simulation_G23.py j60_torque_limits: (lower, upper) per joint —
    driving with the motion, peak * (1 - |dq| / wmax) down to 0; against it,
    the full peak."""
    upper = np.clip(peak * (1.0 - dq / wmax), 0.0, peak)
    lower = -np.clip(peak * (1.0 + dq / wmax), 0.0, peak)
    return lower, upper


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
                 policies: MultiPolicy, transition_s: float = 1.0, log=print,
                 motor_model: str = "ideal", actuator: str = "qfrc", cmd_delay_s: float = 0.0,
                 torque_bw_hz: float = 0.0):
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
        # The motor model's clamp (robot.py).
        if motor_model not in ("ideal", "j60"):
            raise ValueError(f"motor_model {motor_model!r}: ideal | j60")
        self.motor_model = motor_model
        kinds = np.array([joint_kind(f"{leg}_{j}_joint") for leg in LEGS for j in JOINTS])
        self._peak = np.array(J60_PEAK_TORQUE)[kinds]
        self._wmax = np.array(J60_MAX_SPEED)[kinds]
        # The actuator (robot.py): None = qfrc_applied; else the twelve
        # position actuators' ids, in the joint order above.
        if actuator not in ("qfrc", "position"):
            raise ValueError(f"actuator {actuator!r}: qfrc | position")
        self.act: np.ndarray | None = None
        if actuator == "position":
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{leg}_{j}_joint")
                   for leg in LEGS for j in JOINTS]
            if min(ids) < 0:
                raise ValueError("actuator position: the model has no act_<joint> actuators")
            self.act = np.array(ids)
            if torque_bw_hz > 0:
                raise ValueError("torque_bw_hz needs actuator qfrc (a MuJoCo actuator's force "
                                 "cannot be filtered)")
        self._delay = int(round(cmd_delay_s / dt))
        self._queue: collections.deque = collections.deque(maxlen=self._delay + 1)
        self.torque_bw_hz = float(torque_bw_hz)
        self._alpha = dt / (dt + 1.0 / (2.0 * math.pi * torque_bw_hz)) if torque_bw_hz > 0 else 0.0
        self._tau_f: np.ndarray | None = None
        self.t = 0.0
        self.tick = 0
        self.mode: str | None = None
        self.target = Q_DEFAULT.astype(np.float64).copy()
        self.kp = np.zeros(12)
        self.kd = np.zeros(12)
        self._gains_written: tuple[float, ...] | None = None
        self._ease: _Ease | None = None
        self.policy_ms = 0.0

    # -- state -------------------------------------------------------------------
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def dq(self) -> np.ndarray:
        return self.data.qvel[self.vadr].copy()

    def tau(self) -> np.ndarray:
        """The torque on each joint as last applied, in the joint order."""
        if self.act is None:
            return self.data.qfrc_applied[self.vadr].copy()
        return self.data.actuator_force[self.act].copy()

    def set_joints(self, q: np.ndarray) -> None:
        self.data.qpos[self.qadr] = q
        self.data.qvel[self.vadr] = 0.0

    def reset(self, q: np.ndarray = Q_DEFAULT) -> None:
        """After a spawn / teleport: joints at q, at rest, the policy fresh."""
        self.set_joints(q)
        self.target = np.asarray(q, float).copy()
        self.mode = None
        self._ease = None
        self._queue.clear()
        self._tau_f = None
        self.tick = 0
        if self.act is not None:
            self.data.ctrl[self.act] = self.target
        self.policies.reset()

    def roll_pitch(self) -> tuple[float, float]:
        R = self.data.xmat[self.base_body].reshape(3, 3)
        return math.atan2(R[2, 1], R[2, 2]), math.asin(max(-1.0, min(1.0, -R[2, 0])))

    def upset(self) -> bool:
        roll, pitch = self.roll_pitch()
        return abs(roll) > math.radians(30.0) or abs(pitch) > math.radians(45.0)

    def torque_limits(self, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) clamp per joint for the motor model at joint velocity dq."""
        if self.motor_model == "j60":
            return j60_torque_limits(dq, self._peak, self._wmax)
        return -TORQUE_LIMIT, TORQUE_LIMIT

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
        # The command the motors act on: this step's, or the one from
        # cmd_delay_s ago (the queue holds delay + 1 commands; its oldest is
        # the one due).
        self._queue.append((self.target.copy(), self.kp.copy(), self.kd.copy()))
        target, kp, kd = self._queue[0]
        q, dq = self.q(), self.dq()
        lo, hi = self.torque_limits(dq)
        if self.act is None:
            tau = kp * (target - q) - kd * dq
            if self.torque_bw_hz > 0:
                self._tau_f = tau if self._tau_f is None else self._tau_f + self._alpha * (tau - self._tau_f)
                tau = self._tau_f
            self.data.qfrc_applied[self.vadr] = np.clip(tau, lo, hi)
        else:
            self._write_gains(kp, kd)
            self.model.actuator_forcerange[self.act, 0] = lo
            self.model.actuator_forcerange[self.act, 1] = hi
            self.data.ctrl[self.act] = target
        self.tick += 1
        self.t += dt

    def _write_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        """A position actuator's force is gain * ctrl + bias . (1, q, dq):
        kp (q* - q) - kd dq is gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kd."""
        key = (*kp.tolist(), *kd.tolist())
        if key == self._gains_written:
            return
        self.model.actuator_gainprm[self.act, 0] = kp
        self.model.actuator_biasprm[self.act, 1] = -kp
        self.model.actuator_biasprm[self.act, 2] = -kd
        self._gains_written = key

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
