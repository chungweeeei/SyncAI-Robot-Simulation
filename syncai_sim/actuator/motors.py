"""The twelve motors: what the controller's joint command is applied by.

The robot's controller ends every state in the same motor command — a joint
target, a target velocity, PD gains and a feed-forward torque — and the motor
drivers close that loop at 1 kHz. Here the loop is closed on every physics
step:

    tau = kp (q_des - q) + kd (dq_des - dq) + tau_ff

computed here and applied as a generalized force on the joint, clamped to the
motor model's limits (entity/robot.py: the URDF's effort, or the J60's
torque-speed envelope). Not a MuJoCo position actuator: its force is
gain * ctrl + bias . (1, q, dq), which can say kp (q_des - q) - kd dq and
nothing else, and an MPC's whole-body controller sends kp 0, kd 3 and the
torque as tau_ff (the position actuator was a knob for the in-process
policies until 2026-11-13).

Two knobs model the link from the controller to the drivers, off by default:
`cmd_delay_s` holds every command back that long, `torque_bw_hz` filters the
torque first order (the driver's current loop).
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from syncai_sim.entity.robot import (J60_MAX_SPEED, J60_PEAK_TORQUE, JOINTS, LEGS, Q_DEFAULT,
                                     joint_kind)

# URDF <limit effort>; the motors' own clamp (motor_model ideal).
TORQUE_LIMIT = np.array([24.0, 24.0, 36.0] * 4)
# The controller's JointDamping state: kp 0, this kd. What the motors fall
# back to when nothing commands them (the hardware link's watchdog).
DAMPING_KD = 2.0


@dataclass
class JointCommand:
    """One motor command for the twelve joints, in the controller's order
    (entity/robot.py LEGS x JOINTS). Handed to `Motors.apply` and kept there
    until it is due (the command delay), so the sender does not mutate it."""

    q_des: np.ndarray
    dq_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray

    @classmethod
    def pd(cls, q_des: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> JointCommand:
        """A PD target with no velocity target and no feed-forward: what an RL
        controller and the stand-in (hardware_link.py) send."""
        return cls(np.asarray(q_des, float).copy(), np.zeros(12), np.asarray(kp, float).copy(),
                   np.asarray(kd, float).copy(), np.zeros(12))

    @classmethod
    def damping(cls, kd: float) -> JointCommand:
        """kd only: the controller's JointDamping state, and what the motors
        get when no controller is talking."""
        return cls(np.zeros(12), np.zeros(12), np.zeros(12), np.full(12, float(kd)), np.zeros(12))


def j60_torque_limits(dq: np.ndarray, peak: np.ndarray, wmax: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """The J60's envelope, (lower, upper) per joint — driving with the motion,
    peak * (1 - |dq| / wmax) down to 0; against it, the full peak."""
    upper = np.clip(peak * (1.0 - dq / wmax), 0.0, peak)
    lower = -np.clip(peak * (1.0 + dq / wmax), 0.0, peak)
    return lower, upper


class Motors:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, motor_model: str = "ideal",
                 cmd_delay_s: float = 0.0, torque_bw_hz: float = 0.0):
        self.model, self.data = model, data
        jids = []
        for leg in LEGS:
            for j in JOINTS:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{j}_joint")
                if jid < 0:
                    raise ValueError(f"robot has no joint {leg}_{j}_joint")
                jids.append(jid)
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.vadr = np.array([model.jnt_dofadr[j] for j in jids])
        dt = model.opt.timestep
        # The motor model's clamp (entity/robot.py).
        if motor_model not in ("ideal", "j60"):
            raise ValueError(f"motor_model {motor_model!r}: ideal | j60")
        self.motor_model = motor_model
        kinds = np.array([joint_kind(f"{leg}_{j}_joint") for leg in LEGS for j in JOINTS])
        self._peak = np.array(J60_PEAK_TORQUE)[kinds]
        self._wmax = np.array(J60_MAX_SPEED)[kinds]
        # The command the motors act on: this step's, or the one from
        # cmd_delay_s ago (the queue holds delay + 1 commands; its oldest is
        # the one due).
        self._delay = int(round(cmd_delay_s / dt))
        self._queue: collections.deque = collections.deque(maxlen=self._delay + 1)
        self.torque_bw_hz = float(torque_bw_hz)
        self._alpha = dt / (dt + 1.0 / (2.0 * math.pi * torque_bw_hz)) if torque_bw_hz > 0 else 0.0
        self._tau_f: np.ndarray | None = None
        self.last: JointCommand | None = None  # the command last applied (status, benches)

    # -- state -------------------------------------------------------------------
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def dq(self) -> np.ndarray:
        return self.data.qvel[self.vadr].copy()

    def tau(self) -> np.ndarray:
        """The torque on each joint as last applied, in the joint order."""
        return self.data.qfrc_applied[self.vadr].copy()

    def set_joints(self, q: np.ndarray) -> None:
        self.data.qpos[self.qadr] = q
        self.data.qvel[self.vadr] = 0.0

    def reset(self, q: np.ndarray = Q_DEFAULT) -> None:
        """After a spawn / teleport: joints at q, at rest, nothing queued."""
        self.set_joints(q)
        self._queue.clear()
        self._tau_f = None
        self.last = None

    def torque_limits(self, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) clamp per joint for the motor model at joint velocity dq."""
        if self.motor_model == "j60":
            return j60_torque_limits(dq, self._peak, self._wmax)
        return -TORQUE_LIMIT, TORQUE_LIMIT

    # -- one physics step -----------------------------------------------------------
    def apply(self, cmd: JointCommand) -> None:
        """Close the loop for this step on `cmd` (or on the one due from the delay queue)."""
        self._queue.append(cmd)
        cmd = self._queue[0]
        self.last = cmd
        q, dq = self.q(), self.dq()
        lo, hi = self.torque_limits(dq)
        tau = cmd.kp * (cmd.q_des - q) + cmd.kd * (cmd.dq_des - dq) + cmd.tau_ff
        if self.torque_bw_hz > 0:
            self._tau_f = tau if self._tau_f is None else self._tau_f + self._alpha * (tau - self._tau_f)
            tau = self._tau_f
        self.data.qfrc_applied[self.vadr] = np.clip(tau, lo, hi)
