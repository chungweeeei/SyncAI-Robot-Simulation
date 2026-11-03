"""The twelve motors: what every locomotion backend hands its joints to.

The robot's controller ends every state in the same motor command — a joint
target, a target velocity, PD gains and a feed-forward torque — and the motor
drivers close that loop at 1 kHz. Here the loop is closed on every physics
step:

    tau = kp (q_des - q) + kd (dq_des - dq) + tau_ff

either computed here and applied as a generalized force on the joint
(`actuator: qfrc`) or by a MuJoCo position actuator whose gains are written
here (`actuator: position`, entity/robot.py; with the implicitfast integrator
the -kd dq part is integrated implicitly, which is what keeps a stiff PD from
ringing at a coarse step). The torque is clamped to the motor model's limits
(entity/robot.py: the URDF's effort, or the J60's torque-speed envelope).

A position actuator's force is gain * ctrl + bias . (1, q, dq): it can say
kp (q_des - q) - kd dq and nothing else, so a command with a target velocity
or a feed-forward torque — what an MPC's whole-body controller sends, kp 0,
kd 3, tau_ff the torque — needs `qfrc`. `apply` refuses the other two terms
on position actuators instead of dropping them silently. The policies send
dq_des 0 and tau_ff 0, so the RL path is unchanged either way.

Two knobs model the link from the controller to the drivers, off by default:
`cmd_delay_s` holds every command back that long, `torque_bw_hz` filters the
torque first order (the driver's current loop; qfrc only, a MuJoCo actuator's
force cannot be filtered from outside).
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
        """A PD target with no velocity target and no feed-forward: what the
        policies and the stand / lie states send."""
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
                 actuator: str = "qfrc", cmd_delay_s: float = 0.0, torque_bw_hz: float = 0.0):
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
        # The actuator (entity/robot.py): None = qfrc_applied; else the twelve
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
        # The command the motors act on: this step's, or the one from
        # cmd_delay_s ago (the queue holds delay + 1 commands; its oldest is
        # the one due).
        self._delay = int(round(cmd_delay_s / dt))
        self._queue: collections.deque = collections.deque(maxlen=self._delay + 1)
        self.torque_bw_hz = float(torque_bw_hz)
        self._alpha = dt / (dt + 1.0 / (2.0 * math.pi * torque_bw_hz)) if torque_bw_hz > 0 else 0.0
        self._tau_f: np.ndarray | None = None
        self._gains_written: tuple[float, ...] | None = None
        self.last: JointCommand | None = None  # the command last applied (status, benches)

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
        """After a spawn / teleport: joints at q, at rest, nothing queued."""
        self.set_joints(q)
        self._queue.clear()
        self._tau_f = None
        self.last = None
        if self.act is not None:
            self.data.ctrl[self.act] = np.asarray(q, float)

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
        if self.act is None:
            tau = cmd.kp * (cmd.q_des - q) + cmd.kd * (cmd.dq_des - dq) + cmd.tau_ff
            if self.torque_bw_hz > 0:
                self._tau_f = tau if self._tau_f is None else self._tau_f + self._alpha * (tau - self._tau_f)
                tau = self._tau_f
            self.data.qfrc_applied[self.vadr] = np.clip(tau, lo, hi)
            return
        if np.any(cmd.dq_des) or np.any(cmd.tau_ff):
            raise ValueError("actuator position cannot apply a target velocity or a feed-forward "
                             "torque (a position actuator's force is kp (q_des - q) - kd dq); "
                             "use actuator qfrc")
        self._write_gains(cmd.kp, cmd.kd)
        self.model.actuator_forcerange[self.act, 0] = lo
        self.model.actuator_forcerange[self.act, 1] = hi
        self.data.ctrl[self.act] = cmd.q_des

    def _write_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        """kp (q* - q) - kd dq is gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kd."""
        key = (*kp.tolist(), *kd.tolist())
        if key == self._gains_written:
            return
        self.model.actuator_gainprm[self.act, 0] = kp
        self.model.actuator_biasprm[self.act, 1] = -kp
        self.model.actuator_biasprm[self.act, 2] = -kd
        self._gains_written = key
