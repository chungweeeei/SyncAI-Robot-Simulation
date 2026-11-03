"""What the robot's body measures: one record per physics step, the way a
controller's hardware layer reports it.

A legged controller — the robot's own, at the other end of the hardware link
(locomotion/hardware_link.py) — needs each tick: the body's attitude and
angular rate, the body's acceleration (a Kalman filter fuses the specific
force), the twelve joints' q / dq and the torque actually applied (a filter
confirms a planned contact from the foot force it computes with the
Jacobian transpose of `tau`), the base position and world-frame linear
velocity (ground truth, fed straight to an NMPC or used to grade an
estimate), and which feet touch the ground. `Sensors.read` fills one
preallocated float32 record with exactly that, in the order the link sends
it, so the packet is the record's bytes:

    rpy[3]  acc[3]  gyro[3]  q[12]  dq[12]  tau[12]  base_pos[3]  base_linvel[3]  contact[4]

`rpy` is ZYX (yaw about world z, then pitch, then roll); `acc` is the
accelerometer's specific force in the body frame (+9.81 up at rest); `gyro`
is the body-frame angular rate; `base_linvel` is in the world frame (the
free joint's translational velocity is global, its rotational one local);
`contact` is 1.0 where a foot's sphere touches a geom of another root body —
the floor, a stair, a prop — not a force threshold, so a foot resting on the
ground with no load still counts.

Two IMUs, as on the robot: `base_gyro` / `base_acc` sit on `base_link`
(entity/robot.py adds the site; the robot's body IMU is in the trunk) and
are what this record reads; `imu_gyro` / `imu_acc` on `lidar_top_imu` are the
lidar's own, in the model for whoever wants them. The lidar's sits ~0.4 m
up, so its accelerometer sees the lever arm's omega x (omega x r) that a
controller's filter must not.

The RL backend does not call `read` (its runner reads MjData itself at
50 Hz, locomotion/legged.py); the posture check and the trunk's contact
test are separate queries so that the policy path pays for neither the
record nor the packet.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from syncai_sim.actuator.motors import Motors
from syncai_sim.scene.scene import SceneInfo

N_FIELDS = 55
SLICES = {"rpy": slice(0, 3), "acc": slice(3, 6), "gyro": slice(6, 9), "q": slice(9, 21),
          "dq": slice(21, 33), "tau": slice(33, 45), "base_pos": slice(45, 48),
          "base_linvel": slice(48, 51), "contact": slice(51, 55)}


def rpy_of_quat(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """ZYX Euler angles of a wxyz quaternion."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _sensor_adr(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise ValueError(f"the model has no sensor {name!r} (entity/robot.py adds it to the "
                         f"walking body)")
    return int(model.sensor_adr[sid])


class Sensors:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo,
                 motors: Motors):
        self.model, self.data, self.motors = model, data, motors
        self.base_body = info.base_body
        self.f = np.zeros(N_FIELDS, dtype=np.float32)
        for name, s in SLICES.items():
            setattr(self, name, self.f[s])  # views: read() writes the record in place
        self.t = 0.0
        self.quat_wxyz = np.zeros(4)
        self.base_angvel = np.zeros(3)  # body frame, the free joint's rotational qvel
        self._gyro = _sensor_adr(model, "base_gyro")
        self._acc = _sensor_adr(model, "base_acc")
        self._vadr = int(model.jnt_dofadr[model.body_jntadr[info.base_body]])
        # Which geom is which foot (-1 elsewhere), and which geoms are the
        # trunk's and the site's, as lookups over all geoms so that one pass
        # over the contact array answers both questions.
        self._leg_of_geom = np.full(model.ngeom, -1, dtype=np.int64)
        for leg, g in enumerate(info.foot_geoms):
            if g >= 0:
                self._leg_of_geom[g] = leg
        self._trunk = np.zeros(model.ngeom, dtype=bool)
        self._trunk[np.asarray(info.trunk_geoms, dtype=int)] = True
        self._site = np.zeros(model.ngeom, dtype=bool)
        if info.terrain_geoms is not None and info.terrain_geoms.size:
            self._site[np.asarray(info.terrain_geoms, dtype=int)] = True
        if info.stair_geoms:
            self._site[np.asarray(info.stair_geoms, dtype=int)] = True
        self._root = model.body_rootid[model.geom_bodyid]  # root body of every geom

    # -- the record, after mj_step --------------------------------------------------
    def read(self, t: float) -> np.ndarray:
        """Fill the record for time t (seconds, strictly increasing) and return it."""
        d = self.data
        self.t = float(t)
        q = d.xquat[self.base_body]
        self.quat_wxyz[:] = q
        self.rpy[:] = rpy_of_quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        self.acc[:] = d.sensordata[self._acc:self._acc + 3]
        self.gyro[:] = d.sensordata[self._gyro:self._gyro + 3]
        self.q[:] = d.qpos[self.motors.qadr]
        self.dq[:] = d.qvel[self.motors.vadr]
        self.tau[:] = self.motors.tau()
        self.base_pos[:] = d.xpos[self.base_body]
        self.base_linvel[:] = d.qvel[self._vadr:self._vadr + 3]
        self.base_angvel[:] = d.qvel[self._vadr + 3:self._vadr + 6]
        self.contact[:] = 0.0
        n = d.ncon
        if n:
            g = d.contact.geom[:n]
            other_root = self._root[g[:, 0]] != self._root[g[:, 1]]
            for col in (0, 1):
                legs = self._leg_of_geom[g[:, col]]
                hit = legs[other_root & (legs >= 0)]
                if hit.size:
                    self.contact[hit] = 1.0
        return self.f

    # -- queries that need no record ------------------------------------------------
    def roll_pitch(self) -> tuple[float, float]:
        R = self.data.xmat[self.base_body].reshape(3, 3)
        return math.atan2(R[2, 1], R[2, 2]), math.asin(max(-1.0, min(1.0, -R[2, 0])))

    def upset(self) -> bool:
        """The controller's posture check: |roll| > 30 deg or |pitch| > 45 deg."""
        roll, pitch = self.roll_pitch()
        return abs(roll) > math.radians(30.0) or abs(pitch) > math.radians(45.0)

    def trunk_hit(self) -> bool:
        """The trunk touching the site (a terrain box or a fixed stair): what a
        collision is for the walking body."""
        n = self.data.ncon
        if not n:
            return False
        g = self.data.contact.geom[:n]
        return bool(np.any((self._trunk[g[:, 0]] & self._site[g[:, 1]])
                           | (self._trunk[g[:, 1]] & self._site[g[:, 0]])))
