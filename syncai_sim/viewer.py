"""The MuJoCo passive viewer, following the robot, with keyboard teleop."""

from __future__ import annotations

import math

import mujoco
import mujoco.viewer
import numpy as np

# GLFW key codes
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SPACE = 265, 264, 263, 262, 32
KEY_1, KEY_5, KEY_P, KEY_T, KEY_H = 49, 53, 80, 84, 72

HELP = """teleop keys (viewer window focused):
  up/down    vx +-0.1 m/s        left/right  wz +-0.1 rad/s
  space      stop                1 / 5       set_motion_key 1 (RL) / 5 (MPC)
  p          toggle policy 0/1   t           teleport back to the spawn
  h          this help"""


class Teleop:
    def __init__(self, spawn, log=print):
        self.vx = 0.0
        self.wz = 0.0
        self.spawn = spawn
        self.log = log
        self._pending = []  # closures applied under the sim lock

    def key(self, keycode: int) -> None:
        if keycode == KEY_UP:
            self.vx = min(self.vx + 0.1, 0.8)
        elif keycode == KEY_DOWN:
            self.vx = max(self.vx - 0.1, -0.8)
        elif keycode == KEY_LEFT:
            self.wz = min(self.wz + 0.1, 0.65)
        elif keycode == KEY_RIGHT:
            self.wz = max(self.wz - 0.1, -0.65)
        elif keycode == KEY_SPACE:
            self.vx = self.wz = 0.0
        elif keycode == KEY_1:
            self._pending.append(lambda sim: self.log(f"[teleop] key 1 -> {sim.driver.set_motion_key('1')}"))
        elif keycode == KEY_5:
            self._pending.append(lambda sim: self.log(f"[teleop] key 5 -> {sim.driver.set_motion_key('5')}"))
        elif keycode == KEY_P:
            def toggle(sim):
                new = 0 if sim.driver.policy == 1 else 1
                self.log(f"[teleop] policy -> {new} {sim.driver.set_policy_mode(new)}")
            self._pending.append(toggle)
        elif keycode == KEY_T:
            self._pending.append(lambda sim: sim.teleport(*self.spawn[:2], self.spawn[3], self.spawn[2]))
        elif keycode == KEY_H:
            self.log(HELP)
        else:
            return
        self.log(f"[teleop] vx {self.vx:+.1f}  wz {self.wz:+.1f}")

    def apply(self, sim) -> None:
        for fn in self._pending:
            fn(sim)
        self._pending.clear()
        sim.set_cmd(self.vx, 0.0, self.wz)


class Viewer:
    def __init__(self, sim, teleop: Teleop | None = None):
        self.sim = sim
        self.teleop = teleop
        self.handle = mujoco.viewer.launch_passive(
            sim.model, sim.data, key_callback=teleop.key if teleop else None,
            show_left_ui=False, show_right_ui=False)
        cam = self.handle.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = sim.info.base_body
        cam.distance = 9.0
        cam.elevation = -32.0
        cam.azimuth = 150.0
        opt = self.handle.opt
        opt.geomgroup[3] = 0  # collision primitives and plates
        opt.geomgroup[4] = 0  # the mocap handle
        self._scan_n = 0

    def is_running(self) -> bool:
        return self.handle.is_running()

    def sync(self, sim) -> None:
        self.handle.sync()

    def show_scan(self, sim, max_points: int = 400) -> None:
        """Draw a thinned copy of the last scan as small spheres in the body frame."""
        xyzi = sim.scan()  # a fresh, independent draw is fine for a picture
        T = sim.T_map_imu()
        pts = xyzi[:: max(1, xyzi.shape[0] // max_points), :3].astype(np.float64)
        pts_w = pts @ T[:3, :3].T + T[:3, 3]
        # The render thread reads user_scn; editing it unlocked ends in a
        # GLXBadDrawable from the X server a few seconds in.
        with self.handle.lock():
            scn = self.handle.user_scn
            scn.ngeom = 0
            for i in range(min(pts_w.shape[0], scn.maxgeom)):
                g = scn.geoms[i]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.03, 0, 0]),
                                    pts_w[i], np.eye(3).reshape(-1),
                                    np.array([1.0, 0.3, 0.1, 0.8]))
                scn.ngeom = i + 1

    def close(self) -> None:
        try:
            self.handle.close()
        except Exception:
            pass


def yaw_of(sim) -> float:
    T = sim.T_map_base()
    return math.atan2(T[1, 0], T[0, 0])
