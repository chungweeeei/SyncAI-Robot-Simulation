"""The MuJoCo passive viewer, following the robot, with keyboard teleop."""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import time

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


class _FitToDisplay:
    """Keep the viewer's X window the size of the X screen.

    The container's Xvnc has no window manager, so GLFW's window opens at its
    default 2/3 of the screen in the corner and nothing can maximize it; and
    launch_passive hands out no GLFW handle to resize through. So this talks to
    the X server itself (libX11 via ctypes, on a connection of its own, so it
    never races the render thread's) and resizes the top-level window whose
    name starts with "MuJoCo". Polled, not event-driven: the screen changes
    whenever noVNC's remote resize follows the browser (vnc.sh's URL asks for
    it), and a 1 s poll is two round trips to a local server.
    """

    def __init__(self):
        self._x = None
        name = ctypes.util.find_library("X11")
        if name is None:
            return
        x = ctypes.CDLL(name)
        x.XOpenDisplay.restype = ctypes.c_void_p
        x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x.XDefaultRootWindow.restype = ctypes.c_ulong
        x.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        u, i, p = ctypes.c_uint, ctypes.c_int, ctypes.POINTER
        x.XGetGeometry.argtypes = [ctypes.c_void_p, ctypes.c_ulong, p(ctypes.c_ulong), p(i), p(i),
                                   p(u), p(u), p(u), p(u)]
        x.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong, p(ctypes.c_ulong),
                                 p(ctypes.c_ulong), p(p(ctypes.c_ulong)), p(u)]
        x.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, p(ctypes.c_char_p)]
        x.XFree.argtypes = [ctypes.c_void_p]
        x.XMoveResizeWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, i, i, u, u]
        x.XFlush.argtypes = [ctypes.c_void_p]
        self._dpy = x.XOpenDisplay(None)
        if not self._dpy:
            return
        self._x = x
        self._root = x.XDefaultRootWindow(self._dpy)
        self._win = None
        self._next = 0.0

    def _size(self, win) -> tuple[int, int, int, int]:
        root, xi, yi = ctypes.c_ulong(), ctypes.c_int(), ctypes.c_int()
        w, h, b, d = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        self._x.XGetGeometry(self._dpy, win, ctypes.byref(root), ctypes.byref(xi), ctypes.byref(yi),
                             ctypes.byref(w), ctypes.byref(h), ctypes.byref(b), ctypes.byref(d))
        return xi.value, yi.value, w.value, h.value

    def _find(self):
        root, parent = ctypes.c_ulong(), ctypes.c_ulong()
        kids, n = ctypes.POINTER(ctypes.c_ulong)(), ctypes.c_uint()
        if not self._x.XQueryTree(self._dpy, self._root, ctypes.byref(root), ctypes.byref(parent),
                                  ctypes.byref(kids), ctypes.byref(n)):
            return None
        found = None
        for k in range(n.value):
            name = ctypes.c_char_p()
            if self._x.XFetchName(self._dpy, kids[k], ctypes.byref(name)) and name.value:
                if name.value.startswith(b"MuJoCo"):
                    found = kids[k]
                self._x.XFree(name)
            if found is not None:
                break
        if kids:
            self._x.XFree(kids)
        return found

    def poll(self) -> None:
        if self._x is None or time.monotonic() < self._next:
            return
        self._next = time.monotonic() + 1.0
        if self._win is None:
            self._win = self._find()  # the render thread maps it a moment after launch
            if self._win is None:
                return
        _, _, sw, sh = self._size(self._root)
        if self._size(self._win) != (0, 0, sw, sh):
            self._x.XMoveResizeWindow(self._dpy, self._win, 0, 0, sw, sh)
            self._x.XFlush(self._dpy)


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
        self._fit = _FitToDisplay()

    def is_running(self) -> bool:
        return self.handle.is_running()

    def sync(self, sim) -> None:
        self.handle.sync()
        self._fit.poll()

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
