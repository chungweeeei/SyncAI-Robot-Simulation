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
KEY_EQUAL, KEY_MINUS = 61, 45
# Every letter is already a MuJoCo viewer shortcut (mujoco.mjVISSTRING /
# mjRNDSTRING: T transparent, P contact split, H convex hull ... so t / p / h
# below also flip those); the viewer's geom groups take 0-5. 9 is free.
KEY_9 = 57
ARROWS = {KEY_UP: "up", KEY_DOWN: "down", KEY_LEFT: "left", KEY_RIGHT: "right"}

HELP = """teleop keys (viewer window focused):
  arrows     walk / turn while held, stop when released
  = / -      faster / slower     space       stop
  1 / 5      set_motion_key 1 (RL) / 5 (MPC)
  p          toggle policy 0/1   t           teleport back to the spawn
  9          storeys: auto (the robot's and below) / 1F / 2F ... / all
  h          this help"""

# A press without a release to go with it (no X keymap, see _HeldKeys) counts
# as held this long: past a browser's first auto-repeat delay (~0.5 s), so a
# held key keeps refreshing it, and the robot stops this long after release.
PRESS_HOLD_S = 0.6


class Teleop:
    """Hold-to-drive: a velocity only while an arrow key is down.

    The viewer reports which arrows are down (`set_held`, from the X server's
    keymap at 60 Hz); without that, each press holds for PRESS_HOLD_S. The
    old latched increments kept walking after the last key until someone
    pressed space, which with a real gait means into a wall."""

    def __init__(self, spawn, log=print, vx_speed: float = 0.4, wz_speed: float = 0.5):
        self.vx = 0.0
        self.wz = 0.0
        self.vx_speed = vx_speed
        self.wz_speed = wz_speed
        self.spawn = spawn
        self.log = log
        self._pending = []  # closures applied under the sim lock
        self._held: set[str] | None = None  # None: no keymap, use the press timers
        self._pressed_until: dict[str, float] = {}
        self._driving = False

    def set_held(self, held: set[str]) -> None:
        self._held = held

    def held(self, now: float | None = None) -> set[str]:
        if self._held is not None:
            return self._held
        now = time.monotonic() if now is None else now
        return {k for k, t in self._pressed_until.items() if t > now}

    def key(self, keycode: int) -> None:
        if keycode in ARROWS:
            self._pressed_until[ARROWS[keycode]] = time.monotonic() + PRESS_HOLD_S
            return
        if keycode == KEY_SPACE:
            self._pressed_until.clear()
        elif keycode in (KEY_EQUAL, KEY_MINUS):
            f = 1.25 if keycode == KEY_EQUAL else 0.8
            self.vx_speed = min(max(self.vx_speed * f, 0.1), 0.8)
            self.wz_speed = min(max(self.wz_speed * f, 0.1), 2.0)
            self.log(f"[teleop] speed vx {self.vx_speed:.2f} m/s  wz {self.wz_speed:.2f} rad/s")
            return
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

    def apply(self, sim) -> None:
        for fn in self._pending:
            fn(sim)
        self._pending.clear()
        h = self.held()
        self.vx = self.vx_speed * (("up" in h) - ("down" in h))
        self.wz = self.wz_speed * (("left" in h) - ("right" in h))
        # Only while driving, and one stop on release: a teleop that set 0
        # every loop would override cmd_vel / the driver's AXES with the keys
        # untouched (the viewer and the nav stack can run together).
        if h or self._driving:
            sim.set_cmd(self.vx, 0.0, self.wz)
        self._driving = bool(h)


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


class _HeldKeys:
    """Which arrow keys are down right now, from the X server's keymap.

    launch_passive's key_callback gets presses (and auto-repeats) but never a
    release, so it cannot tell "held" from "pressed once". XQueryKeymap can:
    Xvnc keeps each key down from the VNC client's keydown to its keyup, so the
    robot stops the moment the key comes up. A connection of its own, as in
    _FitToDisplay; the keymap is global to the display, which here holds the
    viewer and nothing else."""

    KEYSYMS = {"up": 0xFF52, "down": 0xFF54, "left": 0xFF51, "right": 0xFF53}  # XK_Up ...

    def __init__(self):
        self.ok = False
        name = ctypes.util.find_library("X11")
        if name is None:
            return
        x = ctypes.CDLL(name)
        x.XOpenDisplay.restype = ctypes.c_void_p
        x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x.XKeysymToKeycode.restype = ctypes.c_ubyte
        x.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._dpy = x.XOpenDisplay(None)
        if not self._dpy:
            return
        self._x = x
        self._codes = {k: int(x.XKeysymToKeycode(self._dpy, sym)) for k, sym in self.KEYSYMS.items()}
        self._buf = ctypes.create_string_buffer(32)
        self.ok = all(self._codes.values())

    def poll(self) -> set[str]:
        self._x.XQueryKeymap(self._dpy, self._buf)
        bits = self._buf.raw
        return {k for k, c in self._codes.items() if bits[c >> 3] & (1 << (c & 7))}


class StoreyView:
    """Which storeys of the site the viewer draws (storeys.py, scene.py).

    Modes, cycled with 9: auto — the robot's storey and those below it, so
    the floor above never hides the robot and walking up the stairs brings the
    next storey in; then each storey alone; then all. A hidden storey's site
    geoms move to geom group HIDDEN, which the viewer does not draw; the
    change is made under the viewer's lock, in the main thread."""

    HIDDEN = 5

    def __init__(self, sim, log=print):
        self.sim = sim
        self.log = log
        self.n = len(sim.info.storey_bounds) + 1
        self.modes = ["auto"] + [f"{k + 1}F" for k in range(self.n)] + ["all"]
        self.mode = 0
        self.shown: set[int] | None = None

    def cycle(self) -> None:
        if self.n > 1:
            self.mode = (self.mode + 1) % len(self.modes)
            self.log(f"[viewer] storeys: {self.modes[self.mode]}")

    def wanted(self) -> set[int]:
        name = self.modes[self.mode]
        if name == "all":
            return set(range(self.n))
        if name == "auto":
            from .storeys import storey_of
            here = int(storey_of(self.sim.state.z_ground, self.sim.info.storey_bounds))
            return set(range(here + 1))
        return {self.modes.index(name) - 1}

    def apply(self, handle) -> None:
        if self.n <= 1:
            return
        want = self.wanted()
        if want == self.shown:
            return
        with handle.lock():
            for k, geoms in self.sim.info.site_storeys.items():
                self.sim.model.geom_group[geoms] = 0 if k in want else self.HIDDEN
        self.shown = want


class Viewer:
    def __init__(self, sim, teleop: Teleop | None = None, log=print):
        self.sim = sim
        self.teleop = teleop
        self.storeys = StoreyView(sim, log)
        self.handle = mujoco.viewer.launch_passive(
            sim.model, sim.data, key_callback=self._key,
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
        opt.geomgroup[StoreyView.HIDDEN] = 0  # storeys out of view
        self._scan_n = 0
        self._fit = _FitToDisplay()
        self._keys = _HeldKeys() if teleop else None
        if self._keys is not None and not self._keys.ok:
            self._keys = None  # Teleop falls back to its press timers

    def _key(self, keycode: int) -> None:
        if keycode == KEY_9:
            self.storeys.cycle()
        elif self.teleop is not None:
            self.teleop.key(keycode)

    def is_running(self) -> bool:
        return self.handle.is_running()

    def sync(self, sim) -> None:
        self.storeys.apply(self.handle)
        self.handle.sync()
        self._fit.poll()
        if self._keys is not None:
            self.teleop.set_held(self._keys.poll())

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
