"""The viewer: a GLFW window of our own, following the robot, with keyboard teleop.

Why not `mujoco.viewer.launch_passive`: on llvmpipe (the container has no
GPU) its render loop paces itself on the monitor's vsync, and Xvnc has none,
so it drew the site back to back at whatever rate the CPU allowed — 7-11
frames a second that cost the render thread plus llvmpipe's workers ~3
cores whether or not anything moved (2026-10-24). Its `sync()` also copied
the whole model and the 18 MB MjData to the render thread (12-15 ms, called
60 times a second), it took every letter for its own toggles (t / p / h
flipped display options while teleop used them), reported key presses but
never releases, and gave no handle to its window, so sizing it to the VNC
screen and reading held keys went through libX11 by ctypes. This module is
what that was being bent into: one thread, one window, a frame rate cap
(`RENDER_HZ`), state-only copies (`mj_getState` / `mj_setState`, 0.3 ms),
`glfw.get_key` for held keys, our own key bindings, and the site drawn only
where the robot is (`SiteView`: the storey and the tiles within reach).

Threads: the sim's main loop calls `sync()` (copies the state under the
lock) and `show_scan()`; the viewer thread polls events, reads held keys and
renders. `model.geom_group` is the one thing both touch (SiteView writes it
in sync(), mjv_updateScene reads it): under the same lock.
"""

from __future__ import annotations

import math
import threading
import time

import glfw
import mujoco
import numpy as np

# GLFW key codes
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SPACE = 265, 264, 263, 262, 32
KEY_1, KEY_5, KEY_P, KEY_T, KEY_H = 49, 53, 80, 84, 72
KEY_EQUAL, KEY_MINUS = 61, 45
KEY_0, KEY_9 = 48, 57
ARROWS = {KEY_UP: "up", KEY_DOWN: "down", KEY_LEFT: "left", KEY_RIGHT: "right"}

HELP = """viewer keys (window focused):
  arrows     walk / turn while held, stop when released
  = / -      faster / slower     space       stop
  1 / 5      set_motion_key 1 (RL) / 5 (MPC)
  p          toggle policy 0/1   t           teleport back to the spawn
  9          storeys: auto (the robot's and below) / 1F / 2F ... / all
  0 2 3 4    toggle geom groups: site, props, collision shapes, mocap
  mouse      left-drag orbits, scroll zooms
  h          this help"""

# A press without a release to go with it (no viewer, see Teleop.held) counts
# as held this long: past a browser's first auto-repeat delay (~0.5 s), so a
# held key keeps refreshing it, and the robot stops this long after release.
PRESS_HOLD_S = 0.6

# State copies to the viewer per second. llvmpipe draws the site at 7-11 fps
# (1600x1000, measured 2026-10-24), so 60 syncs a second bought nothing; a
# state-only copy is ~0.3 ms, and 20 a second keep a key release under 50 ms.
VIEW_HZ = 20.0
# Frames per second, at most. llvmpipe managed 7-11 unthrottled while using
# ~3 cores; 10 is what it can do and lets the viewer thread sleep between
# frames instead of rendering the same picture again.
RENDER_HZ = 10.0


class Teleop:
    """Hold-to-drive: a velocity only while an arrow key is down.

    The viewer reports which arrows are down (`set_held`, from glfw.get_key
    at VIEW_HZ); without that, each press holds for PRESS_HOLD_S. The old
    latched increments kept walking after the last key until someone pressed
    space, which with a real gait means into a wall."""

    def __init__(self, spawn, log=print, vx_speed: float = 0.4, wz_speed: float = 0.5):
        self.vx = 0.0
        self.wz = 0.0
        self.vx_speed = vx_speed
        self.wz_speed = wz_speed
        self.spawn = spawn
        self.log = log
        self._pending = []  # closures applied under the sim lock
        self._held: set[str] | None = None  # None: no viewer, use the press timers
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


class SiteView:
    """Which parts of the site the viewer draws: a storey mode and a radius.

    Storeys (storeys.py), cycled with 9: auto — the robot's storey and those
    below it, so the floor above never hides the robot and walking up the
    stairs brings the next storey in; then each storey alone; then all. Of a
    wanted storey only the tiles (scene.TILE_M squares, one geom each) within
    `radius` of the robot are drawn: llvmpipe's cost is per triangle, and dp's
    1F is 1.4 M of them where 30 m around the robot hold a fraction. Hidden
    geoms go to group HIDDEN, which the viewer does not draw; re-decided when
    the robot has moved `move_m` or the mode changed."""

    HIDDEN = 5

    def __init__(self, sim, log=print, radius: float = 30.0, move_m: float = 3.0):
        self.sim = sim
        self.log = log
        self.radius = radius
        self.move_m = move_m
        self.n = len(sim.info.storey_bounds) + 1
        self.modes = ["auto"] + [f"{k + 1}F" for k in range(self.n)] + ["all"]
        self.mode = 0
        self._shown: np.ndarray | None = None
        self._at: np.ndarray | None = None
        self._mode_applied = -1

    def cycle(self) -> None:
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

    def apply(self) -> bool:
        """Set the site geoms' groups for where the robot is; True if any changed.
        Call with the viewer's lock held (mjv_updateScene reads the groups)."""
        info = self.sim.info
        tiles = info.site_tiles
        if tiles is None or tiles.shape[0] == 0:
            return False
        xy = np.array([self.sim.state.x, self.sim.state.y])
        if self._at is not None and self.mode == self._mode_applied \
                and np.linalg.norm(xy - self._at) < self.move_m:
            return False
        want = self.wanted()
        storey_ok = np.isin(tiles[:, 0].astype(int), list(want))
        near = np.hypot(tiles[:, 1] - xy[0], tiles[:, 2] - xy[1]) <= self.radius + tiles[:, 3]
        show = storey_ok & near
        self._at, self._mode_applied = xy, self.mode
        if self._shown is not None and np.array_equal(show, self._shown):
            return False
        self.sim.model.geom_group[np.asarray(info.site_geoms)] = np.where(show, 0, self.HIDDEN)
        self._shown = show
        return True


class Viewer:
    def __init__(self, sim, teleop: Teleop | None = None, log=print):
        self.sim = sim
        self.teleop = teleop
        self.log = log
        self.rate_hz = VIEW_HZ
        self.site = SiteView(sim, log)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = sim.info.base_body
        self.cam.distance = 9.0
        self.cam.elevation = -32.0
        self.cam.azimuth = 150.0
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0  # collision primitives, plates, terrain boxes
        self.opt.geomgroup[4] = 0  # the mocap handle
        self.opt.geomgroup[SiteView.HIDDEN] = 0
        self._lock = threading.Lock()
        self._d = mujoco.MjData(sim.model)  # the render copy, state only
        self._spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state = np.empty(mujoco.mj_stateSize(sim.model, self._spec))
        self._scan: np.ndarray | None = None  # world points to draw as spheres
        self._held: set[str] = set()
        self._mouse = {"x": 0.0, "y": 0.0, "left": False}
        self._running = True
        self._closed = False
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="viewer")
        self._thread.start()
        self._ready.wait(timeout=30.0)
        if self._error is not None:
            raise RuntimeError(f"viewer window failed: {self._error!r}")

    # -- the sim's side (main thread) --------------------------------------------
    def is_running(self) -> bool:
        return self._thread.is_alive() and not self._closed

    def sync(self, sim) -> None:
        """Copy the state to the render copy; place the site; read held keys."""
        with self._lock:
            self.site.apply()
            mujoco.mj_getState(sim.model, sim.data, self._state, self._spec)
            mujoco.mj_setState(sim.model, self._d, self._state, self._spec)
            held = set(self._held)
        if self.teleop is not None:
            self.teleop.set_held(held)

    def show_scan(self, sim, max_points: int = 400) -> None:
        """Draw a thinned copy of the last scan as small spheres in the body frame."""
        xyzi = sim.scan()  # a fresh, independent draw is fine for a picture
        T = sim.T_map_imu()
        pts = xyzi[:: max(1, xyzi.shape[0] // max_points), :3].astype(np.float64)
        with self._lock:
            self._scan = pts @ T[:3, :3].T + T[:3, 3]

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=5.0)

    # -- the viewer thread ----------------------------------------------------------
    def _run(self) -> None:
        try:
            self._open()
        except BaseException as err:  # noqa: BLE001 - reported to the main thread
            self._error = err
            self._ready.set()
            return
        self._ready.set()
        try:
            self._loop()
        finally:
            self._closed = True
            self._ctx.free()
            glfw.destroy_window(self._window)

    def _open(self) -> None:
        if not glfw.init():
            raise RuntimeError("glfw.init failed (no DISPLAY? scripts/vnc.sh starts one)")
        glfw.window_hint(glfw.SAMPLES, 0)  # no multisampling: fill cost on llvmpipe
        w, h = self._screen_size()
        self._window = glfw.create_window(w, h, "MuJoCo : G23", None, None)
        if not self._window:
            raise RuntimeError("glfw.create_window failed")
        glfw.set_window_pos(self._window, 0, 0)
        glfw.make_context_current(self._window)
        glfw.swap_interval(0)
        m = self.sim.model
        self._ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_100)
        self._scn = mujoco.MjvScene(m, maxgeom=m.ngeom + 1000)
        glfw.set_key_callback(self._window, self._on_key)
        glfw.set_cursor_pos_callback(self._window, self._on_cursor)
        glfw.set_mouse_button_callback(self._window, self._on_button)
        glfw.set_scroll_callback(self._window, self._on_scroll)

    @staticmethod
    def _screen_size() -> tuple[int, int]:
        mode = glfw.get_video_mode(glfw.get_primary_monitor())
        return int(mode.size.width), int(mode.size.height)

    def _loop(self) -> None:
        m = self.sim.model
        next_frame = 0.0
        next_fit = 0.0
        while self._running and not glfw.window_should_close(self._window):
            glfw.poll_events()
            self._held = {name for key, name in ARROWS.items()
                          if glfw.get_key(self._window, key) == glfw.PRESS}
            now = time.monotonic()
            if now >= next_fit:
                # Xvnc's screen follows the browser (vnc.sh's resize=remote);
                # keep the window the size of the screen.
                next_fit = now + 1.0
                w, h = self._screen_size()
                if (w, h) != tuple(glfw.get_window_size(self._window)):
                    glfw.set_window_size(self._window, w, h)
            if now < next_frame:
                time.sleep(min(next_frame - now, 0.005))
                continue
            next_frame = now + 1.0 / RENDER_HZ
            with self._lock:
                # Positions from the copied state: bodies and geoms, then the
                # lights (mj_camlight: without it light_xdir stays zero and
                # the floors render in ambient light only, navy instead of
                # the ramp's blue — the first screenshot, 2026-10-24).
                mujoco.mj_kinematics(m, self._d)
                mujoco.mj_camlight(m, self._d)
                mujoco.mjv_updateScene(m, self._d, self.opt, None, self.cam,
                                       mujoco.mjtCatBit.mjCAT_ALL, self._scn)
                self._add_scan()
            w, h = glfw.get_framebuffer_size(self._window)
            mujoco.mjr_render(mujoco.MjrRect(0, 0, w, h), self._scn, self._ctx)
            glfw.swap_buffers(self._window)

    def _add_scan(self) -> None:
        if self._scan is None:
            return
        scn = self._scn
        size = np.array([0.03, 0.0, 0.0])
        rgba = np.array([1.0, 0.3, 0.1, 0.8], np.float32)
        eye = np.eye(3).reshape(-1)
        for p in self._scan[: scn.maxgeom - scn.ngeom]:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, size, p,
                                eye, rgba)
            scn.ngeom += 1

    # -- input (viewer thread) --------------------------------------------------------
    def _on_key(self, window, key, scancode, action, mods) -> None:
        if action != glfw.PRESS:
            return
        if key == KEY_9:
            with self._lock:
                self.site.cycle()
        elif KEY_0 <= key <= KEY_0 + 5 and key not in (KEY_1, KEY_5):
            g = key - KEY_0
            self.opt.geomgroup[g] = 0 if self.opt.geomgroup[g] else 1
        elif key == KEY_H and self.teleop is None:
            self.log(HELP)
        elif self.teleop is not None:
            self.teleop.key(key)

    def _on_cursor(self, window, x, y) -> None:
        dx, dy = x - self._mouse["x"], y - self._mouse["y"]
        self._mouse["x"], self._mouse["y"] = x, y
        if self._mouse["left"]:
            _, h = glfw.get_window_size(window)
            with self._lock:
                mujoco.mjv_moveCamera(self.sim.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                                      dx / h, dy / h, self._scn, self.cam)

    def _on_button(self, window, button, action, mods) -> None:
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse["left"] = action == glfw.PRESS
            self._mouse["x"], self._mouse["y"] = glfw.get_cursor_pos(window)

    def _on_scroll(self, window, xoff, yoff) -> None:
        with self._lock:
            mujoco.mjv_moveCamera(self.sim.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0,
                                  -0.05 * yoff, self._scn, self.cam)


def yaw_of(sim) -> float:
    T = sim.T_map_base()
    return math.atan2(T[1, 0], T[0, 0])
