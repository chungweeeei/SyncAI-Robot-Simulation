"""The browser viewer: the sim's picture drawn by the host's GPU.

Nothing renders in the container. A viser server (a websocket plus a
bundled three.js client) hands the browser the site once — every tile of
the visual mesh as a GLB with vertex colours, the height ramp visual.py
defines — and the robot's decimated STLs once, then only poses: the robot's
bodies and the props at `VIEW_HZ`, the last lidar scan as a point cloud,
the terrain boxes under the feet when asked. The browser draws it all at
its own frame rate with the host's GPU, orbits and zooms itself, and follows
the robot by moving its look-at point while keeping whatever offset the
mouse gave the camera. The panel holds the storey switch, the layers, the
status, the teleport, and with --teleop the sliders and motion keys
(teleop.py applies them under the sim lock).

Foxglove is still the stack's view (plans, costmaps, the map clouds); this
one shows what the sim itself is made of — which tiles exist, what the feet
stand on, where the rays went.

Until 2026-10-26 the picture was a GLFW window of our own on the container's
Xvnc display, Mesa's llvmpipe behind it, served by noVNC: ten frames a
second of the site at best, no shadows, the far tiles cut, three cores busy
(README, "Decisions worth knowing"). The main loop's contract is what that
window had: `rate_hz`, `is_running`, `sync`, `show_scan`, `close`. The site
tiles are exported in a background thread so the sim starts at once; a
browser that connects before they are in sees them arrive.
"""

from __future__ import annotations

import math
import threading
import time

import mujoco
import numpy as np

from . import scene as sc

VIEW_HZ = 20.0  # poses to the browser per second: a state copy is ~0.3 ms, 20 keep the gait fluid
TERRAIN_HZ = 2.0  # the terrain boxes, when shown
STATUS_HZ = 2.0
SCAN_COLOUR = (255, 90, 30)
TERRAIN_COLOUR = (230, 110, 60)
STAIR_COLOUR = (60, 150, 230)
ROBOT_DEFAULT_COLOUR = (70, 70, 78)  # the URDF's visuals carry no colour (visual.py: "dark STL")
MAP_POINT_M = 0.04
MAP_INTENSITY_MAX = 150.0  # Foxglove's layout clamps the clouds' intensity here too
# The light: an HDRI environment for the ambient and the reflections, the
# default directional lights with shadows (the robot on the floor, the
# treads on each other), a pale sky as the background and a fog of the same
# colour from FOG_NEAR to FOG_FAR so the far end of a 100 m site fades
# instead of cluttering the horizon.
ENVIRONMENT = "city"
ENVIRONMENT_INTENSITY = 0.55
SKY_TOP = (168, 190, 214)
SKY_BOTTOM = (226, 230, 236)
FOG_NEAR, FOG_FAR = 35.0, 130.0


def mesh_of(model: mujoco.MjModel, geom: int) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (V, 3), faces (F, 3)) of a mesh geom, in the geom's frame."""
    mid = int(model.geom_dataid[geom])
    va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    return model.mesh_vert[va:va + vn].copy(), model.mesh_face[fa:fa + fn].copy()


def boxes_mesh(centres: np.ndarray, halves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One mesh of axis-aligned boxes: (8 K, 3) vertices, (12 K, 3) faces."""
    corner = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    k = centres.shape[0]
    v = centres[:, None, :] + corner[None, :, :] * halves[:, None, :]
    f = faces[None, :, :] + (np.arange(k) * 8)[:, None, None]
    return v.reshape(-1, 3), f.reshape(-1, 3)


class WebView:
    def __init__(self, sim, teleop=None, port: int = 8080, log=print, host: str = "0.0.0.0",
                 site_in_background: bool = True):
        import viser

        self.sim = sim
        self.teleop = teleop
        self.log = log
        self.rate_hz = VIEW_HZ
        self.server = viser.ViserServer(host=host, port=port, verbose=False)
        self.server.scene.set_up_direction("+z")
        self._light()
        self._d = mujoco.MjData(sim.model)  # the render copy, state only (viewer.py)
        self._spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state = np.empty(mujoco.mj_stateSize(sim.model, self._spec))
        self._lock = threading.Lock()
        self._running = True
        self._site: dict[int, list] = {}  # storey -> tile handles
        self._map: dict[int, object] = {}  # storey -> map cloud handle
        self._shown: set[int] | None = None
        self._robot: list[tuple[int, object]] = []
        self._props: list[tuple[int, object]] = []
        self._terrain_handle = None
        self._next_terrain = 0.0
        self._next_status = 0.0
        self._gui()
        self._build_robot()
        self._build_props()
        self.server.on_client_connect(self._on_connect)
        if site_in_background:
            threading.Thread(target=self._build_site, daemon=True, name="web-site").start()
        else:
            self._build_site()
        self.log(f"[web] viewer serving on port {port} (http://localhost:{port} on the host)")

    # -- the sim's side (main thread) --------------------------------------------
    def is_running(self) -> bool:
        return self._running

    def sync(self, sim) -> None:
        """Copy the state; push the robot's and the props' poses, the terrain
        and the status at their rates; keep every following camera on the robot."""
        with self._lock:
            mujoco.mj_getState(sim.model, sim.data, self._state, self._spec)
            mujoco.mj_setState(sim.model, self._d, self._state, self._spec)
            mujoco.mj_kinematics(sim.model, self._d)
            xpos, xquat = self._d.xpos.copy(), self._d.xquat.copy()
        base = xpos[sim.info.base_body]
        with self.server.atomic():
            for body, handle in self._robot + self._props:
                handle.position = xpos[body]
                handle.wxyz = xquat[body]
        self._apply_storeys()
        now = time.monotonic()
        if self._terrain.value and now >= self._next_terrain:
            self._next_terrain = now + 1.0 / TERRAIN_HZ
            self._push_terrain()
        if now >= self._next_status:
            self._next_status = now + 1.0 / STATUS_HZ
            self._push_status()
        if self._follow.value:
            for client in self.server.get_clients().values():
                offset = client.camera.position - client.camera.look_at
                client.camera.look_at = base
                client.camera.position = base + offset

    def show_scan(self, sim) -> None:
        if not self._scan.value or sim.last_scan is None:
            return
        T = sim.T_map_imu()
        pts = sim.last_scan[:, :3].astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        self.server.scene.add_point_cloud("/scan", pts, colors=SCAN_COLOUR, point_size=0.03,
                                          point_shape="circle")

    def close(self) -> None:
        self._running = False
        self.server.stop()

    # -- the scene ----------------------------------------------------------------
    def _light(self) -> None:
        scene = self.server.scene
        scene.configure_environment_map(hdri=ENVIRONMENT, background=False,
                                        environment_intensity=ENVIRONMENT_INTENSITY)
        scene.configure_default_lights(enabled=True, cast_shadow=True)
        scene.configure_fog(near=FOG_NEAR, far=FOG_FAR, color=SKY_BOTTOM, enabled=True)
        t = np.linspace(0.0, 1.0, 64)[:, None]
        sky = (np.array(SKY_TOP) * (1.0 - t) + np.array(SKY_BOTTOM) * t).astype(np.uint8)
        scene.set_background_image(np.repeat(sky[:, None, :], 2, axis=1), format="png")

    def _build_site(self) -> None:
        from . import site_tiles

        world = getattr(self.sim, "world", None)
        if world is None:
            return
        t0 = time.perf_counter()
        cfg = self.sim.cfg
        tiles = site_tiles.load_or_build(world, cfg.map_dir, cfg.cache_dir, cfg.visual_mesh,
                                         log=self.log)
        for tile in tiles:
            handle = self.server.scene.add_glb(
                tile.name, tile.glb, cast_shadow=False,
                visible=self._shown is None or tile.storey in self._shown)
            self._site.setdefault(tile.storey, []).append(handle)
        self.log(f"[web] site: {len(tiles)} tiles, "
                 f"{sum(t.triangles for t in tiles) / 1e6:.1f} M triangles sent in "
                 f"{time.perf_counter() - t0:.1f} s")

    def _build_robot(self) -> None:
        m, info = self.sim.model, self.sim.info
        root = m.body_rootid[info.base_body]
        for b in range(1, m.nbody):
            if m.body_rootid[b] != root:
                continue
            frame = self.server.scene.add_frame(f"/robot/{m.body(b).name}", show_axes=False)
            self._robot.append((b, frame))
            for g in range(int(m.body_geomadr[b]), int(m.body_geomadr[b] + m.body_geomnum[b])):
                if m.geom_group[g] != 1 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                v, f = mesh_of(m, g)
                rgba = m.geom_rgba[g]
                colour = (tuple(int(c * 255) for c in rgba[:3]) if not np.allclose(rgba[:3], 0.5)
                          else ROBOT_DEFAULT_COLOUR)
                self.server.scene.add_mesh_simple(
                    f"/robot/{m.body(b).name}/g{g}", v, f, color=colour,
                    position=m.geom_pos[g], wxyz=m.geom_quat[g])

    def _build_props(self) -> None:
        m, info = self.sim.model, self.sim.info
        for name, b in info.prop_bodies.items():
            g = int(m.body_geomadr[b])
            size = m.geom_size[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                v, f = _cylinder(float(size[0]), float(size[1]))
            else:
                v, f = boxes_mesh(np.zeros((1, 3)), size[None, :3].astype(float))
            rgba = m.geom_rgba[g]
            handle = self.server.scene.add_mesh_simple(
                f"/props/{name}", v, f, color=tuple(int(c * 255) for c in rgba[:3]),
                flat_shading=True)
            self._props.append((b, handle))

    def _push_terrain(self) -> None:
        m, info = self.sim.model, self.sim.info
        if info.terrain_mocap_ids is None or info.terrain_mocap_ids.size == 0:
            return
        with self._lock:
            centres = self._d.mocap_pos[info.terrain_mocap_ids].copy()
        used = centres[:, 2] > sc.TERRAIN_PARK_Z + 100.0
        half = m.geom_size[int(info.terrain_geoms[0])].astype(float)
        c = [centres[used]]
        h = [np.tile(half, (int(used.sum()), 1))]
        if info.stair_geoms:
            c.append(np.array([m.geom_pos[g] for g in info.stair_geoms], float))
            h.append(np.array([m.geom_size[g] for g in info.stair_geoms], float))
        v, f = boxes_mesh(np.concatenate(c), np.concatenate(h))
        self._terrain_handle = self.server.scene.add_mesh_simple(
            "/terrain", v, f, color=TERRAIN_COLOUR, opacity=0.55, flat_shading=True,
            cast_shadow=False)

    def _load_map(self) -> None:
        from .map_cloud import load_or_build
        cfg = self.sim.cfg
        clouds = load_or_build(cfg.map_dir, cfg.cache_dir, self.sim.world, voxel=MAP_POINT_M,
                               log=self.log)
        if not clouds:
            return
        for k, c in clouds.items():
            grey = np.clip(c[:, 3] / MAP_INTENSITY_MAX, 0.0, 1.0)
            rgb = (np.stack([0.35 + 0.65 * grey] * 3, axis=1) * 255).astype(np.uint8)
            self._map[k] = self.server.scene.add_point_cloud(
                f"/map/s{k}", c[:, :3], colors=rgb, point_size=MAP_POINT_M, point_shape="circle",
                visible=self._shown is None or k in self._shown)

    # -- the GUI --------------------------------------------------------------------
    def _gui(self) -> None:
        gui = self.server.gui
        n = len(self.sim.info.storey_bounds) + 1
        self._storey_modes = ["auto"] + [f"{k + 1}F" for k in range(n)] + ["all"]
        self._storeys = gui.add_dropdown("Storeys", self._storey_modes, initial_value="auto",
                                         hint="auto: the robot's storey and those below it")
        self._follow = gui.add_checkbox("Follow robot", True)
        self._scan = gui.add_checkbox("Lidar scan", True)
        self._terrain = gui.add_checkbox("Terrain boxes", False,
                                         hint="the voxel boxes and fitted stairs under the feet")
        self._map_cloud = gui.add_checkbox("Lidar map", False,
                                           hint="map.pcd per storey, shaded by intensity (Foxglove's)")
        self._status = gui.add_markdown("")

        @self._storeys.on_update
        def _(_event) -> None:
            self._shown = None  # re-decide
            self._apply_storeys()

        @self._terrain.on_update
        def _(_event) -> None:
            if not self._terrain.value and self._terrain_handle is not None:
                self._terrain_handle.remove()
                self._terrain_handle = None

        @self._scan.on_update
        def _(_event) -> None:
            if not self._scan.value:
                self.server.scene.add_point_cloud("/scan", np.zeros((0, 3)), colors=SCAN_COLOUR)

        @self._map_cloud.on_update
        def _(_event) -> None:
            if self._map_cloud.value and not self._map:
                threading.Thread(target=self._load_map, daemon=True, name="web-map").start()
            for k, h in self._map.items():
                h.visible = self._map_cloud.value and (self._shown is None or k in self._shown)

        if self.teleop is None:
            return
        # The teleport is wanted in the session too (a fall leaves the robot
        # in Damping; teleop.py); the rest only when the sim is driven by hand.
        home = gui.add_button("Teleport to the spawn", hint="back on the spawn, standing")
        home.on_click(lambda _e: self.teleop.teleport_home())
        if not self.teleop.drive:
            return
        with gui.add_folder("Teleop"):
            vx = gui.add_slider("vx m/s", -0.6, 0.8, 0.05, 0.0)
            wz = gui.add_slider("wz rad/s", -1.5, 1.5, 0.05, 0.0)
            stop = gui.add_button("Stop")
            key1 = gui.add_button("Motion key 1 (RL)")
            key5 = gui.add_button("Motion key 5 (MPC)")
            policy = gui.add_button("Toggle policy 0 / 1")

        def drive(_event=None) -> None:
            self.teleop.set_cmd(vx.value, wz.value)
        vx.on_update(drive)
        wz.on_update(drive)

        @stop.on_click
        def _(_event) -> None:
            vx.value = 0.0
            wz.value = 0.0
            self.teleop.stop()
        key1.on_click(lambda _e: self.teleop.motion_key("1"))
        key5.on_click(lambda _e: self.teleop.motion_key("5"))
        policy.on_click(lambda _e: self.teleop.toggle_policy())

    def _wanted(self) -> set[int]:
        n = len(self.sim.info.storey_bounds) + 1
        name = self._storeys.value
        if name == "all":
            return set(range(n))
        if name == "auto":
            from .storeys import storey_of
            here = int(storey_of(self.sim.state.z_ground, self.sim.info.storey_bounds))
            return set(range(here + 1))
        return {self._storey_modes.index(name) - 1}

    def _apply_storeys(self) -> None:
        want = self._wanted()
        if want == self._shown:
            return
        self._shown = want
        with self.server.atomic():
            for k, handles in self._site.items():
                for h in handles:
                    h.visible = k in want
            for k, h in self._map.items():
                h.visible = self._map_cloud.value and k in want

    def _push_status(self) -> None:
        st = self.sim.status()
        rt = ""
        if hasattr(self.sim, "realtime"):
            rt = f"realtime x{self.sim.realtime:.2f}  "
        self._status.content = (
            f"**t** {st['sim_time']:.1f} s  {rt}**{st['last_event']}**  \n"
            f"xy ({st['x']:.2f}, {st['y']:.2f}) yaw {math.degrees(st['yaw']):.0f}° "
            f"ground z {st['z_ground']:.2f} pitch {math.degrees(st['pitch']):.1f}°  \n"
            f"motion state {st['motion_state']}  policy {st['policy']}  "
            f"falls {st['falls']}  collisions {st['collisions']}")

    def _on_connect(self, client) -> None:
        with self._lock:
            base = self._d.xpos[self.sim.info.base_body].copy()
        yaw = self.sim.state.yaw
        back = np.array([-math.cos(yaw), -math.sin(yaw), 0.0])
        client.camera.position = base + 7.0 * back + np.array([0.0, 0.0, 4.5])
        client.camera.look_at = base


def _cylinder(radius: float, half_height: float, n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ring = np.column_stack((radius * np.cos(a), radius * np.sin(a)))
    top = np.column_stack((ring, np.full(n, half_height)))
    bottom = np.column_stack((ring, np.full(n, -half_height)))
    v = np.concatenate([bottom, top, [[0.0, 0.0, -half_height], [0.0, 0.0, half_height]]])
    cb, ct = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i], [cb, j, i], [ct, n + i, n + j]]
    return v, np.array(faces)
