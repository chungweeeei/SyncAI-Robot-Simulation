"""The simulator proper: world + robot + physics + sensors, in one object.

`__main__.py` drives it at wall-clock pace (and scripts/controller_bench.py
as fast as it likes); everything the sim tells the outside goes over UDP.
The sim is the robot's hardware and nothing more: the twelve joints are a
controller process's — over the hardware link (locomotion/hardware_link.py),
or as J60 motors on four SocketCAN buses as on the robot (`joint_link: can`,
locomotion/can_link.py) — and the sim decides no joint command of its own. The controller's RL or MPC
turns cmd_vel into joint commands; the panel's keys and AXES go to the
controller (driver/udp_driver.py), not to the body.

One `step()` is one physics tick (`physics_hz`, 1000 by default, the params'
value): the terrain patch is moved under the feet, the controller's command
that came in is applied by the motors, the prop movers are walked, MuJoCo
steps, and the body's record goes out. A fall is counted, not acted on: the
controller's own posture check locks it. `scan()` renders one lidar scan
(`VoxelGrid.cast` for the site, `prop_rays` for the props).

Until 2026-11-13 the sim could also walk itself — the robot's ONNX policies
in this process (`locomotion: policy`), or a rigid statue carried by rules
(`kinematic`) — and answer as the gait controller on the ASCII link
(`driver_link: udp`). Those were the controller's job and moved out with it.

Poses handed out (4x4, metres; MuJoCo's world is the map frame):

    T_map_base   base_link in the map
    T_map_imu    lidar_top_imu in the map: the frame the lidar's cloud is in
    T_map_lidar  lidar_top in the map: where its rays start
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from syncai_sim.actuator.motors import Motors
from syncai_sim.scene import scene as sc
from syncai_sim.driver.udp_driver import UdpDriver
from syncai_sim.envs.placement import BaseState, PlacementConfig, place
from syncai_sim.locomotion.can_link import CanLink, CanLinkConfig
from syncai_sim.locomotion.hardware_link import HardwareLink, HardwareLinkConfig
from syncai_sim.sensor.lidar import LidarConfig, LidarModel, in_firing_frame, rays_along_motion
from syncai_sim.sensor.proprioception import Sensors, sensor_adr
from syncai_sim.sensor.prop_rays import ray_aabb, ray_vertical_cylinder
from syncai_sim.entity.robot import RobotConfig, RobotModel, load_robot
from syncai_sim.terrains.terrain_patch import TerrainPatch, TerrainPatchConfig
from syncai_sim.world.build import World, WorldOptions, build_world

@dataclass
class SimConfig:
    map_dir: str
    cache_dir: str
    urdf_path: str
    world: WorldOptions = field(default_factory=WorldOptions)
    place: PlacementConfig = field(default_factory=PlacementConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    lidar_height: float = 0.481
    # Every physics step is one state datagram; at 1 kHz a controller's 1 ms
    # loop sees a new timestamp each tick (500 works too: it reads each state
    # twice). config.PARAM_DEFAULTS has the same.
    physics_hz: float = 1000.0
    props: list[sc.PropSpec] = field(default_factory=list)
    seed: int = 0
    terrain: TerrainPatchConfig = field(default_factory=TerrainPatchConfig)
    visual_mesh: str = "voxels"  # the browser's site: voxels | pointcloud | tsdf (site_tiles.py)
    visual_look: str = "ramp"  # its look: ramp | photo (visual.py; photo needs config/colour/<map>.npz)
    # fitted = the fitted flights as fixed boxes (stairs.py); voxels = the
    # patch's boxes there too, risers on the 0.1 m column grid (the sim before 2026-10-25)
    stairs: str = "fitted"
    # The body's physics (entity/robot.py RobotConfig says what each is). The
    # defaults are the vendor's — the way a controller's own development
    # simulation models the robot, so what it sees here is what it would see
    # there (README, "The body's physics, as knobs").
    motor_model: str = "j60"  # ideal | j60
    trunk_inertial: str = "vendor"  # urdf | vendor
    foot_solref: tuple[float, float] = (0.005, 1.0)
    foot_friction: tuple[float, float, float] = (1.0, 0.01, 0.01)
    integrator: str = "euler"  # euler | implicitfast (scene.py)
    cmd_delay_ms: float = 0.0
    torque_bw_hz: float = 0.0
    # The hardware link (locomotion/hardware_link.py says the protocol), and
    # where the panel's motion keys and AXES go: the controller's command port.
    link_command_port: int = 20001
    link_state_host: str = "127.0.0.1"
    link_state_port: int = 30010
    link_watchdog_ms: float = 100.0
    # Where the joint commands come from: udp = the hardware link's command
    # datagram; can = J60 frames on four SocketCAN buses, one per leg
    # (locomotion/can_link.py). The state datagram goes out either way: the
    # IMU, the base's truth and the contacts are on no CAN bus.
    joint_link: str = "udp"
    can_interfaces: tuple[str, ...] = ("can1", "can2", "can3", "can4")
    can_motor_ids: tuple[int, ...] = (2, 3, 4)
    can_delay_ms: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    can_drop_rate: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    controller_command_ip: str = "127.0.0.1"
    controller_command_port: int = 50051
    # A late loop catches up at most this much sim time in one burst, and
    # drops a debt past it (envs/pacing.py, the main loop and controller_bench):
    # 8 ms, so that a controller is never handed a quarter second of states in
    # a few ms while its command stays the same.
    max_catch_up_s: float = 0.008


@dataclass
class RawScan:
    """One scan as the lidar itself reports it (sensor/livox_link.py sends it):
    every ray fired, in firing order, each point in the lidar_top frame of the
    moment it was fired — not motion-compensated — and a miss as (0, 0, 0),
    which the MID360 sends too and the stack drops by its blind zone."""

    points: np.ndarray  # (N, 3) float32, metres, lidar_top
    t: np.ndarray  # (N,) float64, sim seconds each ray was fired, ascending
    end: float  # sim time at the scan's end


@dataclass
class Mover:
    name: str
    mocap_id: int
    waypoints: np.ndarray  # (W, 2)
    speed: float
    half_height: float
    seg: int = 0
    forward: bool = True
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    z: float = 0.0



def pose_matrix(pos, R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def yaw_pitch_quat(yaw: float, pitch: float) -> np.ndarray:
    """wxyz quaternion for a yaw about z then a pitch about the body y."""
    qy = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    qp = np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])
    out = np.empty(4)
    mujoco.mju_mulQuat(out, qy, qp)
    return out


class Simulator:
    def __init__(self, cfg: SimConfig, log=print):
        self.cfg = cfg
        self.log = log
        t0 = time.perf_counter()
        self.world: World = build_world(cfg.map_dir, cfg.cache_dir, cfg.world, log=log)
        cache_root = os.path.dirname(os.path.abspath(cfg.cache_dir))
        self.robot: RobotModel = load_robot(RobotConfig(
            urdf_path=cfg.urdf_path, lidar_height=cfg.lidar_height,
            mesh_cache_dir=os.path.join(cache_root, "robot"),
            motor_model=cfg.motor_model, trunk_inertial=cfg.trunk_inertial,
            foot_solref=tuple(float(v) for v in cfg.foot_solref),
            foot_friction=tuple(float(v) for v in cfg.foot_friction)), log=log)
        if cfg.visual_mesh not in ("voxels", "pointcloud", "tsdf"):
            raise ValueError(f"visual_mesh {cfg.visual_mesh!r}: voxels | pointcloud | tsdf")
        if cfg.visual_look not in ("ramp", "photo"):
            raise ValueError(f"visual_look {cfg.visual_look!r}: ramp | photo")
        capacity = cfg.terrain.capacity
        timestep = 1.0 / cfg.physics_hz
        if cfg.stairs not in ("fitted", "voxels"):
            raise ValueError(f"stairs {cfg.stairs!r}: fitted | voxels")
        # The feet's surface, and the flights it fitted, before the scene,
        # which builds those as fixed boxes (stairs.py).
        from syncai_sim.terrains.surface_heights import load_or_build as load_surface
        self.surface = load_surface(cfg.map_dir, cfg.cache_dir, self.world, log=log)
        stairs = None
        if cfg.stairs == "fitted" and self.surface is not None and self.surface.flights:
            from syncai_sim.terrains import stairs as st
            g = self.world.grid
            stairs = st.build(self.surface.flights, g.grid, g.origin_m, g.res, log=log)
        boxes = None if stairs is None else stairs.boxes
        solref = tuple(float(v) for v in cfg.foot_solref)
        key = sc.scene_key(self.world, self.robot, cfg.props, timestep, capacity, stairs=boxes,
                           integrator=cfg.integrator, terrain_solref=solref)
        self.model = sc.cached_model(
            cfg.cache_dir, key,
            lambda: sc.build_scene(self.world, self.robot, cfg.props, timestep=timestep,
                                   terrain_capacity=capacity, stairs=boxes,
                                   integrator=cfg.integrator, terrain_solref=solref)[0], log=log)
        self.info = sc.scene_info(self.model, self.robot, cfg.props, capacity,
                                  list(self.world.storey_bounds))
        self.data = mujoco.MjData(self.model)
        self.falls = 0
        self._fallen = False
        self._last_world_hit = -math.inf
        self.motors = Motors(self.model, self.data, motor_model=cfg.motor_model,
                             cmd_delay_s=cfg.cmd_delay_ms / 1000.0, torque_bw_hz=cfg.torque_bw_hz)
        self.patch = TerrainPatch(self.world.grid, cfg.terrain, self.surface,
                                  None if stairs is None else stairs.regions)
        self.sensors = Sensors(self.model, self.data, self.info, self.motors)
        self._imu_gyro = sensor_adr(self.model, "imu_gyro")
        self._imu_acc = sensor_adr(self.model, "imu_acc")
        self.link = HardwareLink(HardwareLinkConfig(
            command_port=cfg.link_command_port, state_host=cfg.link_state_host,
            state_port=cfg.link_state_port, watchdog_s=cfg.link_watchdog_ms / 1000.0), log=log)
        if cfg.joint_link not in ("udp", "can"):
            raise ValueError(f"joint_link {cfg.joint_link!r}: udp | can")
        self.can: CanLink | None = None
        if cfg.joint_link == "can":
            self.can = CanLink(CanLinkConfig(
                interfaces=tuple(cfg.can_interfaces), motor_ids=tuple(cfg.can_motor_ids),
                timeout_ms=cfg.link_watchdog_ms, delay_ms=tuple(cfg.can_delay_ms),
                drop_rate=tuple(cfg.can_drop_rate)), log=log, seed=cfg.seed)
        self.grid = self.world.grid
        self.lidar = LidarModel(cfg.lidar, cfg.seed)
        # Where the lidar was at each physics step of the last scan window
        # (lidar.rays_along_motion): a ring, `_lidar_i` steps written so far.
        self._steps_per_scan = max(1, int(round(cfg.physics_hz / cfg.lidar.scan_rate_hz)))
        self._lidar_poses = np.tile(np.eye(4), (self._steps_per_scan, 1, 1))
        self._lidar_i = 0
        self.last_scan: np.ndarray | None = None  # the cloud `scan()` last returned
        self.last_raw_scan: RawScan | None = None  # the same scan as the lidar reports it
        self.driver = UdpDriver(host=cfg.controller_command_ip,
                                command_port=cfg.controller_command_port, log=log)
        self.state = BaseState()
        self.cmd = np.zeros(3)
        self.cmd_time = -math.inf
        self.sim_time = 0.0
        self.movers: list[Mover] = []
        self._free_props: dict[str, tuple[int, np.ndarray]] = {}
        for p in cfg.props:
            bid = self.info.prop_bodies[p.name]
            if p.waypoints:
                hh = p.size[1] if p.shape == "cylinder" else p.size[2]
                self.movers.append(Mover(p.name, self.info.prop_mocap_ids[p.name],
                                         np.asarray(p.waypoints, float), p.speed, hh,
                                         pos=np.asarray(p.waypoints[0], float), z=p.pos[2]))
            else:
                self._free_props[p.name] = (bid, np.asarray(p.pos, float))
        self.log(f"[sim] ready in {time.perf_counter() - t0:.1f} s: {self.model.ngeom} geoms, "
                 f"{self.model.nbody} bodies, dt {self.model.opt.timestep * 1000:.1f} ms")

    # -- placement ---------------------------------------------------------------
    def spawn(self, x: float, y: float, yaw: float, z_hint: float | None = None) -> None:
        """Put the robot down at (x, y, yaw), on the floor nearest z_hint."""
        self.state = place(self.state, self.grid, self.cfg.place, x, y, yaw, z_hint)
        self._put_body()
        for name, (bid, pos) in self._free_props.items():
            self._place_free_prop(bid, pos)
        for m in self.movers:
            self._place_mover(m, m.waypoints[0])
        mujoco.mj_forward(self.model, self.data)
        self._lidar_poses[:] = self.T_map_lidar()
        self.log(f"[sim] spawned at ({x:.2f}, {y:.2f}, yaw {yaw:.2f}) on ground z "
                 f"{self.state.z_ground:.2f}")

    def teleport(self, x: float, y: float, yaw: float, z_hint: float | None = None) -> None:
        """Move the robot (a kidnap): the props stay where they are."""
        self.state = place(self.state, self.grid, self.cfg.place, x, y, yaw, z_hint)
        self._put_body()
        self._lidar_poses[:] = self.T_map_lidar()  # a kidnap: no motion to smear over
        self.log(f"[sim] teleported to ({x:.2f}, {y:.2f}, yaw {yaw:.2f}), ground z "
                 f"{self.state.z_ground:.2f}")

    def _put_body(self) -> None:
        """The body at `state`, at rest: level, standing on its feet in the
        default pose, the terrain under it, and the controller's last command
        dropped (it was for the old pose). A fall is forgiven."""
        s = self.state
        pos = np.array([s.x, s.y, self._floor_under(s.x, s.y, s.z_ground) + self.robot.h_stand + 0.005])
        self.motors.reset()  # joints at the default pose, at rest, nothing queued
        self.link.reset()
        if self.can is not None:
            self.can.reset()
        sc.set_base_pose(self.model, self.data, self.info, pos, yaw_pitch_quat(s.yaw, 0.0))
        sc.place_terrain(self.data, self.info, self.patch.boxes_around(pos))
        mujoco.mj_forward(self.model, self.data)
        self.sensors.read(self.link.time(self.sim_time))  # the joints a motor reports before the first step
        self._fallen = False

    def _floor_under(self, x: float, y: float, voxel_top: float) -> float:
        """The measured floor in the voxel whose top is `voxel_top` (the feet
        stand there, terrain_patch.py), or that top without a measurement."""
        if self.surface is None:
            return voxel_top
        g = self.grid
        idx = np.floor((np.array([[x, y, voxel_top - 0.5 * g.res]]) - g.origin_m) / g.res)
        z = float(self.surface.lookup(idx.astype(np.int64))[0])
        return voxel_top if math.isnan(z) else z

    def _place_free_prop(self, bid: int, pos: np.ndarray) -> None:
        g = self.grid.ground_under(pos[0], pos[1], pos[2], 1.0, 6.0)
        if math.isnan(g):
            g = pos[2]
        half = float(self.model.geom_size[self.model.body_geomadr[bid]][2]
                     if self.model.geom_type[self.model.body_geomadr[bid]] == mujoco.mjtGeom.mjGEOM_BOX
                     else self.model.geom_size[self.model.body_geomadr[bid]][1])
        adr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
        self.data.qpos[adr:adr + 3] = (pos[0], pos[1], g + half + 0.01)
        self.data.qpos[adr + 3:adr + 7] = (1, 0, 0, 0)
        vadr = self.model.jnt_dofadr[self.model.body_jntadr[bid]]
        self.data.qvel[vadr:vadr + 6] = 0.0

    def _place_mover(self, m: Mover, xy) -> None:
        g = self.grid.ground_under(float(xy[0]), float(xy[1]), m.z, 1.0, 6.0)
        if not math.isnan(g):
            m.z = g
        m.pos = np.asarray(xy, float)
        self.data.mocap_pos[m.mocap_id] = (m.pos[0], m.pos[1], m.z + m.half_height)

    # -- commands ----------------------------------------------------------------
    def set_cmd(self, vx: float, vy: float, wz: float, now: float | None = None) -> None:
        """cmd_vel from the panel: AXES to the controller, which walks."""
        self.cmd[:] = (vx, vy, wz)
        self.cmd_time = time.monotonic() if now is None else now
        self.driver.send_axes(vx, vy, wz)

    def close(self) -> None:
        """The links' and the driver's sockets."""
        self.link.close()
        if self.can is not None:
            self.can.close()
        self.driver.close()

    # -- one physics tick --------------------------------------------------------
    def step(self) -> None:
        dt = self.model.opt.timestep
        base = self.data.xpos[self.info.base_body]
        if self.patch.due(base):
            sc.place_terrain(self.data, self.info, self.patch.boxes_around(base))
        if self.can is not None:
            self.motors.apply(self.can.poll(self.sim_time, self.sensors))
        else:
            self.motors.apply(self.link.poll(self.sim_time))
        for m in self.movers:
            self._walk_mover(m, dt)
        mujoco.mj_step(self.model, self.data)
        self.sim_time += dt
        # The state this step produced, to the controller.
        self.sensors.read(self.link.time(self.sim_time))
        self.link.send(self.sensors)
        self._record_lidar_pose()
        self._read_state()
        for name, (bid, pos) in self._free_props.items():
            if self.data.xpos[bid][2] < pos[2] - 3.0:  # fell through a floor hole
                self._place_free_prop(bid, pos)

    def _read_state(self) -> None:
        """`state` from the body, for status, the logs and the viewer.

        A collision is the trunk touching the site, counted once per half
        second. Not the legs: a stair climber brushes tread nosings with a
        thigh on every flight of dp (5 contacts on the first one, 2026-10-23),
        which is climbing, not hitting. A fall is the controller's posture
        check (Sensors.upset): counted and logged once, nothing more — the
        controller locks itself, as on the robot."""
        s = self.state
        p = self.data.xpos[self.info.base_body]
        R = self.data.xmat[self.info.base_body].reshape(3, 3)
        s.x, s.y = float(p[0]), float(p[1])
        s.yaw = math.atan2(R[1, 0], R[0, 0])
        s.pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
        g = self.grid.ground_under(s.x, s.y, s.z_ground, self.cfg.place.ground_search_up,
                                   self.cfg.place.ground_search_down)
        if not math.isnan(g):
            s.z_ground = g
        if self.sensors.trunk_hit() and self.sim_time - self._last_world_hit > 0.5:
            s.collisions += 1
            self._last_world_hit = self.sim_time
        if not self._fallen and self.sensors.upset():
            self.falls += 1
            self._fallen = True
            roll, pitch = self.sensors.roll_pitch()
            self.log(f"[sim] FELL at ({s.x:.2f}, {s.y:.2f}) roll {math.degrees(roll):.0f} "
                     f"pitch {math.degrees(pitch):.0f} deg (the controller locks itself; "
                     f"teleport, then RESET_SAFETY and MODE Z to stand it up again)")

    def _walk_mover(self, m: Mover, dt: float) -> None:
        if m.waypoints.shape[0] < 2:
            return
        nxt = m.seg + 1 if m.forward else m.seg - 1
        target = m.waypoints[nxt]
        d = target - m.pos
        dist = float(np.linalg.norm(d))
        step = m.speed * dt
        if dist <= step:
            m.pos = target.copy()
            m.seg = nxt
            if m.seg == m.waypoints.shape[0] - 1 or m.seg == 0:
                m.forward = not m.forward
        else:
            m.pos = m.pos + d / dist * step
        self._place_mover(m, m.pos)

    # -- poses -------------------------------------------------------------------
    def T_map_base(self) -> np.ndarray:
        p, R = sc.body_frame(self.data, self.info.base_body)
        return pose_matrix(p, R)

    def T_map_imu(self) -> np.ndarray:
        p, R = sc.body_frame(self.data, self.info.imu_body)
        return pose_matrix(p, R)

    def T_map_lidar(self) -> np.ndarray:
        p, R = sc.body_frame(self.data, self.info.lidar_body)
        return pose_matrix(p, R)

    # -- sensors -----------------------------------------------------------------
    def _record_lidar_pose(self) -> None:
        self._lidar_poses[self._lidar_i % self._steps_per_scan] = self.T_map_lidar()
        self._lidar_i += 1

    def scan(self) -> np.ndarray:
        """One lidar scan as (K, 4) float32 xyz+intensity in the lidar_top_imu
        frame, at the scan's end. Each ray is cast from where the lidar was
        when it fired during the scan window (the last `_steps_per_scan`
        steps), then the cloud is expressed in the end-of-scan body frame:
        motion-compensated, as a LIO's body cloud is."""
        k = self._steps_per_scan
        dt = self.model.opt.timestep
        chrono = (np.arange(k) + self._lidar_i) % k  # ring slots, oldest first
        poses = self._lidar_poses[chrono]
        dirs_l, t = self.lidar.directions()
        origins, dirs_w = rays_along_motion(poses, t, dt, dirs_l)
        dist, pts = self.grid.cast(origins, dirs_w, self.cfg.lidar.max_range)
        dist = np.where(dist < 0, np.inf, dist)
        for tp in self._prop_distances(origins, dirs_w):
            nearer = tp < dist
            if np.any(nearer):
                pts[nearer] = origins[nearer] + dirs_w[nearer] * tp[nearer, None]
                dist = np.where(nearer, tp, dist)
        dist = np.where(np.isfinite(dist), dist, -1.0)
        ok, pts_w = self.lidar.returns_masked(dirs_w, dist, pts)
        self.last_scan = self.lidar.cloud_xyzi(pts_w, self.T_map_imu())
        raw = np.zeros((t.shape[0], 3), dtype=np.float32)
        raw[ok] = in_firing_frame(poses, t[ok], dt, pts_w)
        self.last_raw_scan = RawScan(points=raw, t=self.sim_time - k * dt + t, end=self.sim_time)
        return self.last_scan

    def lidar_imu(self) -> tuple[np.ndarray, np.ndarray]:
        """The MID360's built-in IMU, on lidar_top_imu: (angular rate rad/s,
        specific force m/s^2), both in that frame, +9.81 up at rest: MuJoCo's
        gyro and accelerometer there."""
        d = self.data
        return (d.sensordata[self._imu_gyro:self._imu_gyro + 3].copy(),
                d.sensordata[self._imu_acc:self._imu_acc + 3].copy())

    def _prop_distances(self, origins: np.ndarray, dirs_w: np.ndarray):
        m, d = self.model, self.data
        for bid in self.info.prop_bodies.values():
            g = m.body_geomadr[bid]
            size = m.geom_size[g]
            pos = d.geom_xpos[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                yield ray_vertical_cylinder(origins, dirs_w, pos[:2], float(size[0]),
                                            float(pos[2] - size[1]), float(pos[2] + size[1]))
            else:
                R = d.geom_xmat[g].reshape(3, 3)
                ext = np.abs(R) @ size  # world-axis half extents of the rotated box
                yield ray_aabb(origins, dirs_w, pos - ext, pos + ext)

    # -- status ------------------------------------------------------------------
    def status(self) -> dict:
        s = self.state
        return {
            "sim_time": round(self.sim_time, 3),
            "x": round(s.x, 3), "y": round(s.y, 3), "yaw": round(s.yaw, 3),
            "z_ground": round(s.z_ground, 3), "pitch": round(s.pitch, 3),
            "collisions": s.collisions,
            "falls": self.falls,
            "motion_state": self.driver.motion_state, "policy": self.driver.policy,
            "link": self.link.stats(),
            **({"can": self.can.stats()} if self.can is not None else {}),
            "cmd_age": round(time.monotonic() - self.cmd_time, 2)
            if self.cmd_time > -math.inf else None,
        }
