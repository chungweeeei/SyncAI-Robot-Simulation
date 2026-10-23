"""The simulator proper, with no ROS in it: world + robot + physics + sensors.

`ros_node.py` wraps this in a node and `__main__.py` drives it; this module can
also be driven by hand (`--no-ros --teleop`) to check a map or the kinematics
without the stack. One `step()` is one physics tick (`physics_hz`, 200 by
default): the fake driver's pending switches are applied, the kinematic base
is advanced (`kinematics.advance`), the mocap handle is placed, the prop movers
are walked, MuJoCo steps, and a trunk pushed off its target by a prop is
detected. `scan()` renders one lidar scan (`VoxelGrid.cast` for the site,
`prop_rays` for the props) and `imu_sample()` one IMU reading.

Frames handed out (4x4, metres):

    T_map_base   base_link in the map (MuJoCo's world IS the map frame)
    T_map_imu    lidar_top_imu = pointlio_body in the map
    T_map_odom   where the sim says Point-LIO's odom frame sits — gravity
                 aligned at the spawn pose, so lio_odom starts at the identity
                 like the real one, and moved on every teleport so that the
                 odom chain stays continuous while map -> odom jumps, which is
                 what the real localizer does on a relocalize (lio_bridge's
                 own comment says so).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import scene as sc
from .fake_driver import FakeDriver
from .kinematics import BaseState, Event, KinematicsConfig, advance, place
from .lidar import LidarConfig, LidarModel
from .prop_rays import ray_aabb, ray_vertical_cylinder
from .robot import RobotConfig, RobotModel, load_robot
from .world import World, WorldOptions, build_world


@dataclass
class SimConfig:
    map_dir: str
    cache_dir: str
    urdf_path: str
    world: WorldOptions = field(default_factory=WorldOptions)
    kin: KinematicsConfig = field(default_factory=KinematicsConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    lidar_height: float = 0.481
    physics_hz: float = 200.0
    switch_delay: float = 1.0
    initial_motion_state: int = 1
    initial_policy: int = 0
    props: list[sc.PropSpec] = field(default_factory=list)
    # A cmd_vel older than this stops the base. The real driver has no watchdog,
    # but a sim that keeps walking after its controller died is a nuisance.
    cmd_timeout: float = 1.0
    seed: int = 0


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


def planar(T: np.ndarray) -> np.ndarray:
    """The yaw-and-translation part of a 4x4 pose (gravity-aligned frame)."""
    yaw = math.atan2(T[1, 0], T[0, 0])
    P = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    P[:2, :2] = [[c, -s], [s, c]]
    P[:3, 3] = T[:3, 3]
    return P


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
        self.robot: RobotModel = load_robot(RobotConfig(urdf_path=cfg.urdf_path,
                                                        lidar_height=cfg.lidar_height))
        self.model, self.info = sc.build_scene(self.world, self.robot, cfg.props,
                                               timestep=1.0 / cfg.physics_hz)
        self.data = mujoco.MjData(self.model)
        self.grid = self.world.grid
        self.lidar = LidarModel(cfg.lidar, cfg.seed)
        self.driver = FakeDriver(switch_delay=cfg.switch_delay,
                                 motion_state=cfg.initial_motion_state, policy=cfg.initial_policy)
        self.state = BaseState()
        self.cmd = np.zeros(3)
        self.cmd_time = -math.inf
        self.sim_time = 0.0
        self.T_map_odom = np.eye(4)
        self.prop_collisions = 0
        self._last_prop_hit = -math.inf
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
        """Put the robot down and define the odom frame there."""
        self.state = place(self.state, self.grid, self.cfg.kin, x, y, yaw, z_hint)
        self._apply_target()
        sc.set_base_pose(self.model, self.data, self.info, self.data.mocap_pos[self.info.mocap_id],
                         self.data.mocap_quat[self.info.mocap_id])
        for name, (bid, pos) in self._free_props.items():
            self._place_free_prop(bid, pos)
        for m in self.movers:
            self._place_mover(m, m.waypoints[0])
        mujoco.mj_forward(self.model, self.data)
        self.T_map_odom = planar(self.T_map_imu())
        self.log(f"[sim] spawned at ({x:.2f}, {y:.2f}, yaw {yaw:.2f}) on ground z "
                 f"{self.state.z_ground:.2f}")

    def teleport(self, x: float, y: float, yaw: float, z_hint: float | None = None) -> None:
        """Move the robot (a kidnap): odom stays continuous, map -> odom jumps."""
        before = planar(self.T_map_imu())
        self.state = place(self.state, self.grid, self.cfg.kin, x, y, yaw, z_hint)
        self._apply_target()
        sc.set_base_pose(self.model, self.data, self.info, self.data.mocap_pos[self.info.mocap_id],
                         self.data.mocap_quat[self.info.mocap_id])
        after = planar(self.T_map_imu())
        self.T_map_odom = after @ np.linalg.inv(before) @ self.T_map_odom
        self.log(f"[sim] teleported to ({x:.2f}, {y:.2f}, yaw {yaw:.2f}), ground z "
                 f"{self.state.z_ground:.2f}; map->odom moved")

    def _apply_target(self) -> None:
        s = self.state
        self.data.mocap_pos[self.info.mocap_id] = (s.x, s.y, s.z_ground + self.robot.h_stand)
        self.data.mocap_quat[self.info.mocap_id] = yaw_pitch_quat(s.yaw, s.pitch)

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
        self.cmd[:] = (vx, vy, wz)
        self.cmd_time = time.monotonic() if now is None else now

    # -- one physics tick --------------------------------------------------------
    def step(self) -> None:
        dt = self.model.opt.timestep
        self.driver.update()
        vx, vy, wz = (self.cmd if time.monotonic() - self.cmd_time < self.cfg.cmd_timeout
                      else (0.0, 0.0, 0.0))
        self.state = advance(self.state, float(vx), float(vy), float(wz), dt, self.grid,
                             self.cfg.kin, self.driver.policy, self.driver.motion_state)
        self._apply_target()
        for m in self.movers:
            self._walk_mover(m, dt)
        mujoco.mj_step(self.model, self.data)
        self.sim_time += dt

        # A prop in the way holds the trunk back from its target; after a few cm
        # of lag give in — snap the kinematic state to where the body really is,
        # so the robot stands against the prop instead of the weld winding up.
        actual = self.data.xpos[self.info.base_body]
        lag = math.hypot(actual[0] - self.state.x, actual[1] - self.state.y)
        if lag > 0.08:
            self.state.x, self.state.y = float(actual[0]), float(actual[1])
            if self.sim_time - self._last_prop_hit > 0.5:
                self.prop_collisions += 1
                self._last_prop_hit = self.sim_time
        for name, (bid, pos) in self._free_props.items():
            if self.data.xpos[bid][2] < pos[2] - 3.0:  # fell through a floor hole
                self._place_free_prop(bid, pos)

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

    def T_odom_imu(self) -> np.ndarray:
        return np.linalg.inv(self.T_map_odom) @ self.T_map_imu()

    # -- sensors -----------------------------------------------------------------
    def scan(self) -> np.ndarray:
        """One lidar scan as (K, 4) float32 xyz+intensity in the pointlio_body frame."""
        T_l = self.T_map_lidar()
        origin = T_l[:3, 3]
        dirs_l = self.lidar.directions()
        dirs_w = dirs_l @ T_l[:3, :3].T
        dist, pts = self.grid.cast(origin, dirs_w, self.cfg.lidar.max_range)
        dist = np.where(dist < 0, np.inf, dist)
        for t in self._prop_distances(origin, dirs_w):
            nearer = t < dist
            if np.any(nearer):
                pts[nearer] = origin[None, :] + dirs_w[nearer] * t[nearer, None]
                dist = np.where(nearer, t, dist)
        dist = np.where(np.isfinite(dist), dist, -1.0)
        pts_w = self.lidar.returns(dirs_w, dist, pts)
        return self.lidar.cloud_xyzi(pts_w, self.T_map_imu())

    def _prop_distances(self, origin: np.ndarray, dirs_w: np.ndarray):
        m, d = self.model, self.data
        for bid in self.info.prop_bodies.values():
            g = m.body_geomadr[bid]
            size = m.geom_size[g]
            pos = d.geom_xpos[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                yield ray_vertical_cylinder(origin, dirs_w, pos[:2], float(size[0]),
                                            float(pos[2] - size[1]), float(pos[2] + size[1]))
            else:
                R = d.geom_xmat[g].reshape(3, 3)
                ext = np.abs(R) @ size  # world-axis half extents of the rotated box
                yield ray_aabb(origin, dirs_w, pos - ext, pos + ext)

    def imu_sample(self) -> tuple[np.ndarray, np.ndarray]:
        """(gyro rad/s, accel in g) in the IMU frame: the yaw rate and gravity."""
        T = self.T_map_imu()
        R = T[:3, :3]
        vx, vy, wz = (self.cmd if time.monotonic() - self.cmd_time < self.cfg.cmd_timeout
                      else (0.0, 0.0, 0.0))
        moving = self.state.last_event is Event.MOVED
        omega_w = np.array([0.0, 0.0, wz if moving else 0.0])
        gyro = R.T @ omega_w
        accel = R.T @ np.array([0.0, 0.0, 1.0])
        return gyro, accel

    # -- status ------------------------------------------------------------------
    def status(self) -> dict:
        s = self.state
        return {
            "sim_time": round(self.sim_time, 3),
            "x": round(s.x, 3), "y": round(s.y, 3), "yaw": round(s.yaw, 3),
            "z_ground": round(s.z_ground, 3), "pitch": round(s.pitch, 3),
            "on_stairs": s.on_stairs, "last_event": s.last_event.value,
            "next_ground": None if math.isnan(s.next_ground) else round(s.next_ground, 3),
            "last_dz": round(s.last_dz, 3),
            "collisions": s.collisions, "refused": s.refused,
            "prop_collisions": self.prop_collisions,
            "motion_state": self.driver.motion_state, "policy": self.driver.policy,
            "cmd_age": round(time.monotonic() - self.cmd_time, 2)
            if self.cmd_time > -math.inf else None,
        }
