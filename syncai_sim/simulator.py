"""The simulator proper, with no ROS in it: world + robot + physics + sensors.

`ros_node.py` wraps this in a node and `__main__.py` drives it; this module can
also be driven by hand (`--no-ros --teleop`) to check a map or the kinematics
without the stack. One `step()` is one physics tick (`physics_hz`, 200 by
default): the fake driver's pending switches are applied, the kinematic base
is advanced (`kinematics.advance`), the mocap handle is placed, the prop movers
are walked, MuJoCo steps, and a trunk pushed off its target by a prop is
detected. `scan()` renders one lidar scan (`VoxelGrid.cast` for the site,
`prop_rays` for the props) and `imu_sample()` one IMU reading.

Two locomotions (`SimConfig.locomotion`):

    kinematic   the body is placed: the step above, a rigid statue on a mocap
                weld, the step rule and the gates of kinematics.py
    policy      the body walks: the G23's own ONNX policies drive twelve PD
                joints (legged.py, policy.py) on a patch of the voxel grid
                (terrain_patch.py); MuJoCo decides where the body goes, and
                `state` is read back from it, so status, the lidar and the
                ROS side are the same in both. No step rule: whether a policy
                climbs a flight is now its own business, which is the point.

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
from .fake_driver import DAMPING, FakeDriver
from .kinematics import BaseState, Event, KinematicsConfig, advance, place
from .legged import LeggedController
from .lidar import LidarConfig, LidarModel
from .policy import load_policies
from .prop_rays import ray_aabb, ray_vertical_cylinder
from .robot import RobotConfig, RobotModel, load_robot
from .terrain_patch import TerrainPatch, TerrainPatchConfig
from .world import World, WorldOptions, build_world

# g2_deploy_rs src/ros_bridge.rs: what the controller accepts from cmd_vel
# before the policy's own +-1 clip.
CMD_LIMITS = ((-0.6, 0.8), (-0.4, 0.4), (-2.5, 2.5))


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
    locomotion: str = "kinematic"  # kinematic | policy
    ppo_policy: str = ""
    himloco_policy: str = ""
    terrain: TerrainPatchConfig = field(default_factory=TerrainPatchConfig)
    visual_mesh: str = "voxels"  # voxels | pointcloud (pcd_mesh.py)


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
        if cfg.locomotion not in ("kinematic", "policy"):
            raise ValueError(f"locomotion {cfg.locomotion!r}: kinematic | policy")
        self.legged = cfg.locomotion == "policy"
        self.robot: RobotModel = load_robot(RobotConfig(urdf_path=cfg.urdf_path,
                                                        lidar_height=cfg.lidar_height,
                                                        articulated=self.legged))
        site = None
        if cfg.visual_mesh == "pointcloud":
            from .pcd_mesh import load_or_build
            site = load_or_build(cfg.map_dir, cfg.cache_dir, self.world, log=log)
        elif cfg.visual_mesh != "voxels":
            raise ValueError(f"visual_mesh {cfg.visual_mesh!r}: voxels | pointcloud")
        self.model, self.info = sc.build_scene(
            self.world, self.robot, cfg.props, timestep=1.0 / cfg.physics_hz,
            terrain_capacity=cfg.terrain.capacity if self.legged else 0, site=site)
        self.data = mujoco.MjData(self.model)
        self.legs: LeggedController | None = None
        self.patch: TerrainPatch | None = None
        self.surface = None
        self.falls = 0
        self._fallen_from: int | None = None
        self._last_world_hit = -math.inf
        if self.legged:
            self.legs = LeggedController(self.model, self.data, self.info.base_body,
                                         load_policies(cfg.ppo_policy, cfg.himloco_policy, log=log),
                                         log=log)
            from .surface_heights import load_or_build as load_surface
            self.surface = load_surface(cfg.map_dir, cfg.cache_dir, self.world, log=log)
            self.patch = TerrainPatch(self.world.grid, cfg.terrain, self.surface)
            self._gyro = self.model.sensor("imu_gyro").adr[0]
            self._acc = self.model.sensor("imu_acc").adr[0]
            self._trunk_geoms = set(self.info.trunk_geoms)
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
                 f"{self.model.nbody} bodies, dt {self.model.opt.timestep * 1000:.1f} ms, "
                 f"locomotion {cfg.locomotion}")

    # -- placement ---------------------------------------------------------------
    def spawn(self, x: float, y: float, yaw: float, z_hint: float | None = None) -> None:
        """Put the robot down and define the odom frame there."""
        self.state = place(self.state, self.grid, self.cfg.kin, x, y, yaw, z_hint)
        self._put_body()
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
        self._put_body()
        after = planar(self.T_map_imu())
        self.T_map_odom = after @ np.linalg.inv(before) @ self.T_map_odom
        self.log(f"[sim] teleported to ({x:.2f}, {y:.2f}, yaw {yaw:.2f}), ground z "
                 f"{self.state.z_ground:.2f}; map->odom moved")

    def _put_body(self) -> None:
        """The body at `state`, at rest. Walking: level, standing on its feet in
        the default pose, the terrain under it, the policy fresh — and a fall
        is forgiven (the motion state it fell from comes back)."""
        if not self.legged:
            self._apply_target()
            sc.set_base_pose(self.model, self.data, self.info,
                             self.data.mocap_pos[self.info.mocap_id],
                             self.data.mocap_quat[self.info.mocap_id])
            return
        s = self.state
        pos = np.array([s.x, s.y, self._floor_under(s.x, s.y, s.z_ground) + self.robot.h_stand + 0.005])
        self.legs.reset()
        sc.set_base_pose(self.model, self.data, self.info, pos, yaw_pitch_quat(s.yaw, 0.0))
        sc.place_terrain(self.data, self.info, self.patch.boxes_around(pos))
        mujoco.mj_forward(self.model, self.data)
        if self._fallen_from is not None:
            self.driver.motion_state = self._fallen_from
            self._fallen_from = None

    def _floor_under(self, x: float, y: float, voxel_top: float) -> float:
        """The measured floor in the voxel whose top is `voxel_top` (the feet
        stand there, terrain_patch.py), or that top without a measurement."""
        if self.surface is None:
            return voxel_top
        g = self.grid
        idx = np.floor((np.array([[x, y, voxel_top - 0.5 * g.res]]) - g.origin_m) / g.res)
        z = float(self.surface.lookup(idx.astype(np.int64))[0])
        return voxel_top if math.isnan(z) else z

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
        if self.legged:
            self._step_legged()
            return
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

    def _step_legged(self) -> None:
        dt = self.model.opt.timestep
        self.driver.update()
        cmd = (self.cmd if time.monotonic() - self.cmd_time < self.cfg.cmd_timeout
               else np.zeros(3))
        cmd = np.array([min(max(c, lo), hi) for c, (lo, hi) in zip(cmd, CMD_LIMITS)])
        base = self.data.xpos[self.info.base_body]
        if self.patch.due(base):
            sc.place_terrain(self.data, self.info, self.patch.boxes_around(base))
        self.legs.step(cmd, self.driver.policy, self.driver.motion_state)
        for m in self.movers:
            self._walk_mover(m, dt)
        mujoco.mj_step(self.model, self.data)
        self.sim_time += dt
        self._read_state(cmd)
        for name, (bid, pos) in self._free_props.items():
            if self.data.xpos[bid][2] < pos[2] - 3.0:
                self._place_free_prop(bid, pos)

    def _read_state(self, cmd: np.ndarray) -> None:
        """`state` from the walking body, so status and the logs read the same.

        A collision is the trunk touching the site — what a blocked move is in
        the kinematic mode — counted once per half second like a prop hit. Not
        the legs: HimLoco brushes tread nosings with a thigh on every flight of
        dp (5 contacts on the first one, 2026-10-23), which is climbing, not
        hitting. A fall is the controller's posture check, and puts the driver
        in Damping as the real controller locks itself."""
        s = self.state
        p = self.data.xpos[self.info.base_body]
        R = self.data.xmat[self.info.base_body].reshape(3, 3)
        s.x, s.y = float(p[0]), float(p[1])
        s.yaw = math.atan2(R[1, 0], R[0, 0])
        s.pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
        g = self.grid.ground_under(s.x, s.y, s.z_ground, self.cfg.kin.ground_search_up,
                                   self.cfg.kin.ground_search_down)
        if not math.isnan(g):
            s.z_ground = g
        if not self.driver.motion_allowed():
            s.last_event = Event.GATED
        elif np.any(np.abs(cmd) > 1e-6):
            s.last_event = Event.MOVED
        else:
            s.last_event = Event.IDLE
        if self._world_hit() and self.sim_time - self._last_world_hit > 0.5:
            s.collisions += 1
            self._last_world_hit = self.sim_time
        if self._fallen_from is None and self.legs.upset():
            self.falls += 1
            self._fallen_from = self.driver.motion_state
            self.driver.motion_state = DAMPING
            roll, pitch = self.legs.roll_pitch()
            self.log(f"[sim] FELL at ({s.x:.2f}, {s.y:.2f}) roll {math.degrees(roll):.0f} "
                     f"pitch {math.degrees(pitch):.0f} deg; driver -> Damping "
                     f"(teleport / initialpose to stand it up again)")

    def _world_hit(self) -> bool:
        # One geom per terrain body, added in a row: their ids are one range.
        terrain = self.info.terrain_geoms
        lo, hi = int(terrain[0]), int(terrain[-1])
        trunk = self._trunk_geoms
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            other = g2 if lo <= g1 <= hi else g1 if lo <= g2 <= hi else -1
            if other in trunk:
                return True
        return False

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

    def joint_states(self) -> tuple[list[str], np.ndarray]:
        """The twelve leg joints by URDF name: the walking body's, or the
        statue's baked standing angles (robot.py), for robot_state_publisher."""
        from .legged import JOINTS, LEGS
        names = [f"{leg}_{j}_joint" for leg in LEGS for j in JOINTS]
        if self.legged:
            return names, self.legs.q()
        standing = self.robot.cfg.standing
        return names, np.array([standing[j] for _ in LEGS for j in JOINTS])

    def controller_telemetry(self) -> dict:
        """What the gait controller reports (controller_link.py), in its body
        frame = base_link: rpy, specific force (m/s^2, gravity only: nothing in
        the stack reads the driver's `imu`), angular rate, and the twelve
        joints in the driver's order (FL/FR/HL/HR x HipX/HipY/Knee, the same as
        joint_states). Kinematic: the baked standing pose, at rest."""
        R = self.data.xmat[self.info.base_body].reshape(3, 3)
        rpy = (math.atan2(R[2, 1], R[2, 2]), math.asin(max(-1.0, min(1.0, -R[2, 0]))),
               math.atan2(R[1, 0], R[0, 0]))
        acc = R.T @ np.array([0.0, 0.0, 9.81])
        _, q = self.joint_states()
        if self.legged:
            omega = self.data.qvel[self.legs.omega_adr:self.legs.omega_adr + 3].copy()
            dq = self.legs.dq()
            tau = self.data.qfrc_applied[self.legs.vadr].copy()
        else:
            moving = self.state.last_event is Event.MOVED
            omega = R.T @ np.array([0.0, 0.0, self.cmd[2] if moving else 0.0])
            dq = tau = np.zeros(12)
        return {"rpy": rpy, "acc": acc, "omega": omega,
                "JOINT_POS": q, "JOINT_VEL": dq, "JOINT_TAU": tau}

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
        """(gyro rad/s, accel in g) in the IMU frame: the yaw rate and gravity,
        or, walking, the gyro and accelerometer on lidar_top_imu."""
        if self.legged:
            gyro = self.data.sensordata[self._gyro:self._gyro + 3].copy()
            acc = self.data.sensordata[self._acc:self._acc + 3] / 9.81
            return gyro, acc
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
            "locomotion": self.cfg.locomotion, "falls": self.falls,
            "motion_state": self.driver.motion_state, "policy": self.driver.policy,
            "cmd_age": round(time.monotonic() - self.cmd_time, 2)
            if self.cmd_time > -math.inf else None,
        }
