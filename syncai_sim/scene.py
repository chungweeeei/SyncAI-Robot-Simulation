"""Assemble the MuJoCo scene: robot + the site's visual mesh and plates + props.

The world's three forms are built in `world.py`; this module only puts them in
one `MjSpec` with the robot (`robot.py`) and the props and compiles it.

Collision bitmasks (contype / conaffinity), so that exactly the pairs the sim
wants touch:

    world plates   1 / 1      the few big boxes; stop props, never the trunk
    trunk boxes    2 / 2      meet props only (the world is VoxelGrid's job)
    props          2 / 3      meet plates (1 & 3) and the trunk (2 & 2)
    visual mesh    0 / 0      the viewer's copy of the site; collides with nothing

Geom groups (the viewer toggles them with keys 0-5): 0 site mesh, 1 robot STL
visuals, 2 props, 3 collision primitives and plates (hidden), 4 the mocap
handle (hidden), 5 reserved for debug markers. The lidar never asks MuJoCo
about any of them — rays go to VoxelGrid for the site and to `prop_rays.py`
for the props (mj_multiRay scans every geom per ray; 18 k plates would cost
hundreds of milliseconds a scan).

Props come from the params as a list of dicts (see `PropSpec`): a pushable box
is a free body with mass; a "person" is a mocap cylinder the node walks between
waypoints. Both are what the planner's dynamic overlay and TerrainLayer's
`obstacle_memory` are for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .robot import RobotModel
from .world import World

PLATE_CONTYPE, PLATE_CONAFFINITY = 1, 1
PROP_CONTYPE, PROP_CONAFFINITY = 2, 3


@dataclass
class PropSpec:
    name: str
    shape: str = "box"  # box | cylinder
    size: tuple[float, ...] = (0.25, 0.25, 0.25)  # half-sizes (box) or (radius, half-height)
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # z = ground under it; the node lifts it
    mass: float = 5.0
    rgba: tuple[float, float, float, float] = (0.9, 0.6, 0.2, 1.0)
    # A mover is kinematic (mocap): it walks its waypoints at `speed` and turns around
    # at the ends; waypoints are (x, y) in the map frame. Empty = a free, pushable body.
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    speed: float = 0.8


@dataclass
class SceneInfo:
    base_body: int
    mocap_body: int
    mocap_id: int  # index into data.mocap_pos
    lidar_body: int
    imu_body: int
    trunk_geoms: list[int]
    prop_bodies: dict[str, int]
    prop_mocap_ids: dict[str, int]  # movers only
    prop_group: int = 2
    site_mesh_geom: int = -1


def build_scene(world: World, robot: RobotModel, props: list[PropSpec],
                timestep: float = 0.005) -> tuple[mujoco.MjModel, SceneInfo]:
    spec = mujoco.MjSpec.from_string(robot.mjcf)
    spec.option.timestep = timestep
    spec.option.gravity[:] = (0.0, 0.0, -9.81)
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.map.znear = 0.02
    spec.visual.map.zfar = 400.0  # a site is 100 m across; the default 50 clips it
    spec.visual.headlight.ambient[:] = (0.35, 0.35, 0.35)
    spec.visual.headlight.diffuse[:] = (0.6, 0.6, 0.6)
    spec.stat.extent = 60.0  # frames the whole site in the viewer's default zoom
    spec.stat.center[:] = (np.array(world.stats.bounds_min_m) + world.stats.bounds_max_m) / 2

    # The site, visual only.
    if world.mesh_faces.shape[0] > 0:
        mesh = spec.add_mesh(name="site_mesh")
        mesh.uservert = world.mesh_vertices.reshape(-1).astype(np.float32)
        mesh.userface = world.mesh_faces.reshape(-1).astype(np.int32)
        spec.worldbody.add_geom(name="site_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
                                meshname="site_mesh", contype=0, conaffinity=0, group=0,
                                rgba=(0.72, 0.72, 0.76, 1.0))

    # Plates: collision for props.
    centres, halves = world.plates_metres()
    for i in range(centres.shape[0]):
        spec.worldbody.add_geom(name=f"plate{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=centres[i], size=halves[i], contype=PLATE_CONTYPE,
                                conaffinity=PLATE_CONAFFINITY, group=3,
                                rgba=(0.3, 0.7, 0.3, 0.35))

    # Props.
    for p in props:
        shape = mujoco.mjtGeom.mjGEOM_BOX if p.shape == "box" else mujoco.mjtGeom.mjGEOM_CYLINDER
        body = spec.worldbody.add_body(name=f"prop_{p.name}", pos=p.pos)
        if p.waypoints:
            body.mocap = True
        else:
            body.add_freejoint(name=f"prop_{p.name}_free")
        geom = body.add_geom(name=f"prop_{p.name}_geom", type=shape, size=list(p.size) + [0.0] * (3 - len(p.size)),
                             contype=PROP_CONTYPE, conaffinity=PROP_CONAFFINITY, group=2,
                             rgba=p.rgba)
        if not p.waypoints:
            geom.mass = p.mass

    spec.worldbody.add_light(name="sun", pos=(0.0, 0.0, 30.0), dir=(0.0, 0.0, -1.0),
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=(0.5, 0.5, 0.5), castshadow=False)

    model = spec.compile()

    def bid(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError(f"scene has no body {name!r}")
        return i

    base = bid(robot.cfg.base_link)
    mocap = bid(robot.cfg.mocap_name)
    trunk = [g for g in range(model.ngeom)
             if model.geom_bodyid[g] == base and model.geom_contype[g] != 0]
    prop_bodies = {p.name: bid(f"prop_{p.name}") for p in props}
    prop_mocap = {p.name: int(model.body_mocapid[prop_bodies[p.name]]) for p in props if p.waypoints}
    info = SceneInfo(
        base_body=base, mocap_body=mocap, mocap_id=int(model.body_mocapid[mocap]),
        lidar_body=bid(robot.cfg.lidar_body), imu_body=bid(robot.cfg.imu_body),
        trunk_geoms=trunk, prop_bodies=prop_bodies, prop_mocap_ids=prop_mocap,
        site_mesh_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "site_visual"),
    )
    return model, info


def set_base_pose(model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo,
                  pos, quat_wxyz) -> None:
    """Teleport: put both the mocap handle and the free base at a pose, at rest."""
    data.mocap_pos[info.mocap_id] = pos
    data.mocap_quat[info.mocap_id] = quat_wxyz
    adr = model.jnt_qposadr[model.body_jntadr[info.base_body]]
    data.qpos[adr:adr + 3] = pos
    data.qpos[adr + 3:adr + 7] = quat_wxyz
    vadr = model.jnt_dofadr[model.body_jntadr[info.base_body]]
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def base_pose(data: mujoco.MjData, info: SceneInfo) -> tuple[np.ndarray, np.ndarray]:
    """(pos, quat wxyz) of base_link in the world."""
    return data.xpos[info.base_body].copy(), data.xquat[info.base_body].copy()


def body_frame(data: mujoco.MjData, body: int) -> tuple[np.ndarray, np.ndarray]:
    """(pos, 3x3 rotation) of a body in the world."""
    return data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()


def trunk_in_contact(data: mujoco.MjData, info: SceneInfo) -> bool:
    trunk = set(info.trunk_geoms)
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in trunk or c.geom2 in trunk:
            return True
    return False
