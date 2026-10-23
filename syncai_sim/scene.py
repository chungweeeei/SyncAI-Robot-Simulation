"""Assemble the MuJoCo scene: robot + the site's visual mesh and plates + props.

The world's three forms are built in `world.py`; this module only puts them in
one `MjSpec` with the robot (`robot.py`) and the props and compiles it.

Collision bitmasks (contype / conaffinity), so that exactly the pairs the sim
wants touch:

    world plates   1 / 1      the few big boxes; stop props, never the trunk
    trunk boxes    2 / 2      meet props only (the world is VoxelGrid's job)
    props          2 / 3      meet plates (1 & 3) and the trunk (2 & 2)
    visual mesh    0 / 0      the viewer's copy of the site; collides with nothing
    terrain patch  0 / 4      articulated robot only: voxel boxes the legs (4 / 0)
                              and the trunk (6 / 2) stand on (terrain_patch.py)

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

from . import visual
from .robot import TERRAIN_BIT, RobotModel
from .world import World

PLATE_CONTYPE, PLATE_CONAFFINITY = 1, 1
PROP_CONTYPE, PROP_CONAFFINITY = 2, 3
TERRAIN_CONTYPE, TERRAIN_CONAFFINITY = 0, TERRAIN_BIT
# Unused terrain boxes wait here, 1 m apart so that no two bounding boxes
# overlap in the broadphase, far below any map.
TERRAIN_PARK_Z = -1000.0


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
    mocap_body: int  # -1 for the articulated robot (no mocap handle)
    mocap_id: int  # index into data.mocap_pos; -1 likewise
    lidar_body: int
    imu_body: int
    trunk_geoms: list[int]
    prop_bodies: dict[str, int]
    prop_mocap_ids: dict[str, int]  # movers only
    prop_group: int = 2
    site_geoms: list[int] = field(default_factory=list)  # the visual site, one per face direction
    terrain_mocap_ids: np.ndarray | None = None  # (capacity,) mocap indices of the terrain pool
    terrain_geoms: np.ndarray | None = None  # (capacity,) their geom ids, consecutive


def build_scene(world: World, robot: RobotModel, props: list[PropSpec],
                timestep: float = 0.005, terrain_capacity: int = 0, site=None
                ) -> tuple[mujoco.MjModel, SceneInfo]:
    """`site`: a pcd_mesh.PcdMesh to draw instead of the voxel mesh."""
    spec = mujoco.MjSpec.from_string(robot.mjcf)
    spec.option.timestep = timestep
    spec.option.gravity[:] = (0.0, 0.0, -9.81)
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.map.znear = 0.02
    spec.visual.map.zfar = 400.0  # a site is 100 m across; the default 50 clips it
    # A dim headlight plus two oblique suns from different azimuths (below):
    # walls facing x and facing y then differ in brightness, which is most of
    # what makes a voxel site read as 3D. No shadows: llvmpipe.
    spec.visual.headlight.ambient[:] = (0.30, 0.30, 0.30)
    spec.visual.headlight.diffuse[:] = (0.35, 0.35, 0.35)
    spec.visual.headlight.specular[:] = (0.0, 0.0, 0.0)
    spec.visual.rgba.haze[:] = (*visual.SKY_TOP, 1.0)
    sky = spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=256, height=256)
    sky.rgb1[:] = visual.SKY_TOP
    sky.rgb2[:] = visual.SKY_BOTTOM
    spec.stat.extent = 60.0  # frames the whole site in the viewer's default zoom
    spec.stat.center[:] = (np.array(world.stats.bounds_min_m) + world.stats.bounds_max_m) / 2

    # The site, visual only: one mesh per face direction, coloured by height (visual.py).
    if site is not None:
        site_names = add_site_meshes(spec, site.vertices, site.faces, site.kinds, lambda k: k)
    else:
        site_names = add_site_meshes(spec, world.mesh_vertices, world.mesh_faces, world.mesh_dirs,
                                     visual.kind_of)

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

    # Terrain pool (terrain_patch.py moves the boxes onto the voxels).
    half = world.grid.res / 2.0 - 0.001
    for i in range(terrain_capacity):
        body = spec.worldbody.add_body(name=f"terrain{i}", mocap=True,
                                       pos=(float(i), 0.0, TERRAIN_PARK_Z))
        body.add_geom(name=f"terrain{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=(half, half, half), contype=TERRAIN_CONTYPE,
                      conaffinity=TERRAIN_CONAFFINITY, group=3, rgba=(0.9, 0.4, 0.2, 0.5))

    for name, d, diffuse in (("sun", (0.35, 0.6, -1.0), 0.45), ("fill", (-0.6, -0.25, -0.8), 0.25)):
        spec.worldbody.add_light(name=name, pos=(0.0, 0.0, 30.0), dir=d,
                                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                 diffuse=(diffuse,) * 3, specular=(0.0, 0.0, 0.0),
                                 castshadow=False)

    model = spec.compile()

    def bid(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError(f"scene has no body {name!r}")
        return i

    base = bid(robot.cfg.base_link)
    mocap = -1 if robot.cfg.articulated else bid(robot.cfg.mocap_name)
    trunk = [g for g in range(model.ngeom)
             if model.geom_bodyid[g] == base and model.geom_contype[g] != 0]
    terrain_bodies = [bid(f"terrain{i}") for i in range(terrain_capacity)]
    prop_bodies = {p.name: bid(f"prop_{p.name}") for p in props}
    prop_mocap = {p.name: int(model.body_mocapid[prop_bodies[p.name]]) for p in props if p.waypoints}
    info = SceneInfo(
        base_body=base, mocap_body=mocap,
        mocap_id=int(model.body_mocapid[mocap]) if mocap >= 0 else -1,
        lidar_body=bid(robot.cfg.lidar_body), imu_body=bid(robot.cfg.imu_body),
        trunk_geoms=trunk, prop_bodies=prop_bodies, prop_mocap_ids=prop_mocap,
        site_geoms=[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in site_names],
        terrain_mocap_ids=np.array([model.body_mocapid[b] for b in terrain_bodies], dtype=int),
        terrain_geoms=np.array([model.body_geomadr[b] for b in terrain_bodies], dtype=int),
    )
    return model, info


def add_site_meshes(spec: mujoco.MjSpec, vertices: np.ndarray, faces: np.ndarray,
                    groups: np.ndarray, kind_of) -> list[str]:
    """One visual geom per face group (a voxel face direction, or a point-cloud
    mesh's surface kind), sharing one height-ramp texture; `kind_of(group)`
    picks the ramp's row. Returns the geom names."""
    if faces.shape[0] == 0:
        return []
    tex = spec.add_texture(name="site_ramp", type=mujoco.mjtTexture.mjTEXTURE_2D)
    rgb = visual.texture_rgb()
    tex.width, tex.height, tex.nchannel = rgb.shape[1], rgb.shape[0], 3
    tex.data = rgb.tobytes()
    mat = spec.add_material(name="site_mat", specular=0.0, shininess=0.0, reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "site_ramp"
    floor = np.isin(groups, [g for g in np.unique(groups) if kind_of(int(g)) == visual.KIND_FLOOR])
    z_range = visual.height_range(vertices[np.unique(faces[floor])][:, 2] if floor.any() else
                                  np.zeros(0))
    names = []
    for k in np.unique(groups):
        f = faces[groups == k]
        if f.shape[0] == 0:
            continue
        used, local = np.unique(f, return_inverse=True)
        v = vertices[used]
        mesh = spec.add_mesh(name=f"site_mesh{k}")
        # One direction's faces enclose no volume; "shell" (area) inertia
        # compiles, and a static visual geom's inertia is never used anyway.
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        mesh.uservert = v.reshape(-1).astype(np.float32)
        mesh.userface = local.reshape(-1).astype(np.int32)
        mesh.usertexcoord = visual.texcoords(v[:, 2], kind_of(int(k)), z_range) \
            .reshape(-1).astype(np.float32)
        name = f"site_visual{k}"
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
                                meshname=f"site_mesh{k}", material="site_mat", contype=0,
                                conaffinity=0, group=0)
        names.append(name)
    return names


def place_terrain(data: mujoco.MjData, info: SceneInfo, centres: np.ndarray) -> None:
    """Move the pool onto `centres` (K, 3); the rest go back to the park."""
    ids = info.terrain_mocap_ids
    k = min(centres.shape[0], ids.shape[0])
    data.mocap_pos[ids[:k]] = centres[:k]
    rest = ids[k:]
    if rest.size:
        data.mocap_pos[rest, 0] = np.arange(k, ids.shape[0], dtype=float)
        data.mocap_pos[rest, 1] = 0.0
        data.mocap_pos[rest, 2] = TERRAIN_PARK_Z


def set_base_pose(model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo,
                  pos, quat_wxyz) -> None:
    """Teleport: put both the mocap handle (if any) and the free base at a pose, at rest."""
    if info.mocap_id >= 0:
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
