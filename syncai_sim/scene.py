"""Assemble the MuJoCo scene: robot + the site's visual mesh and plates + props.

The world's three forms are built in `world.py`; this module only puts them in
one `MjSpec` with the robot (`robot.py`) and the props and compiles it.

Collision bitmasks (contype / conaffinity), so that exactly the pairs the sim
wants touch:

    world plates   1 / 1      the few big boxes; stop props, never the trunk
    trunk boxes    2 / 2      meet props only (the world is VoxelGrid's job)
    props          2 / 3      meet plates (1 & 3) and the trunk (2 & 2)
    visual mesh    0 / 0      the picture's copy of the site; collides with nothing
    terrain patch  0 / 4      articulated robot only: voxel boxes the legs (4 / 0)
                              and the trunk (6 / 2) stand on (terrain_patch.py)
    stairs         0 / 4      the same, the fitted flights as fixed boxes (stairs.py)

Geom groups: 0 site mesh (the tiles web_view.py sends the browser), 1 robot
STL visuals (sent too), 2 props, 3 collision primitives and plates, 4 the
mocap handle. Nothing renders in the container; the groups are what the
browser viewer picks its meshes by, and what MuJoCo's own viewer would show
if the MJB were opened in it. The lidar never asks MuJoCo about any of them —
rays go to VoxelGrid for the site and to `prop_rays.py` for the props
(mj_multiRay scans every geom per ray; 18 k plates would cost hundreds of
milliseconds a scan).

Props come from the params as a list of dicts (see `PropSpec`): a pushable box
is a free body with mass; a "person" is a mocap cylinder the node walks between
waypoints. Both are what the planner's dynamic overlay and TerrainLayer's
`obstacle_memory` are for.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from . import visual
from .robot import TERRAIN_BIT, RobotModel
from .world import World

# The site's visual meshes are cut into squares of this side, one geom each:
# the browser viewer (web_view.py) sends each as its own GLB and shows or
# hides them by storey, and a tile is a message a websocket takes in its
# stride (dp: ~1000 tiles for 2.5 M triangles, under a second). The GLFW
# window of before drew only the tiles within 30 m for the same reason
# (llvmpipe's cost was per triangle).
TILE_M = 10.0
# Bumped when the compiled scene changes without any input changing: the MJB
# cache (cached_model) is keyed on it.
SCENE_FORMAT = 1

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
    site_geoms: list[int] = field(default_factory=list)  # the visual site, one per tile and kind
    site_storeys: dict[int, list[int]] = field(default_factory=dict)  # storey -> its site geoms
    # (G, 4) per site geom: storey, tile centre x, y, half diagonal (m)
    site_tiles: np.ndarray | None = None
    storey_bounds: list[float] = field(default_factory=list)
    terrain_mocap_ids: np.ndarray | None = None  # (capacity,) mocap indices of the terrain pool
    terrain_geoms: np.ndarray | None = None  # (capacity,) their geom ids, consecutive
    stair_geoms: list[int] = field(default_factory=list)  # the fixed stairs (stairs.py)


INTEGRATORS = {"euler": mujoco.mjtIntegrator.mjINT_EULER,
               "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST}


def build_scene(world: World, robot: RobotModel, props: list[PropSpec],
                timestep: float = 0.005, terrain_capacity: int = 0, site=None,
                stairs: np.ndarray | None = None, integrator: str = "euler",
                terrain_solref: tuple[float, float] = (0.02, 1.0)) -> tuple[mujoco.MjModel, SceneInfo]:
    """`site`: a pcd_mesh.PcdMesh to draw instead of the voxel mesh; `stairs`:
    (K, 6) boxes, centre and half-sizes (stairs.StairBoxes.boxes);
    `integrator`: euler (MuJoCo's default) or implicitfast, which integrates
    velocity-dependent forces — joint damping, a position actuator's kv —
    implicitly (the pair for `actuator: position`, robot.py);
    `terrain_solref`: the terrain and stair boxes' contact solref, the feet's
    (RobotConfig.foot_solref) so that the pair's mean is that value."""
    if integrator not in INTEGRATORS:
        raise ValueError(f"integrator {integrator!r}: {' | '.join(INTEGRATORS)}")
    spec = mujoco.MjSpec.from_string(robot.mjcf)
    spec.option.timestep = timestep
    spec.option.integrator = INTEGRATORS[integrator]
    spec.option.gravity[:] = (0.0, 0.0, -9.81)
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.map.znear = 0.02
    spec.visual.map.zfar = 400.0  # a site is 100 m across; the default 50 clips it
    # Lights, sky and haze for whoever opens the MJB in MuJoCo's own viewer
    # (nothing in the container renders it; the browser viewer lights its
    # own scene): a dim headlight plus two oblique suns from different
    # azimuths (below), so walls facing x and facing y differ in brightness,
    # which is most of what makes a voxel site read as 3D.
    spec.visual.headlight.ambient[:] = (0.30, 0.30, 0.30)
    spec.visual.headlight.diffuse[:] = (0.35, 0.35, 0.35)
    spec.visual.headlight.specular[:] = (0.0, 0.0, 0.0)
    spec.visual.rgba.haze[:] = (*visual.SKY_TOP, 1.0)
    sky = spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=256, height=256)
    sky.rgb1[:] = visual.SKY_TOP
    sky.rgb2[:] = visual.SKY_BOTTOM
    spec.stat.extent = 60.0  # frames the whole site in MuJoCo's viewer's default zoom
    spec.stat.center[:] = (np.array(world.stats.bounds_min_m) + world.stats.bounds_max_m) / 2

    # The site, visual only: one mesh per face direction, coloured by height (visual.py).
    if site is not None:
        add_site_meshes(spec, site.vertices, site.faces, site.kinds, lambda k: k, site.storeys)
    else:
        add_site_meshes(spec, world.mesh_vertices, world.mesh_faces, world.mesh_dirs,
                        visual.kind_of, world.mesh_storeys)

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
                      conaffinity=TERRAIN_CONAFFINITY, group=3, rgba=(0.9, 0.4, 0.2, 0.5),
                      solref=terrain_solref)
    for i, b in enumerate(np.zeros((0, 6)) if stairs is None else stairs):
        spec.worldbody.add_geom(name=f"stair{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=tuple(float(v) for v in b[:3]),
                                size=tuple(float(v) for v in b[3:]), contype=TERRAIN_CONTYPE,
                                conaffinity=TERRAIN_CONAFFINITY, group=3, rgba=(0.2, 0.6, 0.9, 0.5),
                                solref=terrain_solref)

    for name, d, diffuse in (("sun", (0.35, 0.6, -1.0), 0.45), ("fill", (-0.6, -0.25, -0.8), 0.25)):
        spec.worldbody.add_light(name=name, pos=(0.0, 0.0, 30.0), dir=d,
                                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                 diffuse=(diffuse,) * 3, specular=(0.0, 0.0, 0.0),
                                 castshadow=False)

    model = spec.compile()
    return model, scene_info(model, robot, props, terrain_capacity, list(world.storey_bounds))


_SITE_GEOM = re.compile(r"^site_visual\d+_s(\d+)_t(-?\d+)_(-?\d+)$")
_STAIR_GEOM = re.compile(r"^stair\d+$")


def scene_info(model: mujoco.MjModel, robot: RobotModel, props: list[PropSpec],
               terrain_capacity: int, storey_bounds: list[float]) -> SceneInfo:
    """The ids the sim needs, read back from a compiled (or loaded) model by name."""
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
    site_geoms, storeys, tiles = [], {}, []
    for g in range(model.ngeom):
        mt = _SITE_GEOM.match(model.geom(g).name)
        if mt is None:
            continue
        st, ix, iy = (int(v) for v in mt.groups())
        site_geoms.append(g)
        storeys.setdefault(st, []).append(g)
        tiles.append((st, (ix + 0.5) * TILE_M, (iy + 0.5) * TILE_M, TILE_M * math.sqrt(0.5)))
    return SceneInfo(
        base_body=base, mocap_body=mocap,
        mocap_id=int(model.body_mocapid[mocap]) if mocap >= 0 else -1,
        lidar_body=bid(robot.cfg.lidar_body), imu_body=bid(robot.cfg.imu_body),
        trunk_geoms=trunk, prop_bodies=prop_bodies, prop_mocap_ids=prop_mocap,
        site_geoms=site_geoms, site_storeys=storeys,
        site_tiles=np.array(tiles, dtype=np.float64).reshape(-1, 4),
        storey_bounds=list(storey_bounds),
        terrain_mocap_ids=np.array([model.body_mocapid[b] for b in terrain_bodies], dtype=int),
        terrain_geoms=np.array([model.body_geomadr[b] for b in terrain_bodies], dtype=int),
        stair_geoms=[g for g in range(model.ngeom) if _STAIR_GEOM.match(model.geom(g).name)],
    )


def scene_key(world: World, robot: RobotModel, props: list[PropSpec], timestep: float,
              terrain_capacity: int, visual_mesh: str, cache_dir: str,
              stairs: np.ndarray | None = None, integrator: str = "euler",
              terrain_solref: tuple[float, float] = (0.02, 1.0)) -> str:
    """Everything the compiled scene depends on, as one string (cached_model).
    The robot's config is in it whole (asdict), so every RobotConfig knob —
    motor model, inertial, contact, actuators — is a different scene."""
    from . import pcd_mesh
    from .world import CACHE_FORMAT

    urdf = os.stat(robot.cfg.urdf_path)
    # The real path: the same URDF reached through a symlinked workspace is
    # the same scene. Where the decimated meshes are cached does not change
    # the model either.
    robot_cfg = asdict(robot.cfg)
    robot_cfg["urdf_path"] = os.path.realpath(robot.cfg.urdf_path)
    robot_cfg.pop("mesh_cache_dir")
    site = {"mesh": visual_mesh}
    if visual_mesh == "pointcloud":
        p = os.path.join(cache_dir, "pcd_mesh.npz")
        st = os.stat(p) if os.path.isfile(p) else None
        site["pcd"] = [pcd_mesh.FORMAT, st.st_size, int(st.st_mtime)] if st else None
    return json.dumps({
        "scene": SCENE_FORMAT, "mujoco": mujoco.__version__, "tile_m": TILE_M,
        "world": [world.stats.cache_key, world.stats.options, CACHE_FORMAT],
        "robot": [urdf.st_size, int(urdf.st_mtime), robot_cfg],
        "props": [asdict(p) for p in props], "timestep": timestep,
        "terrain_capacity": terrain_capacity, "site": site,
        "stairs": [] if stairs is None else np.round(stairs, 4).tolist(),
        "integrator": integrator, "terrain_solref": list(terrain_solref),
    }, sort_keys=True, default=str)


def cached_model(cache_dir: str, key: str, build, log=print) -> mujoco.MjModel:
    """The compiled model for `key` from cache/<map>/scene-<hash>.mjb, else
    `build()` saved there. Compiling the scene takes 4-5 s on dp (2.7 M
    visual triangles, 20 k geoms, measured 2026-10-24); mj_loadModel reads
    the binary in a fraction of that (0.32 s). Older scene-*.mjb files are
    removed: each is 628 MB on dp (MuJoCo keeps a BVH per mesh, and the site
    is 2.5 M triangles) and only the current key's is ever loaded."""
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"scene-{digest}.mjb")
    if os.path.isfile(path):
        t0 = time.perf_counter()
        model = mujoco.MjModel.from_binary_path(path)
        log(f"[scene] cached ({os.path.basename(path)}, {time.perf_counter() - t0:.2f} s)")
        return model
    t0 = time.perf_counter()
    model = build()
    os.makedirs(cache_dir, exist_ok=True)
    for old in glob.glob(os.path.join(cache_dir, "scene-*.mjb")):
        os.remove(old)
    mujoco.mj_saveModel(model, path, None)
    log(f"[scene] compiled in {time.perf_counter() - t0:.1f} s, cached as {os.path.basename(path)}")
    return model


def add_site_meshes(spec: mujoco.MjSpec, vertices: np.ndarray, faces: np.ndarray,
                    groups: np.ndarray, kind_of, storeys: np.ndarray) -> int:
    """One visual geom per (face group, storey, TILE_M tile) — a group is a
    voxel face direction or a point-cloud mesh's surface kind — sharing one
    height-ramp texture; `kind_of(group)` picks the ramp's row. The geom name
    carries the storey and the tile (scene_info reads them back). Returns
    the number of geoms."""
    if faces.shape[0] == 0:
        return 0
    tex = spec.add_texture(name="site_ramp", type=mujoco.mjtTexture.mjTEXTURE_2D)
    rgb = visual.texture_rgb()
    tex.width, tex.height, tex.nchannel = rgb.shape[1], rgb.shape[0], 3
    tex.data = rgb.tobytes()
    mat = spec.add_material(name="site_mat", specular=0.0, shininess=0.0, reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "site_ramp"
    floor = np.isin(groups, [g for g in np.unique(groups) if kind_of(int(g)) == visual.KIND_FLOOR])
    z_range = visual.height_range(vertices[np.unique(faces[floor])][:, 2] if floor.any() else
                                  np.zeros(0))
    # Tile of each face by its centroid; then one sort of the faces by
    # (storey, group, tile) and contiguous runs, not a mask per combination
    # over 2.5 M faces.
    centroid = vertices[faces].mean(axis=1)
    tx = np.floor(centroid[:, 0] / TILE_M).astype(np.int64)
    ty = np.floor(centroid[:, 1] / TILE_M).astype(np.int64)
    tx0, ty0 = tx.min(), ty.min()
    nx, ny = int(tx.max() - tx0 + 1), int(ty.max() - ty0 + 1)
    ng = int(groups.max()) + 1
    key = ((storeys.astype(np.int64) * ng + groups.astype(np.int64)) * nx + (tx - tx0)) * ny + (ty - ty0)
    order = np.argsort(key, kind="stable")
    keys, first, counts = np.unique(key[order], return_index=True, return_counts=True)
    n = 0
    for kk, a, c in zip(keys, first, counts):
        f = faces[order[a:a + c]]
        iy = int(kk % ny + ty0)
        ix = int((kk // ny) % nx + tx0)
        k = int((kk // (ny * nx)) % ng)
        st = int(kk // (ny * nx * ng))
        used, local = np.unique(f, return_inverse=True)
        v = vertices[used]
        mesh_name = f"site_mesh{k}_s{st}_t{ix}_{iy}"
        mesh = spec.add_mesh(name=mesh_name)
        # One direction's faces enclose no volume; "shell" (area) inertia
        # compiles, and a static visual geom's inertia is never used anyway.
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        mesh.uservert = v.reshape(-1).astype(np.float32)
        mesh.userface = local.reshape(-1).astype(np.int32)
        mesh.usertexcoord = visual.texcoords(v[:, 2], kind_of(k), z_range) \
            .reshape(-1).astype(np.float32)
        spec.worldbody.add_geom(name=f"site_visual{k}_s{st}_t{ix}_{iy}",
                                type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name,
                                material="site_mat", contype=0, conaffinity=0, group=0)
        n += 1
    return n


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
