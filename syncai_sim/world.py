"""The site, from `map/<name>/octomap.bt`, in the three forms the sim uses.

    btree.read_occupied_leaves   leaf cubes                    (dp: 3-4 s, 1.6 M)
    cubes_to_voxels              (M, 3) voxel indices          -> VoxelGrid (truth)
    exposed faces, greedy-merged (F, 3) triangles + vertices   -> MuJoCo visual mesh
    greedy_mesh.merge_voxels     big boxes ("plates")          -> MuJoCo collision for props

Why three forms. The *voxel grid* is what the lidar, the ground query and the
trunk's blocking test read (`voxel_grid.py`): exact, fast, and the same tree
the Octo3D planner plans on. MuJoCo never sees it. The *visual mesh* is only
for the viewer — MuJoCo renders it, with contype 0 so nothing collides with its
convex hull — and is the exposed voxel faces merged into rectangles (dp: 4.2 M
faces -> 1.75 M quads -> 3.5 M triangles, under 2 M with the height band). The
*plates* are the few large boxes the greedy merge produces (floor slabs, long
walls) kept as real collision geoms so that props — the box the robot pushes,
the walking cylinder — stand on floors and stop at walls; the small fragments
that make up most of a lidar map's boxes (dp: 360 k, median volume 2 voxels)
would take MuJoCo's compiler minutes and are not needed for props.

Everything is cached under cache/<map>/ keyed on the .bt's size and mtime, so a
map is processed once (dp: ~15 s) and loads in about a second afterwards.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import btree, greedy_mesh
from .voxel_grid import VoxelGrid


@dataclass
class WorldOptions:
    # Visual mesh only: drop faces of voxels higher than this above the LOWEST
    # voxel of their (x, y) column. 0 keeps every face. Ceilings and high walls
    # are most of a lidar map's triangles and matter to nothing the viewer
    # shows about navigation, so this is the frame-rate knob; the voxel grid
    # the sensors read is never filtered. The column's lowest voxel is a crude
    # "ground" (under a balcony it is the floor below), hence a knob, not a
    # default.
    visual_max_height_above_column_floor_m: float = 0.0
    # A merged box with at least this many voxels becomes a collision plate.
    # dp: 16 gives 18 k plates (MuJoCo compiles them in half a second) and a
    # floor under every spot tried; 100 gave 870 and let a prop fall through.
    plate_min_voxels: int = 16
    # Voxels higher than this above the lowest one in the map are dropped from
    # the GRID (not only the visual): a few stray returns 15 m up would
    # otherwise add empty slices to the dense array.
    grid_z_clip_above_lowest_m: float = 12.0
    # A column whose top stands at least this much above the HIGHEST top among
    # its eight neighbours is clipped down to that height (0 = off). This
    # removes the mapping robot's own feet: a lidar map carries one-voxel
    # spikes 0.2-0.3 m tall along the track where a leg was in view (the
    # workspace's notes on dp's octomap name them). The planner tolerates them
    # (its footprint-support rule lets 20 % of a disc be unobserved or odd),
    # but a sim lidar that renders them makes TerrainLayer mark a lethal post
    # on the planned path, and on dp MPPI stopped dead 0.3 m before one on
    # the only route to the stairs (2026-10-23) — a thing the real robot never
    # sees, because its lidar looks at the real floor. A real post wider than
    # one voxel or standing next to anything of its height is kept.
    clip_spikes_above_neighbours_m: float = 0.2
    # Erase occupied voxels inside the volume the MAPPING robot's body swept
    # along `poses.txt`: a cylinder of this radius around each keyframe, from
    # `track_clear_floor_margin_m` above that keyframe's ground (z - 0.481) to
    # `track_clear_top_m` above its body. The robot was physically there, so
    # nothing solid can be: what the map holds there is its own legs, or a
    # person who walked beside it and survived clean_map. On dp a one-voxel
    # thin "fence" 0.3-0.4 m tall spanned the corridor on the only route to
    # the stairs; the planner's footprint rule tolerated it, TerrainLayer in
    # the sim did not, and MPPI stopped dead (2026-10-23). The radius covers
    # the legs' stance (feet at y = +-0.16) plus the track's own pose error:
    # 0.22 left a 0.6 m gap in that fence and MPPI, following a path 0.1 m
    # off the gap's centre, still touched the stubs. A wall the track passed
    # close to loses at most a notch. On stairs the margin is measured from
    # the HIGHEST ground among the keyframes within `track_clear_window` of
    # this one, so the treads ahead of a climbing robot are never erased
    # (with the margin from the current tread alone, dp's first flight was
    # hollowed out along the track: a 0.2 m quantised riser 0.3 m ahead is
    # exactly what the margin and radius reach). 0 = off.
    track_clear_radius_m: float = 0.30
    track_clear_floor_margin_m: float = 0.20
    track_clear_top_m: float = 0.30
    track_clear_window: int = 3  # keyframes each side (~0.5 m apart on dp) for the local ground
    # The converse of the clearing: put a floor voxel under the mapping robot
    # wherever the map has none. Within this radius of a keyframe, a column
    # with no occupied voxel between 0.25 m below and 0.45 m above that
    # keyframe's ground gets one whose top is that ground. The robot stood
    # there, so there was floor under its stance; the lidar's -7 deg lower
    # edge never sees the floor within ~4 m, and what it missed on a landing
    # it crossed once stays a hole in the map. dp's first landing is such a
    # hole along the planner's path (the planner infers those cells; the sim
    # read them as a void and refused to step, 2026-10-23). A riser column
    # ahead (top 0.2 m up) is inside the window and is left alone. 0 = off.
    track_floor_fill_radius_m: float = 0.30
    lidar_height_m: float = 0.481  # poses.txt is the IMU frame; ground = z - this


@dataclass
class WorldStats:
    map_name: str
    octomap: str
    cache_key: str
    resolution: float
    occupied_leaves: int
    voxels: int
    grid_shape: list[int]
    grid_mb: float
    bounds_min_m: list[float]
    bounds_max_m: list[float]
    mesh_vertices: int
    mesh_triangles: int
    plates: int
    options: dict
    parse_s: float
    build_s: float


@dataclass
class World:
    grid: VoxelGrid
    mesh_vertices: np.ndarray  # (V, 3) float32, metres
    mesh_faces: np.ndarray  # (F, 3) int32
    plates: np.ndarray  # (P, 6) int64 voxel boxes: x0 y0 z0 nx ny nz
    stats: WorldStats

    def plates_metres(self) -> tuple[np.ndarray, np.ndarray]:
        """(centre (P, 3), half-size (P, 3)) in metres."""
        res = self.grid.res
        lo = (self.plates[:, :3] - btree.TREE_MAX_VAL) * res
        size = self.plates[:, 3:6] * res
        return lo + size / 2.0, size / 2.0


def _cache_key(bt_path: str) -> str:
    st = os.stat(bt_path)
    return f"{st.st_size}-{int(st.st_mtime)}"


def column_floor_filter(voxels: np.ndarray, resolution: float, max_height_m: float) -> np.ndarray:
    """Keep voxels at most `max_height_m` above the lowest voxel of their column."""
    if max_height_m <= 0:
        return voxels
    order = np.lexsort((voxels[:, 2], voxels[:, 1], voxels[:, 0]))
    v = voxels[order]
    new_col = np.concatenate(([True], (v[1:, 0] != v[:-1, 0]) | (v[1:, 1] != v[:-1, 1])))
    first = np.maximum.accumulate(np.where(new_col, np.arange(v.shape[0]), 0))
    limit = int(round(max_height_m / resolution))
    return v[(v[:, 2] - v[first, 2]) <= limit]


def clip_spikes(voxels: np.ndarray, resolution: float, min_rise_m: float) -> np.ndarray:
    """Remove the voxels of one-column spikes (see WorldOptions.clip_spikes_above_neighbours_m).

    Per (x, y) column the highest occupied voxel is compared with the highest
    among the 8 neighbouring columns; the column is cut down to that height
    when it stands `min_rise_m` or more above all of them. Columns with no
    occupied neighbour at all are left alone (a lone return has no context).
    """
    if min_rise_m <= 0 or voxels.shape[0] == 0:
        return voxels
    lo = voxels.min(axis=0)
    size = voxels.max(axis=0) - lo + 1
    top = np.full((size[0] + 2, size[1] + 2), -1, dtype=np.int64)  # padded top index per column
    ix, iy, iz = (voxels - lo).T
    np.maximum.at(top, (ix + 1, iy + 1), iz)
    nb = np.full_like(top, -1)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                shifted = np.roll(np.roll(top, dx, axis=0), dy, axis=1)
                nb = np.maximum(nb, shifted)
    rise = int(round(min_rise_m / resolution))
    spike = (top >= 0) & (nb >= 0) & (top - nb >= rise)
    limit = np.where(spike, nb, top)  # per column: keep voxels with iz <= limit
    keep = iz <= limit[ix + 1, iy + 1]
    return voxels[keep]


def read_track(map_dir: str) -> np.ndarray:
    """(K, 3) keyframe positions from map/<name>/poses.txt (x y z of the IMU
    frame in the map), or an empty array when the file is absent."""
    path = os.path.join(map_dir, "poses.txt")
    if not os.path.isfile(path):
        return np.zeros((0, 3))
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 3))


def clear_track_volume(voxels: np.ndarray, resolution: float, track: np.ndarray, radius_m: float,
                       floor_margin_m: float, top_m: float, lidar_height_m: float,
                       window: int = 3) -> np.ndarray:
    """Drop voxels inside the mapping robot's swept body volume (see WorldOptions)."""
    if radius_m <= 0 or track.shape[0] == 0 or voxels.shape[0] == 0:
        return voxels
    grounds = track[:, 2] - lidar_height_m
    # The floor margin starts from the highest ground within `window` keyframes
    # either side: on a flight that is the tread two or three steps up, so the
    # risers ahead survive; on a floor it is the floor.
    local_ground = np.array([grounds[max(0, k - window):k + window + 1].max()
                             for k in range(track.shape[0])])
    lo = voxels.min(axis=0)
    size = voxels.max(axis=0) - lo + 1
    grid = np.zeros(tuple(size.tolist()), dtype=bool)
    idx = voxels - lo
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    erase = np.zeros_like(grid)
    r_cells = int(math.ceil(radius_m / resolution))
    offs = np.arange(-r_cells, r_cells + 1)
    ox, oy = np.meshgrid(offs, offs, indexing="ij")
    # Column offsets whose nearest point to the axis is within the radius.
    near = np.maximum(np.abs(ox) - 0.5, 0.0) ** 2 + np.maximum(np.abs(oy) - 0.5, 0.0) ** 2
    disc = near <= (radius_m / resolution) ** 2
    dx, dy = ox[disc], oy[disc]
    origin = (lo - btree.TREE_MAX_VAL) * resolution  # metres of local index 0 (lower corner)
    for (x, y, z), ground in zip(track, local_ground):
        cx = int(math.floor((x - origin[0]) / resolution))
        cy = int(math.floor((y - origin[1]) / resolution))
        z0 = int(math.floor((ground + floor_margin_m - origin[2]) / resolution))
        z1 = int(math.floor((z + top_m - origin[2]) / resolution))
        z0 = max(z0, 0)
        z1 = min(z1, size[2] - 1)
        if z1 < z0:
            continue
        xs = cx + dx
        ys = cy + dy
        ok = (xs >= 0) & (xs < size[0]) & (ys >= 0) & (ys < size[1])
        erase[xs[ok], ys[ok], z0:z1 + 1] = True
    keep = ~erase[idx[:, 0], idx[:, 1], idx[:, 2]]
    return voxels[keep]


def fill_track_floor(voxels: np.ndarray, resolution: float, track: np.ndarray, radius_m: float,
                     lidar_height_m: float, below_m: float = 0.25, above_m: float = 0.45) -> np.ndarray:
    """Add floor voxels under the mapping robot's stance where the map has none
    (see WorldOptions.track_floor_fill_radius_m). Returns the extended voxel set."""
    if radius_m <= 0 or track.shape[0] == 0 or voxels.shape[0] == 0:
        return voxels
    lo = voxels.min(axis=0)
    hi = voxels.max(axis=0) + 1
    # Room for a floor below the lowest voxel if a keyframe's ground is lower.
    zmin_track = int(math.floor((track[:, 2].min() - lidar_height_m - btree.TREE_MAX_VAL * resolution)
                                / resolution)) - 1 + btree.TREE_MAX_VAL
    lo[2] = min(lo[2], zmin_track)
    size = hi - lo
    grid = np.zeros(tuple(size.tolist()), dtype=bool)
    idx = voxels - lo
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    r_cells = int(math.ceil(radius_m / resolution))
    offs = np.arange(-r_cells, r_cells + 1)
    ox, oy = np.meshgrid(offs, offs, indexing="ij")
    near = np.maximum(np.abs(ox) - 0.5, 0.0) ** 2 + np.maximum(np.abs(oy) - 0.5, 0.0) ** 2
    disc = near <= (radius_m / resolution) ** 2
    dx, dy = ox[disc], oy[disc]
    origin = (lo - btree.TREE_MAX_VAL) * resolution
    added = set()
    for x, y, z in track:
        ground = z - lidar_height_m
        iz_top = int(round((ground - origin[2]) / resolution)) - 1  # voxel whose top is `ground`
        z0 = max(0, int(math.floor((ground - below_m - origin[2]) / resolution)))
        z1 = min(size[2] - 1, int(math.floor((ground + above_m - origin[2]) / resolution)))
        if iz_top < 0 or iz_top >= size[2] or z1 < z0:
            continue
        cx = int(math.floor((x - origin[0]) / resolution))
        cy = int(math.floor((y - origin[1]) / resolution))
        xs = cx + dx
        ys = cy + dy
        ok = (xs >= 0) & (xs < size[0]) & (ys >= 0) & (ys < size[1])
        for ix, iy in zip(xs[ok], ys[ok]):
            if not grid[ix, iy, z0:z1 + 1].any():
                added.add((int(ix), int(iy), int(iz_top)))
    if not added:
        return voxels
    new = np.array(sorted(added), dtype=np.int64) + lo
    return np.concatenate([voxels.astype(np.int64), new])


def exposed_face_mesh(voxels: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Triangles of every voxel face that borders empty space, with coplanar
    neighbours merged into rectangles. Returns (vertices (V, 3) float32 metres,
    faces (F, 3) int32), wound counter-clockwise seen from outside."""
    if voxels.shape[0] == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    lo = voxels.min(axis=0) - 1  # one voxel of padding so every face has a neighbour
    hi = voxels.max(axis=0) + 2
    grid = np.zeros(tuple((hi - lo).tolist()), dtype=bool)
    idx = voxels - lo
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    quads = []  # (axis, sign, rects (R, 6) in local voxel units)
    for axis in range(3):
        for sign in (1, -1):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            if sign > 0:
                a[axis] = slice(0, -1)
                b[axis] = slice(1, None)
            else:
                a[axis] = slice(1, None)
                b[axis] = slice(0, -1)
            cells = np.argwhere(grid[tuple(a)] & ~grid[tuple(b)])
            if sign < 0:
                cells[:, axis] += 1
            if cells.shape[0] == 0:
                continue
            # Merge within each face plane: hand merge_voxels the plane axis as
            # its last ("z") column so the runs form along the other two.
            others = [i for i in range(3) if i != axis]
            rects = greedy_mesh.merge_voxels(cells[:, others + [axis]])
            quads.append((axis, sign, rects))

    verts = []
    for axis, sign, rects in quads:
        others = [i for i in range(3) if i != axis]
        a1, a2 = others
        n = rects.shape[0]
        # rect columns: o1, o2, plane, n1, n2, 1  (in the permuted order)
        corner = np.zeros((n, 4, 3), dtype=np.float64)
        base = np.zeros((n, 3))
        base[:, a1] = rects[:, 0]
        base[:, a2] = rects[:, 1]
        base[:, axis] = rects[:, 2] + (1 if sign > 0 else 0)
        e1 = np.zeros((n, 3))
        e1[:, a1] = rects[:, 3]
        e2 = np.zeros((n, 3))
        e2[:, a2] = rects[:, 4]
        # CCW seen from +normal when (a1, a2, axis) is a right-handed triple.
        right_handed = (a1, a2) in ((1, 2), (2, 0), (0, 1))
        flip = (sign > 0) != right_handed
        c0, c1, c2, c3 = base, base + e1, base + e1 + e2, base + e2
        if flip:
            c1, c3 = c3, c1
        corner[:, 0], corner[:, 1], corner[:, 2], corner[:, 3] = c0, c1, c2, c3
        verts.append(corner.reshape(-1, 3))
    v = np.concatenate(verts)
    v = (v + lo[None, :] - btree.TREE_MAX_VAL) * resolution
    # Share vertices between neighbouring rectangles (a 7 M -> ~3 M reduction).
    key = np.round(v / (resolution * 0.5)).astype(np.int64)
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    vertices = (uniq * (resolution * 0.5)).astype(np.float32)
    q = inverse.reshape(-1, 4)
    faces = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).astype(np.int32)
    return vertices, faces


def build_world(map_dir: str, cache_dir: str, opts: WorldOptions | None = None,
                log=print) -> World:
    """Load the cached world for `map_dir` or build it. `log` gets progress lines."""
    opts = opts or WorldOptions()
    bt_path = os.path.join(map_dir, "octomap.bt")
    if not os.path.isfile(bt_path):
        raise FileNotFoundError(f"{bt_path}: the map has no octomap.bt (clean_map --octomap 1)")
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(bt_path)
    name = os.path.basename(os.path.normpath(map_dir))
    stats_path = os.path.join(cache_dir, "world.json")
    data_path = os.path.join(cache_dir, f"world-{key}.npz")

    if os.path.isfile(stats_path) and os.path.isfile(data_path):
        with open(stats_path) as f:
            old = json.load(f)
        if old.get("cache_key") == key and old.get("options") == asdict(opts):
            z = np.load(data_path)
            stats = WorldStats(**old)
            grid = VoxelGrid(z["voxels"], float(z["resolution"]), opts.grid_z_clip_above_lowest_m)
            log(f"[world] {name}: cached ({stats.voxels} voxels, {stats.mesh_triangles} tris, "
                f"{stats.plates} plates)")
            return World(grid, z["mesh_vertices"], z["mesh_faces"], z["plates"], stats)

    t0 = time.perf_counter()
    leaves_path = os.path.join(cache_dir, f"leaves-{key}.npz")
    if os.path.isfile(leaves_path):
        z = np.load(leaves_path)
        cubes, resolution = z["cubes"], float(z["resolution"])
    else:
        log(f"[world] {name}: reading {bt_path}")
        leaves = btree.read_occupied_leaves(bt_path)
        cubes, resolution = leaves.cubes, leaves.resolution
        np.savez_compressed(leaves_path, cubes=cubes, resolution=resolution)
    parse_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    voxels = btree.cubes_to_voxels(cubes)
    n_raw = int(voxels.shape[0])
    voxels = clip_spikes(voxels, resolution, opts.clip_spikes_above_neighbours_m)
    if voxels.shape[0] != n_raw:
        log(f"[world] {name}: clipped {n_raw - voxels.shape[0]} voxels of one-column spikes")
    track = read_track(map_dir)
    n_before = int(voxels.shape[0])
    voxels = fill_track_floor(voxels, resolution, track, opts.track_floor_fill_radius_m,
                              opts.lidar_height_m)
    if voxels.shape[0] != n_before:
        log(f"[world] {name}: added {voxels.shape[0] - n_before} floor voxels under the mapping "
            f"robot's stance where the map had none")
    n_spiked = int(voxels.shape[0])
    voxels = clear_track_volume(voxels, resolution, track, opts.track_clear_radius_m,
                                opts.track_clear_floor_margin_m, opts.track_clear_top_m,
                                opts.lidar_height_m, opts.track_clear_window)
    if voxels.shape[0] != n_spiked:
        log(f"[world] {name}: erased {n_spiked - voxels.shape[0]} voxels inside the mapping "
            f"robot's body volume along {track.shape[0]} keyframes")
    grid = VoxelGrid(voxels, resolution, opts.grid_z_clip_above_lowest_m)
    kept = voxels[voxels[:, 2] <= grid.lo[2] + grid.grid.shape[2] - 1]
    visual = column_floor_filter(kept, resolution, opts.visual_max_height_above_column_floor_m)
    log(f"[world] {name}: {kept.shape[0]} voxels, meshing {visual.shape[0]} for the viewer")
    vertices, faces = exposed_face_mesh(visual, resolution)
    boxes = greedy_mesh.merge_voxels(kept)
    plates = boxes[np.prod(boxes[:, 3:6], axis=1) >= opts.plate_min_voxels]
    build_s = time.perf_counter() - t1

    lo_m, hi_m = grid.bounds_m()
    stats = WorldStats(
        map_name=name, octomap=os.path.abspath(bt_path), cache_key=key, resolution=resolution,
        occupied_leaves=int(cubes.shape[0]), voxels=int(kept.shape[0]),
        grid_shape=[int(s) for s in grid.grid.shape], grid_mb=round(grid.grid.nbytes / 1e6, 1),
        bounds_min_m=[round(float(v), 3) for v in lo_m],
        bounds_max_m=[round(float(v), 3) for v in hi_m],
        mesh_vertices=int(vertices.shape[0]), mesh_triangles=int(faces.shape[0]),
        plates=int(plates.shape[0]), options=asdict(opts), parse_s=round(parse_s, 2),
        build_s=round(build_s, 2),
    )
    np.savez(data_path, voxels=kept.astype(np.int32), resolution=resolution,
             mesh_vertices=vertices, mesh_faces=faces, plates=plates)
    with open(stats_path + ".tmp", "w") as f:
        json.dump(asdict(stats), f, indent=2)
    os.replace(stats_path + ".tmp", stats_path)
    log(f"[world] {name}: built in {parse_s + build_s:.1f} s -> {stats.mesh_triangles} tris, "
        f"{stats.plates} plates, grid {stats.grid_shape} ({stats.grid_mb} MB)")
    return World(grid, vertices, faces, plates, stats)


def default_cache_dir(map_dir: str) -> str:
    name = os.path.basename(os.path.normpath(map_dir))
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", name)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build (or refresh) the cached world of a map")
    p.add_argument("map_dir", help="map/<name> directory holding octomap.bt")
    p.add_argument("--cache", default=None, help="cache directory (default cache/<name>)")
    p.add_argument("--visual-max-height", type=float, default=0.0,
                   help="viewer mesh: drop faces more than this above their column's floor (m)")
    p.add_argument("--plate-min-voxels", type=int, default=16)
    p.add_argument("--keep-spikes", action="store_true", help="do not clip one-column spikes")
    a = p.parse_args(argv)
    opts = WorldOptions(visual_max_height_above_column_floor_m=a.visual_max_height,
                        plate_min_voxels=a.plate_min_voxels,
                        clip_spikes_above_neighbours_m=0.0 if a.keep_spikes else 0.2)
    world = build_world(a.map_dir, a.cache or default_cache_dir(a.map_dir), opts)
    print(json.dumps(asdict(world.stats), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
