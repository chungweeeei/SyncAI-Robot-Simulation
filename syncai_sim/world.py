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

import numba as nb
import numpy as np

from . import btree, greedy_mesh, visual
from .storeys import storey_bounds, storey_of
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
    # Measured per storey (storeys.py): over the column's lowest voxel within
    # its storey, so an upper floor is drawn whole and not cut away as "more
    # than 3 m above" the floor below it (dp's 2F is 5.3 m over its 1F).
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
    # Fill floor holes up to this many columns (0.1 m cells; 0 = off): a region
    # of columns with no surface, enclosed on every side by floor within a
    # 0.5 m height band, gets a floor voxel at the median height of the floor
    # around it. They are what the lidar never saw — the blind cone under the
    # mapper, under furniture — and dp's octomap holds 3965 of them, the same
    # count before and after every step above (2539 single columns, ~1190 of
    # 2-9, 118 of 10-24; 2026-10-23): seen as holes in the viewer, rays went
    # through them, and a walking foot fits in anything over one column.
    # Larger regions are left: dp's 40 of 1 m^2 and more include stairwells.
    floor_hole_max_cells: int = 25


# Bumped when the cache's contents change shape without an option changing
# (2: the visual mesh's per-face directions, visual.py; 3: storeys; 4: the
# track's one-voxel ridges cut, _cut_track_ridges).
CACHE_FORMAT = 4


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
    mesh_dirs: np.ndarray  # (F,) int8, visual.DIRECTIONS
    mesh_storeys: np.ndarray  # (F,) int8, 0 = the lowest storey
    storey_bounds: list[float]  # heights between storeys (storeys.py); [] = one

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
    grid &= ~erase
    _cut_track_ridges(grid, track, grounds, resolution, origin, dx, dy, window)
    keep = grid[idx[:, 0], idx[:, 1], idx[:, 2]]
    return voxels[keep]


def _cut_track_ridges(grid: np.ndarray, track: np.ndarray, grounds: np.ndarray, resolution: float,
                      origin: np.ndarray, dx: np.ndarray, dy: np.ndarray, window: int,
                      flat_m: float = 0.1, reach_m: float = 0.5) -> None:
    """In place: on the flat stretches of the track, cut one-voxel-thick
    ridges inside the swept disc down to their lower neighbour.

    The floor margin above keeps what is shorter than a riser, which is what
    the stairs need and what lets a thin sliver up to ~0.2 m survive on a
    floor: dp's corridor fence at x -7.6 (y -15.1 .. -15.9) kept its lower
    0.2-0.3 m, the kinematic base stepped it (its allowed step is 0.22) and the
    walking robot fell on it in both directions (2026-10-24). A ridge is a
    column whose top stands at least a voxel over both its x neighbours or
    both its y neighbours, cut to the higher of those two: a sliver is one voxel thick, a step, a tread or a
    wall is not (a terrace line of the tilted floor has one lower side only).
    Flat means the keyframes within `window` have grounds within `flat_m`, so
    a flight is never touched."""
    nx, ny, nz = grid.shape
    cols: dict[tuple[int, int], float] = {}
    for k, (x, y, _z) in enumerate(track):
        win = grounds[max(0, k - window):k + window + 1]
        if win.max() - win.min() > flat_m:
            continue
        cx = int(math.floor((x - origin[0]) / resolution))
        cy = int(math.floor((y - origin[1]) / resolution))
        for i, j in zip(cx + dx, cy + dy):
            if 1 <= i < nx - 1 and 1 <= j < ny - 1:
                cols.setdefault((int(i), int(j)), float(grounds[k]))
    for (i, j), ground in cols.items():
        hi = min(nz - 1, int(math.floor((ground + reach_m - origin[2]) / resolution)))
        lo = max(0, int(math.floor((ground - reach_m - origin[2]) / resolution)))

        def top(a, b):
            ks = np.nonzero(grid[a, b, lo:hi + 1])[0]
            return lo + int(ks[-1]) if ks.size else -1

        t = top(i, j)
        if t < 0:
            continue
        pairs = ((top(i - 1, j), top(i + 1, j)), (top(i, j - 1), top(i, j + 1)))
        # Down to the higher of the two lower neighbours: never below the
        # floor beside it (no pit at a terrace line).
        floor = max((max(a, b) for a, b in pairs if a >= 0 and b >= 0 and max(a, b) < t),
                    default=-1)
        if floor >= 0:
            grid[i, j, floor + 1:t + 1] = False


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


def fill_floor_holes(voxels: np.ndarray, max_cells: int, band: int = 5, step: int = 2) -> np.ndarray:
    """Fill enclosed floor holes of at most `max_cells` columns (see
    WorldOptions.floor_hole_max_cells). Bands of `band` voxels every `step`
    voxels, bottom up: overlapping, so a hole in a floor that drifts by a
    voxel over the site still has its whole rim in one band, and a hole filled
    in one band is floor for the next. Returns the extended voxel set."""
    if max_cells <= 0 or voxels.shape[0] == 0:
        return voxels
    v = voxels.astype(np.int64)
    lo = v.min(axis=0) - np.array([1, 1, 0])  # a free border: holes touching it are not enclosed
    hi = v.max(axis=0) + np.array([2, 2, 1])
    grid = np.zeros(tuple((hi - lo).tolist()), dtype=np.bool_)
    idx = v - lo
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    n0 = int(v.shape[0])
    for z0 in range(0, grid.shape[2], step):
        _fill_band(grid, z0, min(z0 + band, grid.shape[2]), max_cells)
    if int(grid.sum()) == n0:
        return voxels
    return np.argwhere(grid) + lo


@nb.njit(cache=True)
def _fill_band(grid, z0, z1, max_cells):
    nx, ny, nz = grid.shape
    top = np.full((nx, ny), -1, np.int64)
    for i in range(nx):
        for j in range(ny):
            for z in range(z1 - 1, z0 - 1, -1):
                if grid[i, j, z] and (z + 1 >= nz or not grid[i, j, z + 1]):
                    top[i, j] = z
                    break
    seen = np.zeros((nx, ny), np.bool_)
    stack = np.empty(nx * ny, np.int64)
    cells = np.empty(nx * ny, np.int64)
    rim = np.empty(4 * max_cells, np.int64)
    for si in range(nx):
        for sj in range(ny):
            if top[si, sj] >= 0 or seen[si, sj]:
                continue
            seen[si, sj] = True
            stack[0] = si * ny + sj
            sp, n, border = 1, 0, False
            while sp:
                sp -= 1
                c = stack[sp]
                cells[n] = c
                n += 1
                i, j = c // ny, c % ny
                if i == 0 or j == 0 or i == nx - 1 or j == ny - 1:
                    border = True
                for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= a < nx and 0 <= b < ny and top[a, b] < 0 and not seen[a, b]:
                        seen[a, b] = True
                        stack[sp] = a * ny + b
                        sp += 1
            if border or n > max_cells:
                continue
            m = 0
            for k in range(n):
                i, j = cells[k] // ny, cells[k] % ny
                for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if top[a, b] >= 0:
                        rim[m] = top[a, b]
                        m += 1
            h = np.sort(rim[:m])[m // 2]
            for k in range(n):
                grid[cells[k] // ny, cells[k] % ny, h] = True


def exposed_face_mesh(voxels: np.ndarray, resolution: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangles of every voxel face that borders empty space, with coplanar
    neighbours merged into rectangles. Returns (vertices (V, 3) float32 metres,
    faces (F, 3) int32 wound counter-clockwise seen from outside, directions
    (F,) int8 indexing visual.DIRECTIONS). Vertices are shared only between
    faces of one direction, so that per-vertex normals stay flat (visual.py)."""
    if voxels.shape[0] == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros(0, np.int8)
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
    dirs = []
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
        dirs.append(np.full(n, visual.DIRECTIONS.index((axis, sign)), np.int8))
    v = np.concatenate(verts)
    v = (v + lo[None, :] - btree.TREE_MAX_VAL) * resolution
    quad_dir = np.concatenate(dirs)
    # Share vertices between neighbouring rectangles of one direction (a
    # 7 M -> ~3 M reduction across all six).
    key = np.round(v / (resolution * 0.5)).astype(np.int64)
    key = np.concatenate([key, np.repeat(quad_dir, 4)[:, None].astype(np.int64)], axis=1)
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    vertices = (uniq[:, :3] * (resolution * 0.5)).astype(np.float32)
    q = inverse.reshape(-1, 4)
    faces = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).astype(np.int32)
    return vertices, faces, np.concatenate([quad_dir, quad_dir])


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
        if old.get("cache_key") == key and old.get("options") == asdict(opts) \
                and old.get("format") == CACHE_FORMAT:
            z = np.load(data_path)
            stats = WorldStats(**{k: v for k, v in old.items() if k != "format"})
            grid = VoxelGrid(z["voxels"], float(z["resolution"]), opts.grid_z_clip_above_lowest_m)
            log(f"[world] {name}: cached ({stats.voxels} voxels, {stats.mesh_triangles} tris, "
                f"{stats.plates} plates)")
            return World(grid, z["mesh_vertices"], z["mesh_faces"], z["plates"], stats,
                         z["mesh_dirs"], z["mesh_storeys"], z["storey_bounds"].tolist())

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
    n_holey = int(voxels.shape[0])
    voxels = fill_floor_holes(voxels, opts.floor_hole_max_cells)
    if voxels.shape[0] != n_holey:
        log(f"[world] {name}: filled {voxels.shape[0] - n_holey} floor voxels into enclosed holes "
            f"of up to {opts.floor_hole_max_cells} columns")
    grid = VoxelGrid(voxels, resolution, opts.grid_z_clip_above_lowest_m)
    kept = voxels[voxels[:, 2] <= grid.lo[2] + grid.grid.shape[2] - 1]
    bounds = storey_bounds(track[:, 2] - opts.lidar_height_m) if track.shape[0] else []
    vertices, faces, dirs, storeys, n_visual = _storey_meshes(
        kept, resolution, bounds, opts.visual_max_height_above_column_floor_m)
    log(f"[world] {name}: {kept.shape[0]} voxels, meshing {n_visual} for the viewer in "
        f"{len(bounds) + 1} storeys (bounds {[round(b, 2) for b in bounds]} m)")
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
             mesh_vertices=vertices, mesh_faces=faces, mesh_dirs=dirs, mesh_storeys=storeys,
             storey_bounds=np.array(bounds, dtype=np.float64), plates=plates)
    with open(stats_path + ".tmp", "w") as f:
        json.dump({**asdict(stats), "format": CACHE_FORMAT}, f, indent=2)
    os.replace(stats_path + ".tmp", stats_path)
    log(f"[world] {name}: built in {parse_s + build_s:.1f} s -> {stats.mesh_triangles} tris, "
        f"{stats.plates} plates, grid {stats.grid_shape} ({stats.grid_mb} MB)")
    return World(grid, vertices, faces, plates, stats, dirs, storeys, bounds)


def _storey_meshes(voxels: np.ndarray, resolution: float, bounds: list[float], max_height_m: float):
    """The visual mesh, storey by storey: each storey's voxels cut at
    `max_height_m` over their column's lowest voxel in that storey, meshed on
    their own. Returns (vertices, faces, dirs, storeys per face, voxel count)."""
    z = (voxels[:, 2] - btree.TREE_MAX_VAL + 0.5) * resolution
    which = storey_of(z, bounds)
    parts, offset, n_visual = [], 0, 0
    for k in range(len(bounds) + 1):
        vis = column_floor_filter(voxels[which == k], resolution, max_height_m)
        n_visual += int(vis.shape[0])
        v, f, d = exposed_face_mesh(vis, resolution)
        parts.append((v, f + offset, d, np.full(f.shape[0], k, np.int8)))
        offset += v.shape[0]
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]), n_visual)


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
