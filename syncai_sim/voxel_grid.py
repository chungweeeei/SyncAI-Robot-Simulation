"""The site as a dense voxel grid, and the three queries the sim asks of it.

This is the sim's ground truth for *sensing and footing*: the lidar's rays, the
ground under the robot and whether the trunk is blocked are all answered here,
on the very voxels `octomap.bt` holds. MuJoCo gets a visual copy (`scene.py`)
and the props, but never decides any of these three.

Why not MuJoCo for the rays: `mj_multiRay` scales linearly with the number of
box geoms (dp's 360 k boxes: seconds per scan) and, against the same site as a
mesh, its BVH still needed 325 ms for 20 k rays (8.5 M triangles). The
Amanatides–Woo walk below, compiled by numba and spread over the cores, does
the same 20 k rays in 1.3 ms. It is also exact for the voxel world: a ray
returns the point where it enters the first occupied voxel, which is where a
lidar return off a voxel face would be.

Memory: dp's occupied space spans 100 x 110 x 12 m, 130 MB as a bool array at
0.1 m. Voxels higher than `z_clip_above_lowest_m` over the lowest one are
dropped (a few stray returns above 10 m would otherwise add another 60 MB of
empty slices).

Coordinates: grid index i on an axis covers metres [origin + i * res, origin +
(i + 1) * res). `origin_m` is the lower corner of index (0, 0, 0).
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from .btree import TREE_MAX_VAL


class VoxelGrid:
    def __init__(self, voxels: np.ndarray, resolution: float, z_clip_above_lowest_m: float = 12.0):
        """voxels: (M, 3) absolute OcTree voxel indices (btree.cubes_to_voxels)."""
        if voxels.shape[0] == 0:
            raise ValueError("empty voxel set")
        v = voxels.astype(np.int64)
        if z_clip_above_lowest_m > 0:
            zmin = int(v[:, 2].min())
            v = v[v[:, 2] <= zmin + int(round(z_clip_above_lowest_m / resolution))]
        self.res = float(resolution)
        self.lo = v.min(axis=0)
        hi = v.max(axis=0) + 1
        self.grid = np.zeros(tuple((hi - self.lo).tolist()), dtype=np.bool_)
        idx = v - self.lo
        self.grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        self.origin_m = ((self.lo - TREE_MAX_VAL) * self.res).astype(np.float64)
        self.n_voxels = int(v.shape[0])

    # -- geometry helpers ------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.grid.shape  # type: ignore[return-value]

    def bounds_m(self) -> tuple[np.ndarray, np.ndarray]:
        return self.origin_m.copy(), self.origin_m + np.array(self.grid.shape) * self.res

    def index_of(self, p_m) -> np.ndarray:
        return np.floor((np.asarray(p_m, dtype=np.float64) - self.origin_m) / self.res).astype(np.int64)

    # -- queries ---------------------------------------------------------------
    def cast(self, origin_m, dirs: np.ndarray, max_range: float) -> tuple[np.ndarray, np.ndarray]:
        """Ray-cast unit `dirs` (N, 3) from `origin_m`. Returns (dist, points):
        the distance along each ray to the first occupied voxel (-1 where
        nothing is hit within `max_range` metres) and the return point (N, 3)
        in metres (undefined where dist < 0).

        The point is where a lidar return off that voxel would plausibly be:
        a millimetre inside the voxel, and, when the voxel is a SURFACE voxel
        (nothing occupied directly above it), at the voxel's mid-height
        instead of on the face that was hit. A mapped floor is a layer of
        voxels whose tops undulate by one voxel; the real floor is somewhere
        inside them, so returns at the faces would put the sim's floor 0 to
        0.1 m above the real one, and a ray grazing the vertical side of a
        voxel one step proud of its neighbour would read 0.1 m above that
        neighbour — which the planner's dynamic overlay marks as an obstacle
        (threshold 0.10). Walls (voxels with occupied voxels above) keep the
        face hit."""
        n = dirs.shape[0]
        dist = np.empty(n, dtype=np.float64)
        pts = np.empty((n, 3), dtype=np.float64)
        _cast(self.grid, self.origin_m, self.res, np.asarray(origin_m, dtype=np.float64),
              np.ascontiguousarray(dirs, dtype=np.float64), float(max_range), dist, pts)
        return dist, pts

    def ground_under(self, x: float, y: float, z_hint: float,
                     search_up: float = 0.3, search_down: float = 3.0) -> float:
        """Top of the highest occupied voxel in column (x, y) whose top is at or
        below z_hint + search_up, looking down to z_hint - search_down.

        The answer is the ground nearest to z_hint among that column and its
        eight neighbours, the centre winning ties within a voxel: a mapped
        floor has one-column holes where nothing was ever seen (the lidar's
        blind cone under the mapper, the base of a handrail post whose lower
        part was never scanned — dp's first flight has one right on the stair
        link), and a stair tread with a cell missing shows the step below
        through the hole (dp's second flight: 1.3 m where the tread is 1.6).
        A foot does not fall through a 0.1 m gap, so neither does the base.
        A real step, up or down, spans the whole patch and is still seen. NaN
        only when the whole 3 x 3 patch is empty (off the map, a real void)."""
        return _ground_under_patch(self.grid, self.origin_m, self.res, x, y, z_hint, search_up,
                                   search_down)

    def disc_blocked(self, cx: float, cy: float, z_ground: float, radius: float,
                     allowed_step: float, z_lo: float, z_hi: float) -> bool:
        """True if the body disc of `radius` at (cx, cy) standing on `z_ground`
        meets an obstacle: a column whose own ground rises more than
        `allowed_step` above z_ground, or any occupied voxel whose centre lies
        in (z_ground + z_lo, z_ground + z_hi] — the trunk's height band."""
        return bool(_disc_blocked(self.grid, self.origin_m, self.res, cx, cy, z_ground, radius,
                                  allowed_step, z_lo, z_hi))


# --- numba kernels -----------------------------------------------------------

@nb.njit(cache=True, parallel=True, fastmath=True)
def _cast(grid, origin, res, pnt, dirs, max_range, out, pts):
    nx, ny, nz = grid.shape
    px = (pnt[0] - origin[0]) / res
    py = (pnt[1] - origin[1]) / res
    pz = (pnt[2] - origin[2]) / res
    tmax_v = max_range / res
    inf = 1e30
    for r in nb.prange(dirs.shape[0]):
        dx = dirs[r, 0]
        dy = dirs[r, 1]
        dz = dirs[r, 2]
        ix = int(math.floor(px))
        iy = int(math.floor(py))
        iz = int(math.floor(pz))
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        sz = 1 if dz > 0 else -1
        tdx = abs(1.0 / dx) if dx != 0 else inf
        tdy = abs(1.0 / dy) if dy != 0 else inf
        tdz = abs(1.0 / dz) if dz != 0 else inf
        tmx = ((ix + (1 if dx > 0 else 0)) - px) / dx if dx != 0 else inf
        tmy = ((iy + (1 if dy > 0 else 0)) - py) / dy if dy != 0 else inf
        tmz = ((iz + (1 if dz > 0 else 0)) - pz) / dz if dz != 0 else inf
        t = 0.0
        hit = -1.0
        while t <= tmax_v:
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                if grid[ix, iy, iz]:
                    hit = t * res + 0.001  # a millimetre inside: see VoxelGrid.cast
                    px_m = pnt[0] + dx * hit
                    py_m = pnt[1] + dy * hit
                    pz_m = pnt[2] + dz * hit
                    surface = iz + 1 >= nz or not grid[ix, iy, iz + 1]
                    if surface:
                        pz_m = origin[2] + (iz + 0.5) * res
                    pts[r, 0] = px_m
                    pts[r, 1] = py_m
                    pts[r, 2] = pz_m
                    break
            else:
                # Outside the grid and heading away on some axis: nothing ahead.
                if ((ix < 0 and dx <= 0) or (ix >= nx and dx >= 0)
                        or (iy < 0 and dy <= 0) or (iy >= ny and dy >= 0)
                        or (iz < 0 and dz <= 0) or (iz >= nz and dz >= 0)):
                    break
            if tmx < tmy:
                if tmx < tmz:
                    ix += sx
                    t = tmx
                    tmx += tdx
                else:
                    iz += sz
                    t = tmz
                    tmz += tdz
            else:
                if tmy < tmz:
                    iy += sy
                    t = tmy
                    tmy += tdy
                else:
                    iz += sz
                    t = tmz
                    tmz += tdz
        out[r] = hit
        if hit < 0:
            pts[r, 0] = 0.0
            pts[r, 1] = 0.0
            pts[r, 2] = 0.0


@nb.njit(cache=True)
def _ground_under(grid, origin, res, x, y, z_hint, search_up, search_down):
    nx, ny, nz = grid.shape
    ix = int(math.floor((x - origin[0]) / res))
    iy = int(math.floor((y - origin[1]) / res))
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
        return math.nan
    # A voxel's top is at origin + (iz + 1) * res; the highest allowed top is
    # z_hint + search_up, so the highest allowed iz is floor((top - origin) / res) - 1.
    top_iz = int(math.floor((z_hint + search_up - origin[2]) / res)) - 1
    bot_iz = int(math.floor((z_hint - search_down - origin[2]) / res))
    if top_iz >= nz:
        top_iz = nz - 1
    if bot_iz < 0:
        bot_iz = 0
    for iz in range(top_iz, bot_iz - 1, -1):
        if grid[ix, iy, iz]:
            return origin[2] + (iz + 1) * res
    return math.nan


@nb.njit(cache=True)
def _ground_under_patch(grid, origin, res, x, y, z_hint, search_up, search_down):
    g = _ground_under(grid, origin, res, x, y, z_hint, search_up, search_down)
    if not math.isnan(g) and abs(g - z_hint) <= res + 1e-6:
        return g  # the centre is within a voxel of where we stand: no doubt
    best = g
    best_d = abs(g - z_hint) if not math.isnan(g) else 1e30
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            gn = _ground_under(grid, origin, res, x + dx * res, y + dy * res, z_hint, search_up,
                               search_down)
            if not math.isnan(gn):
                d = abs(gn - z_hint)
                if d < best_d - 1e-6:
                    best_d = d
                    best = gn
    return best


@nb.njit(cache=True)
def _disc_blocked(grid, origin, res, cx, cy, z_ground, radius, allowed_step, z_lo, z_hi):
    nx, ny, nz = grid.shape
    r_cells = int(math.ceil(radius / res))
    icx = int(math.floor((cx - origin[0]) / res))
    icy = int(math.floor((cy - origin[1]) / res))
    # Occupied voxel CENTRES in (z_ground + z_lo, z_ground + z_hi] block.
    iz_lo = int(math.floor((z_ground + z_lo - origin[2]) / res - 0.5)) + 1
    iz_hi = int(math.floor((z_ground + z_hi - origin[2]) / res - 0.5))
    if iz_lo < 0:
        iz_lo = 0
    if iz_hi >= nz:
        iz_hi = nz - 1
    top_limit = z_ground + allowed_step + 1e-6  # 0.2 - 0.0 > 0.2 in floats otherwise
    r2 = radius * radius
    for ix in range(icx - r_cells, icx + r_cells + 1):
        if ix < 0 or ix >= nx:
            continue
        for iy in range(icy - r_cells, icy + r_cells + 1):
            if iy < 0 or iy >= ny:
                continue
            # Does the column's footprint touch the disc? Nearest point of the
            # cell square to the disc centre, not the cell centre: a wall whose
            # cell centre is just outside the radius still stands in the body.
            x0 = origin[0] + ix * res
            y0 = origin[1] + iy * res
            qx = cx if x0 <= cx <= x0 + res else (x0 if cx < x0 else x0 + res)
            qy = cy if y0 <= cy <= y0 + res else (y0 if cy < y0 else y0 + res)
            ddx = qx - cx
            ddy = qy - cy
            if ddx * ddx + ddy * ddy > r2:
                continue
            for iz in range(iz_lo, iz_hi + 1):
                if grid[ix, iy, iz]:
                    top = origin[2] + (iz + 1) * res
                    if top > top_limit:
                        return True
    return False
