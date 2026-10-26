"""The lidar intensity of each voxel, from `map.pcd`: what shades the picture.

A lidar map's intensity is what makes it legible — floor markings, door
frames, signs, glass, the tape on a stair nosing — and the sim's voxels
carry none of it (an octomap is occupancy). This takes the map's point cloud
(x y z intensity, dp: 19.6 M points) and keeps, per voxel of the sim's grid,
the median intensity of the points inside it; an occupied voxel with no point
inside — 43 % of dp's floor voxels, the octomap's surface sits a voxel off
the cloud's (surface_heights.py) — takes the mean of its 26 neighbours that
have one. The picture (site_tiles.py) then shades every voxel face by it
(visual.intensity_factor). Nothing the lidar, the feet or the planner see
reads this; it is the picture only.

Cached as cache/<map>/voxel_intensity.npz, keyed on the cloud and the world.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np

from .voxel_grid import VoxelGrid
from .world import World

FORMAT = 1
NEIGHBOURS = np.array([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                       if (dx, dy, dz) != (0, 0, 0)], dtype=np.int64)


@dataclass
class VoxelIntensity:
    keys: np.ndarray  # (K,) int64, sorted: linear index of the voxel in the grid
    values: np.ndarray  # (K,) float32
    lo: np.ndarray  # (3,) int64, the grid's lowest absolute voxel index (VoxelGrid.lo)
    shape: tuple[int, int, int]

    def _linear(self, idx: np.ndarray) -> np.ndarray:
        local = np.asarray(idx, dtype=np.int64) - self.lo[None, :]
        nx, ny, nz = self.shape
        inside = np.all((local >= 0) & (local < np.array([nx, ny, nz])[None, :]), axis=1)
        lin = (local[:, 0] * ny + local[:, 1]) * nz + local[:, 2]
        return np.where(inside, lin, -1)

    def lookup(self, idx: np.ndarray) -> np.ndarray:
        """(N,) intensity of absolute voxel indices (N, 3); NaN where unknown."""
        lin = self._linear(idx)
        pos = np.searchsorted(self.keys, lin)
        pos = np.minimum(pos, max(self.keys.shape[0] - 1, 0))
        out = np.full(lin.shape[0], np.nan, dtype=np.float32)
        if self.keys.shape[0] == 0:
            return out
        hit = (lin >= 0) & (self.keys[pos] == lin)
        out[hit] = self.values[pos[hit]]
        return out


def _median_per_key(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((values, keys))
    k, v = keys[order], values[order]
    starts = np.flatnonzero(np.concatenate(([True], k[1:] != k[:-1])))
    counts = np.diff(np.concatenate((starts, [k.shape[0]])))
    return k[starts], v[starts + counts // 2]


def build(points: np.ndarray, intensity: np.ndarray, grid: VoxelGrid) -> VoxelIntensity:
    """Per voxel of `grid` (every voxel a point falls in, occupied or not): the
    median intensity of its points; then every occupied voxel without one
    takes the mean of its 26 neighbours that have one."""
    nx, ny, nz = grid.grid.shape
    local = np.floor((points.astype(np.float64) - grid.origin_m[None, :]) / grid.res).astype(np.int64)
    inside = np.all((local >= 0) & (local < np.array([nx, ny, nz])[None, :]), axis=1)
    local = local[inside]
    lin = (local[:, 0] * ny + local[:, 1]) * nz + local[:, 2]
    keys, values = _median_per_key(lin, intensity[inside].astype(np.float32))
    table = VoxelIntensity(keys, values, grid.lo.astype(np.int64), (nx, ny, nz))

    occ = np.argwhere(grid.grid).astype(np.int64)  # local indices of occupied voxels
    occ_abs = occ + grid.lo[None, :]
    have = ~np.isnan(table.lookup(occ_abs))
    missing = occ_abs[~have]
    if missing.shape[0]:
        total = np.zeros(missing.shape[0], np.float64)
        count = np.zeros(missing.shape[0], np.int64)
        for d in NEIGHBOURS:
            v = table.lookup(missing + d[None, :])
            ok = ~np.isnan(v)
            total[ok] += v[ok]
            count[ok] += 1
        filled = count > 0
        add_lin = table._linear(missing[filled])
        keys = np.concatenate([table.keys, add_lin])
        values = np.concatenate([table.values, (total[filled] / count[filled]).astype(np.float32)])
        order = np.argsort(keys, kind="stable")
        table = VoxelIntensity(keys[order], values[order], table.lo, table.shape)
    return table


def _key(pcd_path: str, world: World) -> str:
    st = os.stat(pcd_path)
    return json.dumps({"format": FORMAT, "pcd": [st.st_size, int(st.st_mtime)],
                       "world": world.stats.cache_key, "world_options": world.stats.options},
                      sort_keys=True)


def load_or_build(map_dir: str, cache_dir: str, world: World, log=print) -> VoxelIntensity | None:
    """The map's voxel intensities, cached; None without map.pcd (the picture
    is then shaded by height alone)."""
    from .pcd_mesh import read_pcd

    pcd_path = os.path.join(map_dir, "map.pcd")
    if not os.path.isfile(pcd_path):
        log(f"[intensity] {pcd_path} not found; the picture is shaded by height alone")
        return None
    key = _key(pcd_path, world)
    path = os.path.join(cache_dir, "voxel_intensity.npz")
    if os.path.isfile(path):
        z = np.load(path)
        if str(z["key"]) == key:
            return VoxelIntensity(z["keys"], z["values"], z["lo"], tuple(int(s) for s in z["shape"]))
    t0 = time.perf_counter()
    xyz, intensity = read_pcd(pcd_path, intensity=True)
    table = build(xyz, intensity, world.grid)
    occ = int(world.grid.n_voxels)
    known = int((~np.isnan(table.lookup(np.argwhere(world.grid.grid) + world.grid.lo[None, :]))).sum())
    log(f"[intensity] {xyz.shape[0]} points -> {table.keys.shape[0]} voxels with a value, "
        f"{known} of {occ} occupied voxels shaded, in {time.perf_counter() - t0:.1f} s")
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, key=key, keys=table.keys, values=table.values, lo=table.lo,
             shape=np.array(table.shape))
    os.replace(tmp, path)
    return table
