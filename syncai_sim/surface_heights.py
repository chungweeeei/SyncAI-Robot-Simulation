"""Where the floor really is inside its voxel, from the map's point cloud.

The feet stand on terrain_patch.py's boxes, one per voxel, and a voxel's top is
the real surface rounded UP to the 0.1 m grid. On a floor that is not level in
the map that rounding is not harmless: dp is tilted ~0.5 deg (docs/findings.md,
finding 3), so every ~13 m the floor crosses a voxel boundary, and along that
line the lidar's noise puts neighbouring columns on either side of it — a strip
of random 0.1 m blocks and pits. PPO tripped on one at (5.8, 25.9) under a
navigation goal, and walking across it from four directions at the speed the
driver sends (0.56 m/s for a 0.4 command) fell four times out of four
(2026-10-24). The robot's floor there is flat.

So for every surface voxel (occupied, free above) the median height of the
`map.pcd` points inside it is kept as that voxel's real top: neighbouring
floor columns then differ by millimetres, and a stair tread sits at its
measured height rather than the 0.1 / 0.2 m the grid made of a 0.17 m riser.
A surface voxel with fewer than `min_points` points (a filled hole, a filled
riser, the mapper's track floor) has none and takes its neighbours'
(terrain_patch.py). This moves where the feet touch, nothing else: the lidar,
the planner and the kinematic base still read the voxels (CLAUDE.md, rule 2).

Cached as cache/<map>/surface_heights.npz (dp: built in a few seconds).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .pcd_mesh import read_pcd
from .world import World

FORMAT = 1


class SurfaceHeights:
    def __init__(self, shape: tuple[int, int, int], keys: np.ndarray, tops: np.ndarray):
        self.shape = shape
        self.keys = keys  # sorted linear voxel indices (ix * ny + iy) * nz + iz
        self.tops = tops  # (K,) float32 real top, metres

    def linear(self, idx: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.shape
        return (idx[..., 0] * ny + idx[..., 1]) * nz + idx[..., 2]

    def lookup(self, idx: np.ndarray) -> np.ndarray:
        """Real tops (NaN where unknown) for (N, 3) grid indices."""
        out = np.full(idx.shape[0], np.nan, np.float32)
        if self.keys.size == 0 or idx.shape[0] == 0:
            return out
        nx, ny, nz = self.shape
        ok = np.all((idx >= 0) & (idx < np.array(self.shape)), axis=1)
        lin = self.linear(idx[ok])
        pos = np.clip(np.searchsorted(self.keys, lin), 0, self.keys.size - 1)
        hit = self.keys[pos] == lin
        vals = np.full(lin.shape[0], np.nan, np.float32)
        vals[hit] = self.tops[pos[hit]]
        out[ok] = vals
        return out


def build(points: np.ndarray, world: World, min_points: int = 3) -> SurfaceHeights:
    g = world.grid
    shape = np.array(g.grid.shape)
    idx = np.floor((points - g.origin_m) / g.res).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < shape - np.array([0, 0, 1])), axis=1)
    idx, z = idx[ok], points[ok, 2]
    surface = g.grid[idx[:, 0], idx[:, 1], idx[:, 2]] & ~g.grid[idx[:, 0], idx[:, 1], idx[:, 2] + 1]
    idx, z = idx[surface], z[surface]
    sh = SurfaceHeights(tuple(int(s) for s in shape), np.zeros(0, np.int64), np.zeros(0, np.float32))
    lin = sh.linear(idx)
    order = np.lexsort((z, lin))
    lin, z, idx = lin[order], z[order], idx[order]
    keys, first, counts = np.unique(lin, return_index=True, return_counts=True)
    enough = counts >= min_points
    med = z[first + counts // 2]  # sorted by z within each voxel: the median
    # Inside the voxel, by construction; clamp the float edge cases.
    iz = idx[first, 2]
    lo = g.origin_m[2] + iz * g.res
    tops = np.clip(med, lo, lo + g.res).astype(np.float32)
    return SurfaceHeights(sh.shape, keys[enough], tops[enough])


def load_or_build(map_dir: str, cache_dir: str, world: World, log=print) -> SurfaceHeights | None:
    """The cached heights, built first if needed; None without map.pcd (voxel tops then)."""
    pcd_path = os.path.join(map_dir, "map.pcd")
    if not os.path.isfile(pcd_path):
        log(f"[surface] {pcd_path} not found; the feet stand on voxel tops")
        return None
    st = os.stat(pcd_path)
    key = json.dumps({"format": FORMAT, "pcd": [st.st_size, int(st.st_mtime)],
                      "world": world.stats.cache_key, "world_options": world.stats.options},
                     sort_keys=True)
    path = os.path.join(cache_dir, "surface_heights.npz")
    if os.path.isfile(path):
        z = np.load(path)
        if str(z["key"]) == key:
            log(f"[surface] cached ({z['keys'].size} surface voxels with a measured top)")
            return SurfaceHeights(tuple(int(s) for s in z["shape"]), z["keys"], z["tops"])
    sh = build(read_pcd(pcd_path), world)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, key=key, shape=np.array(sh.shape), keys=sh.keys, tops=sh.tops)
    os.replace(tmp, path)
    log(f"[surface] {sh.keys.size} surface voxels with a measured top (median of map.pcd)")
    return sh
