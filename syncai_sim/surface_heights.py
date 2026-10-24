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
The medians are then smoothed, edge-preserving: each takes the median of the
measured tops within `smooth_radius` columns (5 x 5 at 0.1 m) that are on its
own surface (within `same_surface` of it, one voxel up or down). Per-voxel
medians alone left dp's ground floor rough in the tail — adjacent columns
differ by 0.4 cm at the median but by more than 2 cm for 8.7 % of pairs and
more than 4 cm for 3.9 % (lidar noise, two passes drifted apart, people in
the map; 2026-10-24) — and a 0.044 m foot meets one every few steps. A median
window flattens blobs narrower than half of it and keeps a real edge sharp: a
riser (0.17 m) or a kerb is outside `same_surface`, so its other side never
votes, and a floor's gentle metre-scale undulation passes through.

Every surface voxel is smoothed this way, also those with no point inside
(57 % of dp's ground floor had one with at least three points, 75 % near the
mapping track; the lidar's lower edge misses the floor close to the mapper):
it takes the median of the measured tops on its surface within the window.
Only one with no measured neighbour there keeps its voxel top
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

FORMAT = 3  # 2: smoothed; 3: every surface voxel, one point is enough


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


def smooth(raw: SurfaceHeights, targets: np.ndarray, origin_z: float, res: float,
           radius: int = 2, same_surface: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """The edge-preserving median over each target surface voxel's own
    surface (see the module docstring). `targets` are (N, 3) grid indices of
    every surface voxel; a measured one is compared with its own median, an
    unmeasured one with its voxel's middle and a tolerance wider by half a
    voxel (its height is only known to the voxel). Returns (keys, tops) of
    the targets that got a value: a target with no measured neighbour on its
    surface within the window has none (the voxel top serves, terrain_patch.py)."""
    nx, ny, nz = raw.shape
    d = np.arange(-radius, radius + 1)
    offs = np.array([(a, b, c) for a in d for b in d for c in (-1, 0, 1)])
    keys = raw.linear(targets)
    order = np.argsort(keys)
    keys, targets = keys[order], targets[order]
    own = raw.lookup(targets)
    mid = (targets[:, 2] + 0.5).astype(np.float32)  # voxel units
    out = np.full(keys.size, np.nan, np.float32)
    chunk = 50_000
    for a in range(0, keys.size, chunk):
        p = targets[a:a + chunk]
        near = raw.lookup((p[:, None, :] + offs[None]).reshape(-1, 3)).reshape(p.shape[0], -1)
        ref = own[a:a + chunk]
        tol = np.where(np.isnan(ref), same_surface + 0.5 * res, same_surface)
        ref = np.where(np.isnan(ref), origin_z + mid[a:a + chunk] * res, ref)
        near = np.where(np.abs(near - ref[:, None]) <= tol[:, None], near, np.nan)
        has = ~np.all(np.isnan(near), axis=1)
        out[a:a + chunk][has] = np.nanmedian(near[has], axis=1)
    got = ~np.isnan(out)
    return keys[got], out[got]


def build(points: np.ndarray, world: World, min_points: int = 1) -> SurfaceHeights:
    """Per-voxel medians of the points (`min_points`: one is a measurement;
    the smoothing removes its noise), then smoothed over every surface voxel."""
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
    raw = SurfaceHeights(sh.shape, keys[enough], tops[enough])
    occ = g.grid
    surf = occ.copy()
    surf[:, :, :-1] &= ~occ[:, :, 1:]
    targets = np.argwhere(surf).astype(np.int64)
    skeys, stops = smooth(raw, targets, float(g.origin_m[2]), float(g.res))
    return SurfaceHeights(sh.shape, skeys, stops)


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
