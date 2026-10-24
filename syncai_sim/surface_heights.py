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
What is still without a value then gets a second, wider pass
(`WIDE_RADIUS`, 13 x 13 columns): the floor the lidar never saw near the
mapper is often wider than the 5 x 5 window. At the top of dp's stairs on 2F
(y -14.05) a strip of unmeasured voxels kept their voxel tops at 5.20 m, a lip
7 cm over the measured floor beyond (5.12); PPO tripped on it and fell at
pitch 45 deg (2026-10-24). Only one with no measured neighbour even there is
left to terrain_patch.py (the voxel's middle).

Last, **seams between two mapping passes** (`blend_track_seams`). Where the
map was put together from passes whose heights drifted apart, the measured
floor carries a step the building does not have: dp's stairwell, landing 2 up
to the top on 2F, sits ~0.13 m under the rest (the mapping robot's own height
there is 2.48 / 5.11 over a measured 2.35 / 4.97, where landing 1 and the 2F
corridor agree with it), so where it meets the corridor the feet found a
0.12-0.16 m step with a groove before it. The planner, on the voxels, sees a
level floor there and hands the robot back to PPO 1.2 m before it; PPO met
the step at 0.7 m/s and fell, twice (2026-10-24), where the real robot walks
on. The mapping track finds them: along a stretch it walked level, the
measured floor under it (`floor - ground`, its own height taken off) jumps
between two keyframes. Around such a jump the floor is replaced by a 1.1 m
box mean, which turns the step into a ramp of a metre (0.16 m: 8 deg) and
leaves everything more than half the window from it as it was. A real step
the robot walked over moved its own height too, so it is no seam; a flight
beside a landing does not change what is under the robot.

All of this moves where the feet touch, nothing else: the lidar, the planner
and the kinematic base still read the voxels (CLAUDE.md, rule 2).

Cached as cache/<map>/surface_heights.npz (dp: built in a few seconds).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .pcd_mesh import read_pcd
from .world import World, read_track

# 2: smoothed; 3: every surface voxel, one point is enough; 4: the wide pass; 5: seams;
# 6: an unmeasured voxel prefers its own layer; 7: seams keyed by column and band
FORMAT = 7
WIDE_RADIUS = 6  # columns, the second pass (module docstring)

# Seams (blend_track_seams). poses.txt is the IMU; its ground is z - this.
LIDAR_HEIGHT = 0.481
SEAM_FLAT = 0.06  # the track's own ground range over +-SEAM_WINDOW keyframes: walked level
SEAM_WINDOW = 3
SEAM_JUMP = 0.06  # floor - ground changing this much between level keyframes is a seam
SEAM_PAIR = 1.0  # m, at most this far apart (keyframes are 0.05-0.5 m apart on dp)
SEAM_UNDER = 0.25  # m, the disc under a keyframe whose floor is read
SEAM_ZONE = 1.2  # m around the jump that is blended
SEAM_BOX = 5  # columns each side: the 1.1 m box mean
# Neighbours further off a cell's top than the seam's own jump plus this are
# not its floor: a riser (0.17 m) beside a 0.07 m seam on dp's landing 3 stays a riser.
SEAM_SAME_FLOOR = 0.05


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
    surface within the window has none (`build` tries a wider window, then
    terrain_patch.py takes the voxel's middle)."""
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
        # An unmeasured voxel takes its own layer's measured neighbours when it
        # has any, the layers above and below only without: a tread filled into
        # the octomap (no point in it) otherwise took the tread under it. On
        # dp's flight 3 the filled 3.4-3.5 voxels between treads at 3.37 and
        # 3.66 got 3.37 from the layer below, and the 0.29 m riser stayed
        # (2026-10-24); their own layer has the tread's measured 3.48-3.53.
        layer = np.where(offs[None, :, 2] == 0, near, np.nan)
        own_layer = np.isnan(own[a:a + chunk]) & ~np.all(np.isnan(layer), axis=1)
        near[own_layer] = layer[own_layer]
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
    left = targets[~np.isin(raw.linear(targets), skeys)]
    wkeys, wtops = smooth(raw, left, float(g.origin_m[2]), float(g.res), radius=WIDE_RADIUS)
    keys = np.concatenate([skeys, wkeys])
    order = np.argsort(keys)
    return SurfaceHeights(sh.shape, keys[order], np.concatenate([stops, wtops])[order])


def _floor_under(sh: SurfaceHeights, origin: np.ndarray, res: float, x: float, y: float,
                 ground: float) -> float:
    """Median top of the surface cells in the SEAM_UNDER disc under (x, y) near
    `ground` (0.35 m under it to 0.15 over: the drift is ~0.13), NaN with
    fewer than three."""
    r = int(np.ceil(SEAM_UNDER / res))
    d = np.arange(-r, r + 1)
    dx, dy = (a.ravel() for a in np.meshgrid(d, d, indexing="ij"))
    disc = (dx * dx + dy * dy) * res * res <= SEAM_UNDER ** 2
    ix = int(np.floor((x - origin[0]) / res)) + dx[disc]
    iy = int(np.floor((y - origin[1]) / res)) + dy[disc]
    zs = np.arange(int(np.floor((ground - 0.35 - origin[2]) / res)),
                   int(np.floor((ground + 0.15 - origin[2]) / res)) + 1)
    idx = np.stack([np.repeat(ix, zs.size), np.repeat(iy, zs.size), np.tile(zs, ix.size)], axis=1)
    t = sh.lookup(idx)
    t = t[(t >= ground - 0.35) & (t <= ground + 0.15)]
    return float(np.median(t)) if t.size >= 3 else float("nan")


def blend_track_seams(sh: SurfaceHeights, origin: np.ndarray, res: float, track: np.ndarray
                      ) -> tuple[SurfaceHeights, list[tuple[int, int, float]]]:
    """Ramp the steps between drifted mapping passes (module docstring).
    `track` is poses.txt's (K, 3) IMU positions. Returns the heights and the
    seams as (keyframe, keyframe, jump in metres)."""
    n = track.shape[0]
    if n < 2 or sh.keys.size == 0:
        return sh, []
    ground = track[:, 2] - LIDAR_HEIGHT
    w = SEAM_WINDOW
    flat = np.array([np.ptp(ground[max(0, k - w):k + w + 1]) <= SEAM_FLAT for k in range(n)])
    off = np.full(n, np.nan)
    for k in np.nonzero(flat)[0]:
        off[k] = _floor_under(sh, origin, res, track[k, 0], track[k, 1], ground[k]) - ground[k]
    seams = []
    for k in range(n - 1):
        if np.isnan(off[k]):
            continue
        for j in range(k + 1, min(n, k + 1 + w)):
            if np.hypot(*(track[j, :2] - track[k, :2])) > SEAM_PAIR:
                break
            if np.isnan(off[j]):
                continue
            if abs(ground[j] - ground[k]) <= 0.04 and abs(off[j] - off[k]) >= SEAM_JUMP:
                seams.append((k, j, float(off[j] - off[k])))
            break
    if not seams:
        return sh, []

    # The cells to blend (within SEAM_ZONE of a seam's segment) and the floor
    # around them the box reads (SEAM_BOX further), as (column, band) -> [(key,
    # top)]. The band (the seam's height window) is part of the key: a
    # stairwell stacks floors on one column, and keyed by column alone the
    # first seam's floor answered for every later one -- landing 2 (2.34 m)
    # stood in for the 2F stair head (5.1 m) above it and its step stayed
    # (2026-10-24).
    nx, ny, nz = sh.shape
    zone: dict[tuple[int, int, int], float] = {}  # (column, band) -> same-floor tolerance
    cols: dict[tuple[int, int, int], list[tuple[int, float]]] = {}
    reach = SEAM_ZONE + (SEAM_BOX + 1) * res
    for k, j, jump in seams:
        tol = abs(jump) + SEAM_SAME_FLOOR
        a, b = track[k, :2], track[j, :2]
        lo_g, hi_g = min(ground[k], ground[j]), max(ground[k], ground[j])
        zs = np.arange(int(np.floor((lo_g - 0.4 - origin[2]) / res)),
                       int(np.floor((hi_g + 0.2 - origin[2]) / res)) + 1)
        band = int(zs[0])
        i0, i1 = (int(np.floor((min(a[0], b[0]) - reach - origin[0]) / res)),
                  int(np.floor((max(a[0], b[0]) + reach - origin[0]) / res)))
        j0, j1 = (int(np.floor((min(a[1], b[1]) - reach - origin[1]) / res)),
                  int(np.floor((max(a[1], b[1]) + reach - origin[1]) / res)))
        ab = b - a
        for ix in range(max(i0, 0), min(i1, nx - 1) + 1):
            for iy in range(max(j0, 0), min(j1, ny - 1) + 1):
                p = origin[:2] + (np.array([ix, iy]) + 0.5) * res
                t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
                dist = float(np.hypot(*(p - (a + t * ab))))
                if dist > reach:
                    continue
                if dist <= SEAM_ZONE:
                    zone[(ix, iy, band)] = max(tol, zone.get((ix, iy, band), 0.0))
                if (ix, iy, band) not in cols:
                    idx = np.stack([np.full(zs.size, ix), np.full(zs.size, iy), zs], axis=1)
                    tops = sh.lookup(idx)
                    hit = ~np.isnan(tops)
                    cols[(ix, iy, band)] = list(zip(sh.linear(idx[hit]).tolist(), tops[hit].tolist()))
    new: dict[int, float] = {}
    for (ix, iy, band), tol in zone.items():
        for key, top in cols.get((ix, iy, band), ()):
            near = [t for dx in range(-SEAM_BOX, SEAM_BOX + 1) for dy in range(-SEAM_BOX, SEAM_BOX + 1)
                    for _, t in cols.get((ix + dx, iy + dy, band), ()) if abs(t - top) <= tol]
            new[key] = float(np.mean(near))
    keys = np.fromiter(new.keys(), np.int64)
    pos = np.searchsorted(sh.keys, keys)
    tops = sh.tops.copy()
    tops[pos] = np.fromiter(new.values(), np.float32)
    return SurfaceHeights(sh.shape, sh.keys, tops), seams


def load_or_build(map_dir: str, cache_dir: str, world: World, log=print) -> SurfaceHeights | None:
    """The cached heights, built first if needed; None without map.pcd (voxel tops then)."""
    pcd_path = os.path.join(map_dir, "map.pcd")
    if not os.path.isfile(pcd_path):
        log(f"[surface] {pcd_path} not found; the feet stand on voxel tops")
        return None
    st = os.stat(pcd_path)
    poses = os.path.join(map_dir, "poses.txt")
    pst = os.stat(poses) if os.path.isfile(poses) else None
    key = json.dumps({"format": FORMAT, "pcd": [st.st_size, int(st.st_mtime)],
                      "poses": [pst.st_size, int(pst.st_mtime)] if pst else None,
                      "world": world.stats.cache_key, "world_options": world.stats.options},
                     sort_keys=True)
    path = os.path.join(cache_dir, "surface_heights.npz")
    if os.path.isfile(path):
        z = np.load(path)
        if str(z["key"]) == key:
            log(f"[surface] cached ({z['keys'].size} surface voxels with a measured top)")
            return SurfaceHeights(tuple(int(s) for s in z["shape"]), z["keys"], z["tops"])
    sh = build(read_pcd(pcd_path), world)
    g = world.grid
    track = read_track(map_dir)
    sh, seams = blend_track_seams(sh, g.origin_m, g.res, track)
    for k, j, jump in seams:
        x, y = track[j, :2]
        log(f"[surface] seam between passes at keyframes {k}-{j} ({x:.2f}, {y:.2f}): "
            f"the floor jumps {jump:+.2f} m under a level track; ramped")
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, key=key, shape=np.array(sh.shape), keys=sh.keys, tops=sh.tops)
    os.replace(tmp, path)
    log(f"[surface] {sh.keys.size} surface voxels with a measured top (median of map.pcd)")
    return sh
