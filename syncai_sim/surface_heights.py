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

And last, **the flights the track climbed are fitted as uniform staircases**
(`fit_track_flights`). Per voxel, a riser that crosses a column puts its face
in that column's top voxel, and the median of a face is half a step: dp's
flight 3 came out as 2.56 / 2.61 / 2.69 / 2.91 ... 3.06 / 3.17 / 3.23 ...
(risers of 0.05-0.22 m on treads of 0.1-0.3 m), not the building's equal
risers. HimLoco tipped over backwards or sideways on all four flights of
the stairwell in 7 of 20 climbs over that surface; over the fitted one it
did in 3 of 10 (2026-10-25), so the slivers were not what fells it -- the
fit is here because the stairs are now the building's. Where the points disagree,
two passes overlap half a riser apart (flight 3's upper half shows tread
levels 3.39 / 3.45, 3.57 / 3.69), and on flight 4 the lidar saw the risers'
faces but hardly a tread. A per-voxel rule could not tell a face from a
noisy floor (15 % of dp's ground-floor voxels spread 5 cm or more, against
half of a flight's; marking them faces made the floors rougher and the
flights no better: rejected). So the flight is fitted whole: along the
track, the upper envelope of the points between 0.35 m under the track's
ground and 0.3 m over it (the flight's own soffit, seen from the flight
under it, is ~0.45 m down; a handrail ~0.9 m up), against a staircase of n
equal risers and equal treads whose two ends are the floor the surface
already has under the level keyframes before and after the flight (so the
stairs meet their landings, seams and fixes included). n is held to risers
of 0.13-0.20 m between those two floors -- free, the fit shortened a flight
by calling its last treads a landing -- and the points may sit at most 0.07
m off the landings as a whole (free, an offset of one riser shifted every
tread by one). Points under the staircase cost a quarter of points over it:
a tread the lidar never saw leaves only what is under it. On dp: eight
risers of 0.165-0.174 m on 0.24-0.25 m treads for flights 1 and 2, of 0.163
on 0.225 for flight 3, seven of 0.170 on 0.245 for flight 4 (its top is the
stair head's drifted 5.02, not 5.13), residuals 1-1.6 cm. The fitted height
replaces every surface top within 0.6 m of the track's line and 0.3 m of
the staircase, from the landing floor before the flight to the one after;
between an end riser and the landing (at most 0.6 m) it runs from the
fitted floor to what the landing has there, so a landing's own slope stays
(dp's landing 2 rises 0.11 m off flight 2 where its seam was ramped:
flattened to the fitted 2.42, it ended in a 0.11 m step) and a bump before
it goes (5.12 at the top of flight 4, on a stair head of 5.02).

**Stairs the track never climbed** get the same fit from a stand-in walk
(`read_stair_walks`): `config/stair_walks/<map>.yaml` in this repository lists,
per stair, a polyline of (x, y, ground) through it, level for a metre at both
ends and on each landing, which is resampled every `WALK_STEP` into keyframes
and fitted exactly as the track is. The workspace's planner detects such
stairs off the map since 2026-10-25 (`stair_detect`) and routes over them, so
the robot climbs them; without a walk their treads stay the measured ones.
dp's middle stair is the first (its walk is the planner's detected link).

**Ramps** (`smooth_ramps`): a floor that slopes across voxel boundaries keeps
a step at each, because the edge-preserving median above does not vote
across one. dp's 1F has a long arched ridge (on site a gentle hump, 0.2 m
over ~1.2 m each side, x 10-12.6 from y -1.5 to 7.5) between the middle
stair and the east hall; walked across, the feet met 9 and 6 cm lips at x
10.5 and 10.8 where the measured tops jumped a layer (-0.41 -> -0.32 ->
-0.25), and PPO fell there 5 times in 19 crossings in the dp patrol batch
(2026-10-25/26), never anywhere else on 1F. `config/ramps/<map>.yaml` lists
such places as boxes with a height band; inside one every top in the band
becomes the plain mean of the band's tops within `RAMP_RADIUS` columns
(0.7 m), so a lip is spread over its neighbours and the arch stays.

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
# 6: an unmeasured voxel prefers its own layer; 7: seams keyed by column and band;
# 8: flights fitted as staircases; 9: the fits saved too (stairs.py builds them as boxes);
# 10: stair walks; 11: ramps
FORMAT = 11
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

# Flights (fit_track_flights; the module docstring has the why of each).
# A keyframe is on a flight where the track's ground changes more than this
# over +-2 keyframes, and a run of them that rises more than FLIGHT_RISE is
# one: the workspace's scripts/map_repair/find_pits.py finds them the same way.
FLIGHT_SLOPE = 0.15
FLIGHT_RISE = 0.5
FLIGHT_BAND = (-0.35, 0.30)  # m around the track's ground: the points that are the stairs
FLIGHT_FIT_HALF_WIDTH = 0.35  # m of points either side of the track's line, for the fit
FLIGHT_HALF_WIDTH = 0.6  # m either side of it that the staircase replaces (dp's flights: 1.4-1.6 wide)
# m: a top this close to the staircase is part of it -- a whole riser off
# (up to 0.20), the points' offset to the landings (FLIGHT_OFFSET) and 3 cm
# of noise; the flight's soffit is ~0.45 m under it, a handrail ~0.9 m over.
# At 0.25 a 0.27 m pit at the top of dp's flight 1 stayed.
FLIGHT_WINDOW = 0.30
FLIGHT_BIN = 0.02  # m along the flight, the envelope's bins
FLIGHT_ENVELOPE_Q = 0.95
FLIGHT_UNDER_COST = 0.25  # what a point under the staircase costs, against one over it
FLIGHT_RISERS = (0.13, 0.20)  # m
FLIGHT_TREADS = (0.22, 0.33)  # m
FLIGHT_OFFSET = 0.07  # m, the points against the landings
FLIGHT_MAX_ERR = 0.03  # m, mean (truncated) residual of a fit that is used
FLIGHT_END = 0.6  # m at most between a flight's end riser and the landing floor it is joined to
# Stair walks (read_stair_walks): keyframes this far apart, as on dp's track
# over a flight (0.05-0.5 m), so _flight_runs' +-2 keyframes span 0.8 m.
WALK_STEP = 0.2
WALKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "config", "stair_walks")

# Ramps (smooth_ramps): the box mean's half width in columns. 0.7 m spreads a
# 9 cm lip over seven columns (1.3 cm each) and is under half of dp's hump.
RAMP_RADIUS = 3
RAMPS_DIR = os.path.join(os.path.dirname(WALKS_DIR), "ramps")


class SurfaceHeights:
    def __init__(self, shape: tuple[int, int, int], keys: np.ndarray, tops: np.ndarray,
                 flights: list | None = None):
        self.shape = shape
        self.keys = keys  # sorted linear voxel indices (ix * ny + iy) * nz + iz
        self.tops = tops  # (K,) float32 real top, metres
        self.flights: list[FlightFit] = flights or []  # fit_track_flights' fits

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


class FlightFit:
    """A flight fitted as a staircase: along `axis` (0 x, 1 y), `sign` so that
    u = sign * coordinate climbs; risers at u0 + k * tread, k < n; floors
    `bottom` before the first and `top` after the last, running to the
    landing's own `end_lo` at u_lo and `end_hi` at u_hi. `keyframes` index the
    track, or stair walk `walk`'s keyframes when that is not -1."""

    def __init__(self, keyframes, axis, sign, across, u0, tread, n, bottom, top, err, walk=-1):
        self.keyframes, self.axis, self.sign, self.across = keyframes, axis, sign, across
        self.u0, self.tread, self.n, self.bottom, self.top = u0, tread, n, bottom, top
        self.err, self.walk = err, walk
        self.u_lo = self.u_hi = self.end_lo = self.end_hi = float("nan")

    @property
    def name(self) -> str:
        where = "" if self.walk < 0 else f"stair walk {self.walk} "
        return f"{where}keyframes {self.keyframes[0]}-{self.keyframes[1]}"

    # The cache's row: every field, keyframes, n and walk as floats.
    def row(self) -> list[float]:
        return [*self.keyframes, self.axis, self.sign, self.across, self.u0, self.tread, self.n,
                self.bottom, self.top, self.err, self.u_lo, self.u_hi, self.end_lo, self.end_hi,
                self.walk]

    @classmethod
    def from_row(cls, r) -> FlightFit:
        f = cls((int(r[0]), int(r[1])), int(r[2]), float(r[3]), float(r[4]), float(r[5]),
                float(r[6]), int(r[7]), float(r[8]), float(r[9]), float(r[10]), int(r[15]))
        f.u_lo, f.u_hi, f.end_lo, f.end_hi = (float(v) for v in r[11:15])
        return f

    @property
    def riser(self) -> float:
        return (self.top - self.bottom) / self.n

    @property
    def u1(self) -> float:
        return self.u0 + (self.n - 1) * self.tread

    def height(self, u: np.ndarray) -> np.ndarray:
        k = np.clip(np.floor((u - self.u0) / self.tread) + 1, 0, self.n)
        h = self.bottom + self.riser * k
        a = np.clip((self.u0 - u) / max(self.u0 - self.u_lo, 1e-9), 0.0, 1.0)
        b = np.clip((u - self.u1) / max(self.u_hi - self.u1, 1e-9), 0.0, 1.0)
        h = np.where(u < self.u0, self.bottom + (self.end_lo - self.bottom) * a, h)
        return np.where(u > self.u1, self.top + (self.end_hi - self.top) * b, h)


def _floor_across(sh: SurfaceHeights, origin: np.ndarray, res: float, f: FlightFit, u: float,
                  near: float) -> float:
    """Median surface top across the flight's width at u, within FLIGHT_WINDOW
    of `near`; `near` itself where there is none."""
    vals = []
    for c in np.arange(f.across - FLIGHT_HALF_WIDTH, f.across + FLIGHT_HALF_WIDTH + 1e-9, res):
        p = [0.0, 0.0]
        p[f.axis], p[1 - f.axis] = f.sign * u, c
        ix, iy = (int(np.floor((p[i] - origin[i]) / res)) for i in range(2))
        zs = np.arange(int(np.floor((near - FLIGHT_WINDOW - origin[2]) / res)),
                       int(np.floor((near + FLIGHT_WINDOW - origin[2]) / res)) + 1)
        t = sh.lookup(np.stack([np.full(zs.size, ix), np.full(zs.size, iy), zs], axis=1))
        t = t[np.abs(t - near) <= FLIGHT_WINDOW]
        if t.size:
            vals.append(float(t.max()))
    return float(np.median(vals)) if vals else near


def _flight_runs(ground: np.ndarray) -> list[np.ndarray]:
    n = ground.size
    on = np.array([abs(ground[min(k + 2, n - 1)] - ground[max(k - 2, 0)]) > FLIGHT_SLOPE
                   for k in range(n)], bool)
    runs, cur = [], []
    for k in range(n):
        if on[k]:
            cur.append(k)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [np.array(r) for r in runs if np.ptp(ground[r]) > FLIGHT_RISE]


def _fit_staircase(uc: np.ndarray, env: np.ndarray, bottom: float, top: float
                   ) -> tuple[float, int, float, float] | None:
    """(err, n, tread, u0) of the staircase from `bottom` to `top` that best
    fits the envelope `env` at `uc` (the flight's keyframes' stretch), or None."""
    rise = top - bottom
    best = None
    lo_u, hi_u = float(uc.min()), float(uc.max())
    for n in range(max(2, int(np.ceil(rise / FLIGHT_RISERS[1]))), int(np.floor(rise / FLIGHT_RISERS[0])) + 1):
        r = rise / n
        for d in np.arange(FLIGHT_TREADS[0], FLIGHT_TREADS[1] + 1e-9, 0.005):
            u0s = np.arange(lo_u - 0.8, hi_u + 0.8 - (n - 1) * d, 0.01)
            if u0s.size == 0:
                continue
            k = np.clip(np.floor((uc[None, :] - u0s[:, None]) / d) + 1, 0, n)
            res = env[None, :] - (bottom + r * k)
            off = np.clip(np.quantile(res, 0.75, axis=1), -FLIGHT_OFFSET, FLIGHT_OFFSET)
            res = res - off[:, None]
            err = np.mean(np.where(res > 0, np.minimum(res, 0.08),
                                   FLIGHT_UNDER_COST * np.minimum(-res, 0.08)), axis=1)
            j = int(np.argmin(err))
            if best is None or err[j] < best[0]:
                best = (float(err[j]), n, float(d), float(u0s[j]))
    return best


def fit_track_flights(sh: SurfaceHeights, occ: np.ndarray, origin: np.ndarray, res: float,
                      track: np.ndarray, points: np.ndarray, walk: int = -1
                      ) -> tuple[SurfaceHeights, list[FlightFit], list[str]]:
    """Replace the surface of every flight on the track by a fitted staircase
    (module docstring). `occ` is the voxel grid, `points` map.pcd; `track` may
    be stair walk `walk`'s keyframes (read_stair_walks). Returns the heights,
    the fits used and why the others were not."""
    if track.shape[0] < 5 or sh.keys.size == 0:
        return sh, [], []
    ground = track[:, 2] - LIDAR_HEIGHT
    fits: list[FlightFit] = []
    skipped: list[str] = []
    nx, ny, nz = sh.shape
    new: dict[int, float] = {}
    for r in _flight_runs(ground):
        name = ("" if walk < 0 else f"stair walk {walk} ") + f"keyframes {r[0]}-{r[-1]}"
        xy = track[r, :2]
        ax = 0 if np.ptp(xy[:, 0]) > np.ptp(xy[:, 1]) else 1
        climbs = ground[r[-1]] > ground[r[0]]
        sign = 1.0 if (xy[-1, ax] - xy[0, ax] > 0) == climbs else -1.0
        across = float(np.median(xy[:, 1 - ax]))
        u_kf = sign * xy[:, ax]
        order = np.argsort(u_kf)
        u_kf, g_kf = u_kf[order], np.maximum.accumulate(ground[r][order])

        # The floors at both ends: under the level keyframes before and after.
        ends = []
        for ks in (range(max(0, r[0] - 3), r[0]), range(r[-1] + 1, min(track.shape[0], r[-1] + 4))):
            ks = list(ks)
            f = [_floor_under(sh, origin, res, track[k, 0], track[k, 1], ground[k]) for k in ks]
            ok = [i for i, v in enumerate(f) if v == v]
            ends.append((float(np.median([f[i] for i in ok])) if ok else float("nan"),
                         [sign * track[ks[i], ax] for i in ok]))
        (f0, u0_ends), (f1, u1_ends) = ends if climbs else ends[::-1]
        if f0 != f0 or f1 != f1:
            skipped.append(f"{name}: no level floor measured at an end")
            continue

        # The points' upper envelope along the keyframes' stretch.
        u = sign * points[:, ax]
        m = ((u >= u_kf[0]) & (u <= u_kf[-1])
             & (np.abs(points[:, 1 - ax] - across) <= FLIGHT_FIT_HALF_WIDTH))
        p, pu = points[m], u[m]
        g = np.interp(pu, u_kf, g_kf)
        keep = (p[:, 2] >= g + FLIGHT_BAND[0]) & (p[:, 2] <= g + FLIGHT_BAND[1])
        p, pu = p[keep], pu[keep]
        b = np.floor((pu - u_kf[0]) / FLIGHT_BIN).astype(np.int64)
        order = np.lexsort((p[:, 2], b))
        b, z = b[order], p[order, 2]
        bins, first, counts = np.unique(b, return_index=True, return_counts=True)
        enough = counts >= 3
        env = z[first + np.minimum(counts - 1, (counts * FLIGHT_ENVELOPE_Q).astype(np.int64))][enough]
        uc = u_kf[0] + (bins[enough] + 0.5) * FLIGHT_BIN
        if uc.size < 10:
            skipped.append(f"{name}: too few points on the flight")
            continue
        fit = _fit_staircase(uc, env, f0, f1)
        if fit is None or fit[0] > FLIGHT_MAX_ERR:
            skipped.append(f"{name}: no staircase fits"
                           + ("" if fit is None else f" (residual {fit[0]:.3f} m)"))
            continue
        err, n, d, u0 = fit
        ff = FlightFit((int(r[0]), int(r[-1])), ax, sign, across, u0, d, n, f0, f1, err, walk)
        u1 = ff.u1
        u_lo = ff.u_lo = float(np.clip(min(u0_ends, default=u0 - FLIGHT_END), u0 - FLIGHT_END, u0 - res))
        u_hi = ff.u_hi = float(np.clip(max(u1_ends, default=u1 + FLIGHT_END), u1 + res, u1 + FLIGHT_END))
        ff.end_lo = _floor_across(sh, origin, res, ff, u_lo, f0)
        ff.end_hi = _floor_across(sh, origin, res, ff, u_hi, f1)
        fits.append(ff)

        # Every surface voxel there that is on the staircase takes its height.
        cmin = across - FLIGHT_HALF_WIDTH
        cmax = across + FLIGHT_HALF_WIDTH
        lo_xy, hi_xy = [0.0, 0.0], [0.0, 0.0]
        lo_xy[ax], hi_xy[ax] = sorted((sign * u_lo, sign * u_hi))
        lo_xy[1 - ax], hi_xy[1 - ax] = cmin, cmax
        i0, j0 = (int(np.floor((v - origin[i]) / res)) for i, v in enumerate(lo_xy))
        i1, j1 = (int(np.floor((v - origin[i]) / res)) for i, v in enumerate(hi_xy))
        for ix in range(max(i0, 0), min(i1, nx - 1) + 1):
            for iy in range(max(j0, 0), min(j1, ny - 1) + 1):
                c = origin[:2] + (np.array([ix, iy]) + 0.5) * res
                if not (u_lo <= sign * c[ax] <= u_hi and cmin <= c[1 - ax] <= cmax):
                    continue
                h = float(ff.height(np.array([sign * c[ax]]))[0])
                for iz in range(max(int(np.floor((h - 0.3 - origin[2]) / res)), 0),
                                min(int(np.floor((h + 0.3 - origin[2]) / res)), nz - 2) + 1):
                    if not occ[ix, iy, iz] or occ[ix, iy, iz + 1]:
                        continue
                    lo = origin[2] + iz * res
                    # No bound but the window: a top lowered takes its stack
                    # down (terrain_patch.py), and one raised past its voxel
                    # floats over the stack -- the gap is under the tread's
                    # nose, where no foot goes. Held to one voxel, the column
                    # a fitted riser moves over kept its half step (1.06 among
                    # 0.95 / 1.25 in tests/test_surface_heights.py).
                    key = (ix * ny + iy) * nz + iz
                    cur = sh.lookup(np.array([[ix, iy, iz]]))[0]
                    cur = lo + 0.5 * res if cur != cur else cur
                    if abs(cur - h) <= FLIGHT_WINDOW:
                        new[key] = h
    if not new:
        return sh, fits, skipped
    keys = np.fromiter(new.keys(), np.int64)
    vals = np.fromiter(new.values(), np.float32)
    have = np.isin(keys, sh.keys)
    tops = sh.tops.copy()
    tops[np.searchsorted(sh.keys, keys[have])] = vals[have]
    all_keys = np.concatenate([sh.keys, keys[~have]])
    all_tops = np.concatenate([tops, vals[~have]])
    order = np.argsort(all_keys)
    return SurfaceHeights(sh.shape, all_keys[order], all_tops[order]), fits, skipped


def read_stair_walks(path: str) -> list[tuple[str, np.ndarray]]:
    """(name, (K, 3) keyframes as poses.txt's: IMU z = ground + LIDAR_HEIGHT)
    per stair in a stair-walks file (module docstring); [] without one."""
    if not os.path.isfile(path):
        return []
    import yaml

    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    out = []
    for w in spec.get("walks", []):
        p = np.asarray(w["points"], np.float64).reshape(-1, 3)
        seg = np.hypot(*np.diff(p[:, :2], axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        t = np.linspace(0.0, s[-1], max(2, int(round(s[-1] / WALK_STEP)) + 1))
        kf = np.stack([np.interp(t, s, p[:, i]) for i in range(3)], axis=1)
        kf[:, 2] += LIDAR_HEIGHT
        out.append((str(w["name"]), kf))
    return out


def read_ramps(path: str) -> list[tuple[str, tuple[float, ...]]]:
    """(name, (x0, x1, y0, y1, z0, z1)) per ramp in a ramps file (module
    docstring); [] without one."""
    if not os.path.isfile(path):
        return []
    import yaml

    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    return [(str(r["name"]), tuple(float(v) for v in list(r["box"]) + list(r["band"])))
            for r in spec.get("ramps", [])]


def smooth_ramps(sh: SurfaceHeights, origin: np.ndarray, res: float,
                 ramps: list[tuple[str, tuple[float, ...]]]) -> SurfaceHeights:
    """Every top inside a ramp's box and band becomes the mean of the band's
    tops (the highest per column) within RAMP_RADIUS columns, read also from
    the margin outside the box so that its edges meet the floor around it."""
    if not ramps or sh.keys.size == 0:
        return sh
    nx, ny, nz = sh.shape
    ix, iy = sh.keys // (nz * ny), (sh.keys // nz) % ny
    tops = sh.tops.copy()
    r = RAMP_RADIUS
    for _, (x0, x1, y0, y1, z0, z1) in ramps:
        i0, i1 = (int(np.floor((v - origin[0]) / res)) for v in (x0, x1))
        j0, j1 = (int(np.floor((v - origin[1]) / res)) for v in (y0, y1))
        band = (sh.tops >= z0) & (sh.tops <= z1)
        near = band & (ix >= i0 - r) & (ix <= i1 + r) & (iy >= j0 - r) & (iy <= j1 + r)
        w, h = i1 - i0 + 1, j1 - j0 + 1
        top = np.full((w + 2 * r, h + 2 * r), np.nan)
        np.fmax.at(top, (ix[near] - i0 + r, iy[near] - j0 + r), sh.tops[near])
        have = ~np.isnan(top)
        val = np.where(have, top, 0.0)
        total, count = np.zeros((w, h)), np.zeros((w, h))
        for a in range(2 * r + 1):
            for b in range(2 * r + 1):
                total += val[a:a + w, b:b + h]
                count += have[a:a + w, b:b + h]
        inner = near & (ix >= i0) & (ix <= i1) & (iy >= j0) & (iy <= j1)
        tops[inner] = (total / np.maximum(count, 1))[ix[inner] - i0, iy[inner] - j0]
    return SurfaceHeights(sh.shape, sh.keys, tops, sh.flights)


def load_or_build(map_dir: str, cache_dir: str, world: World, log=print,
                  walks_dir: str = WALKS_DIR, ramps_dir: str = RAMPS_DIR
                  ) -> SurfaceHeights | None:
    """The cached heights, built first if needed; None without map.pcd (voxel tops then)."""
    pcd_path = os.path.join(map_dir, "map.pcd")
    if not os.path.isfile(pcd_path):
        log(f"[surface] {pcd_path} not found; the feet stand on voxel tops")
        return None
    st = os.stat(pcd_path)
    poses = os.path.join(map_dir, "poses.txt")
    pst = os.stat(poses) if os.path.isfile(poses) else None
    walks_path = os.path.join(walks_dir, os.path.basename(os.path.normpath(map_dir)) + ".yaml")
    walks = read_stair_walks(walks_path)
    ramps_path = os.path.join(ramps_dir, os.path.basename(os.path.normpath(map_dir)) + ".yaml")
    ramps = read_ramps(ramps_path)
    key = json.dumps({"format": FORMAT, "pcd": [st.st_size, int(st.st_mtime)],
                      "poses": [pst.st_size, int(pst.st_mtime)] if pst else None,
                      "walks": [[n, np.round(kf, 4).tolist()] for n, kf in walks],
                      "ramps": ramps,
                      "world": world.stats.cache_key, "world_options": world.stats.options},
                     sort_keys=True)
    path = os.path.join(cache_dir, "surface_heights.npz")
    if os.path.isfile(path):
        z = np.load(path)
        if str(z["key"]) == key:
            log(f"[surface] cached ({z['keys'].size} surface voxels with a measured top)")
            return SurfaceHeights(tuple(int(s) for s in z["shape"]), z["keys"], z["tops"],
                                  [FlightFit.from_row(r) for r in z["flights"]])
    points = read_pcd(pcd_path)
    sh = build(points, world)
    g = world.grid
    track = read_track(map_dir)
    sh, seams = blend_track_seams(sh, g.origin_m, g.res, track)
    for k, j, jump in seams:
        x, y = track[j, :2]
        log(f"[surface] seam between passes at keyframes {k}-{j} ({x:.2f}, {y:.2f}): "
            f"the floor jumps {jump:+.2f} m under a level track; ramped")
    sh = smooth_ramps(sh, g.origin_m, g.res, ramps)
    for name, box in ramps:
        log(f"[surface] ramp: {name} ({ramps_path}), box {box[:4]}, band {box[4:]}")
    sh, fits, skipped = fit_track_flights(sh, g.grid, g.origin_m, g.res, track, points)
    for i, (name, kf) in enumerate(walks):
        log(f"[surface] stair walk {i}: {name} ({kf.shape[0]} keyframes, {walks_path})")
        sh, wf, ws = fit_track_flights(sh, g.grid, g.origin_m, g.res, kf, points, walk=i)
        fits += wf
        skipped += ws
    for f in fits:
        rs = [f.sign * (f.u0 + k * f.tread) for k in range(f.n)]
        log(f"[surface] flight at {f.name} "
            f"({'xy'[1 - f.axis]} {f.across:.2f}): {f.n} risers of {f.riser:.3f} m on "
            f"{f.tread:.3f} m treads, {f.bottom:.2f} -> {f.top:.2f}, residual {f.err:.3f} m; "
            f"risers at {'xy'[f.axis]} {rs[0]:.2f} .. {rs[-1]:.2f}")
    for s in skipped:
        log(f"[surface] flight at {s}; left as measured")
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp.npz"
    sh.flights = fits
    np.savez(tmp, key=key, shape=np.array(sh.shape), keys=sh.keys, tops=sh.tops,
             flights=np.array([f.row() for f in fits], np.float64).reshape(-1, 16))
    os.replace(tmp, path)
    log(f"[surface] {sh.keys.size} surface voxels with a measured top (median of map.pcd)")
    return sh
