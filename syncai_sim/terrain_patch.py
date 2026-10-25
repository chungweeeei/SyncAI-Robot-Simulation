"""The feet's ground: a patch of the voxel grid, as MuJoCo boxes, around the robot.

A walking robot needs the site to be solid under its feet and against its
legs, and the voxel grid is the site (CLAUDE.md, rule 2). MuJoCo cannot take
the grid whole: dp is 1.6 M voxels, and its own mesh collision is a convex
hull. So it gets a fixed pool of voxel-sized boxes (`scene.py`, one mocap body
each: a mocap pose is the input MuJoCo reads every step, the supported way to
move geometry at run time) and this module moves them onto the voxels in a box
around the robot, rebuilding when the base has moved `refresh_dist` from where
the patch was centred. A foot is at most ~0.45 m from the base, so a 0.8 m
half-width with a 0.15 m refresh leaves the next foothold covered before it is
reached.

The grid is what a lidar saw — surfaces — and two of its gaps are footholds a
real robot never meets, so the patch closes them before choosing boxes:

* **Hollow steps.** A mapped stair tread is one voxel thick with nothing under
  it (the planner CLI's synthetic flight: floor top 0.0, the first tread the
  voxel 0.1-0.2, air between). A foot pushed into the riser at floor height
  slides under the tread's lip and wedges there; HimLoco stood at the first
  riser for 12 s that way, where it climbs the same flight built of solid
  boxes (scratch harness, 2026-10-23). So a column is filled from each
  surface down to the lowest surface of its four neighbours, when that is at
  most `solid_step` lower. A riser becomes a wall; a table top, a bench or
  a ceiling more than `solid_step` above what is beside it stays open.
* **One-voxel slots** in the ground under the body: a column whose top is
  lower than both its neighbours along x, or both along y, is raised to the
  lower of the two (at most `solid_step`). dp's first flight has a whole row
  of them across the stairs at y -11.3 (top 0.2 between treads at 0.5 and
  0.6: a tread the lidar never saw); HimLoco put a foot in it and stood
  nose-up at 40 deg (2026-10-23). The ground is the highest voxel below the
  base, so a ceiling or a tread overhead never counts.
* **One-column holes,** per height layer: a free voxel with at least
  `fill_min_neighbours` of its eight neighbours occupied is filled. A mapped
  floor has them where nothing was ever seen (VoxelGrid.ground_under bridges
  the same holes for the base); a 0.022 m foot would drop into a 0.1 m hole
  the planner steps over.

Then the occupied voxels with at least one free face-neighbour (an interior
voxel can never be touched) in the height band from `below` under the base to
`above` over it become boxes, nearest first. A box that is the top of its
column stands with its top at the surface's measured height inside the voxel
(surface_heights.py: the median of map.pcd's points there; without points,
the median of its neighbours'; with none of those either, the voxel's middle),
not at the voxel's top: the grid rounds a flat
but tilted floor into a strip of 0.1 m blocks every ~13 m, and PPO fell on one
four times out of four. Without map.pcd the tops are the voxel tops, the
height `VoxelGrid.ground_under` calls the ground. The boxes are
2 mm smaller than a voxel so that neighbours' bounding boxes do not touch (no
broadphase pairs between them); a 0.044 m foot does not notice the seam.

The fitted flights are not built here: a box per column puts every riser on
the column grid, so stairs.py makes them fixed boxes, and the `regions` it
returns are left out of the patch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from .voxel_grid import VoxelGrid


@dataclass
class TerrainPatchConfig:
    capacity: int = 1500  # boxes in the pool
    half_xy: float = 0.8
    below: float = 0.9  # the feet are ~0.35 under the base, a step down another 0.2
    above: float = 0.4  # the trunk's top is ~0.1 over the base; walls above it never meet a leg
    refresh_dist: float = 0.15
    # dp's flights have 0.3 m jumps between quantised treads (kinematics.py,
    # ground_search_up); a table is ~0.7 m.
    solid_step: float = 0.3
    fill_min_neighbours: int = 6


class TerrainPatch:
    def __init__(self, grid: VoxelGrid, cfg: TerrainPatchConfig | None = None, surface=None,
                 regions: np.ndarray | None = None):
        self.grid = grid
        self.cfg = cfg or TerrainPatchConfig()
        self.surface = surface  # surface_heights.SurfaceHeights or None
        # (R, 6) x0 x1 y0 y1 z0 z1: left out, the fixed stairs stand there (stairs.py)
        self.regions = np.zeros((0, 6)) if regions is None else np.asarray(regions, np.float64)
        self.centre: np.ndarray | None = None
        self.n_used = 0
        self.truncated = 0  # rebuilds that had more boxes than the pool

    def due(self, base_pos: np.ndarray) -> bool:
        if self.centre is None:
            return True
        return bool(np.linalg.norm(np.asarray(base_pos) - self.centre) > self.cfg.refresh_dist)

    def boxes_around(self, base_pos) -> np.ndarray:
        """(K, 3) box centres in metres, nearest to the base first, K <= capacity."""
        g, c = self.grid, self.cfg
        res = g.res
        x, y, z = (float(v) for v in base_pos)
        lo = np.floor((np.array([x - c.half_xy, y - c.half_xy, z - c.below]) - g.origin_m) / res)
        hi = np.floor((np.array([x + c.half_xy, y + c.half_xy, z + c.above]) - g.origin_m) / res)
        lo = lo.astype(np.int64)
        hi = hi.astype(np.int64) + 1
        # A margin taken from the real grid on every side, so that the filling
        # and the exposure test see the neighbours outside the window: one
        # voxel sideways, and `solid_step` downwards for the step filling.
        k = int(round(c.solid_step / res))
        pad_lo = lo - np.array([1, 1, k + 1])
        pad_hi = hi + 1
        occ = _slice_padded(g.grid, pad_lo, pad_hi)
        base_iz = int(np.floor((z - g.origin_m[2]) / res)) - int(pad_lo[2])
        _fill_slots(occ, min(max(base_iz, 0), occ.shape[2]), k)
        _fill_hollow_steps(occ, k)
        _fill_holes(occ, c.fill_min_neighbours)
        keep = _exposed(occ)
        off = lo - pad_lo
        inner = keep[off[0]:off[0] + (hi - lo)[0], off[1]:off[1] + (hi - lo)[1],
                     off[2]:off[2] + (hi - lo)[2]]
        rel = np.argwhere(inner)
        idx = rel + lo
        centres = g.origin_m + (idx + 0.5) * res
        if self.surface is not None and idx.shape[0]:
            p = rel + off  # the same cells in the padded slice
            top = ~occ[p[:, 0], p[:, 1], p[:, 2] + 1]
            voxel_z = centres[:, 2].copy()
            centres[top, 2] = self._tops(idx[top], centres[top, 2] + 0.5 * res) - 0.5 * res
            # A top lowered below its voxel takes the boxes of its stack down
            # with it: left in place, the exposed voxel under it stands over
            # the surface. A seam ramped across a voxel boundary on dp's 2F
            # (surface_heights.py) put 5.05 m tops on 5.1-5.2 voxels, and the
            # 5.0-5.1 ones beneath, exposed to the lower side, made a 5 cm lip.
            drop = {c: dz for c, dz in zip(map(tuple, p[top].tolist()),
                                           (centres[top, 2] - voxel_z[top]).tolist()) if dz < 0}
            if drop:
                for i in np.nonzero(~top)[0]:
                    a, b, h = (int(v) for v in p[i])
                    while h + 1 < occ.shape[2] and occ[a, b, h + 1]:
                        h += 1
                    centres[i, 2] += drop.get((a, b, h), 0.0)
        for x0, x1, y0, y1, z0, z1 in self.regions:
            centres = centres[~((centres[:, 0] > x0) & (centres[:, 0] < x1)
                                & (centres[:, 1] > y0) & (centres[:, 1] < y1)
                                & (centres[:, 2] > z0) & (centres[:, 2] < z1))]
        d = np.hypot(centres[:, 0] - x, centres[:, 1] - y)
        if centres.shape[0] > c.capacity:
            # Keep what a foot can reach first: near, and near the feet's height.
            d = d + 0.5 * np.abs(centres[:, 2] - (z - 0.35))
            self.truncated += 1
        centres = centres[np.argsort(d)[:c.capacity]]
        self.centre = np.array([x, y, z])
        self.n_used = int(centres.shape[0])
        return centres


    def _tops(self, idx: np.ndarray, voxel_tops: np.ndarray) -> np.ndarray:
        """Measured tops for surface cells; the median of the measured
        neighbours (8 around, one voxel up and down: the far side of a
        terrace line is a voxel off) where a cell has none; the voxel's middle
        where neither has. Not its top: a measured top lies anywhere in the
        voxel, so the top is 5 cm high on average and up to 10, and on dp's
        last step to 2F such tops made a lip PPO fell on (surface_heights.py)."""
        tops = self.surface.lookup(idx)
        missing = np.isnan(tops)
        if np.any(missing):
            m = idx[missing]
            offs = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                             for dz in (-1, 0, 1) if dx or dy])
            near = self.surface.lookup((m[:, None, :] + offs[None]).reshape(-1, 3))
            near = near.reshape(m.shape[0], offs.shape[0])
            known = ~np.isnan(near)
            fill = np.full(m.shape[0], np.nan, np.float32)
            rows = known.any(axis=1)
            fill[rows] = np.nanmedian(near[rows], axis=1)
            tops[missing] = fill
        return np.where(np.isnan(tops), voxel_tops - 0.5 * self.grid.res, tops)


def _slice_padded(grid: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """grid[lo:hi] as a copy, False outside the grid."""
    out = np.zeros(tuple((hi - lo).tolist()), dtype=np.bool_)
    src_lo = np.maximum(lo, 0)
    src_hi = np.minimum(hi, np.array(grid.shape))
    if np.any(src_hi <= src_lo):
        return out
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)
    out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        grid[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out


@nb.njit(cache=True)
def _fill_slots(occ, below_iz, k):
    """In place: on the height map of the highest occupied voxel under
    `below_iz` (-1 = none), raise a column that is lower than both its x or
    both its y neighbours to the lower of them, when that is within k voxels."""
    nx, ny, nz = occ.shape
    top = np.full((nx, ny), -1, np.int64)
    for i in range(nx):
        for j in range(ny):
            for z in range(below_iz - 1, -1, -1):
                if occ[i, j, z]:
                    top[i, j] = z
                    break
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            t = top[i, j]
            if t < 0:
                continue
            fill = t
            for a, b in ((top[i - 1, j], top[i + 1, j]), (top[i, j - 1], top[i, j + 1])):
                m = min(a, b)
                if m > fill and m - t <= k:
                    fill = m
            for z in range(t + 1, fill + 1):
                occ[i, j, z] = True


@nb.njit(cache=True)
def _fill_hollow_steps(occ, k):
    """In place: under every surface voxel (occupied, free above), fill down to
    the lowest surface among the four side neighbours found within k voxels
    below it. Neighbours with nothing within k below (a void, a far floor) do
    not count. Decided on the input grid, written to a copy, then copied back."""
    nx, ny, nz = occ.shape
    out = occ.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for t in range(1, nz):
                if not occ[i, j, t] or (t + 1 < nz and occ[i, j, t + 1]):
                    continue
                if occ[i, j, t - 1]:
                    continue  # already solid underneath
                lowest = t
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    for d in range(1, k + 1):
                        z = t - d
                        if z < 0:
                            break
                        if occ[i + di, j + dj, z]:
                            if z < lowest:
                                lowest = z
                            break
                for z in range(t - 1, lowest, -1):
                    if occ[i, j, z]:
                        break  # this column's own lower stack
                    out[i, j, z] = True
    occ[:] = out


@nb.njit(cache=True)
def _fill_holes(occ, min_neighbours):
    """In place: free voxels with >= min_neighbours of their 8 in-layer
    neighbours occupied (decided on the input, as above)."""
    nx, ny, nz = occ.shape
    out = occ.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for z in range(nz):
                if occ[i, j, z]:
                    continue
                n = 0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if (di or dj) and occ[i + di, j + dj, z]:
                            n += 1
                if n >= min_neighbours:
                    out[i, j, z] = True
    occ[:] = out


@nb.njit(cache=True)
def _exposed(occ):
    """Occupied voxels with a free face-neighbour; the outer layer is never kept
    (its neighbours lie outside the slice)."""
    nx, ny, nz = occ.shape
    out = np.zeros_like(occ)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for z in range(1, nz - 1):
                if occ[i, j, z] and not (occ[i + 1, j, z] and occ[i - 1, j, z] and occ[i, j + 1, z]
                                         and occ[i, j - 1, z] and occ[i, j, z + 1]
                                         and occ[i, j, z - 1]):
                    out[i, j, z] = True
    return out
