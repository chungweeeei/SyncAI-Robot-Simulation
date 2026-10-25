"""The fitted flights as fixed boxes: one solid step per tread, where the feet meet them.

surface_heights.py fits every flight the mapping track climbed (or a stair walk
crosses) as a staircase
of equal risers on equal treads, but terrain_patch.py builds the feet's ground
from 0.1 m voxel boxes, each with the fitted height at its column's centre.
That keeps the risers (0.163-0.174 m on dp) and rounds every riser's position
to the column grid: the treads come out 0.20 or 0.30 m deep, in a pattern
that follows where the risers fall in their columns, where the building's are
0.22-0.255 and all alike (flight 1: 0.20 / 0.20 / 0.30 / 0.20 / 0.30 for a
fitted 0.240; measured on the patch, 2026-10-25). HimLoco then meets a
different tread at every step, and a nosing up to 0.1 m from where it is.

So each fit becomes boxes, static geoms built into the scene: a slab of the
floor from the column holding the first riser up to it, one box per tread
(fitted depth, from riser to riser), and a slab from the last riser to the end
of its column; each box is the step's own height plus `UNDER` thick, so every
riser face is whole and nothing hangs under a flight into the one below. The
patch leaves its voxels out of the same columns, in the band from under the
lowest box to `ABOVE` over the top (`regions`, terrain_patch.py). The boxes
also stay put: the patch re-deals its pool as the robot walks, these never move.

How wide: the fit replaced the surface `FLIGHT_HALF_WIDTH` (0.6 m) either side
of the track's line, and dp's flights are 1.4-1.6 m. Here each side grows a
column at a time while the column still looks like the flight -- an occupied
voxel within a step of the tread's height and nothing over it from
`CLEAR_LOW` to `CLEAR_HIGH` (a wall or a handrail) -- at most `MAX_HALF_WIDTH`.
A flight whose own line fails that test is left to the voxels. On a stair
walk's flight (surface_heights.read_stair_walks) a column with no voxel near
the tread at all also counts, as long as it is clear above: those stairs were
never walked, so the lidar saw them only from afar, and dp's north stair has
no voxel at the tread in a third of its columns (the floor 2 m under it
shows through; the boxes were 0.2 m wide). The walk was drawn down a stair
someone looked at, and the walls either side still stop the growth. A track
flight keeps the strict test: the well between a U stair's flights is just
such a column.

The lidar, the planner and the kinematic base still read the voxels
(CLAUDE.md, rule 2): these boxes are what the feet touch, nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNDER = 0.10  # m of box under a step's own riser: the box below overlaps it by this
ABOVE = 0.30  # m over a flight's top that the patch leaves out in its columns
MAX_HALF_WIDTH = 0.8  # m either side of the track's line; dp's flights are 1.4-1.6 wide
CLEAR_LOW, CLEAR_HIGH = 0.15, 0.8  # m over a tread that must be free in a column of the flight
STEP_BELOW = 0.35  # m under a tread's height where the column must have a voxel
COLUMN_OK = 0.7  # share of a column's treads that must pass for the column to count


@dataclass
class StairBoxes:
    boxes: np.ndarray  # (K, 6) centre x y z, half-sizes x y z; axis-aligned
    regions: np.ndarray  # (F, 6) x0 x1 y0 y1 z0 z1 the patch leaves to the boxes
    widths: list[tuple[float, float]]  # per flight kept, metres either side of its line


def _column_ok(occ: np.ndarray, origin: np.ndarray, res: float, f, c: float) -> bool:
    """The column at `c` across the flight looks like the flight at most of its treads
    (on a stair walk's flight, an unseen tread counts: module docstring)."""
    unseen_ok = getattr(f, "walk", -1) >= 0
    hits = 0
    for k in range(f.n - 1):
        u = f.u0 + (k + 0.5) * f.tread
        h = f.bottom + f.riser * (k + 1)
        p = [0.0, 0.0]
        p[f.axis], p[1 - f.axis] = f.sign * u, c
        ix, iy = (int(np.floor((p[i] - origin[i]) / res)) for i in range(2))
        if not (0 <= ix < occ.shape[0] and 0 <= iy < occ.shape[1]):
            continue
        col = occ[ix, iy]
        z = lambda v: int(np.clip(np.floor((v - origin[2]) / res), 0, occ.shape[2]))  # noqa: E731
        step = col[z(h - STEP_BELOW):z(h) + 1].any()
        clear = not col[z(h + CLEAR_LOW):z(h + CLEAR_HIGH) + 1].any()
        hits += bool((step or unseen_ok) and clear)
    return hits >= COLUMN_OK * max(f.n - 1, 1)


def build(flights: list, occ: np.ndarray, origin: np.ndarray, res: float, log=print) -> StairBoxes:
    """Boxes and patch regions for `flights` (surface_heights.FlightFit) on the
    voxel grid `occ` (origin, res: VoxelGrid's)."""
    boxes, regions, widths = [], [], []
    for f in flights:
        ax = f.axis
        cell = lambda v, i: origin[i] + np.floor((v - origin[i]) / res) * res  # noqa: E731
        # Across: the column the line is in, then outwards a column at a time.
        c_cell = cell(f.across, 1 - ax)
        if not _column_ok(occ, origin, res, f, c_cell + 0.5 * res):
            log(f"[stairs] flight at {f.name}: its own line "
                f"does not look like the fitted flight; left to the voxels")
            continue
        lo, hi = c_cell, c_cell + res
        ok = lambda c: _column_ok(occ, origin, res, f, c)  # noqa: E731
        while f.across - (lo - res) <= MAX_HALF_WIDTH and ok(lo - 0.5 * res):
            lo -= res
        while hi + res - f.across <= MAX_HALF_WIDTH and ok(hi + 0.5 * res):
            hi += res
        # Along, in u (climbing): the risers, out to the edges of their end columns.
        risers = f.u0 + f.tread * np.arange(f.n)
        ends = sorted(f.sign * risers[[0, -1]])
        a_lo = cell(ends[0], ax)
        a_hi = cell(ends[1], ax) + res
        u_lo, u_hi = sorted((f.sign * a_lo, f.sign * a_hi))
        edges = np.concatenate([[u_lo], risers, [u_hi]])
        tops = np.concatenate([[float(f.height(np.array([0.5 * (u_lo + risers[0])]))[0])],
                               f.bottom + f.riser * np.arange(1, f.n),
                               [float(f.height(np.array([0.5 * (risers[-1] + u_hi)]))[0])]])
        for k in range(edges.size - 1):
            u0, u1 = edges[k], edges[k + 1]
            if u1 - u0 < 1e-3:
                continue
            thick = f.riser + UNDER
            centre, half = [0.0, 0.0, tops[k] - 0.5 * thick], [0.0, 0.0, 0.5 * thick]
            centre[ax], half[ax] = f.sign * 0.5 * (u0 + u1), 0.5 * (u1 - u0)
            centre[1 - ax], half[1 - ax] = 0.5 * (lo + hi), 0.5 * (hi - lo)
            boxes.append([*centre, *half])
        r = [0.0] * 6
        r[2 * ax], r[2 * ax + 1] = a_lo, a_hi
        r[2 * (1 - ax)], r[2 * (1 - ax) + 1] = lo, hi
        r[4] = float(tops.min()) - f.riser - UNDER
        r[5] = float(tops.max()) + ABOVE
        regions.append(r)
        widths.append((f.across - lo, hi - f.across))
        log(f"[stairs] flight at {f.name}: {f.n} risers of "
            f"{f.riser:.3f} m on {f.tread:.3f} m treads as {edges.size - 1} boxes, "
            f"{hi - lo:.1f} m wide ({'xy'[1 - ax]} {lo:.2f} .. {hi:.2f})")
    return StairBoxes(np.array(boxes, np.float64).reshape(-1, 6),
                      np.array(regions, np.float64).reshape(-1, 6), widths)
