"""Merge occupied voxels into as few axis-aligned boxes as a greedy pass gives.

MuJoCo collides a `mesh` geom as its convex hull, so a building handed over as
one mesh collapses into a single convex blob — but a `box` geom is exact, and a
non-convex world is just many boxes. 1.6 M voxels as 1.6 M geoms is too many
for the broadphase and the viewer; merging runs them down by one to two orders
of magnitude: a floor slab becomes a handful of plates, a wall a few long bars.

The merge is the classic greedy one, in a fixed axis order:

    1. runs along x       — consecutive voxels with the same (y, z)
    2. runs merge along y — runs with the same (x0, x1, z) and consecutive y
    3. rectangles merge along z — same (x0, x1, y0, y1) and consecutive z

Each voxel ends up in exactly one box, so the total box volume equals the voxel
count (the test checks that). It is not the minimal box cover (that problem is
NP-hard, and a different axis order gives a different count), but it is fully
vectorised in numpy and takes a second or two for a site.

Everything here is in integer voxel indices; `world.py` turns the result into
metres with the tree's resolution.
"""

from __future__ import annotations

import numpy as np


def _group_runs(same: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From a boolean `same[i]` = row i+1 continues row i, return (start, length)."""
    starts = np.flatnonzero(np.concatenate(([True], ~same)))
    ends = np.concatenate((starts[1:], [same.shape[0] + 1]))
    return starts, ends - starts


def merge_voxels(voxels: np.ndarray) -> np.ndarray:
    """Greedy-merge (M, 3) int voxel indices into (B, 6) int boxes.

    Each box row is (x0, y0, z0, nx, ny, nz): lowest voxel index and edge
    lengths in voxels. Duplicate voxels are dropped first.
    """
    if voxels.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.int64)
    v = np.unique(voxels.astype(np.int64), axis=0)

    # 1. x-runs: sort by (z, y, x), merge consecutive x within the same (z, y).
    order = np.lexsort((v[:, 0], v[:, 1], v[:, 2]))
    v = v[order]
    same = (
        (v[1:, 2] == v[:-1, 2]) & (v[1:, 1] == v[:-1, 1]) & (v[1:, 0] == v[:-1, 0] + 1)
    )
    start, length = _group_runs(same)
    runs = np.column_stack((v[start, 0], v[start, 1], v[start, 2], length))  # x0 y z nx

    # 2. merge runs along y: equal (z, x0, nx), consecutive y.
    order = np.lexsort((runs[:, 1], runs[:, 3], runs[:, 0], runs[:, 2]))
    r = runs[order]
    same = (
        (r[1:, 2] == r[:-1, 2])
        & (r[1:, 0] == r[:-1, 0])
        & (r[1:, 3] == r[:-1, 3])
        & (r[1:, 1] == r[:-1, 1] + 1)
    )
    start, length = _group_runs(same)
    rects = np.column_stack(
        (r[start, 0], r[start, 1], r[start, 2], r[start, 3], length)
    )  # x0 y0 z nx ny

    # 3. merge rectangles along z: equal (x0, nx, y0, ny), consecutive z.
    order = np.lexsort((rects[:, 2], rects[:, 4], rects[:, 1], rects[:, 3], rects[:, 0]))
    q = rects[order]
    same = (
        (q[1:, 0] == q[:-1, 0])
        & (q[1:, 3] == q[:-1, 3])
        & (q[1:, 1] == q[:-1, 1])
        & (q[1:, 4] == q[:-1, 4])
        & (q[1:, 2] == q[:-1, 2] + 1)
    )
    start, length = _group_runs(same)
    boxes = np.column_stack(
        (q[start, 0], q[start, 1], q[start, 2], q[start, 3], q[start, 4], length)
    )
    return boxes


def boxes_volume(boxes: np.ndarray) -> int:
    """Total volume of (B, 6) boxes in voxels — equals the voxel count after a merge."""
    if boxes.shape[0] == 0:
        return 0
    return int(np.prod(boxes[:, 3:6], axis=1).sum())
