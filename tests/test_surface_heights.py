import numpy as np

from syncai_sim.btree import TREE_MAX_VAL
from syncai_sim.surface_heights import build
from syncai_sim.terrain_patch import TerrainPatch
from syncai_sim.voxel_grid import VoxelGrid
from syncai_sim.world import _cut_track_ridges

RES = 0.1
ORG = TREE_MAX_VAL


def _grid(x, y, z):
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)


class FakeWorld:
    def __init__(self, grid):
        self.grid = grid


def test_measured_tops_smooth_a_quantised_terrace():
    # A floor at z top 0.0 for x < 2 and 0.1 for x >= 2 in voxels — but the
    # real floor is a gentle ramp from 0.04 to 0.06 (a tilted map).
    lo = _grid(range(ORG, ORG + 20), range(ORG, ORG + 40), [ORG - 1])
    hi = _grid(range(ORG + 20, ORG + 40), range(ORG, ORG + 40), [ORG])
    g = VoxelGrid(np.concatenate([lo, hi]), RES, z_clip_above_lowest_m=0)
    xs, ys = np.meshgrid(np.arange(0.05, 4.0, 0.1), np.arange(0.05, 4.0, 0.1), indexing="ij")
    x, y = xs.ravel(), ys.ravel()
    z = 0.04 + 0.02 * x / 4.0
    z = np.where(x < 2.0, z - 0.1, z)  # the low side's points sit in the voxel below the line
    pts = np.repeat(np.stack([x, y, z], axis=1), 3, axis=0)
    sh = build(pts.astype(np.float32), FakeWorld(g))
    patch = TerrainPatch(g, surface=sh)
    c = patch.boxes_around((2.0, 2.0, 0.4))
    row = c[np.abs(c[:, 1] - 2.05) < 1e-6]
    tops = row[np.argsort(row[:, 0]), 2] + 0.049 + 0.001  # box centre + half -> top
    assert np.max(np.abs(np.diff(tops[(row[:, 0] > 1.5) & (row[:, 0] < 2.5)]))) < 0.1


def test_ridge_across_a_flat_track_is_cut():
    nx, ny, nz = 30, 30, 10
    grid = np.zeros((nx, ny, nz), bool)
    grid[:, :, 2] = True  # floor, top at z index 2
    grid[15, :, 3:6] = True  # a one-voxel wall across the corridor, 0.3 m
    grid[5:8, :, 3:5] = True  # a three-voxel-wide step: not a ridge
    track = np.array([[x, 1.5, 0.3 + 0.481] for x in np.arange(0.5, 2.6, 0.5)])
    origin = np.zeros(3)
    d = np.arange(-3, 4)
    dx, dy = (a.ravel() for a in np.meshgrid(d, d, indexing="ij"))
    _cut_track_ridges(grid, track, track[:, 2] - 0.481, RES, origin, dx, dy, window=2)
    assert not grid[15, 15, 3:].any()  # the wall is gone where the track crosses it
    assert grid[15, 0, 3:6].all()  # and kept outside the swept disc
    assert grid[6, 15, 3:5].all()  # the step stays
