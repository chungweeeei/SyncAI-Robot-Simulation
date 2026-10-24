import numpy as np

from syncai_sim.btree import TREE_MAX_VAL
from syncai_sim.surface_heights import SurfaceHeights, blend_track_seams, build, smooth
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


def test_unmeasured_floor_beyond_the_window_takes_the_wide_pass():
    # A floor whose voxel tops are 0.1 but the points (x < 1 m only) say 0.03:
    # a column 6 voxels past the last measured one is outside the 5 x 5 window
    # and gets the measured height from the wide pass, not the voxel top.
    # (+ one voxel up in a corner: build reads a layer above every surface)
    g = VoxelGrid(np.concatenate([_grid(range(ORG, ORG + 30), range(ORG, ORG + 10), [ORG]),
                                  [[ORG + 29, ORG + 9, ORG + 2]]]), RES, z_clip_above_lowest_m=0)
    xs, ys = np.meshgrid(np.arange(0.05, 1.0, 0.1), np.arange(0.05, 1.0, 0.1), indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 0.03)], axis=1)
    sh = build(np.repeat(pts, 3, axis=0).astype(np.float32), FakeWorld(g))
    near, wide, far = sh.lookup(np.array([[11, 5, 0], [15, 5, 0], [25, 5, 0]]))
    assert abs(near - 0.03) < 1e-6 and abs(wide - 0.03) < 1e-6
    assert np.isnan(far)  # past both windows: terrain_patch.py takes the voxel's middle


def _seam_case(track_follows_floor):
    # One floor layer (iz 0) over 4 x 2 m whose measured top is 0.05 for
    # x < 2 m and 0.18 beyond; the mapping track runs along y 1 m.
    nx, ny = 40, 20
    ix, iy = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"))
    tops = np.where(ix < 20, 0.05, 0.18).astype(np.float32)
    sh = SurfaceHeights((nx, ny, 3), (ix * ny + iy) * 3, tops)
    x = np.arange(0.5, 3.6, 0.2)
    ground = np.where(x < 2.0, 0.05, 0.18) if track_follows_floor else np.full(x.size, 0.05)
    track = np.stack([x, np.full(x.size, 1.0), ground + 0.481], axis=1)
    return sh, blend_track_seams(sh, np.zeros(3), RES, track)


def test_a_seam_under_a_level_track_becomes_a_ramp():
    sh, (out, seams) = _seam_case(track_follows_floor=False)
    assert len(seams) == 1 and abs(seams[0][2] - 0.13) < 0.01
    row = out.lookup(np.array([[i, 10, 0] for i in range(40)]))
    assert np.max(np.abs(np.diff(row))) < 0.03  # no step left, a slope
    assert abs(row[5] - 0.05) < 1e-6 and abs(row[35] - 0.18) < 1e-6  # far off: untouched


def test_a_real_step_the_track_climbed_is_kept():
    sh, (out, seams) = _seam_case(track_follows_floor=True)
    assert seams == []
    assert out is sh


def test_an_unmeasured_voxel_prefers_its_own_layer():
    # A filled tread voxel (iz 4: 0.4-0.5, no point) with measured neighbours in
    # its own layer at 0.48 and in the layer below (the tread under it) at 0.37.
    shape = (5, 5, 8)
    cells = {(1, 2, 4): 0.48, (3, 2, 4): 0.48, (2, 1, 3): 0.37, (2, 3, 3): 0.37, (1, 1, 3): 0.37}
    raw = SurfaceHeights(shape, np.zeros(0, np.int64), np.zeros(0, np.float32))
    keys = raw.linear(np.array(list(cells)))
    order = np.argsort(keys)
    raw = SurfaceHeights(shape, keys[order], np.array(list(cells.values()), np.float32)[order])
    k, t = smooth(raw, np.array([[2, 2, 4]]), 0.0, RES)
    assert k.size == 1 and abs(t[0] - 0.48) < 1e-6


def test_seams_on_stacked_floors_are_both_ramped():
    # Two floors on the same columns, 3 m apart, each with a 0.13 m seam at
    # x 2 m under a level stretch of track: the upper one must not be read
    # through the lower one's cells (a stairwell's landings and stair head).
    nx, ny, nz = 40, 20, 40
    ix, iy = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"))
    lo = np.where(ix < 20, 0.05, 0.18)
    keys = np.concatenate([(ix * ny + iy) * nz + 0, (ix * ny + iy) * nz + 30])
    tops = np.concatenate([lo, lo + 3.0]).astype(np.float32)
    order = np.argsort(keys)
    sh = SurfaceHeights((nx, ny, nz), keys[order], tops[order])
    x = np.arange(0.5, 3.6, 0.2)
    leg = lambda g: np.stack([x, np.full(x.size, 1.0), np.full(x.size, g + 0.481)], axis=1)  # noqa: E731
    out, seams = blend_track_seams(sh, np.zeros(3), RES, np.concatenate([leg(0.05), leg(3.05)]))
    assert len(seams) == 2
    for iz, base in ((0, 0.05), (30, 3.05)):
        row = out.lookup(np.array([[i, 10, iz] for i in range(40)]))
        assert np.max(np.abs(np.diff(row))) < 0.03, f"floor at {base} m kept its step"
