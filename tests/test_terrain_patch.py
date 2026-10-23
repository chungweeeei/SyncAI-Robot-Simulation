import numpy as np

from syncai_sim.btree import TREE_MAX_VAL
from syncai_sim.terrain_patch import TerrainPatch, TerrainPatchConfig
from syncai_sim.voxel_grid import VoxelGrid

RES = 0.1
ORG = TREE_MAX_VAL  # voxel index of x = 0


def _grid(x, y, z):
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)


def _floor():
    """4 x 4 m floor, one voxel thick, top at z = 0."""
    return _grid(range(ORG, ORG + 40), range(ORG, ORG + 40), [ORG - 1])


def _patch(voxels, **cfg):
    g = VoxelGrid(np.concatenate(voxels), RES, z_clip_above_lowest_m=0)
    return TerrainPatch(g, TerrainPatchConfig(**cfg))


def _has(centres, x, y, z):
    return bool(np.any(np.all(np.abs(centres - [x, y, z]) < 1e-6, axis=1)))


BASE = (2.0, 2.0, 0.35)  # standing on the floor, mid map


def test_floor_only_surface_and_window():
    p = _patch([_floor()])
    c = p.boxes_around(BASE)
    # The floor voxels under the window (the voxels holding x +- half_xy, both
    # ends included: 17 x 17) and nothing else: no layer below a one-voxel floor.
    assert c.shape[0] == 17 * 17
    assert np.allclose(c[:, 2], -0.05)
    assert np.all(np.abs(c[:, :2] - 2.0) <= 0.85 + 1e-9)
    # Nearest first.
    d = np.hypot(c[:, 0] - 2.0, c[:, 1] - 2.0)
    assert np.all(np.diff(d) >= -1e-9)


def test_hollow_tread_is_filled_down_to_the_floor():
    # A one-voxel tread at z 0.1-0.2 over x >= 2.3 m, air beneath (octomap's hollow stair).
    tread = _grid(range(ORG + 23, ORG + 40), range(ORG, ORG + 40), [ORG + 1])
    c = _patch([_floor(), tread]).boxes_around(BASE)
    # The riser face at x = 2.35 is solid from the floor up: the voxel 0-0.1 exists.
    assert _has(c, 2.35, 2.05, 0.05)
    assert _has(c, 2.35, 2.05, 0.15)


def test_table_is_not_filled():
    # A 1-voxel top 0.7 m over the floor at x 2.3-2.6: open underneath.
    table = _grid(range(ORG + 23, ORG + 26), range(ORG + 15, ORG + 25), [ORG + 6])
    c = _patch([_floor(), table], above=0.6).boxes_around(BASE)
    assert _has(c, 2.35, 2.05, 0.65)
    assert not _has(c, 2.35, 2.05, 0.35)
    assert not _has(c, 2.35, 2.05, 0.05)


def test_slot_across_a_flight_is_raised():
    # Treads at 0.4 (x < 2.3) and 0.5 (x > 2.3) with a missing row at x 2.3 whose
    # top is 0.2: the slot is filled to the lower tread.
    lower = _grid(range(ORG + 10, ORG + 23), range(ORG, ORG + 40), range(ORG, ORG + 4))
    slot = _grid([ORG + 23], range(ORG, ORG + 40), range(ORG, ORG + 2))
    upper = _grid(range(ORG + 24, ORG + 40), range(ORG, ORG + 40), range(ORG, ORG + 5))
    c = _patch([_floor(), lower, slot, upper]).boxes_around((2.0, 2.0, 0.75))
    assert _has(c, 2.35, 2.05, 0.35)  # top of the filled slot = the lower tread's 0.4


def test_one_column_hole_is_filled():
    floor = _floor()
    hole = (floor[:, 0] == ORG + 20) & (floor[:, 1] == ORG + 20)
    c = _patch([floor[~hole]]).boxes_around(BASE)
    assert _has(c, 2.05, 2.05, -0.05)


def test_capacity_and_refresh():
    p = _patch([_floor()], capacity=50, refresh_dist=0.15)
    c = p.boxes_around(BASE)
    assert c.shape[0] == 50 and p.truncated == 1
    assert np.hypot(c[:, 0] - 2.0, c[:, 1] - 2.0).max() < 0.5  # the nearest ones were kept
    assert not p.due(np.array([2.1, 2.0, 0.35]))
    assert p.due(np.array([2.2, 2.0, 0.35]))
