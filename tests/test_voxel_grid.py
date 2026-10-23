import math

import numpy as np

from syncai_sim.btree import TREE_MAX_VAL
from syncai_sim.voxel_grid import VoxelGrid

RES = 0.1
ORG = TREE_MAX_VAL  # voxel index of x = 0


def _grid(x, y, z):
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)


def make_room():
    """10 x 6 m floor (z in [-0.1, 0)), a wall at x = 5 m from 0 to 2 m high, and a
    0.17 m step at x in [7, 10) m (two voxels thick: z in [0, 0.17) ~ voxels 0 and 1)."""
    floor = _grid(range(ORG, ORG + 100), range(ORG, ORG + 60), [ORG - 1])
    wall = _grid([ORG + 50], range(ORG, ORG + 60), range(ORG, ORG + 20))
    step = _grid(range(ORG + 70, ORG + 100), range(ORG, ORG + 60), range(ORG, ORG + 2))  # top at 0.2 m
    return VoxelGrid(np.concatenate([floor, wall, step]), RES, z_clip_above_lowest_m=0)


def test_bounds_and_origin():
    g = make_room()
    lo, hi = g.bounds_m()
    assert np.allclose(lo, [0.0, 0.0, -0.1])
    assert np.allclose(hi, [10.0, 6.0, 2.0])  # the wall reaches 2 m


def test_cast_hits_wall_floor_and_misses():
    g = make_room()
    origin = np.array([1.0, 3.0, 0.5])
    dirs = np.array([
        [1.0, 0.0, 0.0],        # +x into the wall at x = 5 -> 4.0 m
        [0.0, 0.0, -1.0],       # straight down onto the floor top z = 0 -> 0.5 m
        [0.0, 0.0, 1.0],        # up: nothing
        [-1.0, 0.0, 0.0],       # -x: leaves the grid at x = 0 with nothing hit
    ])
    d, pts = g.cast(origin, dirs, 40.0)
    # Hits are reported 1 mm inside the voxel (see VoxelGrid.cast), hence the tolerance.
    assert math.isclose(d[0], 4.0, abs_tol=2e-3)
    assert math.isclose(d[1], 0.5, abs_tol=2e-3)
    assert d[2] == -1.0
    assert d[3] == -1.0
    # The wall voxel at x in [5.0, 5.1) has occupied voxels above it: the point
    # stays on the ray. The floor voxel is a surface voxel: its z is the
    # voxel's mid-height, -0.05, not the top face at 0.
    assert np.allclose(pts[0], [5.001, 3.0, 0.5], atol=1e-6)
    assert np.allclose(pts[1], [1.0, 3.0, -0.05], atol=1e-6)


def test_cast_respects_max_range_and_grazing_down():
    g = make_room()
    d, _ = g.cast(np.array([1.0, 3.0, 0.5]), np.array([[1.0, 0.0, 0.0]]), 2.0)
    assert d[0] == -1.0
    # A shallow -7 deg ray from 0.48 m up reaches the floor ~3.9 m out.
    el = math.radians(-7.0)
    d, _ = g.cast(np.array([0.5, 3.0, 0.48]), np.array([[math.cos(el), 0.0, math.sin(el)]]), 40.0)
    assert 3.5 < d[0] < 4.2


def test_ground_under_floor_step_and_hole():
    g = make_room()
    assert math.isclose(g.ground_under(2.0, 3.0, 0.0), 0.0, abs_tol=1e-9)
    assert math.isclose(g.ground_under(8.0, 3.0, 0.0), 0.2, abs_tol=1e-9)
    # A wall column is occupied at every height. Standing on the floor next
    # to it, the ground there is the floor the neighbours agree on (the disc
    # test, not the ground, is what stops the robot at a wall); with nothing
    # else in the patch (a hint up at 1.9) it is the wall's top.
    assert math.isclose(g.ground_under(5.05, 3.0, 0.0), 0.0, abs_tol=1e-9)
    assert math.isclose(g.ground_under(5.05, 3.0, 1.9), 2.0, abs_tol=1e-9)
    assert math.isnan(g.ground_under(20.0, 3.0, 0.0))  # off the map


def test_ground_under_bridges_a_one_column_hole_and_a_rail_post():
    floor = _grid(range(ORG, ORG + 20), range(ORG, ORG + 20), [ORG - 1])
    hole = (floor[:, 0] == ORG + 5) & (floor[:, 1] == ORG + 5)
    post = _grid([ORG + 10], [ORG + 10], range(ORG + 13, ORG + 16))  # 1.3-1.6 m, nothing below
    vox = np.concatenate([floor[~hole], post])
    g = VoxelGrid(vox, RES, z_clip_above_lowest_m=0)
    assert math.isclose(g.ground_under(0.55, 0.55, 0.0), 0.0, abs_tol=1e-9)  # the hole
    assert math.isclose(g.ground_under(1.05, 1.05, 0.0), 0.0, abs_tol=1e-9)  # under the post
    # A 3 x 3 void is still a void.
    big = (floor[:, 0] >= ORG + 14) & (floor[:, 0] <= ORG + 16) & (floor[:, 1] >= ORG + 14) \
        & (floor[:, 1] <= ORG + 16)
    g2 = VoxelGrid(floor[~big], RES, z_clip_above_lowest_m=0)
    assert math.isnan(g2.ground_under(1.55, 1.55, 0.0))


def test_disc_blocked_by_wall_not_by_step():
    g = make_room()
    # Radius 0.25: at x = 4.8 the disc reaches 5.05, into the wall column [5.0, 5.1).
    assert g.disc_blocked(4.8, 3.0, 0.0, 0.25, 0.17, 0.1, 0.6)
    assert g.disc_blocked(4.76, 3.0, 0.0, 0.25, 0.17, 0.1, 0.6)  # touches at 5.01
    assert not g.disc_blocked(4.74, 3.0, 0.0, 0.25, 0.17, 0.1, 0.6)
    assert not g.disc_blocked(4.5, 3.0, 0.0, 0.25, 0.17, 0.1, 0.6)
    # The 0.2 m step is a riser taller than 0.17: blocked at flat policy's allowance,
    # not when 0.3 m steps are allowed.
    assert g.disc_blocked(6.9, 3.0, 0.0, 0.25, 0.17, 0.1, 0.6)
    assert not g.disc_blocked(6.9, 3.0, 0.0, 0.25, 0.3, 0.1, 0.6)


def test_ground_under_sees_through_a_tread_hole_but_takes_a_real_step():
    # A tread at 0.6 m with one cell missing; the step below (0.4 m) shows through.
    lower = _grid(range(ORG, ORG + 10), range(ORG, ORG + 10), [ORG + 3])  # top 0.4
    tread = _grid(range(ORG + 3, ORG + 10), range(ORG, ORG + 10), [ORG + 5])  # top 0.6
    hole = (tread[:, 0] == ORG + 6) & (tread[:, 1] == ORG + 5)
    g = VoxelGrid(np.concatenate([lower, tread[~hole]]), RES, z_clip_above_lowest_m=0)
    assert math.isclose(g.ground_under(0.65, 0.55, 0.6, 0.5, 3.0), 0.6, abs_tol=1e-9)  # the hole
    # One cell past the tread's edge the tread is still a neighbour (the base
    # overhangs by a voxel, as a foot would); two cells out it is a real step.
    assert math.isclose(g.ground_under(0.25, 0.55, 0.6, 0.5, 3.0), 0.6, abs_tol=1e-9)
    assert math.isclose(g.ground_under(0.15, 0.55, 0.6, 0.5, 3.0), 0.4, abs_tol=1e-9)
