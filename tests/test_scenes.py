"""scripts/make_scenes.py: the committed config files are its own, and the
sites are what its docstring says."""

import os
import sys

import numpy as np

from syncai_sim import REPO_DIR
from syncai_sim.world.btree import TREE_MAX_VAL

sys.path.insert(0, os.path.join(REPO_DIR, "scripts"))
import make_scenes as ms  # noqa: E402


def test_the_committed_config_files_are_the_scripts():
    for rel, text in ms.config_files().items():
        with open(os.path.join(REPO_DIR, rel)) as f:
            assert f.read() == text, f"{rel}: rerun scripts/make_scenes.py"


def test_the_stair_has_six_15_cm_risers_up_and_down():
    x = np.arange(0.0, 13.0, 0.01)
    h = ms.stair_profile(x)
    steps = np.diff(h)
    assert np.isclose(steps[steps > 0], 0.15).all() and (steps > 0).sum() == 6
    assert np.isclose(steps[steps < 0], -0.15).all() and (steps < 0).sum() == 6
    assert np.isclose(h.max(), 0.9)
    up = x[1:][steps > 0]
    assert np.allclose(np.diff(up), 0.30, atol=0.011)


def test_each_lane_climbs_at_its_angle():
    s = ms.slope()
    for _, a, yl, yr in ms.slope_lanes():
        y = (yl + yr) / 2
        x = np.array([ms.RAMP_START + 0.2, ms.RAMP_START + 0.3])
        h = s.height(x, np.full(2, y))
        assert np.isclose(np.degrees(np.arctan2(h[1] - h[0], 0.1)), a, atol=0.01)


def test_a_surface_on_a_voxel_boundary_is_in_the_voxel_under_it():
    # The flat floor's top is 0.0: its voxels end there and its points sit in them.
    s = ms.flat()
    vox = ms.voxels(s)
    floor = vox[(vox[:, 0] == TREE_MAX_VAL) & (vox[:, 1] == TREE_MAX_VAL)]
    assert sorted(floor[:, 2] - TREE_MAX_VAL) == [-2, -1]
    pts = ms.points(s)
    near = pts[(np.abs(pts[:, 0]) < 1.0) & (np.abs(pts[:, 1]) < 1.0)]
    assert (np.floor(near[:, 2] / ms.RES) == -1).all()


def test_the_friction_lanes_fill_the_floor_between_the_walls_side_by_side():
    lanes = ms.friction_lanes()
    assert [mu for mu, _, _ in lanes] == list(ms.FRICTION_VALUES)
    assert all(np.isclose(yl - yr, ms.FRICTION_LANE_W) for _, yl, yr in lanes)
    assert all(a[2] == b[1] for a, b in zip(lanes[:-1], lanes[1:]))
    s = ms.friction()
    assert np.isclose(lanes[0][1], s.y1 - ms.WALL_T) and np.isclose(lanes[-1][2], s.y0 + ms.WALL_T)
    # The spawn faces a strip, a metre or more of plain floor before it.
    x, y, _, yaw = s.spawn
    assert yaw == 0.0 and x + 1.0 <= ms.FRICTION_X[0]
    assert any(yr < y < yl for _, yl, yr in lanes)
    # Level everywhere inside the walls.
    xs, ys = np.meshgrid(np.arange(-3.0, 15.0, 0.5), np.arange(-3.5, 3.6, 0.5))
    assert (s.height(xs.ravel(), ys.ravel()) == 0.0).all()


def test_the_courses_sections_follow_one_another_with_plain_floor_between():
    strips = ms.COURSE_STRIPS
    assert all(a[2] <= b[1] for a, b in zip(strips[:-1], strips[1:]))
    hills = [ms.hill_x(start, up, down) for _, start, up, down, _ in ms.COURSE_HILLS]
    assert strips[-1][2] + 1.0 <= hills[0][0]
    assert all(a[3] + 1.0 <= b[0] for a, b in zip(hills[:-1], hills[1:]))
    s = ms.course()
    assert hills[-1][3] + 2.0 <= s.x1 - ms.WALL_T
    # The strips are level; each hill climbs and descends at its angles.
    for _, x0, x1, _ in strips:
        assert (s.height(np.linspace(x0, x1, 20), np.zeros(20)) == 0.0).all()
    for (_, start, up, down, _), (x0, top0, top1, x1) in zip(ms.COURSE_HILLS, hills):
        for x, angle in ((x0 + 0.2, up), (x1 - 0.3, -down)):
            h = s.height(np.array([x, x + 0.1]), np.zeros(2))
            assert np.isclose(np.degrees(np.arctan2(h[1] - h[0], 0.1)), angle, atol=0.01)
        assert np.isclose(s.height(np.array([(top0 + top1) / 2]), np.zeros(1)), ms.RAMP_RISE)


def test_a_hill_with_a_friction_of_its_own_has_it_on_the_way_up_only():
    zones = {name: (box, mu) for name, box, mu in ms.course_zones()}
    for name, start, up, down, mu in ms.COURSE_HILLS:
        key = f"course {name}, the way up"
        if mu is None:
            assert key not in zones
            continue
        box, value = zones[key]
        x0, top0, _, _ = ms.hill_x(start, up, down)
        assert value == mu and box[0] == x0 and np.isclose(box[1], top0, atol=0.01)


def test_all_holds_every_scene_in_a_bay_of_its_own_open_onto_a_level_apron():
    s = ms.all_scene()
    bays = ms.all_bays()
    assert [b[0] for b in bays] == list(ms.ALL_BAYS)
    assert set(ms.ALL_BAYS) == set(ms.SCENES) - {"all"}
    # Side by side from +y, a wall's thickness apart, the outer walls round them.
    assert np.isclose(bays[0][4], s.y1 - ms.WALL_T) and np.isclose(bays[-1][3], s.y0 + ms.WALL_T)
    assert all(np.isclose(a[3] - b[4], ms.WALL_T) for a, b in zip(bays[:-1], bays[1:]))
    for name, b, dy, y0, y1, x_end in bays:
        # Inside its bay a scene is what it is on its own, moved by dy.
        x, y = (a.ravel() for a in np.meshgrid(np.arange(0.05, x_end, 0.1),
                                               np.arange(y0 + 0.05, y1, 0.1)))
        assert (s.height(x, y) == b.height(x, y - dy)).all(), name
        # Walled at its far end, and from the next bay all along it.
        assert s.height(np.array([x_end + 0.05]), np.array([(y0 + y1) / 2]))[0] == ms.WALL_H
        if y0 > s.y0 + ms.WALL_T:
            xs = np.arange(0.05, x_end, 0.1)
            assert (s.height(xs, np.full(xs.size, y0 - ms.WALL_T / 2)) == ms.WALL_H).all(), name
    x, y = (a.ravel() for a in np.meshgrid(np.arange(s.x0 + ms.WALL_T + 0.05, 0.0, 0.1),
                                           np.arange(s.y0 + ms.WALL_T + 0.05, s.y1 - ms.WALL_T, 0.1)))
    assert (s.height(x, y) == 0.0).all()  # the apron
    sx, sy, _, yaw = s.spawn
    assert sx < 0.0 and yaw == 0.0 and s.track[:, 0].max() < 0.0  # start and track on the apron
