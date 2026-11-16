#!/usr/bin/env python3
"""Test sites made rather than mapped: level ground, ramps, a 15 cm stair,
floors of different grip, a course of several of these in a row, and all of
them side by side in one site.

    python3 scripts/make_scenes.py              # all of them into scenes/<name>/
    python3 scripts/make_scenes.py stairs15     # one
    python3 -m syncai_sim --profile /home/syncrobotic/sim/config/params/scenes/stairs15.yaml

Each is written as a mapped site is, so the sim reads it the same way
(CLAUDE.md, rule 2): `octomap.bt` (the voxels: the lidar, the ground, the
viewer), `map.pcd` (the surface's true heights, which surface_heights.py
puts under the feet) and `poses.txt` (a mapping track on the level floor).
What a mapped site keeps in this repository is written here too, from the
same numbers: the ramps' boxes (`config/ramps/<name>.yaml`), the floors'
friction (`config/friction/<name>.yaml`, terrains/friction.py), and the
stair's walk and tape measure (`config/stair_walks/stairs15.yaml`,
`config/stair_shapes/stairs15.yaml`), so its flights are fitted to
0.15 m risers and built as boxes (terrains/stairs.py). The spawn is in
`config/params/scenes/<name>.yaml`.

Every site is a height field: per 0.1 m column a top, solid down to
`FLOOR_BOTTOM`; walls and pillars are tall columns. The voxels hold each
column's height at its centre; the points sample the surface itself every
`POINT_STEP`, `POINT_DROP` under it (a top on a voxel boundary, 0.0 or
0.3 m, would otherwise put its points in the free voxel above). Walls are
there for the lidar too: on an open plane Point-LIO has nothing to hold to.

    flat      20 x 8 m level floor, four pillars to the sides
    slope     four lanes side by side, 5 / 10 / 15 / 20 deg up to a 0.5 m
              platform and down again, walls between them
    stairs15  six 0.15 m risers on 0.30 m treads up to a 0.9 m platform and
              six down, 1.6 m wide between walls
    friction  level floor, four lanes side by side with a 6 m strip each of
              sliding friction 0.6 / 0.4 / 0.2 / 0.1
    course    one corridor along x with, in turn: a strip of friction 0.4, one
              of 0.2, a hill (10 deg up, 15 deg down), and a second hill
              (10 deg both ways) whose way up has friction 0.4
    all       the five above as bays side by side (from +y: flat, friction,
              slope, stairs15, course), each as it is on its own, open onto
              one level apron at x < 0 where the robot starts: one site, one
              sim, every test a walk away

The feet stand on 0.1 m boxes whatever the site (terrains/terrain_patch.py),
so a ramp is a stair of 0.1 m treads with risers of 0.1 tan(angle): 0.9 cm
at 5 deg, 3.6 cm at 20 deg. The stair's flights are boxes of their own, at
the fitted risers.

The output directories are generated, not committed (`.gitignore`); the
config files are committed, and `tests/test_scenes.py` checks they are this
script's.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)

from syncai_sim.world.btree import TREE_MAX_VAL, write_occupied  # noqa: E402

RES = 0.1  # the voxel, as dp's octomap
FLOOR_BOTTOM = -0.2  # m: two voxels of slab under the floor
POINT_STEP = 0.025  # m between the points on the surface (16 per voxel top)
POINT_DROP = 0.0005  # m under the surface (module docstring)
TRACK_STEP = 0.2  # m between keyframes, as surface_heights.WALK_STEP
LIDAR_HEIGHT = 0.481  # poses.txt is the IMU: ground + this
WALL_T = 0.3  # m, the outer walls
WALL_H = 2.0

# slope: lanes along x, each up `RAMP_RISE` at its angle from `RAMP_START`, a
# platform, and down at the same angle.
RAMP_ANGLES = (5.0, 10.0, 15.0, 20.0)  # deg, from +y to -y
RAMP_RISE = 0.5
RAMP_START = 2.0
RAMP_TOP = 2.0  # m of platform
LANE_EDGES = (4.7, 2.6, 2.2, 0.2, -0.2, -2.2, -2.6, -4.7)  # y, (left, right) per lane
DIVIDER_H = 1.5  # m, the walls between lanes (4 columns thick: a ramp's mean never reads the next lane)

# friction: a strip per lane, from +y to -y. The controller's MPC and WBC plan
# for 0.6 (its task.info, frictionCone): at it, and three steps under it.
FRICTION_VALUES = (0.6, 0.4, 0.2, 0.1)
FRICTION_LANE_W = 2.0
FRICTION_X = (3.0, 9.0)  # the strips' start and end

# course: sections along x in one corridor, each after a stretch of plain floor.
COURSE_HALF_W = 2.0
COURSE_STRIPS = (("wet floor", 2.0, 5.0, 0.4), ("ice", 5.0, 8.0, 0.2))  # name, x0, x1, friction
COURSE_HILLS = (("hill", 9.5, 10.0, 15.0, None),  # name, start, deg up, deg down, friction up
                ("slippery hill", 17.5, 10.0, 10.0, 0.4))

# stairs15: up along x from STAIR_START, the platform, down.
RISER = 0.15
TREAD = 0.30
RISERS = 6
STAIR_START = 3.0
STAIR_TOP = 3.0  # m of platform
STAIR_HALF_W = 0.8  # = terrains/stairs.MAX_HALF_WIDTH: the boxes cover it all
STAIR_WALL_T = 0.3
STAIR_WALL_H = 1.0  # m over the platform


class Scene:
    def __init__(self, name, x0, x1, y0, y1, height, track, spawn, about):
        self.name, self.x0, self.x1, self.y0, self.y1 = name, x0, x1, y0, y1
        self.height = height  # (x, y) arrays -> top (m)
        self.inner = None  # the same without the outer walls (`all` puts it in a bay)
        self.track = track  # (K, 2) keyframes on the level floor
        self.spawn = spawn  # x y z yaw
        self.about = about


def _walls(x, y, s: Scene, h):
    inside = (x > s.x0 + WALL_T) & (x < s.x1 - WALL_T) & (y > s.y0 + WALL_T) & (y < s.y1 - WALL_T)
    return np.where(inside, h, WALL_H)


def _boxes(x, y, h, boxes):
    for bx0, bx1, by0, by1, top in boxes:
        h = np.where((x >= bx0) & (x < bx1) & (y >= by0) & (y < by1), np.maximum(h, top), h)
    return h


def _line(x0, x1, y):
    n = int(round((x1 - x0) / TRACK_STEP)) + 1
    return np.stack([np.linspace(x0, x1, n), np.full(n, y)], axis=1)


# --- flat ------------------------------------------------------------------
FLAT_PILLARS = [(2.7, 3.3, 2.2, 2.8, 1.2), (5.7, 6.3, -2.8, -2.2, 1.2),
                (8.7, 9.3, 2.2, 2.8, 1.2), (11.7, 12.3, -2.8, -2.2, 1.2)]


def flat() -> Scene:
    s = Scene("flat", -4.0, 16.0, -4.0, 4.0, None, _line(-3.0, 15.0, 0.0), (0.0, 0.0, 0.0, 0.0),
              "20 x 8 m of level floor, four pillars to the sides")
    s.inner = lambda x, y: _boxes(x, y, np.zeros_like(x), FLAT_PILLARS)
    s.height = lambda x, y: _walls(x, y, s, s.inner(x, y))
    return s


# --- slope -----------------------------------------------------------------
def ramp_length(angle_deg: float) -> float:
    return RAMP_RISE / math.tan(math.radians(angle_deg))


def _lane_profile(x, angle_deg):
    run = ramp_length(angle_deg)
    up = np.clip((x - RAMP_START) / run, 0.0, 1.0)
    down = np.clip(1.0 - (x - (RAMP_START + run + RAMP_TOP)) / run, 0.0, 1.0)
    return RAMP_RISE * np.minimum(up, down)


def slope_lanes():
    """(name, angle, y_left, y_right) per lane."""
    e = LANE_EDGES
    return [(f"{a:g} deg lane", a, e[2 * i], e[2 * i + 1]) for i, a in enumerate(RAMP_ANGLES)]


def _slope_end() -> float:
    return RAMP_START + 2 * ramp_length(min(RAMP_ANGLES)) + RAMP_TOP


def slope() -> Scene:
    lanes = slope_lanes()
    x1 = math.ceil(_slope_end() + 3.0)
    s = Scene("slope", -4.0, float(x1), -5.0, 5.0, None, _line(-3.0, 1.0, 0.0),
              (0.0, 1.2, 0.0, 0.0),
              "lanes at " + " / ".join(f"{a:g}" for a in RAMP_ANGLES)
              + f" deg up to a {RAMP_RISE} m platform and down, walls between them")

    def inner(x, y):
        h = np.zeros_like(x)
        for _, a, yl, yr in lanes:
            h = np.where((y < yl) & (y >= yr), _lane_profile(x, a), h)
        divider = (x >= RAMP_START - 0.5) & (x < _slope_end() + 0.5)
        for (_, _, _, yr), (_, _, yl, _) in zip(lanes[:-1], lanes[1:]):
            h = np.where(divider & (y < yr) & (y >= yl), DIVIDER_H, h)
        return h

    s.inner = inner
    s.height = lambda x, y: _walls(x, y, s, inner(x, y))
    return s


def ramp_boxes():
    """(name, box, band) per lane for config/ramps/slope.yaml."""
    out = []
    for name, a, yl, yr in slope_lanes():
        end = RAMP_START + 2 * ramp_length(a) + RAMP_TOP
        out.append((f"slope {name}", [RAMP_START - 0.3, round(end + 0.3, 2), yr, yl],
                    [-0.1, RAMP_RISE + 0.1]))
    return out


# --- friction --------------------------------------------------------------
def friction_lanes():
    """(friction, y_left, y_right) per lane, from +y to -y."""
    top = FRICTION_LANE_W * len(FRICTION_VALUES) / 2
    return [(mu, top - i * FRICTION_LANE_W, top - (i + 1) * FRICTION_LANE_W)
            for i, mu in enumerate(FRICTION_VALUES)]


def friction() -> Scene:
    lanes = friction_lanes()
    half = lanes[0][1]
    s = Scene("friction", -4.0, 16.0, -half - WALL_T, half + WALL_T, None, _line(-3.0, 15.0, 0.0),
              (0.0, 1.0, 0.0, 0.0),
              f"level floor, lanes {FRICTION_LANE_W:g} m wide with a strip of sliding friction "
              + " / ".join(f"{mu:g}" for mu in FRICTION_VALUES)
              + f" from x {FRICTION_X[0]:g} to {FRICTION_X[1]:g}")
    s.inner = lambda x, y: np.zeros_like(x)
    s.height = lambda x, y: _walls(x, y, s, s.inner(x, y))
    return s


def friction_zones():
    """(name, box, friction) per lane for config/friction/friction.yaml."""
    return [(f"friction {mu:g} lane", [FRICTION_X[0], FRICTION_X[1], yr, yl], mu)
            for mu, yl, yr in friction_lanes()]


# --- course ----------------------------------------------------------------
def hill_x(start: float, up_deg: float, down_deg: float):
    """x of a hill's foot, of its platform's start and end, and of its far foot."""
    top0 = start + ramp_length(up_deg)
    top1 = top0 + RAMP_TOP
    return start, top0, top1, top1 + ramp_length(down_deg)


def hill_profile(x, start: float, up_deg: float, down_deg: float):
    x0, top0, top1, x1 = hill_x(start, up_deg, down_deg)
    up = np.clip((x - x0) / (top0 - x0), 0.0, 1.0)
    down = np.clip((x1 - x) / (x1 - top1), 0.0, 1.0)
    return RAMP_RISE * np.minimum(up, down)


def _course_end() -> float:
    return max(hill_x(start, up, down)[3] for _, start, up, down, _ in COURSE_HILLS)


def course() -> Scene:
    s = Scene("course", -4.0, float(math.ceil(_course_end() + 3.0)), -COURSE_HALF_W - WALL_T,
              COURSE_HALF_W + WALL_T, None, _line(-3.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0),
              "one corridor along x: "
              + ", ".join(f"friction {mu:g} from x {x0:g} to {x1:g}" for _, x0, x1, mu in COURSE_STRIPS)
              + "".join(f", a {RAMP_RISE} m hill at x {start:g} ({up:g} deg up, {down:g} down"
                        + (f", friction {mu:g} on the way up)" if mu else ")")
                        for _, start, up, down, mu in COURSE_HILLS))

    def inner(x, y):
        h = np.zeros_like(x)
        for _, start, up, down, _ in COURSE_HILLS:
            h = np.maximum(h, hill_profile(x, start, up, down))
        return h

    s.inner = inner
    s.height = lambda x, y: _walls(x, y, s, inner(x, y))
    return s


def course_ramp_boxes():
    """(name, box, band) per hill for config/ramps/course.yaml."""
    return [(f"course {name}", [round(hill_x(start, up, down)[0] - 0.3, 2),
                                round(hill_x(start, up, down)[3] + 0.3, 2),
                                -COURSE_HALF_W, COURSE_HALF_W], [-0.1, RAMP_RISE + 0.1])
            for name, start, up, down, _ in COURSE_HILLS]


def course_zones():
    """(name, box, friction) for config/friction/course.yaml: the strips, and
    the way up of every hill that has a friction of its own."""
    zones = [(f"course {name}", [x0, x1, -COURSE_HALF_W, COURSE_HALF_W], mu)
             for name, x0, x1, mu in COURSE_STRIPS]
    for name, start, up, down, mu in COURSE_HILLS:
        if mu is not None:
            x0, top0, _, _ = hill_x(start, up, down)
            zones.append((f"course {name}, the way up", [x0, round(top0, 2), -COURSE_HALF_W,
                                                         COURSE_HALF_W], mu))
    return zones


# --- stairs15 --------------------------------------------------------------
def stair_x():
    """x of the first riser up, of the platform's start, of the first riser
    down, and of the floor after the last."""
    top0 = STAIR_START + (RISERS - 1) * TREAD
    down0 = top0 + STAIR_TOP
    return STAIR_START, top0, down0, down0 + (RISERS - 1) * TREAD


def stair_profile(x):
    up0, _, down0, _ = stair_x()
    k_up = np.clip(np.floor((x - up0) / TREAD) + 1, 0, RISERS)
    k_down = np.clip(RISERS - (np.floor((x - down0) / TREAD) + 1), 0, RISERS)
    return RISER * np.minimum(k_up, k_down)


def stairs15() -> Scene:
    up0, top0, down0, end = stair_x()
    s = Scene("stairs15", -4.0, math.ceil(end + 4.0), -3.0, 3.0, None, _line(-3.0, up0 - 0.6, 0.0),
              (0.0, 0.0, 0.0, 0.0),
              f"{RISERS} risers of {RISER} m on {TREAD} m treads up to a {RISERS * RISER:.2f} m "
              f"platform and {RISERS} down, {2 * STAIR_HALF_W} m wide between walls")

    def inner(x, y):
        h = np.where(np.abs(y) < STAIR_HALF_W, stair_profile(x), 0.0)
        wall = ((np.abs(y) >= STAIR_HALF_W) & (np.abs(y) < STAIR_HALF_W + STAIR_WALL_T)
                & (x >= up0 - TREAD) & (x < end + TREAD))
        return np.where(wall, RISERS * RISER + STAIR_WALL_H, h)

    s.inner = inner
    s.height = lambda x, y: _walls(x, y, s, inner(x, y))
    return s


def stair_walk():
    """The walk up and down the stair's middle (config/stair_walks/stairs15.yaml):
    level a metre either side of each flight, the ground a line over it."""
    up0, top0, down0, end = stair_x()
    top = RISERS * RISER
    return [[up0 - 2.0, 0.0, 0.0], [up0 - 0.15, 0.0, 0.0], [top0 + 0.15, 0.0, top],
            [down0 - 0.15, 0.0, top], [end + 0.15, 0.0, 0.0], [end + 2.0, 0.0, 0.0]]


# --- all -------------------------------------------------------------------
# The bays from +y to -y. Each keeps its own x (every test starts 1.5 m or
# more past x 0, its entrance) and the y between its own walls; a wall
# WALL_T thick between two bays, and one across each bay's far end.
ALL_BAYS = ("flat", "friction", "slope", "stairs15", "course")
ALL_APRON = 6.0  # m of level floor before the bays, the whole site wide: the start


def all_bays():
    """(name, scene, dy, y0, y1, x_end) per bay: its scene, the shift from the
    scene's y to the site's, the bay's y between its walls and its far wall's x."""
    scenes = [SCENES[name]() for name in ALL_BAYS]
    widths = [(s.y1 - WALL_T) - (s.y0 + WALL_T) for s in scenes]
    top = round((sum(widths) + WALL_T * (len(scenes) - 1)) / 2, 6)
    out = []
    for name, s, w in zip(ALL_BAYS, scenes, widths):
        dy = top - (s.y1 - WALL_T)
        out.append((name, s, dy, top - w, top, s.x1 - WALL_T))
        top = round(top - w - WALL_T, 6)
    return out


def all_scene() -> Scene:
    bays = all_bays()
    half = bays[0][4]
    track_y = np.arange(-half + 1.0, half - 1.0 + 1e-9, TRACK_STEP)
    s = Scene("all", -ALL_APRON - WALL_T, max(b[5] for b in bays) + WALL_T, -half - WALL_T,
              half + WALL_T, None,
              np.stack([np.full(track_y.size, -ALL_APRON / 2), track_y], axis=1),
              (-ALL_APRON / 2, 0.0, 0.0, 0.0),
              "bays side by side, from +y " + ", ".join(
                  f"{name} (y {y0:.1f} .. {y1:.1f})" for name, _, _, y0, y1, _ in bays)
              + f", each as its own scene, open onto a {ALL_APRON:g} m level apron at x < 0")

    def inner(x, y):
        h = np.where(x >= 0.0, WALL_H, 0.0)  # the walls between the bays, unless a bay is there
        for _, b, dy, y0, y1, x_end in bays:
            in_bay = (y >= y0) & (y < y1) & (x >= 0.0)
            h = np.where(in_bay, np.where(x < x_end, b.inner(x, y - dy), WALL_H), h)
        return h

    s.inner = inner
    s.height = lambda x, y: _walls(x, y, s, inner(x, y))
    return s


def _bay_dy(name: str) -> float:
    return next(dy for n, _, dy, _, _, _ in all_bays() if n == name)


def _shift_y(box, dy):
    """[x0, x1, y0, y1] with y moved by dy."""
    return [box[0], box[1], box[2] + dy, box[3] + dy]


def all_ramp_boxes():
    """config/ramps/all.yaml: the slope's lanes and the course's hills in their bays."""
    return ([(f"all {n}", _shift_y(b, _bay_dy("slope")), band) for n, b, band in ramp_boxes()]
            + [(f"all {n}", _shift_y(b, _bay_dy("course")), band)
               for n, b, band in course_ramp_boxes()])


def all_zones():
    """config/friction/all.yaml: the friction lanes and the course's floors in their bays."""
    return ([(f"all {n}", _shift_y(b, _bay_dy("friction")), mu) for n, b, mu in friction_zones()]
            + [(f"all {n}", _shift_y(b, _bay_dy("course")), mu) for n, b, mu in course_zones()])


SCENES = {"flat": flat, "slope": slope, "stairs15": stairs15, "friction": friction,
          "course": course, "all": all_scene}


# --- writing ---------------------------------------------------------------
def voxels(s: Scene) -> np.ndarray:
    i0, i1 = (int(round(v / RES)) for v in (s.x0, s.x1))
    j0, j1 = (int(round(v / RES)) for v in (s.y0, s.y1))
    ix, iy = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()
    top = s.height((ix + 0.5) * RES, (iy + 0.5) * RES)
    k_top = np.floor((top - POINT_DROP) / RES).astype(np.int64)  # the voxel the surface is in
    k_bottom = int(round(FLOOR_BOTTOM / RES))
    n = k_top - k_bottom + 1
    col = np.repeat(np.arange(ix.size), n)
    k = k_bottom + np.arange(col.size) - np.repeat(np.cumsum(n) - n, n)
    return np.stack([ix[col], iy[col], k], axis=1) + TREE_MAX_VAL


def points(s: Scene) -> np.ndarray:
    xs = np.arange(s.x0 + POINT_STEP / 2, s.x1, POINT_STEP)
    ys = np.arange(s.y0 + POINT_STEP / 2, s.y1, POINT_STEP)
    x, y = (a.ravel() for a in np.meshgrid(xs, ys, indexing="ij"))
    return np.stack([x, y, s.height(x, y) - POINT_DROP], axis=1).astype(np.float32)


def write_pcd(path: str, pts: np.ndarray) -> None:
    """Binary x y z intensity, float32: what Point-LIO / clean_map write and
    visuals/pcd_mesh.read_pcd reads."""
    data = np.concatenate([pts, np.full((pts.shape[0], 1), 50.0, np.float32)], axis=1)
    n = pts.shape[0]
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\n"
              "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii") + data.astype("<f4").tobytes())


def write_poses(path: str, track: np.ndarray) -> None:
    """poses.txt as the workspace's mapping writes it: name, IMU x y z, the
    orientation (w x y z) level."""
    with open(path, "w") as f:
        for k, (x, y) in enumerate(track):
            f.write(f"{k}.pcd {x:.4f} {y:.4f} {LIDAR_HEIGHT:.4f} 1 0 0 0\n")


def _num(v) -> str:
    return f"{round(float(v), 4):g}"


def config_files() -> dict[str, str]:
    """Repository-relative path -> content of the committed config files."""
    head = "# Written by scripts/make_scenes.py from its constants; edit there and rerun.\n"
    def ramps_file(about: str, boxes) -> str:
        text = head + about + "\n\nramps:\n"
        for name, box, band in boxes:
            text += (f"  - name: {name}\n    box: [{', '.join(_num(v) for v in box)}]\n"
                     f"    band: [{', '.join(_num(v) for v in band)}]\n")
        return text

    def friction_file(about: str, zones) -> str:
        text = head + about + "\n\nzones:\n"
        for name, box, mu in zones:
            text += (f"  - name: {name}\n    box: [{', '.join(_num(v) for v in box)}]\n"
                     f"    friction: {_num(mu)}\n")
        return text

    ramps = ramps_file(
        "# The slope scene's lanes (config/ramps/dp.yaml says what a ramp box does):\n"
        "# the band holds the floor, the ramp and the platform, not the walls.", ramp_boxes())
    def walk_file(name: str, dy: float) -> str:
        text = head + ("# The stair is never climbed by the mapping track (poses.txt stays on the\n"
                       "# level floor, so the platform is no storey of its own), so it is fitted\n"
                       "# from this walk, as dp's middle stair is.\n\nwalks:\n"
                       f"  - name: {name} up and down\n    points:\n")
        for x, y, z in stair_walk():
            text += f"      - [{', '.join(_num(v) for v in (x, y + dy, z))}]\n"
        return text

    def shape_file(name: str, dy: float) -> str:
        up0, _, _, end = stair_x()
        return head + ("# The stair as built, so both flights are fitted to it.\n\nstairs:\n"
                       f"  - name: {name}\n"
                       f"    box: [{_num(up0 - 1.0)}, {_num(end + 1.0)}, {round(dy - 1.0, 4)}, "
                       f"{round(dy + 1.0, 4)}]\n"
                       f"    tread: {_num(TREAD)}\n    riser: {_num(RISER)}\n")

    walk, shape = walk_file("stairs15", 0.0), shape_file("stairs15", 0.0)
    out = {"config/ramps/slope.yaml": ramps, "config/stair_walks/stairs15.yaml": walk,
           "config/stair_shapes/stairs15.yaml": shape,
           "config/ramps/course.yaml": ramps_file(
               "# The course scene's hills: the band holds the floor, the ramps and the platforms.",
               course_ramp_boxes()),
           "config/friction/friction.yaml": friction_file(
               "# The friction scene's lanes (syncai_sim/terrains/friction.py says what a zone does):\n"
               "# box is x0, x1, y0, y1; elsewhere the floor is the feet's own friction.",
               friction_zones()),
           "config/friction/course.yaml": friction_file(
               "# The course scene's strips of floor, and the way up its slippery hill.",
               course_zones()),
           "config/ramps/all.yaml": ramps_file(
               "# The all scene's ramps: the slope's lanes and the course's hills, in their bays.",
               all_ramp_boxes()),
           "config/friction/all.yaml": friction_file(
               "# The all scene's floors: the friction lanes and the course's, in their bays.",
               all_zones()),
           "config/stair_walks/all.yaml": walk_file("all stairs15", _bay_dy("stairs15")),
           "config/stair_shapes/all.yaml": shape_file("all stairs15", _bay_dy("stairs15"))}
    for name, make in SCENES.items():
        s = make()
        x, y, z, yaw = s.spawn
        out[f"config/params/scenes/{name}.yaml"] = (
            head + f"# The {name} scene: {s.about}.\n"
            f"#   python3 -m syncai_sim --profile "
            f"/home/syncrobotic/sim/config/params/scenes/{name}.yaml\n\n"
            f"map_dir: /home/syncrobotic/sim/scenes/{name}\n"
            "use_ini_initial_pose: false\n"
            f"spawn: [{', '.join(_num(v) for v in (x, y, z, yaw))}]   # x y z(hint) yaw\n")
    return out


def write_scene(s: Scene, out_dir: str) -> None:
    d = os.path.join(out_dir, s.name)
    os.makedirs(d, exist_ok=True)
    vox = voxels(s)
    write_occupied(os.path.join(d, "octomap.bt"), vox, RES)
    pts = points(s)
    write_pcd(os.path.join(d, "map.pcd"), pts)
    write_poses(os.path.join(d, "poses.txt"), s.track)
    print(f"{d}: {s.about}; {vox.shape[0]} voxels, {pts.shape[0]} points, "
          f"{s.track.shape[0]} keyframes")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", metavar="name",
                    help=f"scenes to write, of {', '.join(SCENES)} (default: all)")
    ap.add_argument("--out", default=os.path.join(REPO, "scenes"))
    a = ap.parse_args(argv)
    unknown = [n for n in a.names if n not in SCENES]
    if unknown:
        ap.error(f"no scene {', '.join(unknown)}")
    for name in a.names or SCENES:
        write_scene(SCENES[name](), os.path.abspath(a.out))
    for rel, text in config_files().items():
        path = os.path.join(REPO, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
