"""Floors that grip differently: per map, boxes of floor with their own
sliding friction, for walking a controller over wet tiles or ice.

`config/friction/<map>.yaml` lists them:

    zones:
      - name: wet tiles
        box: [2.0, 5.0, -2.0, 2.0]     # x0, x1, y0, y1 (m, map frame)
        friction: 0.4                  # sliding coefficient

Everywhere else, and on a map without such a file, the floor is as it was:
the feet's own friction decides (RobotConfig.foot_friction). Where zones
overlap, the later one in the file wins.

How it reaches the contact. MuJoCo takes, of the two geoms that touch, the
larger friction, so a slippery floor under a foot of friction 1.0 would not be
slippery at all; unless one geom has the higher `priority`, and then its
friction is used alone. So on a map with zones the terrain pool's boxes are
given priority 1 and, each time the pool is dealt under the robot
(terrain_patch.py), every box gets the sliding friction of where it now
lies; torsional and rolling stay the feet's, so outside the zones a contact
is exactly what the larger-of-two rule gave it. Both are arrays of the
compiled model written at run time, so the cached scene (scene.py) is the
same with and without zones.

The fitted stairs' boxes (stairs.py) are static geoms, not of the pool: a
zone does not reach them. A box is one 0.1 m voxel, so a zone's edge is where
its box's centre falls: within 0.05 m of the file's number.
"""

from __future__ import annotations

import os

import numpy as np

from syncai_sim import REPO_DIR

FRICTION_DIR = os.path.join(REPO_DIR, "config", "friction")

Zone = tuple[str, tuple[float, float, float, float], float]


def zones_path(map_dir: str, friction_dir: str = FRICTION_DIR) -> str:
    """The map's zones file, named after the map's folder as its ramps are."""
    return os.path.join(friction_dir, os.path.basename(os.path.normpath(map_dir)) + ".yaml")


def read_zones(path: str) -> list[Zone]:
    """(name, (x0, x1, y0, y1), sliding friction) per zone of a file (module
    docstring); [] without one."""
    if not os.path.isfile(path):
        return []
    import yaml

    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    zones = []
    for z in spec.get("zones", []):
        box = tuple(float(v) for v in z["box"])
        mu = float(z["friction"])
        if len(box) != 4 or box[0] >= box[1] or box[2] >= box[3]:
            raise ValueError(f"{path}: zone {z['name']!r}: box is x0, x1, y0, y1 with x0 < x1, y0 < y1")
        if not 0.0 < mu <= 2.0:
            # 0 is no friction at all, which MuJoCo's cone does not take; above 2 is no floor.
            raise ValueError(f"{path}: zone {z['name']!r}: friction {mu} is not within (0, 2]")
        zones.append((str(z["name"]), box, mu))
    return zones


def sliding(zones: list[Zone], xy: np.ndarray, default: float) -> np.ndarray:
    """The sliding friction at each of `xy` (K, 2): `default` outside every
    zone, the last listed zone's where several overlap."""
    mu = np.full(xy.shape[0], float(default))
    for _, (x0, x1, y0, y1), value in zones:
        inside = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
        mu[inside] = value
    return mu
