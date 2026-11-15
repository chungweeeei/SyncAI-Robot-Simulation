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
