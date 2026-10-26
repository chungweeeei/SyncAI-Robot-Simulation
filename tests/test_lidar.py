import os

import numpy as np
import pytest

from syncai_sim.lidar import (DEFAULT_PATTERN_FILE, LidarConfig, LidarModel, LidarPattern,
                              rays_along_motion)
from syncai_sim.livox_bag import build_pattern, save_pattern

from tests.test_livox_bag import sweep_frame


def test_directions_are_unit_and_inside_fov():
    lm = LidarModel(LidarConfig(points_per_scan=5000), seed=3)
    d, t = lm.directions()
    assert d.shape == (5000, 3) and t.shape == (5000,)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-9)
    el = np.degrees(np.arcsin(d[:, 2]))
    assert el.min() >= -7.0 - 1e-6 and el.max() <= 52.0 + 1e-6
    # Roughly even in azimuth: each quadrant gets about a quarter.
    az = np.arctan2(d[:, 1], d[:, 0])
    q = np.histogram(az, bins=4, range=(-np.pi, np.pi))[0]
    assert q.min() > 1000 and q.max() < 1500
    # Fired evenly over the scan, inside the period.
    assert t[0] == 0.0 and np.all(np.diff(t) > 0) and t[-1] < 0.1


def test_scans_differ():
    lm = LidarModel(LidarConfig(points_per_scan=100), seed=0)
    assert not np.allclose(lm.directions()[0], lm.directions()[0])


def test_unknown_pattern_is_refused():
    with pytest.raises(ValueError):
        LidarModel(LidarConfig(pattern="rosette"))


@pytest.fixture
def bank(tmp_path):
    frames = [sweep_frame(az0_deg=0.0, k=0), sweep_frame(az0_deg=90.0, gaps=(8,), k=1)]
    path = str(tmp_path / "bank.npz")
    save_pattern(path, build_pattern(frames, source="unit"))
    return path


def test_recorded_pattern_fires_frame_by_frame_and_cycles(bank):
    pat = LidarPattern(bank)
    assert pat.n_frames == 2
    d0, t0 = pat.frame(0)
    assert d0.shape == (400, 3) and np.allclose(np.linalg.norm(d0, axis=1), 1.0, atol=1e-6)
    assert np.allclose(t0[:3], [0.0, 250e-6, 500e-6])  # 400 samples over 0.1 s
    lm = LidarModel(LidarConfig(pattern="mid360", pattern_file=bank, points_per_scan=20000))
    a, ta = lm.directions()
    b, tb = lm.directions()
    c, _ = lm.directions()
    assert a.shape == (400, 3) and b.shape == (400, 3)  # frame 1's gap was filled
    assert np.allclose(a, d0) and not np.allclose(a, b) and np.allclose(c, a)  # cycles
    assert np.all(np.diff(ta) > 0) and ta[-1] < 0.1 and tb[-1] < 0.1
    # The first sample of frame 1 points 90 deg from frame 0's.
    assert abs(np.degrees(np.arctan2(b[0, 1], b[0, 0])) - 90.0) < 0.02


def test_recorded_pattern_is_thinned_to_points_per_scan(bank):
    lm = LidarModel(LidarConfig(pattern="mid360", pattern_file=bank, points_per_scan=100))
    d, t = lm.directions()
    assert d.shape == (100, 3) and t.shape == (100,)
    assert np.all(np.diff(t) > 0)  # evenly along the sweep, in time order
    assert t[0] == 0.0 and abs(t[-1] - 399 * 250e-6) < 1e-12


def test_committed_mid360_bank_loads():
    if not os.path.isfile(DEFAULT_PATTERN_FILE):
        pytest.skip("config/lidar/mid360_pattern.npz not present")
    pat = LidarPattern(DEFAULT_PATTERN_FILE)
    assert pat.n_frames == 20
    el = np.degrees(np.arcsin(pat.dirs[:, 2]))
    assert -7.5 < el.min() < -6.5 and 52.0 < el.max() < 53.5  # the MID360's -7..52 (datasheet)
    sizes = [pat.frame(k)[0].shape[0] for k in range(pat.n_frames)]
    assert min(sizes) > 15000 and max(sizes) < 21000  # ~20 k fired, ~16 k that returned
    for k in range(pat.n_frames):
        t = pat.frame(k)[1]
        assert np.all(np.diff(t) >= 0) and t[-1] < 0.102
    lm = LidarModel(LidarConfig(pattern="mid360"))
    d, t = lm.directions()
    assert d.shape[0] == sizes[0] and t[-1] < 0.1


def test_rays_along_motion_take_the_pose_of_their_firing_step():
    # Five steps of 10 ms: the lidar moves +x 0.1 m a step and yaws 10 deg a step.
    k, dt = 5, 0.01
    poses = np.tile(np.eye(4), (k, 1, 1))
    for j in range(k):
        yaw = np.radians(10.0 * j)
        poses[j, :3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        poses[j, :3, 3] = [0.1 * j, 0.0, 0.5]
    t = np.array([0.0, 0.021, 0.049, 0.0999])  # steps 0, 2, 4 and (clipped) 4
    dirs_l = np.tile([1.0, 0.0, 0.0], (4, 1))
    o, d = rays_along_motion(poses, t, dt, dirs_l)
    assert np.allclose(o[:, 0], [0.0, 0.2, 0.4, 0.4]) and np.allclose(o[:, 2], 0.5)
    assert np.allclose(np.degrees(np.arctan2(d[:, 1], d[:, 0])), [0.0, 20.0, 40.0, 40.0])
