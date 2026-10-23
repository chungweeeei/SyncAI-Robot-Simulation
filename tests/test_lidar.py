import numpy as np

from syncai_sim.lidar import LidarConfig, LidarModel


def test_directions_are_unit_and_inside_fov():
    lm = LidarModel(LidarConfig(points_per_scan=5000), seed=3)
    d = lm.directions()
    assert d.shape == (5000, 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-9)
    el = np.degrees(np.arcsin(d[:, 2]))
    assert el.min() >= -7.0 - 1e-6 and el.max() <= 52.0 + 1e-6
    # Roughly even in azimuth: each quadrant gets about a quarter.
    az = np.arctan2(d[:, 1], d[:, 0])
    q = np.histogram(az, bins=4, range=(-np.pi, np.pi))[0]
    assert q.min() > 1000 and q.max() < 1500


def test_scans_differ():
    lm = LidarModel(LidarConfig(points_per_scan=100), seed=0)
    assert not np.allclose(lm.directions(), lm.directions())


def test_returns_filter_and_noise():
    lm = LidarModel(LidarConfig(points_per_scan=4, range_noise_sigma=0.0), seed=0)
    dirs = np.eye(3)[[0, 1, 2, 0]].astype(float)
    dist = np.array([-1.0, 0.05, 5.0, 45.0])  # miss, blind zone, ok, beyond reach
    pts = dirs * np.where(dist > 0, dist, 0)[:, None]
    kept = lm.returns(dirs, dist, pts)
    assert kept.shape == (1, 3) and np.allclose(kept[0], [0.0, 0.0, 5.0])


def test_cloud_in_body_frame():
    lm = LidarModel(LidarConfig(points_per_scan=1, range_noise_sigma=0.0))
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    yaw = np.pi / 2
    T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    # A world point 2 m "ahead" of a body facing +y is at body x = 2.
    xyzi = lm.cloud_xyzi(np.array([[1.0, 4.0, 3.0]]), T)
    assert xyzi.dtype == np.float32
    assert np.allclose(xyzi[0], [2.0, 0.0, 0.0, 100.0], atol=1e-6)
