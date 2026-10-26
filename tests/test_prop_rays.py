import numpy as np

from syncai_sim.prop_rays import ray_aabb, ray_vertical_cylinder


def test_ray_aabb_hits_and_misses():
    o = np.array([0.0, 0.0, 0.5])
    d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    lo, hi = np.array([2.0, -0.5, 0.0]), np.array([3.0, 0.5, 1.0])
    t = ray_aabb(o, d, lo, hi)
    assert np.isclose(t[0], 2.0)
    assert np.isinf(t[1]) and np.isinf(t[2])
    # A ray with a zero component along an axis the box spans still hits.
    assert np.isclose(t[3], 2.0)


def test_one_origin_per_ray():
    d = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    origins = np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [2.4, 0.0, 3.0]])
    lo, hi = np.array([2.0, -0.5, 0.0]), np.array([3.0, 0.5, 1.0])
    assert np.allclose(ray_aabb(origins, d, lo, hi), [2.0, 1.0, 2.0])
    t = ray_vertical_cylinder(origins, d, np.array([3.0, 0.0]), 0.5, 0.0, 1.8)
    assert np.allclose(t[:2], [2.5, 1.5]) and np.isinf(t[2])  # the third is outside the radius


def test_ray_cylinder_side_and_cap():
    o = np.array([0.0, 0.0, 1.0])
    d = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    t = ray_vertical_cylinder(o, d, np.array([3.0, 0.0]), 0.5, 0.0, 1.8)
    assert np.isclose(t[0], 2.5)
    # Too high: the ray at z = 1 passes a cylinder that ends at 0.8.
    t2 = ray_vertical_cylinder(o, d, np.array([3.0, 0.0]), 0.5, 0.0, 0.8)
    assert np.isinf(t2[0])
    # Straight down onto the top cap of a cylinder under the origin.
    t3 = ray_vertical_cylinder(o, d, np.array([0.0, 0.0]), 0.5, 0.0, 0.3)
    assert np.isclose(t3[2], 0.7)
