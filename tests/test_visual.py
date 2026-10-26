import numpy as np
import pytest

from syncai_sim import visual
from syncai_sim.map_cloud import thin
from syncai_sim.pcd_mesh import read_pcd


def test_ramp_ends():
    assert np.allclose(visual.ramp(np.array([0.0]))[0], visual.PALETTE[0][1])
    assert np.allclose(visual.ramp(np.array([1.0, 7.0]))[1], visual.PALETTE[-1][1])


def test_height_range_is_clamped():
    lo, hi = visual.height_range(np.array([0.0, 0.1]))  # one storey: still a 3 m ramp
    assert lo == pytest.approx(0.0, abs=0.01) and hi - lo == pytest.approx(visual.MIN_RANGE_M)
    lo, hi = visual.height_range(np.linspace(0.0, 30.0, 1000))
    assert hi - lo == pytest.approx(visual.MAX_RANGE_M)


def test_intensity_factor_spans_dark_to_bright_and_defaults_unknown():
    f = visual.intensity_factor(np.array([0.0, visual.INTENSITY_LO, visual.INTENSITY_HI, 250.0,
                                          np.nan]))
    assert f[0] == pytest.approx(visual.FACTOR_LO) and f[1] == pytest.approx(visual.FACTOR_LO)
    assert f[2] == pytest.approx(visual.FACTOR_HI) and f[3] == pytest.approx(visual.FACTOR_HI)
    assert f[4] == pytest.approx(visual.UNKNOWN_FACTOR)
    assert np.all(np.diff(visual.intensity_factor(np.linspace(0, 150, 30))) >= 0)


def test_shade_by_kind_height_and_intensity():
    kind = np.array([visual.KIND_FLOOR, visual.KIND_FLOOR, visual.KIND_WALL, visual.KIND_UNDER])
    z = np.array([0.0, 10.0, 0.0, 0.0])
    plain = visual.shade(kind, z, None, (0.0, 10.0))
    assert plain.dtype == np.uint8 and plain.shape == (4, 3)
    assert np.array_equal(plain[0], (np.array(visual.PALETTE[0][1]) * 255 + 0.5).astype(np.uint8))
    assert np.array_equal(plain[1], (np.array(visual.PALETTE[-1][1]) * 255 + 0.5).astype(np.uint8))
    assert np.all(plain[2] > plain[0])  # the wall is a pale copy
    assert np.array_equal(plain[3], (visual.UNDER_GREY * 255 + 0.5).astype(np.uint8))
    shaded = visual.shade(kind, z, np.array([10.0, 120.0, 10.0, 10.0]), (0.0, 10.0))
    assert np.all(shaded[0] < plain[0])  # a dark floor voxel
    assert np.all(shaded[1] >= plain[1])  # a marking: brighter, clipped at white


def test_surface_kinds_by_normal():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    f = np.array([[0, 1, 2],    # normal +z: floor
                  [0, 2, 1],    # -z: underside
                  [0, 1, 3]])   # -y: wall
    assert visual.surface_kinds(v, f).tolist() == [visual.KIND_FLOOR, visual.KIND_UNDER,
                                                   visual.KIND_WALL]


def test_read_pcd_binary(tmp_path):
    pts = np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.7]], np.float32)
    head = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
            "COUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2\nDATA binary\n")
    p = tmp_path / "m.pcd"
    p.write_bytes(head.encode() + pts.tobytes())
    assert np.allclose(read_pcd(str(p)), pts[:, :3])
    p.write_bytes(head.replace("binary", "ascii").encode())
    with pytest.raises(ValueError):
        read_pcd(str(p))


def test_thin_averages_one_point_per_cell():
    pts = np.array([[0.01, 0.01, 0.0], [0.03, 0.03, 0.0], [0.5, 0.0, 0.0]], np.float32)
    val = np.array([10.0, 30.0, 7.0], np.float32)
    p, v = thin(pts, val, 0.1)
    order = np.argsort(p[:, 0])
    assert np.allclose(p[order], [[0.02, 0.02, 0.0], [0.5, 0.0, 0.0]], atol=1e-6)
    assert np.allclose(v[order], [20.0, 7.0])
