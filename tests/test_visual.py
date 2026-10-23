import numpy as np
import pytest

from syncai_sim import visual
from syncai_sim.pcd_mesh import read_pcd


def test_texture_rows_and_ramp_ends():
    rgb = visual.texture_rgb()
    assert rgb.shape == (3 * visual.ROW_H, visual.TEX_W, 3) and rgb.dtype == np.uint8
    floor = rgb[0, :, :] / 255.0
    assert np.allclose(floor[0], visual.PALETTE[0][1], atol=1 / 255)
    assert np.allclose(floor[-1], visual.PALETTE[-1][1], atol=1 / 255)
    under = rgb[-1, :, :] / 255.0
    assert np.allclose(under, visual.UNDER_GREY, atol=1 / 255)


def test_texcoords_clamp_and_pick_the_row():
    uv = visual.texcoords(np.array([-5.0, 0.0, 1.5, 3.0, 99.0]), visual.KIND_WALL, (0.0, 3.0))
    assert np.all((uv[:, 0] > 0) & (uv[:, 0] < 1))
    assert uv[2, 0] == pytest.approx(0.5)
    assert np.allclose(uv[:, 1], 0.5)  # the middle of three rows


def test_height_range_is_clamped():
    lo, hi = visual.height_range(np.array([0.0, 0.1]))  # one storey: still a 3 m ramp
    assert lo == pytest.approx(0.0, abs=0.01) and hi - lo == pytest.approx(visual.MIN_RANGE_M)
    lo, hi = visual.height_range(np.linspace(0.0, 30.0, 1000))
    assert hi - lo == pytest.approx(visual.MAX_RANGE_M)


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
