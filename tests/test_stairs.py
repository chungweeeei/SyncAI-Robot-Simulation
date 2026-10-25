import numpy as np

from syncai_sim import stairs
from syncai_sim.surface_heights import FlightFit

RES = 0.1
ORIGIN = np.array([0.0, 0.0, -0.1])  # voxel z 0 is -0.1 .. 0: the floor's top at 0


def _flight(sign=1.0):
    """Five risers of 0.15 on 0.24 treads, climbing along y (+y for sign +1)
    at x 1.0, the first riser at u 1.03; floors 0 and 0.75."""
    f = FlightFit((0, 9), 1, sign, 1.0, 1.03, 0.24, 5, 0.0, 0.75, 0.01)
    f.u_lo, f.u_hi, f.end_lo, f.end_hi = 0.73, f.u1 + 0.3, 0.0, 0.75
    return f


def _occ(f, wall_ix=17):
    """The flight in voxels, every column solid up to the fitted height at its
    centre (the 0.1 m rounding stairs.py is there to undo), and a wall 2 m
    high from column `wall_ix` on."""
    occ = np.zeros((30, 40, 25), bool)
    for iy in range(40):
        u = f.sign * (iy + 0.5) * RES
        h = float(f.height(np.array([u]))[0])
        occ[:, iy, :int(round((h - ORIGIN[2]) / RES))] = True
    occ[wall_ix:, :, :21] = True
    return occ


def test_one_box_per_tread_at_the_fitted_depth():
    f = _flight()
    sb = stairs.build([f], _occ(f), ORIGIN, RES, log=lambda *a: None)
    b = sb.boxes
    # Two end slabs (1.0 .. 1.03 and 1.99 .. 2.0) and the four treads between risers.
    assert b.shape == (6, 6)
    depth = 2 * b[1:-1, 4]
    assert np.allclose(depth, 0.24)
    tops = b[:, 2] + b[:, 5]
    assert np.allclose(np.diff(tops[:-1]), 0.15)
    assert np.isclose(tops[-1], 0.75) and np.isclose(tops[0], 0.0)
    # The risers where the fit puts them, not on the column grid.
    assert np.allclose(b[1:, 1] - b[1:, 4], 1.03 + 0.24 * np.arange(5))
    # Each box reaches UNDER below the box before it: every riser face is whole.
    assert np.all(b[1:, 2] - b[1:, 5] <= tops[:-1] - stairs.UNDER + 1e-9)


def test_width_grows_to_the_wall_and_the_cap():
    f = _flight()
    sb = stairs.build([f], _occ(f), ORIGIN, RES, log=lambda *a: None)
    x0, x1, y0, y1, z0, z1 = sb.regions[0]
    assert np.isclose(x1, 1.7)  # the wall's first column
    assert np.isclose(x0, 1.0 - stairs.MAX_HALF_WIDTH)
    assert np.isclose(y0, 1.0) and np.isclose(y1, 2.0)
    assert z0 < 0.0 and np.isclose(z1, 0.75 + stairs.ABOVE)
    assert np.allclose(sb.boxes[:, 0] - sb.boxes[:, 3], x0)
    assert np.allclose(sb.boxes[:, 0] + sb.boxes[:, 3], x1)


def test_descending_axis_mirrors():
    f = _flight(sign=-1.0)
    f.u0 = -2.97  # the same staircase climbing towards -y, risers at y 2.97 .. 2.01
    f.u_lo, f.u_hi = f.u0 - 0.3, f.u1 + 0.3
    sb = stairs.build([f], _occ(f), ORIGIN, RES, log=lambda *a: None)
    b = sb.boxes
    tops = b[:, 2] + b[:, 5]
    order = np.argsort(-b[:, 1])  # climbing = going to -y
    assert np.allclose(np.diff(tops[order][:-1]), 0.15)
    assert np.allclose(2 * b[order][1:-1, 4], 0.24)


def test_a_line_that_is_not_the_flight_is_left_alone():
    f = _flight()
    occ = _occ(f)
    occ[:, :, 12:] = True  # a ceiling 1.1 m up over everything: no column is clear
    sb = stairs.build([f], occ, ORIGIN, RES, log=lambda *a: None)
    assert sb.boxes.shape == (0, 6) and sb.regions.shape == (0, 6)


def test_fit_row_round_trip():
    f = _flight()
    g = FlightFit.from_row(f.row())
    assert g.row() == f.row()
    u = np.linspace(0.5, 2.5, 50)
    assert np.allclose(g.height(u), f.height(u))
