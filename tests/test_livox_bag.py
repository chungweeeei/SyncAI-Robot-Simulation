"""livox_bag.py against hand-built CustomMsg blobs: the CDR layout, the gap
filling along a laser, and the pattern bank it writes."""

import json
import struct

import numpy as np

from syncai_sim.livox_bag import (LivoxFrame, build_pattern, directions_of, fill_gaps,
                                  parse_custom_msg, save_pattern)

PERIOD_NS = 4947


def custom_msg_blob(points, frame_id="robot01/laser", timebase=1792131354082800791,
                    stamp=(1792131354, 82800791)):
    """CDR as rmw_cyclonedds serialises livox_ros_driver2/CustomMsg: the
    4-byte encapsulation, primitives aligned from its end, no padding after
    the last point (the one place a naive stride would overrun)."""
    b = bytearray(b"\x00\x01\x00\x00")
    b += struct.pack("<iI", *stamp)
    fid = frame_id.encode() + b"\0"
    b += struct.pack("<I", len(fid)) + fid
    while (len(b) - 4) % 8:
        b += b"\0"
    b += struct.pack("<Q", timebase)
    b += struct.pack("<I", len(points))
    b += bytes([1, 0, 0, 0])
    b += struct.pack("<I", len(points))
    for i, p in enumerate(points):
        b += struct.pack("<Ifff3B", *p)
        if i < len(points) - 1:
            b += b"\0"
    return bytes(b)


def sweep_frame(n=400, az0_deg=0.0, gaps=(), k=0):
    """n samples of four lasers sweeping 0.3 deg a sample at elevations 10 +
    line deg; the slots in `gaps` have no return (stored as zeros)."""
    pts = []
    for i in range(n):
        line = i % 4
        az = np.radians(az0_deg + 0.3 * i)
        el = np.radians(10.0 + line)
        if i in gaps:
            x = y = z = 0.0
        else:
            r = 3.0 + 0.001 * i
            x, y, z = r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)
        pts.append((i * PERIOD_NS, x, y, z, 20, 0, line))
    return parse_custom_msg(custom_msg_blob(pts, timebase=10 ** 12 + 10 ** 8 * k))  # 0.1 s apart


def test_parse_custom_msg_fields_and_last_point():
    pts = [(0, 1.0, 2.0, 3.0, 7, 0, 0), (PERIOD_NS, 0.0, 0.0, 0.0, 0, 192, 1),
           (2 * PERIOD_NS, -1.5, 0.5, 0.25, 255, 4, 2)]
    f = parse_custom_msg(custom_msg_blob(pts))
    assert isinstance(f, LivoxFrame)
    assert f.frame_id == "robot01/laser" and f.lidar_id == 1
    assert f.timebase_ns == 1792131354082800791
    assert abs(f.stamp_s - 1792131354.082800791) < 1e-6
    assert f.points.shape == (3,)
    assert list(f.points["offset_time"]) == [0, PERIOD_NS, 2 * PERIOD_NS]
    assert np.allclose(f.points["x"], [1.0, 0.0, -1.5])
    assert list(f.points["reflectivity"]) == [7, 0, 255]
    assert list(f.points["tag"]) == [0, 192, 4] and list(f.points["line"]) == [0, 1, 2]
    dirs, valid = directions_of(f.points)
    assert list(valid) == [True, False, True]
    assert np.allclose(np.linalg.norm(dirs[valid], axis=1), 1.0)


def test_fill_gaps_interpolates_short_runs_per_laser_only():
    f = sweep_frame(gaps=(40, 44, 48, 100, 104, 108, 112, 116))  # line 0: a run of 3, a run of 5
    dirs, valid = directions_of(f.points)
    filled = fill_gaps(dirs, valid, f.points["line"].astype(int), max_gap=3)
    assert filled[40] and filled[44] and filled[48]
    assert not (filled[100] or filled[116])  # five in a row: left out
    # The filled direction is the sweep's: between its neighbours of the same laser.
    az44 = np.degrees(np.arctan2(dirs[44, 1], dirs[44, 0]))
    assert abs(az44 - 0.3 * 44) < 0.02
    el44 = np.degrees(np.arcsin(dirs[44, 2]))
    assert abs(el44 - 10.0) < 0.02  # line 0's elevation, not a neighbouring laser's


def test_build_pattern_and_bank_file(tmp_path):
    frames = [sweep_frame(az0_deg=0.0, gaps=(40, 44), k=0),
              sweep_frame(az0_deg=37.0, gaps=tuple(range(200, 240, 4)), k=1)]
    pat = build_pattern(frames, max_gap=3, source="unit", start_frame=5)
    meta = json.loads(pat["meta"])
    assert meta["firing_period_ns"] == 250_000 and meta["frames"] == 2  # 0.1 s / 400 samples
    assert abs(meta["frame_period_s"] - 0.1) < 1e-9
    assert meta["samples_per_frame"] == [400, 390]
    assert meta["filled"] == 2 and meta["dropped"] == 10
    assert pat["slot"].dtype == np.uint16 and pat["frame"].dtype == np.uint8
    assert list(pat["slot"][:3]) == [0, 1, 2]
    # Frame 1's dropped run leaves its slots out: the slot is the place in the frame.
    f1 = pat["slot"][pat["frame"] == 1]
    assert f1.shape[0] == 390 and 200 not in f1 and 196 in f1 and 240 in f1
    assert abs(pat["az"][1] / 100.0 - 0.3) < 0.011  # 0.01 deg quantisation
    path = str(tmp_path / "p.npz")
    save_pattern(path, pat)
    z = np.load(path)
    assert set(z.files) == {"az", "el", "slot", "line", "frame", "meta"}
    assert z["az"].shape[0] == 790
