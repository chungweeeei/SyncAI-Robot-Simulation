"""Livox CustomMsg frames straight out of a rosbag2 sqlite file, and the scan
pattern built from them — no ROS, no livox_ros_driver2 on the path.

Why read the bag by hand: the sim's MID360 pattern (lidar.py) comes from a
real recording of the robot's own lidar (the workspace's record/dp, the
mapping run of dp), and the file that holds it is built on a Mac as well as in
the container. rosbag2's sqlite3 storage is one table of CDR blobs, and
CustomMsg is flat enough to parse with struct:

    livox_ros_driver2/msg/CustomMsg        livox_ros_driver2/msg/CustomPoint
      std_msgs/Header header                 uint32  offset_time   ns from timebase
      uint64 timebase                        float32 x, y, z       m, lidar frame
      uint32 point_num                       uint8   reflectivity  0..255
      uint8  lidar_id                        uint8   tag
      uint8[3] rsvd                          uint8   line          laser 0..3
      CustomPoint[] points

CDR (XCDR1, little endian): a 4-byte encapsulation header, then every
primitive aligned to its own size counted from the end of that header, so
`timebase` pads to 8 and each 19-byte point pads to a 20-byte stride.

What the recording says about the MID360 (dp_0.db3, the first 60 frames,
2026-10-26): frames every 99.9 ms of ~20 k samples (19 776-20 160) in packets
of 96, in firing order (`build_pattern` on why the per-point offset_time is
not); `line == index % 4`; a laser's consecutive samples are
1.2 deg apart (p99 1.39), so the sweep is smooth per line; elevation -7.0 to
52.7 deg; 19.5 % of the samples had no return and sit in the message as
(0, 0, 0) with their time; those come in runs of 2-3 samples (p90 3, the
longest 191). The same slot in consecutive frames points 55 deg apart: the
pattern does not repeat frame to frame, which is the property the sim wants.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import struct
from dataclasses import dataclass

import numpy as np

POINT_DTYPE = np.dtype([
    ("offset_time", "<u4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("reflectivity", "u1"), ("tag", "u1"), ("line", "u1"), ("_pad", "u1"),
])
POINT_STRIDE = 20  # 19 bytes of fields, padded to the next uint32
NO_RETURN_RANGE_M = 0.05  # a sample with no return is stored as (0, 0, 0)
PATTERN_FORMAT = 1


@dataclass
class LivoxFrame:
    stamp_s: float  # header.stamp
    timebase_ns: int  # the first point's absolute time
    frame_id: str
    lidar_id: int
    points: np.ndarray  # POINT_DTYPE, one per fired sample, returns or not


def parse_custom_msg(blob: bytes) -> LivoxFrame:
    """One CDR-serialised CustomMsg -> LivoxFrame (points view the blob's bytes)."""
    b = bytes(blob)
    off = 4  # encapsulation header; alignment counts from here
    sec, nsec = struct.unpack_from("<iI", b, off)
    off += 8
    n = struct.unpack_from("<I", b, off)[0]
    off += 4
    frame_id = b[off:off + n - 1].decode()
    off += n
    off += (-(off - 4)) % 8
    timebase = struct.unpack_from("<Q", b, off)[0]
    off += 8
    point_num = struct.unpack_from("<I", b, off)[0]
    off += 4
    lidar_id = b[off]
    off += 4  # lidar_id + rsvd[3]
    count = struct.unpack_from("<I", b, off)[0]
    off += 4
    need = off + count * POINT_STRIDE
    if count != point_num or need - 1 > len(b):
        raise ValueError(f"CustomMsg: {count} points declared, {point_num} point_num, "
                         f"{len(b)} bytes")
    # The last point has no padding byte after it: pad the buffer, not the parse.
    buf = b[off:need - 1] + b"\0" if need > len(b) else b[off:need]
    points = np.frombuffer(buf, dtype=POINT_DTYPE, count=count)
    return LivoxFrame(stamp_s=sec + nsec * 1e-9, timebase_ns=int(timebase), frame_id=frame_id,
                      lidar_id=int(lidar_id), points=points)


def bag_files(bag: str) -> list[str]:
    """The .db3 files of a rosbag2 directory in recording order (or the one
    file given). The compressed copies (.db3.zstd) are not read: `zstd -d`
    them first, as rosbag2 would."""
    if os.path.isfile(bag):
        return [bag]
    files = glob.glob(os.path.join(bag, "*.db3"))
    if not files:
        raise FileNotFoundError(f"no .db3 under {bag} (only .zstd? decompress them first)")

    def key(p):
        m = re.search(r"_(\d+)\.db3$", p)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)


def iter_frames(bag: str, topic: str = "/robot01/livox/lidar", start: int = 0,
                count: int | None = None):
    """Yield LivoxFrame for `topic`, in time order, skipping the first `start`
    and stopping after `count`."""
    seen = 0
    yielded = 0
    for path in bag_files(bag):
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT id FROM topics WHERE name = ?", (topic,)).fetchone()
            if row is None:
                continue
            cur = con.execute("SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                              (row[0],))
            for (blob,) in cur:
                if seen >= start:
                    yield parse_custom_msg(blob)
                    yielded += 1
                    if count is not None and yielded >= count:
                        return
                seen += 1
        finally:
            con.close()


# --- the pattern -------------------------------------------------------------

def directions_of(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(dirs (N, 3) unit, valid (N,)) of a frame's points; a sample without a
    return has no direction (valid False, dirs zero)."""
    xyz = np.column_stack((points["x"], points["y"], points["z"])).astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    valid = r > NO_RETURN_RANGE_M
    dirs = np.zeros_like(xyz)
    dirs[valid] = xyz[valid] / r[valid, None]
    return dirs, valid


def fill_gaps(dirs: np.ndarray, valid: np.ndarray, line: np.ndarray, max_gap: int) -> np.ndarray:
    """Directions for samples without a return, by interpolating between the
    nearest returned samples of the same laser when at most `max_gap` samples
    are missing in a row (the sweep moves ~1.2 deg a sample, so a straight
    blend of the neighbours, renormalised, is within a few hundredths of a
    degree). Returns `valid` with the filled samples set; `dirs` is updated in
    place. Longer runs stay invalid: the lidar's own blind spots and the long
    outages behind the robot's body are not a direction the sim should fire."""
    filled = valid.copy()
    for lsr in np.unique(line):
        idx = np.flatnonzero(line == lsr)
        v = valid[idx]
        good = np.flatnonzero(v)
        if good.size < 2:
            continue
        # Runs of invalid between two valid samples of this laser.
        for a, b in zip(good[:-1], good[1:]):
            gap = b - a - 1
            if gap == 0 or gap > max_gap:
                continue
            da, db = dirs[idx[a]], dirs[idx[b]]
            w = (np.arange(1, gap + 1) / (gap + 1))[:, None]
            d = (1.0 - w) * da[None, :] + w * db[None, :]
            d /= np.linalg.norm(d, axis=1)[:, None]
            dirs[idx[a + 1:b]] = d
            filled[idx[a + 1:b]] = True
    return filled


def build_pattern(frames: list[LivoxFrame], max_gap: int = 3, source: str = "",
                  start_frame: int = 0) -> dict:
    """The bank of fired directions from consecutive frames, as the arrays
    `save_pattern` writes: per kept sample its azimuth / elevation, the firing
    slot (its index in the frame; time = slot x firing period), the laser and
    the frame."""
    if not frames:
        raise ValueError("no frames")
    # A sample's firing time is its place in the frame times the frame's mean
    # firing interval, not its `offset_time`. The message's point order is the
    # firing order — the sweep runs on from each 96-sample packet into the
    # next (dp_0.db3 frame 100: packets cover azimuth -86..-53, -54..-21,
    # -22..10, 10..32 ... in order) — while the per-point offset_time is not a
    # time within the frame: packets come in pairs whose spans overlap
    # (0..470 us and 7..477 us), then jump ~180 us, so sorting by it tears
    # the sweep apart. 20 064 samples in 100 ms is 4 984 ns a sample, the
    # 4 947 ns of consecutive offset_times within a packet near enough.
    stamps = np.array([f.timebase_ns for f in frames], dtype=np.float64)
    frame_period = float(np.median(np.diff(stamps)) * 1e-9) if len(frames) > 1 else 0.1
    counts = np.array([f.points.shape[0] for f in frames], dtype=np.float64)
    period_ns = int(round(frame_period * 1e9 / float(np.median(counts))))
    az_all, el_all, slot_all, line_all, frame_all = [], [], [], [], []
    per_frame, n_filled, n_dropped = [], 0, 0
    for k, f in enumerate(frames):
        dirs, valid = directions_of(f.points)
        line = f.points["line"].astype(np.int64)
        keep = fill_gaps(dirs, valid, line, max_gap)
        n_filled += int(keep.sum() - valid.sum())
        n_dropped += int((~keep).sum())
        d = dirs[keep]
        az_all.append(np.degrees(np.arctan2(d[:, 1], d[:, 0])))
        el_all.append(np.degrees(np.arcsin(np.clip(d[:, 2], -1.0, 1.0))))
        slot_all.append(np.flatnonzero(keep).astype(np.float64))
        line_all.append(line[keep])
        frame_all.append(np.full(int(keep.sum()), k))
        per_frame.append(int(keep.sum()))
    az = np.concatenate(az_all)
    el = np.concatenate(el_all)
    meta = {
        "format": PATTERN_FORMAT, "source": source, "start_frame": start_frame,
        "frames": len(frames), "firing_period_ns": period_ns, "frame_period_s": frame_period,
        "samples_per_frame": per_frame, "fov_deg": [float(el.min()), float(el.max())],
        "filled": n_filled, "dropped": n_dropped, "max_gap": max_gap,
        "frame_id": frames[0].frame_id,
    }
    slot = np.concatenate(slot_all)
    if slot.max() >= 65536:
        raise ValueError(f"firing slot {slot.max()} does not fit uint16")
    return {
        "az": np.rint(az * 100).astype(np.int16), "el": np.rint(el * 100).astype(np.int16),
        "slot": slot.astype(np.uint16), "line": np.concatenate(line_all).astype(np.uint8),
        "frame": np.concatenate(frame_all).astype(np.uint8), "meta": json.dumps(meta),
    }


def save_pattern(path: str, pattern: dict) -> None:
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **pattern)
    os.replace(tmp, path)
