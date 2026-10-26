#!/usr/bin/env python3
"""Build the sim's MID360 scan pattern from a recording of the real lidar.

    python3 scripts/lidar_pattern.py --bag record/dp --frames 20 --start 100 \\
        --out config/lidar/mid360_pattern.npz

Reads livox_ros_driver2 CustomMsg frames out of the rosbag2 sqlite files
(syncai_sim/livox_bag.py: no ROS needed), keeps the samples that returned,
fills short no-return gaps along each laser's sweep, and writes the bank of
directions with their firing times that lidar.py fires frame by frame
(`lidar_pattern: mid360`). config/lidar/README.md records which recording the
committed file came from.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from syncai_sim.livox_bag import build_pattern, iter_frames, save_pattern  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="rosbag2 directory (its .db3 files) or one .db3")
    ap.add_argument("--topic", default="/robot01/livox/lidar")
    ap.add_argument("--start", type=int, default=100,
                    help="first frame to take (default 100: 10 s in, past the driver's start)")
    ap.add_argument("--frames", type=int, default=20,
                    help="consecutive frames in the bank (20 = 2 s before the pattern repeats)")
    ap.add_argument("--max-gap", type=int, default=3,
                    help="fill no-return runs up to this many samples along a laser")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    frames = list(iter_frames(a.bag, a.topic, start=a.start, count=a.frames))
    if len(frames) < a.frames:
        print(f"only {len(frames)} frames from {a.start} on {a.topic} in {a.bag}", file=sys.stderr)
        return 1
    pattern = build_pattern(frames, max_gap=a.max_gap, source=os.path.basename(
        os.path.normpath(a.bag)), start_frame=a.start)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_pattern(a.out, pattern)
    meta = json.loads(pattern["meta"])
    n = pattern["az"].shape[0]
    print(f"{a.out}: {n} samples from {meta['frames']} frames "
          f"({min(meta['samples_per_frame'])}-{max(meta['samples_per_frame'])} a frame), "
          f"firing every {meta['firing_period_ns']} ns, frame period {meta['frame_period_s']:.4f} s, "
          f"elevation {meta['fov_deg'][0]:.1f}..{meta['fov_deg'][1]:.1f} deg, "
          f"{meta['filled']} gaps filled, {meta['dropped']} samples dropped; "
          f"{os.path.getsize(a.out) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
