#!/usr/bin/env python3
"""Colour a map's voxels from its mapping run's bag: config/colour/<map>.npz,
what the viewer's `photo` look paints (syncai_sim/voxel_colour.py).

    python3 scripts/colour_map.py --bag record/rec_20261027_031456_replay \\
        --odom record/rec_20261027_031456_replay/odom.txt --map map/outdoor_1027 \\
        [--out config/colour/outdoor_1027.npz] [--preview /tmp/outdoor_1027.png]

Reads the bag's lidar scans, images, camera_info and /tf_static straight out
of its sqlite files (no ROS; syncai_sim/livox_bag.py, bag_camera.py) and
needs, besides, the robot's pose along the bag: Point-LIO's odometry
replayed from it by scripts/bag_odom.py, which says how. The bag must be
the map's own mapping run, so that the odometry's frame is the map's.
Paths are relative to the cwd; in the container the workspace is
/home/syncrobotic/robot_ws. outdoor_1027 (5240 scans, 14836 frames): 48 s
and 9 MB, 77 % of the occupied voxels. config/colour/README.md records
which bag each committed table came from.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from syncai_sim import bag_camera, voxel_colour  # noqa: E402


def preview(table: voxel_colour.VoxelColour, path: str) -> None:
    """A top-down PNG of the table: each column its highest coloured voxel,
    north up, a pixel per voxel; black where nothing is coloured."""
    from PIL import Image

    nx, ny, nz = table.shape
    ix = table.keys // (ny * nz)
    iy = (table.keys // nz) % ny
    iz = table.keys % nz
    order = np.lexsort((iz, iy, ix))  # per column, the highest last
    image = np.zeros((ny, nx, 3), np.uint8)
    image[ny - 1 - iy[order], ix[order]] = table.rgb[order]
    Image.fromarray(image).save(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="rosbag2 directory (its .db3 files)")
    ap.add_argument("--odom", required=True, help="scripts/bag_odom.py's file for this bag")
    ap.add_argument("--map", required=True, help="map/<name> directory (octomap.bt)")
    ap.add_argument("--cache", default=None, help="the sim's cache directory (default cache/<name>)")
    ap.add_argument("--out", default=None, help="default config/colour/<name>.npz")
    ap.add_argument("--every-scan", type=int, default=1,
                    help="colour every Nth scan (1 = all; 2 halves the time, hardly the coverage)")
    ap.add_argument("--max-range", type=float, default=voxel_colour.MAX_RANGE_M,
                    help=f"returns farther than this are not coloured ({voxel_colour.MAX_RANGE_M} m)")
    ap.add_argument("--lidar-topic", default="/robot01/livox/lidar")
    ap.add_argument("--image-topic", default=bag_camera.IMAGE_TOPIC)
    ap.add_argument("--camera-info-topic", default=bag_camera.CAMERA_INFO_TOPIC)
    ap.add_argument("--tf-topic", default=bag_camera.TF_STATIC_TOPIC)
    ap.add_argument("--preview", default=None, help="also write a top-down PNG of the result")
    a = ap.parse_args(argv)

    from syncai_sim.config import PARAM_DEFAULTS, world_options
    from syncai_sim.world import build_world, default_cache_dir

    cache = a.cache or default_cache_dir(a.map)
    world = build_world(a.map, cache, world_options(PARAM_DEFAULTS))
    out = a.out or voxel_colour.table_path(world.stats.map_name)

    model = bag_camera.read_camera_info(a.bag, a.camera_info_topic)
    camera = voxel_colour.Camera.from_model(model)
    tfs = bag_camera.read_static_transforms(a.bag, a.tf_topic)
    lidar_to_camera = bag_camera.find_transform(tfs, "lidar_top", "camera_optical_frame")
    lidar_to_body = bag_camera.find_transform(tfs, "lidar_top", "lidar_top_imu")
    track = voxel_colour.read_odom(a.odom)
    print(f"[colour] camera {model.width} x {model.height} {model.model}, f {camera.fx:.1f} / "
          f"{camera.fy:.1f}, centre ({camera.cx:.1f}, {camera.cy:.1f}); lidar -> camera "
          f"{np.round(lidar_to_camera.t, 3).tolist()}; lidar -> body "
          f"{np.round(lidar_to_body.t, 3).tolist()}; odometry {track.t.shape[0]} samples over "
          f"{track.span[1] - track.span[0]:.0f} s", flush=True)

    table, meta = voxel_colour.build_from_bag(
        a.bag, track, world.grid, camera, lidar_to_camera, lidar_to_body,
        every_scan=a.every_scan, max_range=a.max_range, lidar_topic=a.lidar_topic,
        image_topic=a.image_topic, log=lambda s: print(s, flush=True))
    g = world.grid
    occupied = np.argwhere(g.grid) + g.lo[None, :]
    coloured = int((~np.isnan(table.lookup(occupied)[:, 0])).sum())
    meta.update({"map": world.stats.map_name, "world_cache_key": world.stats.cache_key,
                 "odom": os.path.basename(a.odom), "occupied_voxels": int(occupied.shape[0]),
                 "occupied_coloured": coloured,
                 "camera": {"width": model.width, "height": model.height, "model": model.model,
                            "k": np.round(model.k, 3).tolist(), "d": np.round(model.d, 5).tolist()}})
    voxel_colour.save(out, table, meta)
    print(f"[colour] {out}: {coloured} of {occupied.shape[0]} occupied voxels coloured "
          f"({100.0 * coloured / max(occupied.shape[0], 1):.0f} %), "
          f"{os.path.getsize(out) / 1e6:.1f} MB; {json.dumps(meta)[:200]}...")
    if a.preview:
        preview(table, a.preview)
        print(f"[colour] preview {a.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
