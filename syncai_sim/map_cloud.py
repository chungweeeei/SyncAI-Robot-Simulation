"""The site as a point cloud for Foxglove: `map.pcd`, thinned, one per storey.

The MuJoCo viewer renders in software (llvmpipe over VNC), so the detail of a
lidar map is out of its reach; Foxglove on the host draws millions of points
on the GPU. The sim publishes this once, latched (`sim/map_cloud/storey<N>`,
frame `map`), with the points' intensity, which is what makes a lidar map
legible — floor markings, signs, glass — and a colour-by-height mode is one
click in the panel.

The points are those the picture shows elsewhere (pcd_mesh.on_site: on the
cleaned voxels, under the per-storey ceiling cut), then a `voxel` grid keeps
one averaged point per cell. 4 cm keeps 6.1 M on dp, 16 bytes each, more than
a websocket hands a browser comfortably; 6 cm keeps 2.22 M on 1F and 1.46 M on
2F (59 MB, built in 11 s, 2026-10-24). Cached as cache/<map>/map_cloud.npz.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .pcd_mesh import on_site, read_pcd
from .storeys import storey_of
from .world import World

FORMAT = 1


def thin(points: np.ndarray, values: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """One point per `voxel` cell: the mean position and value of the points in it."""
    if points.shape[0] == 0:
        return points, values
    key = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    out = np.zeros((counts.size, 3))
    np.add.at(out, inverse, points)
    val = np.zeros(counts.size)
    np.add.at(val, inverse, values)
    return (out / counts[:, None]).astype(np.float32), (val / counts).astype(np.float32)


def load_or_build(map_dir: str, cache_dir: str, world: World, voxel: float = 0.06,
                  max_height: float = 3.0, log=print) -> dict[int, np.ndarray] | None:
    """storey -> (N, 4) float32 x y z intensity; None without map.pcd."""
    pcd_path = os.path.join(map_dir, "map.pcd")
    if not os.path.isfile(pcd_path):
        log(f"[map_cloud] {pcd_path} not found; no map cloud for Foxglove")
        return None
    st = os.stat(pcd_path)
    key = json.dumps({"format": FORMAT, "pcd": [st.st_size, int(st.st_mtime)],
                      "world": world.stats.cache_key, "world_options": world.stats.options,
                      "storeys": world.storey_bounds, "voxel": voxel, "max_height": max_height},
                     sort_keys=True)
    path = os.path.join(cache_dir, "map_cloud.npz")
    if os.path.isfile(path):
        z = np.load(path)
        if str(z["key"]) == key:
            clouds = {int(k[1:]): z[k] for k in z.files if k.startswith("s")}
            log(f"[map_cloud] cached ({sum(c.shape[0] for c in clouds.values())} points)")
            return clouds
    xyz, intensity = read_pcd(pcd_path, intensity=True)
    keep = on_site(xyz, world, max_height)
    xyz, intensity = xyz[keep], intensity[keep]
    which = storey_of(xyz[:, 2], world.storey_bounds)
    clouds = {}
    for k in range(len(world.storey_bounds) + 1):
        p, i = thin(xyz[which == k], intensity[which == k], voxel)
        clouds[k] = np.concatenate([p, i[:, None]], axis=1).astype(np.float32)
    log(f"[map_cloud] {keep.size} points -> {int(keep.sum())} on the site -> "
        f"{ {k + 1: c.shape[0] for k, c in clouds.items()} } per storey at {voxel:g} m")
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, key=key, **{f"s{k}": c for k, c in clouds.items()})
    os.replace(tmp, path)
    return clouds
