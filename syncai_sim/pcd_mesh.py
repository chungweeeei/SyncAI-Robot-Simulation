"""The viewer's site from `map/<name>/map.pcd`: a smooth surface instead of voxels.

The voxel mesh (world.py) is exact about what the sim knows — 0.1 m cubes —
and looks like it. The map's point cloud has the detail (dp: 19.6 M points),
so for the picture only this builds a surface from it, once per map, cached
next to the world:

1. Points are kept only where the cleaned voxel grid is occupied (the voxel or
   a face neighbour), so the picture loses what the world's cleanups removed
   — the mapper's feet, the companion sliver (world.py) — and nothing else;
   and only up to `max_height` over their column's lowest voxel in their
   storey (storeys.py), the voxel mesh's ceiling cut.
2. A `voxel` down-sample (4 cm: 4.2 M points on dp), normals from the
   neighbours, turned towards the nearest mapping-track pose (the lidar saw
   every surface from the track side).
3. Screened Poisson reconstruction at `depth` (11: ~6 cm cells over dp's
   120 m), single-threaded — Open3D 0.18's threaded iso-surface extraction
   aborted on dp with "Failed to close loop" (2026-10-23). Vertices of the
   lowest `density_quantile` and those farther than `max_dist` from every
   point are dropped: Poisson closes the scene into bubbles there. 2 % and
   0.25 m keep the floor whole across the lidar's blind cone under the
   mapper; 10 % and 8 cm cut it full of holes (same date).
4. Quadric decimation to `target_triangles`, the voxel mesh's size (dp: 1.8 M
   voxel triangles; a size the browser takes in a second, and that the
   llvmpipe window of before drew at ~120 ms per 2 M).

dp: ~3.5 min and ~4 GB. The surface is a picture: the lidar, the feet and the
planner still read the voxels (CLAUDE.md, rule 2), whose floor tops are up
to a voxel above the real floor this draws — the robot may stand a few
centimetres above the drawn floor.

    python3 -m syncai_sim.pcd_mesh map/dp     # build (or refresh) the cache
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import visual
from .storeys import storey_of
from .world import World, read_track

# Bumped when the mesh changes without an option changing (2: storeys).
FORMAT = 2


@dataclass
class PcdMeshOptions:
    voxel: float = 0.04
    max_height: float = 3.0  # over the column's lowest voxel; the voxel mesh's visual band
    depth: int = 11
    density_quantile: float = 0.02
    max_dist: float = 0.25
    target_triangles: int = 1_500_000


@dataclass
class PcdMesh:
    vertices: np.ndarray  # (V, 3) float32
    faces: np.ndarray  # (F, 3) int32
    kinds: np.ndarray  # (F,) int8, visual.KIND_*
    storeys: np.ndarray  # (F,) int8, by the triangle's centroid (storeys.py)


def read_pcd(path: str, intensity: bool = False):
    """(N, 3) float32 xyz of a binary PCD with float32 x y z first (Point-LIO /
    clean_map's); with `intensity`, also its (N,) intensity field (zeros if none)."""
    with open(path, "rb") as f:
        hdr: dict[str, list[str]] = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no DATA line")
            parts = line.decode("ascii", "replace").split()
            if not parts or parts[0].startswith("#"):
                continue
            hdr[parts[0]] = parts[1:]
            if parts[0] == "DATA":
                break
        fields = hdr["FIELDS"]
        if hdr["DATA"][0] != "binary" or fields[:3] != ["x", "y", "z"] \
                or any(t != "F" for t in hdr["TYPE"]) or any(s != "4" for s in hdr["SIZE"]):
            raise ValueError(f"{path}: expected binary float32 fields x y z ..., got "
                             f"{hdr['DATA']} {fields} {hdr['TYPE']} {hdr['SIZE']}")
        n = int(hdr["POINTS"][0])
        data = np.fromfile(f, dtype=np.float32, count=n * len(fields)).reshape(n, -1)
    if not intensity:
        return data[:, :3]
    i = data[:, fields.index("intensity")] if "intensity" in fields else np.zeros(n, np.float32)
    return data[:, :3], i


def _key(pcd_path: str, world: World, opts: PcdMeshOptions) -> str:
    st = os.stat(pcd_path)
    return json.dumps({"format": FORMAT, "pcd": [st.st_size, int(st.st_mtime)],
                       "world": world.stats.cache_key, "storeys": world.storey_bounds,
                       "world_options": world.stats.options, "options": asdict(opts)},
                      sort_keys=True)


def load_or_build(map_dir: str, cache_dir: str, world: World,
                  opts: PcdMeshOptions | None = None, log=print) -> PcdMesh | None:
    """The cached mesh, built first if needed; None (with a log line) when the
    map has no map.pcd or Open3D is missing, so the caller falls back to voxels."""
    opts = opts or PcdMeshOptions()
    pcd_path = os.path.join(map_dir, "map.pcd")
    if not os.path.isfile(pcd_path):
        log(f"[pcd_mesh] {pcd_path} not found; the viewer shows the voxels")
        return None
    key = _key(pcd_path, world, opts)
    data_path = os.path.join(cache_dir, "pcd_mesh.npz")
    if os.path.isfile(data_path):
        z = np.load(data_path)
        if str(z["key"]) == key:
            log(f"[pcd_mesh] cached ({z['faces'].shape[0]} tris)")
            return PcdMesh(z["vertices"], z["faces"], z["kinds"], z["storeys"])
    try:
        import open3d  # noqa: F401
    except ImportError:
        log("[pcd_mesh] open3d not installed (requirements.txt); the viewer shows the voxels")
        return None
    log(f"[pcd_mesh] building from {pcd_path} (dp: ~3.5 min, once)")
    mesh = build(read_pcd(pcd_path), world, read_track(map_dir), opts, log)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = data_path + ".tmp.npz"
    np.savez(tmp, key=key, vertices=mesh.vertices, faces=mesh.faces, kinds=mesh.kinds,
             storeys=mesh.storeys)
    os.replace(tmp, data_path)
    return mesh


def on_site(points: np.ndarray, world: World, max_height: float) -> np.ndarray:
    """(N,) bool: the point's voxel or a face neighbour is occupied in the
    cleaned grid, and it is at most `max_height` over its column's lowest voxel
    within its storey (the viewer's ceiling cut, per storey)."""
    g = world.grid
    shape = np.array(g.grid.shape)
    idx = np.floor((points - g.origin_m) / g.res).astype(np.int64)
    inside = np.all((idx >= 1) & (idx < shape - 1), axis=1)
    keep = np.zeros(points.shape[0], bool)
    pts, idx = points[inside], idx[inside]
    near = g.grid[idx[:, 0], idx[:, 1], idx[:, 2]].copy()
    for axis in range(3):
        for step in (-1, 1):
            j = idx.copy()
            j[:, axis] += step
            near |= g.grid[j[:, 0], j[:, 1], j[:, 2]]
    bounds = world.storey_bounds
    which = storey_of(pts[:, 2], bounds)
    cuts = [0] + [int(np.clip(np.floor((b - g.origin_m[2]) / g.res), 0, shape[2])) for b in bounds] \
        + [int(shape[2])]
    floor_z = np.full(pts.shape[0], np.inf)
    for k in range(len(bounds) + 1):
        band = g.grid[:, :, cuts[k]:cuts[k + 1]]
        lowest = np.where(band.any(axis=2), np.argmax(band, axis=2) + cuts[k], shape[2])
        sel = which == k
        floor_z[sel] = g.origin_m[2] + lowest[idx[sel, 0], idx[sel, 1]] * g.res
    keep[inside] = near & (pts[:, 2] <= floor_z + max_height)
    return keep


def build(points: np.ndarray, world: World, track: np.ndarray, opts: PcdMeshOptions,
          log=print) -> PcdMesh:
    import open3d as o3d

    t0 = time.perf_counter()
    keep = on_site(points, world, opts.max_height)
    points = points[keep]
    log(f"[pcd_mesh] {keep.size} points, {points.shape[0]} on the site's voxels under "
        f"{opts.max_height:g} m")

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    pc = pc.voxel_down_sample(opts.voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * opts.voxel, max_nn=24))
    P = np.asarray(pc.points)
    N = np.asarray(pc.normals).copy()
    poses = track[::5, :3] if track.shape[0] else np.zeros((1, 3))
    chunk = 100_000
    for a in range(0, P.shape[0], chunk):
        p = P[a:a + chunk]
        d = ((p[:, None, :] - poses[None, :, :]) ** 2).sum(-1)
        towards = poses[np.argmin(d, axis=1)] - p
        flip = np.einsum("ij,ij->i", N[a:a + chunk], towards) < 0
        N[a:a + chunk][flip] *= -1
    pc.normals = o3d.utility.Vector3dVector(N)
    log(f"[pcd_mesh] {P.shape[0]} points at {opts.voxel:g} m, normals "
        f"({time.perf_counter() - t0:.0f} s)")

    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=opts.depth, n_threads=1)
    density = np.asarray(density)
    mesh.remove_vertices_by_mask(density < np.quantile(density, opts.density_quantile))
    dist = np.asarray(o3d.geometry.PointCloud(mesh.vertices).compute_point_cloud_distance(pc))
    mesh.remove_vertices_by_mask(dist > opts.max_dist)
    log(f"[pcd_mesh] Poisson depth {opts.depth}: {len(mesh.triangles)} triangles kept "
        f"({time.perf_counter() - t0:.0f} s)")
    if len(mesh.triangles) > opts.target_triangles:
        mesh = mesh.simplify_quadric_decimation(opts.target_triangles)
    mesh.remove_unreferenced_vertices()
    V = np.asarray(mesh.vertices, np.float32)
    F = np.asarray(mesh.triangles, np.int32)
    log(f"[pcd_mesh] {F.shape[0]} triangles, {V.shape[0]} vertices "
        f"({time.perf_counter() - t0:.0f} s)")
    return PcdMesh(V, F, visual.surface_kinds(V, F),
                   storey_of(V[F].mean(axis=1)[:, 2], world.storey_bounds).astype(np.int8))


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .world import build_world, default_cache_dir

    p = argparse.ArgumentParser(description="Build (or refresh) a map's point-cloud viewer mesh")
    p.add_argument("map_dir", help="map/<name> directory holding octomap.bt and map.pcd")
    p.add_argument("--cache", default=None, help="cache directory (default cache/<name>)")
    a = p.parse_args(argv)
    cache = a.cache or default_cache_dir(a.map_dir)
    # The world options the sim uses by default, so that the cache key matches.
    from .config import PARAM_DEFAULTS, world_options
    world = build_world(a.map_dir, cache, world_options(PARAM_DEFAULTS))
    return 0 if load_or_build(a.map_dir, cache, world) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
