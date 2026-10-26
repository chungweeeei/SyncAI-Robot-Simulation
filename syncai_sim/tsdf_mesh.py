"""The site as a TSDF surface from the mapping run's own keyframes: the third
picture (`visual_mesh: tsdf`), the one that keeps thin structure.

The voxel picture is exact and blocky; the Poisson surface (pcd_mesh.py) is
smooth but blurs anything thin — handrails, door frames, the edge of a
tread — and bubbles where the cloud is sparse, because a surface fitted
to points has no idea where the air was. The mapping run knows: every
keyframe's patch (`map/<name>/patches/<k>.pcd`, the keyframe's own returns
in its own frame; `poses.txt` has the keyframe's pose, `name x y z qw qx qy
qz`) is a set of rays from a known origin. A truncated signed distance
field integrates those rays — free space along each one, the surface
within `sdf_trunc` of the return — and marching cubes reads the surface
off it at the voxel size, thin things kept, unseen places left open (open3d's ScalableTSDFVolume,
which wants depth images: each patch is rendered to six 90 deg cube-face
depth images of `face_px` a side first).

What is kept afterwards is what the voxel world keeps: faces off the
cleaned site (pcd_mesh.on_site: not within a voxel of an occupied one, or
above the storey's height band) go, so the mapper's own feet, the corridor
fence and the track-cleared volume are not drawn either. Then kinds and
storeys as pcd_mesh.py; the shading is voxel_intensity.py's, as for the
other two sites (site_tiles.py). No colour in the volume: carrying the
intensity there tripled the memory per voxel, and with it dp passed 6 GB at
2600 of 3026 keyframes (2026-10-26). dp: 3026 keyframes in 12 blocks of 35 m, 6 min and 2.5 GB at
most, 2 M triangles, once; cached as cache/<map>/tsdf_mesh.npz.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import visual
from .pcd_mesh import on_site, read_pcd
from .storeys import storey_of
from .world import World

FORMAT = 3  # 2: no colour in the volume; 3: by blocks


@dataclass
class TsdfOptions:
    # 7 cm and 15 m: at 5 cm and 25 m dp's volume passed 11.5 GB in 7 min
    # (2026-10-26) on a 15 GB container running the session beside it; far
    # returns are the sparse, noisy ones and add volume, not surface.
    voxel: float = 0.07
    sdf_trunc: float = 0.25  # ~3.5 voxels: the lidar's few cm of noise average out inside it
    face_px: int = 192  # 0.47 deg a pixel on a 90 deg face; a patch is sparser than that
    min_range: float = 0.3  # the robot's own body
    max_range: float = 15.0
    max_rss_gb: float = 6.0  # abort the build past this resident size (a guard, not a target)
    block_m: float = 35.0  # built block by block (build); dp is 3 x 4 blocks
    block_margin_m: float = 1.0
    max_height: float = 3.0  # over the column's lowest voxel; the voxel mesh's visual band
    target_triangles: int = 2_000_000
    every: int = 1  # every n-th keyframe


@dataclass
class TsdfMesh:
    vertices: np.ndarray  # (V, 3) float32
    faces: np.ndarray  # (F, 3) int32
    kinds: np.ndarray  # (F,) int8, visual.KIND_*
    storeys: np.ndarray  # (F,) int8


def quat_wxyz_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_poses(map_dir: str) -> list[tuple[str, np.ndarray]]:
    """(patch file name, 4x4 T_map_keyframe) per line of poses.txt. The
    quaternion is w x y z: with x y z w the patches land off the walls
    (checked on dp: 82-89 % of a patch's points fall in occupied voxels
    transformed this way, 3-14 % the other, 2026-10-26)."""
    out = []
    with open(os.path.join(map_dir, "poses.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            T = np.eye(4)
            T[:3, :3] = quat_wxyz_matrix(p[4:8])
            T[:3, 3] = [float(v) for v in p[1:4]]
            out.append((p[0], T))
    return out


def cube_faces() -> np.ndarray:
    """(6, 3, 3) rotations R_cam_kf, one per cube face: p_cam = R @ p_kf, the
    camera looking along +x, -x, +y, -y, +z, -z of the keyframe frame with
    the pinhole convention (x right, y down, z forward)."""
    up = np.array([0.0, 0.0, 1.0])
    faces = []
    for d in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]):
        z = np.array(d, float)
        side = up if abs(z[2]) < 0.5 else np.array([1.0, 0.0, 0.0])
        x = np.cross(z, side)  # right, for a camera whose "up" is `side`
        x /= np.linalg.norm(x)
        y = np.cross(z, x)  # down: right-handed (x, y, z)
        faces.append(np.stack([x, y, z]))
    return np.array(faces)


FACES = cube_faces()


def depth_faces(points: np.ndarray, intensity: np.ndarray, face_px: int, min_range: float,
                max_range: float) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """A keyframe's returns (N, 3) in its own frame as cube-face depth images:
    [(face, depth (W, W) float32 metres along the optical axis, 0 = nothing,
    colour (W, W, 3) uint8 = the intensity)], only the faces with returns.
    A point goes to the face its largest coordinate points at, the nearest
    return per pixel wins."""
    p = points.astype(np.float64)
    r = np.linalg.norm(p, axis=1)
    ok = (r >= min_range) & (r <= max_range)
    p, inten = p[ok], np.asarray(intensity, np.float64)[ok]
    if p.shape[0] == 0:
        return []
    face = np.argmax(np.abs(p), axis=1) * 2 + (p[np.arange(p.shape[0]), np.argmax(np.abs(p), axis=1)] < 0)
    half = face_px / 2.0
    out = []
    for k in range(6):
        sel = face == k
        if not sel.any():
            continue
        c = p[sel] @ FACES[k].T  # (M, 3) camera coords
        z = c[:, 2]
        u = np.floor(half * c[:, 0] / z + half).astype(np.int64)
        v = np.floor(half * c[:, 1] / z + half).astype(np.int64)
        inside = (u >= 0) & (u < face_px) & (v >= 0) & (v < face_px)
        u, v, z, i = u[inside], v[inside], z[inside], inten[sel][inside]
        order = np.argsort(-z, kind="stable")  # the nearest written last
        depth = np.zeros((face_px, face_px), np.float32)
        depth[v[order], u[order]] = z[order]
        grey = np.clip(i[order], 0.0, 255.0).astype(np.uint8)
        colour = np.zeros((face_px, face_px, 3), np.uint8)
        colour[v[order], u[order]] = grey[:, None]
        out.append((k, depth, colour))
    return out


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def build(map_dir: str, world: World, opts: TsdfOptions, log=print) -> TsdfMesh:
    """Block by block (`block_m` squares of the site): each block's volume
    integrates only the returns that fall inside it (plus `block_margin_m`,
    so the surface is whole up to the block's edge), is meshed, cropped to
    the block and freed. One volume for all of dp grew past 6 GB at 2600 of
    3026 keyframes whatever its voxel or colour (2026-10-26); a block holds
    a ninth of the site."""
    import open3d as o3d

    t0 = time.perf_counter()
    poses = read_poses(map_dir)[::max(1, opts.every)]
    patches = os.path.join(map_dir, "patches")
    half = opts.face_px / 2.0
    intr = o3d.camera.PinholeCameraIntrinsic(opts.face_px, opts.face_px, half, half, half, half)
    # Every keyframe's returns once, in the map frame (dp: 3026 x ~6 k points).
    frames = []
    for name, T in poses:
        path = os.path.join(patches, name)
        if not os.path.isfile(path):
            continue
        pts = read_pcd(path).astype(np.float64)
        r = np.linalg.norm(pts, axis=1)
        pts = pts[(r >= opts.min_range) & (r <= opts.max_range)]
        frames.append((T, pts @ T[:3, :3].T + T[:3, 3]))
    lo, hi = world.grid.bounds_m()
    nbx = max(1, int(np.ceil((hi[0] - lo[0]) / opts.block_m)))
    nby = max(1, int(np.ceil((hi[1] - lo[1]) / opts.block_m)))
    parts_v, parts_f, offset, n_images, peak = [], [], 0, 0, 0.0
    for bx in range(nbx):
        for by in range(nby):
            x0, y0 = lo[0] + bx * opts.block_m, lo[1] + by * opts.block_m
            x1, y1 = x0 + opts.block_m, y0 + opts.block_m
            m = opts.block_margin_m
            vol = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=opts.voxel, sdf_trunc=opts.sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
            used = 0
            for T, world_pts in frames:
                inside = ((world_pts[:, 0] >= x0 - m) & (world_pts[:, 0] < x1 + m)
                          & (world_pts[:, 1] >= y0 - m) & (world_pts[:, 1] < y1 + m))
                if not inside.any():
                    continue
                local = (world_pts[inside] - T[:3, 3]) @ T[:3, :3]
                T_kf_map = np.linalg.inv(T)
                for face, depth, colour in depth_faces(local, np.zeros(local.shape[0]),
                                                       opts.face_px, opts.min_range,
                                                       opts.max_range):
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = FACES[face]
                    extrinsic = extrinsic @ T_kf_map  # map -> keyframe -> camera
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        o3d.geometry.Image(colour), o3d.geometry.Image(depth), depth_scale=1.0,
                        depth_trunc=opts.max_range, convert_rgb_to_intensity=False)
                    vol.integrate(rgbd, intr, extrinsic)
                    n_images += 1
                used += 1
            rss = _rss_gb()
            peak = max(peak, rss)
            if rss > opts.max_rss_gb:
                raise MemoryError(f"TSDF block ({bx}, {by}) at {rss:.1f} GB, past max_rss_gb "
                                  f"{opts.max_rss_gb}: lower block_m or raise voxel")
            mesh = vol.extract_triangle_mesh()
            del vol
            V = np.asarray(mesh.vertices)
            F = np.asarray(mesh.triangles)
            if F.shape[0]:
                c = V[F].mean(axis=1)
                keep = ((c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)
                        & on_site(c.astype(np.float32), world, opts.max_height))
                F = F[keep]
                usedv, local_f = np.unique(F, return_inverse=True)
                parts_v.append(V[usedv])
                parts_f.append(local_f.reshape(-1, 3) + offset)
                offset += usedv.shape[0]
            log(f"[tsdf] block {bx * nby + by + 1} / {nbx * nby}: {used} keyframes, "
                f"{int(F.shape[0]) if F.shape[0] else 0} triangles kept, {rss:.1f} GB "
                f"({time.perf_counter() - t0:.0f} s)")
    if not parts_f:
        V = np.zeros((0, 3), np.float32)
        F = np.zeros((0, 3), np.int32)
    else:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(parts_v)),
                                         o3d.utility.Vector3iVector(np.concatenate(parts_f)))
        if len(mesh.triangles) > opts.target_triangles:
            mesh = mesh.simplify_quadric_decimation(opts.target_triangles)
            mesh.remove_unreferenced_vertices()
        V = np.asarray(mesh.vertices, np.float32)
        F = np.asarray(mesh.triangles, np.int32)
    log(f"[tsdf] {len(frames)} keyframes, {n_images} images, peak {peak:.1f} GB -> "
        f"{F.shape[0]} triangles, {V.shape[0]} vertices on the site "
        f"({time.perf_counter() - t0:.0f} s)")
    return TsdfMesh(V, F, visual.surface_kinds(V, F),
                    storey_of(V[F].mean(axis=1)[:, 2] if F.shape[0] else np.zeros(0),
                              world.storey_bounds).astype(np.int8))


def _key(map_dir: str, world: World, opts: TsdfOptions) -> str:
    st = os.stat(os.path.join(map_dir, "poses.txt"))
    patches = os.path.join(map_dir, "patches")
    n = len(os.listdir(patches)) if os.path.isdir(patches) else 0
    return json.dumps({"format": FORMAT, "poses": [st.st_size, int(st.st_mtime)], "patches": n,
                       "world": world.stats.cache_key, "storeys": world.storey_bounds,
                       "world_options": world.stats.options, "options": asdict(opts)},
                      sort_keys=True)


def load_or_build(map_dir: str, cache_dir: str, world: World, opts: TsdfOptions | None = None,
                  log=print) -> TsdfMesh | None:
    """The cached TSDF surface, built first if needed; None (with a log line)
    when the map has no keyframe patches or Open3D is missing."""
    opts = opts or TsdfOptions()
    if not os.path.isfile(os.path.join(map_dir, "poses.txt")) \
            or not os.path.isdir(os.path.join(map_dir, "patches")):
        log(f"[tsdf] {map_dir} has no poses.txt + patches/; no TSDF surface")
        return None
    key = _key(map_dir, world, opts)
    path = os.path.join(cache_dir, "tsdf_mesh.npz")
    if os.path.isfile(path):
        z = np.load(path)
        if str(z["key"]) == key:
            log(f"[tsdf] cached ({z['faces'].shape[0]} tris)")
            return TsdfMesh(z["vertices"], z["faces"], z["kinds"], z["storeys"])
    try:
        import open3d  # noqa: F401
    except ImportError:
        log("[tsdf] open3d not installed (requirements.txt); no TSDF surface")
        return None
    log(f"[tsdf] building from {map_dir}/patches (dp: ~6 min, once)")
    mesh = build(map_dir, world, opts, log)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, key=key, vertices=mesh.vertices, faces=mesh.faces, kinds=mesh.kinds,
             storeys=mesh.storeys)
    os.replace(tmp, path)
    return mesh


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .config import PARAM_DEFAULTS, world_options
    from .world import build_world, default_cache_dir

    p = argparse.ArgumentParser(description="Build (or refresh) a map's TSDF viewer surface")
    p.add_argument("map_dir", help="map/<name> directory holding octomap.bt, poses.txt, patches/")
    p.add_argument("--cache", default=None, help="cache directory (default cache/<name>)")
    p.add_argument("--every", type=int, default=1, help="use every n-th keyframe")
    a = p.parse_args(argv)
    cache = a.cache or default_cache_dir(a.map_dir)
    world = build_world(a.map_dir, cache, world_options(PARAM_DEFAULTS))
    mesh = load_or_build(a.map_dir, cache, world, TsdfOptions(every=a.every))
    return 0 if mesh is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
