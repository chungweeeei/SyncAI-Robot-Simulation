"""The real colour of each voxel, from the camera of the mapping run: what the
viewer's `photo` look paints (visual.py, site_tiles.py).

An octomap is occupancy and map.pcd carries intensity; neither knows that
the kerb is grey, the line on the road red and the hedge green. The mapping
run's bag does: its camera frames (bag_camera.py), its lidar scans
(livox_bag.py) and, replayed through Point-LIO, the pose of the robot at
every instant (scripts/bag_odom.py). This colours the scans and keeps, per
voxel of the sim's grid, the median colour of the returns inside it:

1. Every return of a scan is placed in the map frame from the robot's pose
   at its own firing time (poses interpolated between Point-LIO's 10 Hz
   samples; the lidar frame -> Point-LIO's body, the MID360's IMU, is the
   bag's /tf_static lidar_top -> lidar_top_imu).
2. The image nearest the scan's middle (28 Hz: within 20 ms) is the one the
   returns are coloured from. They are moved into the camera at the image's
   own time (the robot walks ~8 cm in a scan) and projected with the bag's
   camera_info (equidistant: the G23's lens is a fisheye).
3. Occlusion: the lidar sees through the gaps in a hedge, the camera does
   not. The returns are binned on a coarse pixel grid (DEPTH_CELL_PX) and
   only those within DEPTH_SLACK of the nearest return of their cell keep
   the pixel's colour; the rest, behind something, take nothing.
4. Per voxel, the median of each channel over every observation: a passing
   person, a frame's exposure or a mis-timed pixel is outvoted by the frames
   around it.

What it cannot do: colour what the camera never saw — the robot's own back
and sides, the ceilings, the far side of anything — so the photo look
falls back to a neutral shading there (visual.shade_photo), and shadows,
exposure and white balance are baked in as the camera had them.

The table lives in config/colour/<map>.npz (COLOUR_DIR, next to the stair
walks and ramps: a per-map file of the sim's, built by scripts/colour_map.py
from the bag and the replayed odometry, not from the map directory alone, so
not a cache). The sim reads it when the look is `photo` and checks it is for
this grid (lo and shape); a map whose octomap bounds changed needs a rebuild.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np

from .bag_camera import CameraModel, StaticTransform, rotation_matrix
from .voxel_grid import VoxelGrid
from .voxel_intensity import _median_per_key

FORMAT = 1
COLOUR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "config", "colour")
# Returns nearer than this are the robot's own body (the MID360 sees its
# back and the operator's legs); farther than MAX_RANGE a voxel is under a
# pixel wide in a 1280 px fisheye (0.1 m at 30 m = 1.4 px) and its colour
# is haze.
MIN_RANGE_M = 0.6
MAX_RANGE_M = 30.0
# A scan without a frame this close to its middle is left uncoloured: at
# 28 Hz the gap is 18 ms at most unless the feeder dropped frames.
MAX_IMAGE_DT_S = 0.06
# The occlusion buffer: the nearest return per DEPTH_CELL_PX square is what
# the camera saw there; a return within DEPTH_SLACK (relative, absolute) of it
# is on the same surface. 6 px is ~0.5 deg, a voxel at 10 m; 8 % + 0.15 m
# keeps a wall's returns across a cell and drops the one behind the hedge.
DEPTH_CELL_PX = 6
DEPTH_SLACK = (0.08, 0.15)
# Pixels this close to the image's edge are vignetted and most distorted.
EDGE_PX = 8
# Beyond this angle off the optical axis the fisheye model is extrapolated.
MAX_THETA = np.radians(80.0)


# --- the camera --------------------------------------------------------------

@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    d: np.ndarray  # equidistant: k1..k4; plumb_bob: k1 k2 p1 p2 k3
    width: int
    height: int
    model: str = "equidistant"

    @classmethod
    def from_model(cls, m: CameraModel) -> Camera:
        k = m.k
        return cls(fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]),
                   d=np.asarray(m.d, np.float64), width=int(m.width), height=int(m.height),
                   model=m.model)

    def project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(u, v, valid) of (N, 3) points in the optical frame (z forward, x
        right, y down): pixel coordinates, and whether the point is in front
        of the camera and inside the image less EDGE_PX."""
        p = np.asarray(p, np.float64)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = np.hypot(x, y)
        valid = z > 1e-6
        zs = np.where(valid, z, 1.0)
        if self.model == "equidistant":
            theta = np.arctan2(r, zs)
            th2 = theta * theta
            k = np.zeros(4)
            k[:min(4, self.d.shape[0])] = self.d[:4]
            theta_d = theta * (1 + th2 * (k[0] + th2 * (k[1] + th2 * (k[2] + th2 * k[3]))))
            scale = np.where(r > 1e-9, theta_d / np.maximum(r, 1e-9), 1.0 / zs)
            xd, yd = x * scale, y * scale
            valid &= theta < MAX_THETA
        else:  # plumb_bob (OpenCV's radial-tangential)
            xn, yn = x / zs, y / zs
            r2 = xn * xn + yn * yn
            k = np.zeros(5)
            k[:min(5, self.d.shape[0])] = self.d[:5]
            radial = 1 + r2 * (k[0] + r2 * (k[1] + r2 * k[4]))
            xd = xn * radial + 2 * k[2] * xn * yn + k[3] * (r2 + 2 * xn * xn)
            yd = yn * radial + k[2] * (r2 + 2 * yn * yn) + 2 * k[3] * xn * yn
            valid &= np.arctan2(np.sqrt(r2), 1.0) < MAX_THETA
        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy
        valid &= (u >= EDGE_PX) & (u < self.width - EDGE_PX) & (v >= EDGE_PX) \
            & (v < self.height - EDGE_PX)
        return u, v, valid


def colour_points(p_cam: np.ndarray, camera: Camera, image: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """(rgb (N, 3) uint8, keep (N,)) of points in the camera's optical frame:
    the pixel each projects to, for those the camera saw (in the image and
    not behind a nearer return of the same DEPTH_CELL_PX cell)."""
    u, v, ok = camera.project(p_cam)
    depth = p_cam[:, 2]
    cols = -(-camera.width // DEPTH_CELL_PX)
    rows = -(-camera.height // DEPTH_CELL_PX)
    cell = np.zeros(u.shape[0], np.int64)
    cell[ok] = (v[ok].astype(np.int64) // DEPTH_CELL_PX) * cols + u[ok].astype(np.int64) // DEPTH_CELL_PX
    nearest = np.full(rows * cols, np.inf)
    np.minimum.at(nearest, cell[ok], depth[ok])
    rel, absolute = DEPTH_SLACK
    keep = ok & (depth <= nearest[cell] * (1.0 + rel) + absolute)
    rgb = np.zeros((u.shape[0], 3), np.uint8)
    rgb[keep] = image[v[keep].astype(np.int64), u[keep].astype(np.int64), :3]
    return rgb, keep


# --- the poses ---------------------------------------------------------------

class PoseTrack:
    """Point-LIO's odometry (t, p, q) sampled at 10 Hz, interpolated: the
    translation linearly, the rotation by normalised lerp between the two
    samples around t (the heading turns a few degrees between samples at
    most, where nlerp and slerp agree to a fraction of a degree)."""

    def __init__(self, t: np.ndarray, p: np.ndarray, q: np.ndarray):
        order = np.argsort(t, kind="stable")
        self.t = np.asarray(t, np.float64)[order]
        self.p = np.asarray(p, np.float64)[order]
        q = np.asarray(q, np.float64)[order]
        # One hemisphere along the track, so the lerp never goes the long way.
        flip = np.cumsum(np.concatenate(([False], (q[1:] * q[:-1]).sum(axis=1) < 0))) % 2 == 1
        q[flip] *= -1
        self.q = q / np.linalg.norm(q, axis=1, keepdims=True)

    @property
    def span(self) -> tuple[float, float]:
        return float(self.t[0]), float(self.t[-1])

    def at(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(R (N, 3, 3), p (N, 3)) of the body in the odometry frame at times t."""
        t = np.asarray(t, np.float64)
        j = np.clip(np.searchsorted(self.t, t), 1, self.t.shape[0] - 1)
        i = j - 1
        dt = self.t[j] - self.t[i]
        w = np.clip((t - self.t[i]) / np.where(dt > 0, dt, 1.0), 0.0, 1.0)[:, None]
        p = self.p[i] * (1 - w) + self.p[j] * w
        q = self.q[i] * (1 - w) + self.q[j] * w
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return rotation_matrices(q), p


def rotation_matrices(q: np.ndarray) -> np.ndarray:
    """(N, 3, 3) of unit quaternions (N, 4) w x y z."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)


def read_odom(path: str) -> PoseTrack:
    """scripts/bag_odom.py's file: `stamp x y z qw qx qy qz` per line."""
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape[1] < 8:
        raise ValueError(f"{path}: expected `stamp x y z qw qx qy qz` per line")
    return PoseTrack(rows[:, 0], rows[:, 1:4], rows[:, 4:8])


# --- the table ---------------------------------------------------------------

@dataclass
class VoxelColour:
    keys: np.ndarray  # (K,) int64, sorted: linear index of the voxel in the grid
    rgb: np.ndarray  # (K, 3) uint8
    count: np.ndarray  # (K,) observations behind each
    lo: np.ndarray  # (3,) int64, the grid's lowest absolute voxel index (VoxelGrid.lo)
    shape: tuple[int, int, int]

    def _linear(self, idx: np.ndarray) -> np.ndarray:
        local = np.asarray(idx, dtype=np.int64) - self.lo[None, :]
        nx, ny, nz = self.shape
        inside = np.all((local >= 0) & (local < np.array([nx, ny, nz])[None, :]), axis=1)
        lin = (local[:, 0] * ny + local[:, 1]) * nz + local[:, 2]
        return np.where(inside, lin, -1)

    def lookup(self, idx: np.ndarray) -> np.ndarray:
        """(N, 3) float32 rgb (0-255) of absolute voxel indices (N, 3); NaN
        where the camera never saw the voxel."""
        lin = self._linear(idx)
        out = np.full((lin.shape[0], 3), np.nan, dtype=np.float32)
        if self.keys.shape[0] == 0:
            return out
        pos = np.minimum(np.searchsorted(self.keys, lin), self.keys.shape[0] - 1)
        hit = (lin >= 0) & (self.keys[pos] == lin)
        out[hit] = self.rgb[pos[hit]]
        return out


class Builder:
    """Observations (point in the map frame, colour) -> VoxelColour."""

    def __init__(self, grid: VoxelGrid):
        self.lo = grid.lo.astype(np.int64)
        self.shape = tuple(int(s) for s in grid.grid.shape)
        self.origin_m = grid.origin_m.astype(np.float64)
        self.res = float(grid.res)
        self._keys: list[np.ndarray] = []
        self._rgb: list[np.ndarray] = []
        self.observations = 0

    def add(self, points_m: np.ndarray, rgb: np.ndarray) -> int:
        """Keep the observations that fall inside the grid; returns how many."""
        nx, ny, nz = self.shape
        local = np.floor((np.asarray(points_m, np.float64) - self.origin_m[None, :]) / self.res
                         ).astype(np.int64)
        inside = np.all((local >= 0) & (local < np.array([nx, ny, nz])[None, :]), axis=1)
        local = local[inside]
        if local.shape[0] == 0:
            return 0
        self._keys.append((local[:, 0] * ny + local[:, 1]) * nz + local[:, 2])
        self._rgb.append(np.asarray(rgb, np.uint8)[inside])
        self.observations += local.shape[0]
        return local.shape[0]

    def table(self) -> VoxelColour:
        if not self._keys:
            return VoxelColour(np.zeros(0, np.int64), np.zeros((0, 3), np.uint8),
                               np.zeros(0, np.int64), self.lo, self.shape)
        keys = np.concatenate(self._keys)
        rgb = np.concatenate(self._rgb)
        channels = []
        for c in range(3):
            k, v = _median_per_key(keys, rgb[:, c].astype(np.float32))
            channels.append(v)
        _, count = np.unique(keys, return_counts=True)  # the same sorted keys as k
        return VoxelColour(k, (np.stack(channels, axis=1) + 0.5).astype(np.uint8), count,
                           self.lo, self.shape)


def build_from_bag(bag: str, track: PoseTrack, grid: VoxelGrid, camera: Camera,
                   lidar_to_camera: StaticTransform, lidar_to_body: StaticTransform,
                   every_scan: int = 1, max_range: float = MAX_RANGE_M,
                   lidar_topic: str = "/robot01/livox/lidar",
                   image_topic: str | None = None, log=print) -> tuple[VoxelColour, dict]:
    """Colour the bag's scans from its images along `track` (the module
    docstring) and return the table with its build notes."""
    from . import bag_camera
    from .livox_bag import iter_frames

    image_topic = image_topic or bag_camera.IMAGE_TOPIC
    R_lc, t_lc = rotation_matrix(lidar_to_camera.q), lidar_to_camera.t
    R_lb, t_lb = rotation_matrix(lidar_to_body.q), lidar_to_body.t
    images = bag_camera.iter_images(bag, image_topic)
    prev = nxt = None  # the last frame before the scan's middle, the first at or after
    decoded_stamp, image = None, None
    builder = Builder(grid)
    t0 = time.perf_counter()
    lo_t, hi_t = track.span
    scans = used = no_image = off_track = 0
    t_last_log = t0
    for k, frame in enumerate(iter_frames(bag, lidar_topic)):
        if k % every_scan:
            continue
        scans += 1
        t_mid = frame.timebase_ns * 1e-9 + 0.05
        if not lo_t <= t_mid <= hi_t:
            off_track += 1
            continue
        # Advance the images to the two around the scan's middle (the scans
        # come in time order, so the images are read once, in step).
        while nxt is None or nxt.stamp_s < t_mid:
            following = next(images, None)
            if following is None:
                break
            prev, nxt = (nxt if nxt is not None else prev), following
        around = [f for f in (prev, nxt) if f is not None]
        if not around:
            no_image += 1
            continue
        img = min(around, key=lambda f: abs(f.stamp_s - t_mid))
        if abs(img.stamp_s - t_mid) > MAX_IMAGE_DT_S:
            no_image += 1
            continue
        if decoded_stamp != img.stamp_s:
            decoded_stamp, image = img.stamp_s, bag_camera.decode_image(img)

        pts = frame.points
        p_l = np.column_stack((pts["x"], pts["y"], pts["z"])).astype(np.float64)
        rng = np.linalg.norm(p_l, axis=1)
        sel = (rng > MIN_RANGE_M) & (rng < max_range)
        if not sel.any():
            continue
        p_l = p_l[sel]
        t_pt = frame.timebase_ns * 1e-9 + pts["offset_time"][sel].astype(np.float64) * 1e-9
        R_wb, t_wb = track.at(t_pt)
        p_b = (p_l - t_lb) @ R_lb  # lidar -> body (R_lb^T applied row-wise)
        p_map = np.einsum("nij,nj->ni", R_wb, p_b) + t_wb
        # Into the camera at the image's time.
        R_i, t_i = track.at(np.array([img.stamp_s]))
        p_b_i = (p_map - t_i[0]) @ R_i[0]
        p_l_i = p_b_i @ R_lb.T + t_lb
        p_cam = (p_l_i - t_lc) @ R_lc
        rgb, keep = colour_points(p_cam, camera, image)
        if keep.any():
            builder.add(p_map[keep], rgb[keep])
            used += 1
        if time.perf_counter() - t_last_log > 30:
            t_last_log = time.perf_counter()
            log(f"[colour] {scans} scans, {used} coloured, {builder.observations} observations, "
                f"{t_last_log - t0:.0f} s")
    table = builder.table()
    meta = {"format": FORMAT, "bag": os.path.basename(os.path.normpath(bag)),
            "odom_span_s": [lo_t, hi_t], "scans": scans, "scans_coloured": used,
            "scans_without_image": no_image, "scans_off_track": off_track,
            "every_scan": every_scan, "max_range_m": max_range,
            "observations": int(builder.observations), "voxels": int(table.keys.shape[0]),
            "lo": table.lo.tolist(), "shape": list(table.shape),
            "built": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.perf_counter() - t0, 1)}
    log(f"[colour] {scans} scans ({used} coloured, {no_image} without an image, {off_track} off "
        f"the track), {builder.observations} observations -> {table.keys.shape[0]} voxels in "
        f"{meta['seconds']} s")
    return table, meta


# --- the file ----------------------------------------------------------------

def save(path: str, table: VoxelColour, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, format=FORMAT, keys=table.keys, rgb=table.rgb,
                        count=table.count.astype(np.int32), lo=table.lo,
                        shape=np.array(table.shape), meta=json.dumps(meta))
    os.replace(tmp, path)


def load(path: str) -> tuple[VoxelColour, dict]:
    z = np.load(path)
    if int(z["format"]) != FORMAT:
        raise ValueError(f"{path}: format {int(z['format'])}, this is {FORMAT}; rebuild it")
    table = VoxelColour(z["keys"], z["rgb"], z["count"], z["lo"].astype(np.int64),
                        tuple(int(s) for s in z["shape"]))
    return table, json.loads(str(z["meta"]))


def table_path(map_name: str, colour_dir: str = COLOUR_DIR) -> str:
    return os.path.join(colour_dir, f"{map_name}.npz")


def load_for(world, log=print, colour_dir: str = COLOUR_DIR) -> VoxelColour | None:
    """The map's table if there is one for this grid; None (and why) otherwise."""
    path = table_path(world.stats.map_name, colour_dir)
    if not os.path.isfile(path):
        log(f"[colour] no {path}: scripts/colour_map.py builds it from the mapping run's bag")
        return None
    table, meta = load(path)
    g = world.grid
    if tuple(table.lo.tolist()) != tuple(int(v) for v in g.lo) \
            or table.shape != tuple(int(s) for s in g.grid.shape):
        log(f"[colour] {path} is for another grid (lo {table.lo.tolist()}, shape "
            f"{list(table.shape)}; this map's is lo {g.lo.tolist()}, shape "
            f"{list(g.grid.shape)}): rebuild it")
        return None
    log(f"[colour] {path}: {table.keys.shape[0]} voxels from {meta.get('bag')} "
        f"({meta.get('observations')} observations, built {meta.get('built')})")
    return table


def file_key(map_name: str, colour_dir: str = COLOUR_DIR) -> str | None:
    """What a cache keyed on the table should carry: size and mtime, or None."""
    path = table_path(map_name, colour_dir)
    if not os.path.isfile(path):
        return None
    st = os.stat(path)
    return f"{st.st_size}-{int(st.st_mtime)}-{FORMAT}"
