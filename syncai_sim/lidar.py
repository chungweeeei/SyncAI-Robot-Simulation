"""A Livox MID360 by ray casting: the scan pattern, the returns, the frames.

What matters to the navigation stack and is therefore modelled:

* **The field of view.** 360° around, −7° to +52° in elevation. The lower edge
  is the one that shapes the whole terrain pipeline: from 0.48 m up a −7° ray
  reaches the floor ~3.9 m out, so the floor under and just ahead of the robot
  is never in the current scan and TerrainLayer has to remember it from earlier
  (`ground_memory`). A sim that saw the floor at its feet would never test
  that.
* **Density, rate and pattern.** ~20 k samples per 0.1 s scan (the real
  driver's `publish_freq` 10 Hz at ~200 k samples/s). Two patterns
  (`LidarConfig.pattern`): `mid360` fires the directions of a recording of
  the robot's own lidar, frame after frame (`LidarPattern`, built by
  scripts/lidar_pattern.py from the workspace's record/dp with
  livox_bag.py, which also says what the recording showed); `random` draws
  fresh pseudo-random directions each scan, uniform in azimuth and in
  sin(elevation), the sim before 2026-10-26 and the fallback without the
  pattern file. Both keep the property that matters most — no two scans
  sample the same spots — but only the recorded one has the MID360's
  rosette: four lasers sweeping smoothly (1.2 deg a sample), dense along the
  sweep and sparse across it, which is what the stack's plane fits and
  TerrainLayer's cells actually see.
* **Time within the scan.** Every sample has a firing time (the recording's
  `offset_time`; evenly spaced for `random`), and the simulator casts each
  ray from where the lidar was at that time (`rays_along_motion`): walking at
  0.4 m/s the sensor moves 4 cm during a scan and turning at 1 rad/s 5.7 deg,
  which smears the pattern over the world the way it does on the robot. The
  cloud handed out is still expressed in the end-of-scan body frame, i.e.
  already motion-compensated, which is what Point-LIO's `body_cloud` is.
* **Range.** 0.1 m blind zone, 40 m reach, Gaussian range noise of 1 cm (see
  `LidarConfig` for why not the nominal 2).

What is not: intensity (a constant 100; the recording's reflectivity has a
median of 18 and nothing in the stack reads it), multi-return, rain / glass.
Dynamic props live in MuJoCo and are cast separately by the node, which keeps
the nearer of the two hits per ray.

Frames: directions are generated in the lidar's own frame (`lidar_top`, x
forward, z up the lidar's axis, pitched 14° nose-down on the G23), rotated into
the world by the body pose, cast (`VoxelGrid.cast` decides where on the voxel
the return lands), and the points brought into the frame the stack expects
clouds in — `pointlio_body` = the IMU frame.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .livox_bag import PATTERN_FORMAT

DEFAULT_PATTERN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config",
                                    "lidar", "mid360_pattern.npz")


@dataclass
class LidarConfig:
    points_per_scan: int = 20000  # random: exactly; mid360: at most (the frame's own count)
    scan_rate_hz: float = 10.0
    fov_lower_deg: float = -7.0  # random only; mid360 fires what the recording fired
    fov_upper_deg: float = 52.0
    min_range: float = 0.1
    max_range: float = 40.0
    # 1 cm, not the MID360's nominal 2 cm: the noise is along the ray, and on
    # a voxel world a return pushed back across a 0.1 m floor step lands in
    # the lower cell a tenth of a metre above its surface — the planner's
    # overlay marks that. At 1 cm it practically never crosses.
    range_noise_sigma: float = 0.01
    intensity: float = 100.0
    pattern: str = "random"  # random | mid360 (module docstring)
    pattern_file: str = ""  # mid360: the bank; "" = config/lidar/mid360_pattern.npz


class LidarPattern:
    """The recorded bank of fired directions: F consecutive frames, each a
    list of (direction, firing time) in the lidar frame (livox_bag.build_pattern)."""

    def __init__(self, path: str):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        if self.meta.get("format") != PATTERN_FORMAT:
            raise ValueError(f"{path}: pattern format {self.meta.get('format')}, "
                             f"this code reads {PATTERN_FORMAT}")
        az = np.radians(z["az"].astype(np.float64) / 100.0)
        el = np.radians(z["el"].astype(np.float64) / 100.0)
        c = np.cos(el)
        self.dirs = np.column_stack((c * np.cos(az), c * np.sin(az), np.sin(el)))
        self.t = z["slot"].astype(np.float64) * (self.meta["firing_period_ns"] * 1e-9)
        frame = z["frame"].astype(np.int64)
        if np.any(np.diff(frame) < 0):
            raise ValueError(f"{path}: frames are not stored in order")
        self.n_frames = int(frame.max()) + 1 if frame.size else 0
        ks = np.arange(self.n_frames)
        self._start = np.searchsorted(frame, ks)
        self._end = np.searchsorted(frame, ks, side="right")
        self.frame_period_s = float(self.meta["frame_period_s"])

    def frame(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(dirs (N, 3), t (N,) seconds from the frame's start) of frame k."""
        a, b = int(self._start[k]), int(self._end[k])
        return self.dirs[a:b], self.t[a:b]


class LidarModel:
    def __init__(self, cfg: LidarConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._s_lo = np.sin(np.radians(cfg.fov_lower_deg))
        self._s_hi = np.sin(np.radians(cfg.fov_upper_deg))
        self.pattern: LidarPattern | None = None
        self._frame = 0
        if cfg.pattern == "mid360":
            self.pattern = LidarPattern(cfg.pattern_file or DEFAULT_PATTERN_FILE)
            if self.pattern.n_frames == 0:
                raise ValueError("empty lidar pattern")
        elif cfg.pattern != "random":
            raise ValueError(f"lidar pattern {cfg.pattern!r}: random | mid360")

    def directions(self) -> tuple[np.ndarray, np.ndarray]:
        """One scan's rays in the lidar frame: (N, 3) unit vectors and (N,)
        firing times in seconds from the scan's start, ascending, all below
        the scan period."""
        n = self.cfg.points_per_scan
        period = 1.0 / self.cfg.scan_rate_hz
        if self.pattern is None:
            az = self.rng.uniform(0.0, 2.0 * np.pi, n)
            s = self.rng.uniform(self._s_lo, self._s_hi, n)  # uniform in sin(el): even per solid angle
            c = np.sqrt(1.0 - s * s)
            return np.column_stack((c * np.cos(az), c * np.sin(az), s)), np.arange(n) * (period / n)
        dirs, t = self.pattern.frame(self._frame % self.pattern.n_frames)
        self._frame += 1
        if dirs.shape[0] > n:  # thin evenly along the sweep, keeping the time order
            pick = np.linspace(0, dirs.shape[0] - 1, n).astype(np.int64)
            dirs, t = dirs[pick], t[pick]
        # The recording's last samples can sit a few hundred microseconds past
        # the nominal period (packets are not cut exactly at 100 ms).
        return dirs, np.minimum(t, np.nextafter(period, 0.0))

    def returns(self, dirs_world: np.ndarray, dist: np.ndarray,
                pts_world: np.ndarray) -> np.ndarray:
        """Apply the blind zone, the reach and the range noise.

        dist: distance along each ray (-1 = no hit); pts_world: the return
        points (see VoxelGrid.cast). Returns the (K, 3) world points of the
        rays that produced a return, noise added along the ray."""
        ok = (dist >= self.cfg.min_range) & (dist <= self.cfg.max_range)
        pts = pts_world[ok]
        if self.cfg.range_noise_sigma > 0:
            noise = self.rng.normal(0.0, self.cfg.range_noise_sigma, pts.shape[0])
            pts = pts + dirs_world[ok] * noise[:, None]
        return pts

    def cloud_xyzi(self, pts_world: np.ndarray, T_map_body: np.ndarray) -> np.ndarray:
        """(K, 4) float32 points in the body (IMU) frame, intensity constant."""
        R = T_map_body[:3, :3]
        pts = (pts_world - T_map_body[:3, 3][None, :]) @ R
        out = np.empty((pts.shape[0], 4), dtype=np.float32)
        out[:, :3] = pts
        out[:, 3] = self.cfg.intensity
        return out


def rays_along_motion(poses: np.ndarray, t: np.ndarray, dt: float,
                      dirs_lidar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where each ray starts and points in the world, given where the lidar was
    during the scan: `poses` (K, 4, 4) is the lidar's pose at each of the K
    physics steps of the scan window, oldest first (the last one is the
    scan's end), `t` the rays' firing times from the window's start, `dt` the
    step. A ray fired at t takes the pose of step floor(t / dt). Returns
    (origins (N, 3), directions (N, 3))."""
    k = poses.shape[0]
    j = np.clip((np.asarray(t, dtype=np.float64) / dt).astype(np.int64), 0, k - 1)
    T = poses[j]
    return np.ascontiguousarray(T[:, :3, 3]), np.einsum("nij,nj->ni", T[:, :3, :3], dirs_lidar)


def rotate_rows(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate (N, 3) row vectors by the 3x3 matrix R."""
    return v @ R.T
