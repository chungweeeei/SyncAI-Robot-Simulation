"""A Livox MID360 by ray casting: the scan pattern, the returns, the frames.

What matters to the navigation stack and is therefore modelled:

* **The field of view.** 360° around, −7° to +52° in elevation. The lower edge
  is the one that shapes the whole terrain pipeline: from 0.48 m up a −7° ray
  reaches the floor ~3.9 m out, so the floor under and just ahead of the robot
  is never in the current scan and TerrainLayer has to remember it from earlier
  (`ground_memory`). A sim that saw the floor at its feet would never test
  that.
* **Density and rate.** ~20 k returns per 0.1 s scan (the real driver's
  `publish_freq` 10 Hz at ~200 k points/s). The MID360's pattern is a
  non-repeating rosette; here each scan draws fresh pseudo-random directions,
  uniform in azimuth and in sin(elevation) so the coverage per solid angle is
  even, which has the same effect that matters — no two scans sample the same
  spots, so a surface fills in over time.
* **Range.** 0.1 m blind zone, 40 m reach, Gaussian range noise of 1 cm (see
  `LidarConfig` for why not the nominal 2).

What is not: intensity (a constant 100), per-point timestamps (ground-truth
mode publishes a cloud already motion-compensated to the end-of-scan pose,
exactly what Point-LIO's `body_cloud` is), multi-return, rain / glass. Dynamic
props live in MuJoCo and are cast separately by the node, which keeps the
nearer of the two hits per ray.

Frames: directions are generated in the lidar's own frame (`lidar_top`, x
forward, z up the lidar's axis, pitched 14° nose-down on the G23), rotated into
the world by the body pose, cast (`VoxelGrid.cast` decides where on the voxel
the return lands), and the points brought into the frame the stack expects
clouds in — `pointlio_body` = the IMU frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LidarConfig:
    points_per_scan: int = 20000
    scan_rate_hz: float = 10.0
    fov_lower_deg: float = -7.0
    fov_upper_deg: float = 52.0
    min_range: float = 0.1
    max_range: float = 40.0
    # 1 cm, not the MID360's nominal 2 cm: the noise is along the ray, and on
    # a voxel world a return pushed back across a 0.1 m floor step lands in
    # the lower cell a tenth of a metre above its surface — the planner's
    # overlay marks that. At 1 cm it practically never crosses.
    range_noise_sigma: float = 0.01
    intensity: float = 100.0


class LidarModel:
    def __init__(self, cfg: LidarConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._s_lo = np.sin(np.radians(cfg.fov_lower_deg))
        self._s_hi = np.sin(np.radians(cfg.fov_upper_deg))

    def directions(self) -> np.ndarray:
        """(N, 3) unit vectors in the lidar frame for one scan."""
        n = self.cfg.points_per_scan
        az = self.rng.uniform(0.0, 2.0 * np.pi, n)
        s = self.rng.uniform(self._s_lo, self._s_hi, n)  # uniform in sin(el): even per solid angle
        c = np.sqrt(1.0 - s * s)
        return np.column_stack((c * np.cos(az), c * np.sin(az), s))

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


def rotate_rows(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate (N, 3) row vectors by the 3x3 matrix R."""
    return v @ R.T
