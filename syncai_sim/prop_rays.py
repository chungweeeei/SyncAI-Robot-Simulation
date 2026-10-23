"""Analytic ray tests for the props, so the lidar sees them without MuJoCo.

`mj_multiRay` tests every geom against every ray (a bounding-sphere check per
pair, no BVH for primitives), which with the site's 18 k collision plates in
the model would cost hundreds of milliseconds per scan. The props are a
handful of boxes and vertical cylinders, so they are intersected here in
numpy — a few milliseconds for 20 k rays — and the node keeps the nearer of
the site hit (`VoxelGrid.cast`) and the prop hit per ray.

Boxes are treated as world-axis-aligned (a pushed crate may rotate a little;
its AABB is what is tested), cylinders as vertical.
"""

from __future__ import annotations

import numpy as np


def ray_aabb(origin: np.ndarray, dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Distance along each unit ray (N, 3) from `origin` to the box [lo, hi];
    inf where the ray misses or starts inside and exits (no entry)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t0 = (lo[None, :] - origin[None, :]) * inv
        t1 = (hi[None, :] - origin[None, :]) * inv
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)
    # A zero direction component yields +-inf or nan (0 * inf) — treat nan as "no constraint".
    tmin = np.where(np.isnan(tmin), -np.inf, tmin)
    tmax = np.where(np.isnan(tmax), np.inf, tmax)
    t_enter = tmin.max(axis=1)
    t_exit = tmax.min(axis=1)
    hit = (t_enter <= t_exit) & (t_exit > 0) & (t_enter > 0)
    return np.where(hit, t_enter, np.inf)


def ray_vertical_cylinder(origin: np.ndarray, dirs: np.ndarray, centre_xy: np.ndarray,
                          radius: float, z_lo: float, z_hi: float) -> np.ndarray:
    """Distance to a capped vertical cylinder; inf where missed."""
    ox, oy = origin[0] - centre_xy[0], origin[1] - centre_xy[1]
    dx, dy = dirs[:, 0], dirs[:, 1]
    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4.0 * a * c
    out = np.full(dirs.shape[0], np.inf)
    ok = (disc >= 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t_side = np.where(ok, (-b - sq) / np.where(ok, 2.0 * a, 1.0), np.inf)
    z_side = origin[2] + t_side * dirs[:, 2]
    side_hit = ok & (t_side > 0) & (z_side >= z_lo) & (z_side <= z_hi)
    out = np.where(side_hit, t_side, out)
    # Caps: the ray meets the planes z = z_lo / z_hi inside the radius.
    for z_plane in (z_lo, z_hi):
        with np.errstate(divide="ignore", invalid="ignore"):
            t_cap = (z_plane - origin[2]) / dirs[:, 2]
        px = origin[0] + t_cap * dirs[:, 0] - centre_xy[0]
        py = origin[1] + t_cap * dirs[:, 1] - centre_xy[1]
        cap_hit = np.isfinite(t_cap) & (t_cap > 0) & (px * px + py * py <= radius * radius)
        out = np.where(cap_hit & (t_cap < out), t_cap, out)
    return out
