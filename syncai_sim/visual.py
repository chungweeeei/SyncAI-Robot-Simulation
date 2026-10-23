"""How the site looks in the viewer: one colour ramp by height, and flat shading.

The viewer is the only consumer (scene.py, contype 0 geoms); nothing here
touches what the lidar, the feet or the planner see.

* **Flat shading.** MuJoCo shades a mesh with per-vertex normals averaged over
  the faces that share the vertex. The voxel mesh used to share vertices
  between a floor and the wall beside it, so every floor rectangle with a
  corner on a wall had a tilted normal there and its whole surface shaded in
  a smear. The faces are now split by direction (`DIRECTIONS`, six meshes)
  and share vertices only within one direction: every normal of a mesh is the
  same, and a floor is one flat colour.
* **Colour by height.** One 2D texture, a row per kind of surface: floors and
  treads (faces up) take `PALETTE` by absolute height over the site's floor
  range, so storeys, landings and flights read apart; walls take a pale copy
  of the same ramp; undersides are a flat grey. The texture coordinate is
  u = height on the ramp, v = the kind's row; on a planar quad u is linear in z,
  so interpolation across the merged rectangles is exact.
"""

from __future__ import annotations

import numpy as np

# Face directions, the order exposed_face_mesh numbers them: (axis, sign).
DIRECTIONS = ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))
UP, DOWN = 4, 5
KIND_FLOOR, KIND_WALL, KIND_UNDER = 0, 1, 2

# Colour stops on [0, 1] of the height ramp: steel blue low (the ground floor,
# most of what is on screen, so not saturated), through teal, green and sand,
# to terracotta high. Soft on purpose: the robot (dark STL) and the
# props (orange) must stand out against it.
PALETTE = (
    (0.00, (0.42, 0.56, 0.68)),
    (0.25, (0.36, 0.64, 0.64)),
    (0.50, (0.52, 0.70, 0.45)),
    (0.75, (0.86, 0.76, 0.48)),
    (1.00, (0.80, 0.50, 0.40)),
)
WALL_GREY = np.array([0.86, 0.86, 0.88])
WALL_TINT = 0.35  # how much of the ramp's colour a wall keeps
UNDER_GREY = np.array([0.42, 0.42, 0.45])
TEX_W, ROW_H = 256, 8  # rows 8 px tall so mipmaps do not bleed one kind into the next
MIN_RANGE_M, MAX_RANGE_M = 3.0, 10.0

# The sky: a gradient instead of MuJoCo's black, so the gaps a lidar map is
# full of read as openings rather than holes; haze softens the far site.
SKY_TOP = (0.62, 0.70, 0.80)
SKY_BOTTOM = (0.20, 0.23, 0.28)


def ramp(t: np.ndarray) -> np.ndarray:
    """PALETTE at t in [0, 1] -> (N, 3) rgb."""
    xs = np.array([s for s, _ in PALETTE])
    cs = np.array([c for _, c in PALETTE])
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    return np.stack([np.interp(t, xs, cs[:, k]) for k in range(3)], axis=-1)


def texture_rgb() -> np.ndarray:
    """(3 * ROW_H, TEX_W, 3) uint8: floor ramp, pale wall ramp, underside grey."""
    floor = ramp(np.linspace(0.0, 1.0, TEX_W))
    wall = WALL_TINT * floor + (1.0 - WALL_TINT) * WALL_GREY
    under = np.tile(UNDER_GREY, (TEX_W, 1))
    rows = [np.repeat(r[None], ROW_H, axis=0) for r in (floor, wall, under)]
    return (np.concatenate(rows) * 255.0 + 0.5).astype(np.uint8)


def height_range(floor_z: np.ndarray) -> tuple[float, float]:
    """The ramp's span: from the site's low floor to its high one, clamped so
    a one-storey site still has some colour and a tall one is not washed out."""
    if floor_z.size == 0:
        return 0.0, MIN_RANGE_M
    lo = float(np.percentile(floor_z, 1))
    hi = float(np.percentile(floor_z, 99))
    return lo, lo + min(max(hi - lo, MIN_RANGE_M), MAX_RANGE_M)


def texcoords(z: np.ndarray, kind: int, z_range: tuple[float, float]) -> np.ndarray:
    """(N, 2) texture coordinates for vertices at heights z of one kind."""
    lo, hi = z_range
    u = np.clip((np.asarray(z, float) - lo) / (hi - lo), 0.002, 0.998)
    v = np.full_like(u, (kind + 0.5) / 3.0)
    return np.stack([u, v], axis=-1)


def kind_of(direction: int) -> int:
    return KIND_FLOOR if direction == UP else KIND_UNDER if direction == DOWN else KIND_WALL


# A triangle whose normal is within ~40 deg of vertical is a floor (or a tread)
# or an underside; anything steeper is a wall (or a riser).
FLAT_NZ = 0.75


def surface_kinds(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(F,) KIND_* of each triangle of a smooth mesh, by its normal (CCW winding)."""
    a, b, c = (vertices[faces[:, i]].astype(np.float64) for i in range(3))
    n = np.cross(b - a, c - a)
    nz = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    kinds = np.full(faces.shape[0], KIND_WALL, np.int8)
    kinds[nz >= FLAT_NZ] = KIND_FLOOR
    kinds[nz <= -FLAT_NZ] = KIND_UNDER
    return kinds
