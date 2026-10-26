"""How the site looks in the browser: a colour ramp by height, shaded by the
lidar's intensity, and flat shading.

The picture is the only consumer (site_tiles.py, web_view.py); nothing here
touches what the lidar, the feet or the planner see.

* **Flat shading.** The voxel mesh's faces are split by direction
  (`DIRECTIONS`, six sets) and share vertices only within one direction, so
  every normal of a tile is the same and a floor is one flat colour rather
  than the smear per-vertex normals averaged over a floor-wall corner gave.
* **Colour by height.** Floors and treads (faces up) take `PALETTE` by
  absolute height over the site's floor range, so storeys, landings and
  flights read apart; walls take a pale copy of the same ramp; undersides are
  a flat grey.
* **Shaded by intensity** (`intensity_factor`, from voxel_intensity.py). A
  lidar map's intensity is what makes it legible — floor markings, door
  frames, signs, the tape on a nosing — so the ramp's colour is scaled by
  the voxel's intensity: dark for the matt floor's ~15-30, up to a third
  brighter for a marking's 90-150 (dp's map.pcd: median 18, 99th percentile
  ~115). A voxel without an intensity keeps a little under the neutral
  brightness, so a map without map.pcd looks like the ramp alone.
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
MIN_RANGE_M, MAX_RANGE_M = 3.0, 10.0

# Intensity -> brightness: the map's intensities from INTENSITY_LO to
# INTENSITY_HI (with a gamma that opens up the dark end, where most of a
# floor is) scale the ramp's colour from FACTOR_LO to FACTOR_HI; brighter
# than that clips to white. UNKNOWN_FACTOR for a voxel with no intensity.
INTENSITY_LO, INTENSITY_HI = 5.0, 90.0
INTENSITY_GAMMA = 0.7
FACTOR_LO, FACTOR_HI = 0.55, 1.30
UNKNOWN_FACTOR = 0.85
SHADE_VERSION = 1  # bump when the constants above change: site_tiles.py's cache key


def ramp(t: np.ndarray) -> np.ndarray:
    """PALETTE at t in [0, 1] -> (N, 3) rgb."""
    xs = np.array([s for s, _ in PALETTE])
    cs = np.array([c for _, c in PALETTE])
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    return np.stack([np.interp(t, xs, cs[:, k]) for k in range(3)], axis=-1)


def height_range(floor_z: np.ndarray) -> tuple[float, float]:
    """The ramp's span: from the site's low floor to its high one, clamped so
    a one-storey site still has some colour and a tall one is not washed out."""
    if floor_z.size == 0:
        return 0.0, MIN_RANGE_M
    lo = float(np.percentile(floor_z, 1))
    hi = float(np.percentile(floor_z, 99))
    return lo, lo + min(max(hi - lo, MIN_RANGE_M), MAX_RANGE_M)


def kind_of(direction: int) -> int:
    return KIND_FLOOR if direction == UP else KIND_UNDER if direction == DOWN else KIND_WALL


def intensity_factor(intensity: np.ndarray) -> np.ndarray:
    """(N,) brightness factor of lidar intensities; NaN -> UNKNOWN_FACTOR."""
    i = np.asarray(intensity, dtype=np.float64)
    t = np.clip((i - INTENSITY_LO) / (INTENSITY_HI - INTENSITY_LO), 0.0, 1.0) ** INTENSITY_GAMMA
    f = FACTOR_LO + (FACTOR_HI - FACTOR_LO) * t
    return np.where(np.isnan(i), UNKNOWN_FACTOR, f)


def shade(kind: np.ndarray, z: np.ndarray, intensity: np.ndarray | None,
          z_range: tuple[float, float]) -> np.ndarray:
    """(N, 3) uint8 colours of surface samples: by `kind` (KIND_*) and height
    on the ramp, scaled by the intensity (None: the ramp alone)."""
    lo, hi = z_range
    t = np.clip((np.asarray(z, float) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    floor = ramp(t)
    rgb = np.where((kind == KIND_FLOOR)[:, None], floor,
                   np.where((kind == KIND_WALL)[:, None],
                            WALL_TINT * floor + (1.0 - WALL_TINT) * WALL_GREY,
                            np.tile(UNDER_GREY, (t.shape[0], 1))))
    if intensity is not None:
        rgb = rgb * intensity_factor(intensity)[:, None]
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


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
