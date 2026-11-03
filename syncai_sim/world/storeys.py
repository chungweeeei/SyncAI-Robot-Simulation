"""The site's storeys, from where the mapping robot walked.

The viewer draws one storey at a time (or the robot's and those below it), so
it needs the heights that separate them. The map itself does not say where a
storey ends, but `poses.txt` does: the mapping robot spent most of its
keyframes on the floors and only a few on the stairs. dp, by the ground under
each keyframe (2026-10-23):

    -1.0 .. -0.2 m   2066   1F (the range is the map's 0.5 deg tilt)
     1.1 / 2.5 / 3.8    ~90   landings
     4.4 .. 5.6 m     ~700   2F

A storey is a run of 0.2 m height bins each holding at least `min_bin`
keyframes, with at least `min_share` of all keyframes in the run (the
landings, 1-2 % each, are not storeys); the boundary between two storeys sits
`below_floor` under the upper one's low floor (its 5th percentile), so the
stairs and landings go with the storey below and the upper floor's slab, tilt
included, with the one above.
"""

from __future__ import annotations

import numpy as np


def storey_bounds(ground_z: np.ndarray, bin_m: float = 0.2, min_bin: int = 3,
                  min_share: float = 0.05, below_floor: float = 0.3) -> list[float]:
    """Heights (m, ascending) between consecutive storeys; [] for one storey."""
    z = np.asarray(ground_z, float)
    if z.size == 0:
        return []
    edges = np.arange(np.floor(z.min() / bin_m) * bin_m, z.max() + 2 * bin_m, bin_m)
    counts, edges = np.histogram(z, bins=edges)
    runs, start = [], None
    for i, c in enumerate(np.append(counts, 0)):
        if c >= min_bin and start is None:
            start = i
        elif c < min_bin and start is not None:
            runs.append((start, i))
            start = None
    storeys = []
    for a, b in runs:
        inside = z[(z >= edges[a]) & (z < edges[b])]
        if inside.size >= min_share * z.size:
            storeys.append(inside)
    return [float(np.percentile(s, 5)) - below_floor for s in storeys[1:]]


def storey_of(z, bounds: list[float]) -> np.ndarray:
    """Storey index (0 = lowest) of each height."""
    return np.searchsorted(np.asarray(bounds, float), np.asarray(z, float), side="right")
