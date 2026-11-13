"""Where the base is put down, and where it is read back.

A spawn or a teleport names (x, y, yaw); `place` finds the ground there on
the voxel grid (`VoxelGrid.ground_under`, the current z as hint so a 2F floor
over 1F is not mistaken for the ground) and the slope between two probes
±`pitch_probe` ahead and behind. The simulator then stands the body there on
its legs and, every step, reads `BaseState` back from it (status, the
viewer's storey). Nothing here moves the body: the controller does, through
the joints.

Plain Python over a `Ground` protocol. Until 2026-11-13 this was the
kinematic base (locomotion/kinematics.py), which also integrated cmd_vel
under a step rule; only the placement outlived it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol


class Ground(Protocol):
    def ground_under(self, x: float, y: float, z_hint: float,
                     search_up: float = ..., search_down: float = ...) -> float: ...


@dataclass
class PlacementConfig:
    pitch_probe: float = 0.25
    # How far above / below the current ground a column's top may be and still
    # count as the ground under the base. dp's first flight has a 0.3 m jump
    # between quantised treads; at 0.3 the search missed it on a float
    # boundary (2026-10-23).
    ground_search_up: float = 0.50
    ground_search_down: float = 3.0


@dataclass
class BaseState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    z_ground: float = 0.0
    pitch: float = 0.0
    collisions: int = 0  # the trunk touching the site (Simulator._read_state)

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z_ground


def place(state: BaseState, ground: Ground, cfg: PlacementConfig, x: float, y: float,
          yaw: float, z_hint: float | None = None) -> BaseState:
    """Put the base at (x, y, yaw) on the ground nearest z_hint (or the current
    z). Raises if there is no ground there."""
    hint = state.z_ground if z_hint is None else z_hint
    g = ground.ground_under(x, y, hint, cfg.ground_search_up, cfg.ground_search_down)
    if math.isnan(g):
        # Look a little further both ways before giving up: a spawn pose from an
        # INI is often a hand-rounded number.
        g = ground.ground_under(x, y, hint, 1.0, 6.0)
    if math.isnan(g):
        raise ValueError(f"no ground under ({x:.2f}, {y:.2f}) near z {hint:.2f}")
    return replace(state, x=x, y=y, yaw=yaw, z_ground=g, pitch=_pitch(ground, cfg, x, y, yaw, g))


def _pitch(ground: Ground, cfg: PlacementConfig, x, y, yaw, g) -> float:
    c, s = math.cos(yaw), math.sin(yaw)
    p = cfg.pitch_probe
    gf = ground.ground_under(x + p * c, y + p * s, g, cfg.ground_search_up, cfg.ground_search_down)
    gb = ground.ground_under(x - p * c, y - p * s, g, cfg.ground_search_up, cfg.ground_search_down)
    if math.isnan(gf):
        gf = g
    if math.isnan(gb):
        gb = g
    # REP-103: positive pitch is nose DOWN, so a higher front gives a negative pitch.
    return math.atan2(gb - gf, 2.0 * p)
