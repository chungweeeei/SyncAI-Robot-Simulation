"""The kinematic base: where the trunk goes next, given cmd_vel and the ground.

There is no gait policy to simulate (the robot's HIMLOCO / MPC controller is a
black box on the real robot and its weights are not available), so the G23 is
a rigid body in a standing pose that the sim *places*: this module integrates
`cmd_vel`, follows the ground, pitches on slopes and decides whether a move is
allowed. MuJoCo then pulls the trunk to that pose with a mocap weld
(`scene.py`), which is what makes a contact with a prop physical.

What is deliberately modelled, because the navigation stack depends on it:

* **Ground following.** z is the ground under the body centre (`VoxelGrid.
  ground_under` with the current z as hint, so a 2F floor over 1F is not
  mistaken for the ground), pitch is the slope between two probes ±`pitch_probe`
  ahead and behind, so the lidar looks down a flight of stairs the way the
  real one does and the planner's `lidar_height` subtraction sees the known
  pitch error.
* **The step rule.** A ground change under `riser_min` (0.15) is terrain and
  walked in any policy. A riser of at most `max_step_height` (0.17, the
  planner's limit) is a kerb and is walked in any policy; a second riser within
  `stair_tread_window` (0.45 m) of travel is a staircase and needs the stair
  policy (`set_policy_mode` 1 = HIMLOCO), as does any single riser taller than
  that — which, on a 0.1 m grid, is every real 0.17 m riser stored as 0.2; a
  drop deeper than `max_drop` is never walked. The second descending riser in
  a row, in the stair policy, additionally needs vx < 0 when
  `require_backward_descent` is on: the controllers walk stairs down backwards
  (`stair_descend_backwards`), and this is the only way the sim can tell
  whether they did (the first step down is allowed forwards — a landing's own
  one-voxel nosing). The planner CLI's
  synthetic stairs have 0.17 m risers — exactly the limit — so a rule on
  single-step height alone could not tell a kerb from a flight.
* **Blocking.** The trunk disc (radius `robot_radius`) is checked against the
  voxels in its height band (`body_clearance` over the ground, from
  `min_obstacle_height`); a column whose ground rises more than the allowed
  step is a wall or a too-tall riser. A blocked move leaves the base where it
  is and counts a collision — the test driver fails a run on that count.
* **Gates.** Nothing moves unless the fake driver's motion state is Locomotion
  (1) or MPC (5), the two in which the real controller accepts `cmd_vel`.
  The optional gait dead zones (`walk_vx_min`, `in_place_wz_min`) reproduce
  the RL gait standing still below them; off by default because MPPI already
  floors its own output.

Everything is plain Python over a `Ground` protocol so it is unit-tested
without MuJoCo or numba (`tests/test_kinematics.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Protocol

LOCOMOTION = 1
MPC = 5


class Ground(Protocol):
    def ground_under(self, x: float, y: float, z_hint: float,
                     search_up: float = ..., search_down: float = ...) -> float: ...

    def disc_blocked(self, cx: float, cy: float, z_ground: float, radius: float,
                     allowed_step: float, z_lo: float, z_hi: float) -> bool: ...


@dataclass
class KinematicsConfig:
    max_step_height: float = 0.17
    stair_tread_window: float = 0.45
    max_drop: float = 0.30
    # A ground change smaller than this is terrain (a kerb, a slope sampled on
    # a grid, the one-voxel jitter of a lidar floor) and is walked in any
    # policy without stair bookkeeping; from here up it is a riser. The grid
    # is 0.1 m, so a real 0.17 m riser is stored as 0.1 or 0.2 (a flight shows
    # as 0.2, 0.1, 0.2, 0.2 ...) and a mapped floor undulates by 0.1 — 0.15
    # separates the two, at the price of a flight of 0.1-quantised risers
    # reading as kerbs. dp's ground floor steps between -0.4 and -0.3 within
    # metres; with 0.04 here two of those within a tread window were "a
    # flight" and the robot stood refusing on a flat floor (2026-10-23).
    riser_min: float = 0.15
    # What the body DISC may stand next to without being "blocked", above the
    # ground under its centre. The disc reaches a riser before the centre does,
    # so this is what lets the centre arrive at the step where the step rule
    # judges it. Flat policy: one step plus half a voxel — the grid is 0.1 m,
    # so a 0.17 m riser is stored as 0.1 or 0.2, and 0.17 alone would block
    # the 0.2 ones as walls before the rule could refuse them as steps. Stair
    # policy: room for the next riser of a steep flight inside the disc.
    disc_step_tolerance: float = 0.05
    stair_disc_step_allow: float = 0.45
    require_backward_descent: bool = True
    stair_policy: int = 1
    robot_radius: float = 0.25
    body_clearance: float = 0.60
    min_obstacle_height: float = 0.10
    pitch_probe: float = 0.25
    # How far above the current ground the next column's top may be and still
    # count as ground (a riser to judge) rather than "no ground". dp's first
    # flight has a 0.3 m jump between quantised treads; at 0.3 the search
    # missed it on a float boundary and the base refused "no_ground" forever.
    ground_search_up: float = 0.50
    ground_search_down: float = 3.0
    walk_vx_min: float = 0.0  # 0 = no dead zone
    in_place_wz_min: float = 0.0


class Event(Enum):
    MOVED = "moved"
    GATED = "gated"  # motion state does not accept cmd_vel
    IDLE = "idle"  # zero command
    REFUSED_NO_GROUND = "no_ground"
    REFUSED_DROP = "drop"
    REFUSED_STEP_POLICY = "step_needs_stair_policy"
    REFUSED_BACKWARD = "descent_needs_backward"
    BLOCKED = "blocked"


@dataclass
class BaseState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    z_ground: float = 0.0
    pitch: float = 0.0
    travel_since_step: float = math.inf  # metres walked since the last step edge
    on_stairs: bool = False
    last_step_down: bool = False  # the last riser was a descent (for the backward rule)
    collisions: int = 0
    refused: int = 0
    last_event: Event = field(default=Event.IDLE)
    # Diagnostics of the last decision: the ground the move would have put
    # the centre on and its height change, for reading a refusal off `sim/status`.
    next_ground: float = math.nan
    last_dz: float = 0.0

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z_ground


def place(state: BaseState, ground: Ground, cfg: KinematicsConfig, x: float, y: float,
          yaw: float, z_hint: float | None = None) -> BaseState:
    """Teleport: put the base at (x, y, yaw) on the ground nearest z_hint (or the
    current z). Stair memory is cleared. Raises if there is no ground there."""
    hint = state.z_ground if z_hint is None else z_hint
    g = ground.ground_under(x, y, hint, cfg.ground_search_up, cfg.ground_search_down)
    if math.isnan(g):
        # Look a little further both ways before giving up: a spawn pose from an
        # INI is often a hand-rounded number.
        g = ground.ground_under(x, y, hint, 1.0, 6.0)
    if math.isnan(g):
        raise ValueError(f"no ground under ({x:.2f}, {y:.2f}) near z {hint:.2f}")
    return replace(state, x=x, y=y, yaw=yaw, z_ground=g, pitch=_pitch(ground, cfg, x, y, yaw, g),
                   travel_since_step=math.inf, on_stairs=False, last_event=Event.IDLE)


def _pitch(ground: Ground, cfg: KinematicsConfig, x, y, yaw, g) -> float:
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


def advance(state: BaseState, vx: float, vy: float, wz: float, dt: float, ground: Ground,
            cfg: KinematicsConfig, policy: int, motion_state: int) -> BaseState:
    """One kinematic step of `dt` seconds under the rules in the module docstring."""
    if motion_state not in (LOCOMOTION, MPC):
        return replace(state, last_event=Event.GATED)
    if cfg.walk_vx_min > 0 and 0 < abs(vx) < cfg.walk_vx_min:
        vx = 0.0
    if cfg.in_place_wz_min > 0 and vx == 0.0 and vy == 0.0 and 0 < abs(wz) < cfg.in_place_wz_min:
        wz = 0.0
    if vx == 0.0 and vy == 0.0 and wz == 0.0:
        return replace(state, last_event=Event.IDLE)

    c, s = math.cos(state.yaw), math.sin(state.yaw)
    nx = state.x + (vx * c - vy * s) * dt
    ny = state.y + (vx * s + vy * c) * dt
    nyaw = math.atan2(math.sin(state.yaw + wz * dt), math.cos(state.yaw + wz * dt))
    ds = math.hypot(nx - state.x, ny - state.y)

    g = ground.ground_under(nx, ny, state.z_ground, cfg.ground_search_up, cfg.ground_search_down)
    state = replace(state, next_ground=g, last_dz=(g - state.z_ground) if not math.isnan(g) else 0.0)
    if math.isnan(g):
        return _refuse(state, Event.REFUSED_NO_GROUND)
    dz = g - state.z_ground

    stair_policy = policy == cfg.stair_policy
    allowed_step = (cfg.stair_disc_step_allow if stair_policy
                    else cfg.max_step_height + cfg.disc_step_tolerance)
    travel = state.travel_since_step + ds
    on_stairs = state.on_stairs
    if abs(dz) >= cfg.riser_min:
        if dz < -cfg.max_drop:
            return _refuse(state, Event.REFUSED_DROP)
        is_flight = travel < cfg.stair_tread_window  # a second riser within one tread
        tall = abs(dz) > cfg.max_step_height
        if (is_flight or tall) and not stair_policy:
            return _refuse(state, Event.REFUSED_STEP_POLICY)
        # Descending a flight must be done backwards — but only from the
        # SECOND descending riser: the first step down after a climb is the
        # landing's own quantisation (a mapped tread nosing stands one voxel
        # proud of the landing behind it, dp's first landing drops 1.2 -> 1.0
        # right past the top tread) and a real robot walks onto a landing
        # forwards. Two descents in a row within a tread window is a flight.
        if dz < 0 and stair_policy and (is_flight or tall) and cfg.require_backward_descent \
                and vx >= 0 and state.last_step_down and is_flight:
            return _refuse(state, Event.REFUSED_BACKWARD)
        on_stairs = is_flight or tall
        travel = 0.0
        last_step_down = dz < 0
    else:
        last_step_down = state.last_step_down
        if travel >= cfg.stair_tread_window:
            on_stairs = False
            last_step_down = False

    translating = ds > 1e-9
    if translating and ground.disc_blocked(nx, ny, g, cfg.robot_radius, allowed_step,
                                           cfg.min_obstacle_height, cfg.body_clearance):
        # Blocked: the body stays, but it may still turn — a robot pressed
        # against a wall can rotate in place, and the controllers' recovery
        # from a blocked spot IS a turn (`Stairs: turning in place ... onto the
        # path` repeated for 30 s against a landing wall on dp when the sim
        # refused rotation too, 2026-10-23). A pure rotation is never blocked.
        return replace(state, yaw=nyaw, collisions=state.collisions + 1,
                       last_event=Event.BLOCKED)

    return replace(
        state, x=nx, y=ny, yaw=nyaw, z_ground=g, pitch=_pitch(ground, cfg, nx, ny, nyaw, g),
        travel_since_step=travel, on_stairs=on_stairs, last_step_down=last_step_down,
        last_event=Event.MOVED,
    )


def _refuse(state: BaseState, event: Event) -> BaseState:
    return replace(state, refused=state.refused + 1, last_event=event)
