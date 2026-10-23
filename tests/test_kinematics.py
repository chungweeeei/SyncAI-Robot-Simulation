"""The step / blocking rules on a hand-made ground: a floor, a kerb, a flight of
0.17 m stairs, a wall and a cliff. No MuJoCo, no numba."""

import math

import pytest

from syncai_sim.kinematics import (
    LOCOMOTION,
    MPC,
    BaseState,
    Event,
    KinematicsConfig,
    advance,
    place,
)

CFG = KinematicsConfig()


class Terrain:
    """Ground height as a function of x (y ignored); walls as x intervals."""

    def __init__(self, height, walls=()):
        self.height = height
        self.walls = walls

    def ground_under(self, x, y, z_hint, search_up=0.3, search_down=3.0):
        h = self.height(x)
        if h is None or h > z_hint + search_up or h < z_hint - search_down:
            return math.nan
        return h

    def disc_blocked(self, cx, cy, z_ground, radius, allowed_step, z_lo, z_hi):
        for x0, x1 in self.walls:
            if cx + radius > x0 and cx - radius < x1:
                return True
        # Columns in the disc whose ground rises more than allowed_step block.
        for dx in (-radius, 0.0, radius):
            h = self.height(cx + dx)
            if h is not None and h - z_ground > allowed_step:
                return True
        return False


def flat(x):
    return 0.0


def kerb(x):  # one 0.15 m kerb at x = 2
    return 0.15 if x >= 2.0 else 0.0


def stairs(x):  # 0.17 m risers every 0.3 m from x = 2 to x = 3.5, then a landing
    if x < 2.0:
        return 0.0
    n = min(int((x - 2.0) // 0.3) + 1, 5)
    return 0.17 * n


def cliff(x):
    return -1.0 if x >= 2.0 else 0.0


def walk(state, ground, vx, policy=0, motion=LOCOMOTION, seconds=10.0, dt=0.01):
    events = []
    for _ in range(int(seconds / dt)):
        state = advance(state, vx, 0.0, 0.0, dt, ground, CFG, policy, motion)
        events.append(state.last_event)
    return state, events


def test_gated_unless_locomotion_or_mpc():
    g = Terrain(flat)
    s = place(BaseState(), g, CFG, 0, 0, 0)
    for motion in (0, 2, 3, 4, 8):
        s2 = advance(s, 0.5, 0, 0, 0.1, g, CFG, 0, motion)
        assert s2.last_event is Event.GATED and s2.x == 0
    for motion in (LOCOMOTION, MPC):
        assert advance(s, 0.5, 0, 0, 0.1, g, CFG, 0, motion).x == pytest.approx(0.05)


def test_integrates_body_frame_velocity_and_yaw():
    g = Terrain(flat)
    s = place(BaseState(), g, CFG, 0, 0, math.pi / 2)
    s = advance(s, 1.0, 0.0, 0.0, 0.5, g, CFG, 0, LOCOMOTION)
    assert s.x == pytest.approx(0.0, abs=1e-9) and s.y == pytest.approx(0.5)
    s = advance(s, 0.0, 0.0, 1.0, 0.25, g, CFG, 0, LOCOMOTION)
    assert s.yaw == pytest.approx(math.pi / 2 + 0.25)


def test_kerb_is_walked_in_any_policy():
    s, _ = walk(place(BaseState(), Terrain(kerb), CFG, 1.5, 0, 0), Terrain(kerb), 0.5)
    assert s.x > 3.0 and s.z_ground == pytest.approx(0.15) and s.refused == 0


def test_stairs_refused_without_stair_policy_after_first_riser():
    g = Terrain(stairs)
    s, events = walk(place(BaseState(), g, CFG, 1.5, 0, 0), g, 0.5, policy=0)
    assert Event.REFUSED_STEP_POLICY in events
    assert s.z_ground == pytest.approx(0.17)  # got onto the first tread, no further
    assert 2.0 <= s.x < 2.35
    assert s.refused > 0 and s.collisions == 0


def test_stairs_climbed_with_stair_policy():
    g = Terrain(stairs)
    s, events = walk(place(BaseState(), g, CFG, 1.5, 0, 0), g, 0.5, policy=1)
    assert s.z_ground == pytest.approx(0.85)
    assert s.x > 4.0 and s.refused == 0 and s.collisions == 0
    assert Event.REFUSED_STEP_POLICY not in events


def test_descent_needs_backward_velocity_in_stair_policy():
    # The flight descends towards -x. Facing -x (yaw = pi) and walking forwards
    # (vx > 0) is refused at the first riser; facing +x (yaw = 0) and walking
    # backwards (vx < 0) — the way the controllers descend — goes down.
    g = Terrain(stairs)
    facing_down = place(BaseState(), g, CFG, 5.0, 0, math.pi, z_hint=0.85)
    s, events = walk(facing_down, g, 0.5, policy=1, seconds=6)
    assert Event.REFUSED_BACKWARD in events
    # The first step down is allowed forwards (a landing nosing); the second is not.
    assert s.z_ground == pytest.approx(0.68)
    facing_up = place(BaseState(), g, CFG, 5.0, 0, 0.0, z_hint=0.85)
    s, events = walk(facing_up, g, -0.5, policy=1, seconds=8)
    assert s.z_ground == pytest.approx(0.0) and s.refused == 0


def test_cliff_is_never_walked():
    g = Terrain(cliff)
    s, events = walk(place(BaseState(), g, CFG, 1.5, 0, 0), g, 0.5, policy=1, seconds=3)
    assert s.z_ground == 0.0 and s.x < 2.0 + 0.01
    assert Event.REFUSED_DROP in events or Event.REFUSED_NO_GROUND in events


def test_wall_blocks_and_counts_collisions():
    g = Terrain(flat, walls=[(3.0, 3.2)])
    s, events = walk(place(BaseState(), g, CFG, 1.5, 0, 0), g, 0.5, seconds=6)
    assert 2.7 <= s.x <= 2.76  # stopped with the disc edge at the wall
    assert s.collisions > 0 and Event.BLOCKED in events
    # Pressed against the wall it can still turn in place.
    turned = advance(s, 0.3, 0.0, 0.5, 0.1, g, CFG, 0, LOCOMOTION)
    assert turned.last_event is Event.BLOCKED and turned.yaw == pytest.approx(0.05)
    spun = advance(s, 0.0, 0.0, 0.5, 0.1, g, CFG, 0, LOCOMOTION)
    assert spun.last_event is Event.MOVED and spun.yaw == pytest.approx(0.05)


def test_pitch_follows_slope_nose_down_positive():
    ramp = Terrain(lambda x: -0.2 * x)  # descending ahead: nose down -> positive pitch
    s = place(BaseState(), ramp, CFG, 1.0, 0, 0, z_hint=-0.2)
    assert s.pitch > 0
    up = Terrain(lambda x: 0.2 * x)
    s = place(BaseState(), up, CFG, 1.0, 0, 0, z_hint=0.2)
    assert s.pitch < 0


def test_place_snaps_to_ground_and_rejects_void():
    g = Terrain(stairs)
    s = place(BaseState(), g, CFG, 5.0, 0.0, 0.0, z_hint=1.0)
    assert s.z_ground == pytest.approx(0.85)
    with pytest.raises(ValueError):
        place(BaseState(), Terrain(lambda x: None), CFG, 0, 0, 0)


def test_one_step_down_onto_a_landing_is_walked_forwards():
    # Climb two risers, then a one-voxel dip onto the landing, then flat.
    def landing(x):
        if x < 2.0:
            return 0.0
        if x < 2.3:
            return 0.2
        if x < 2.6:
            return 0.4
        return 0.2  # the landing sits one quantised step below the top tread
    g = Terrain(landing)
    s, events = walk(place(BaseState(), g, CFG, 1.5, 0, 0), g, 0.5, policy=1, seconds=6)
    assert Event.REFUSED_BACKWARD not in events
    assert s.x > 3.5 and s.z_ground == pytest.approx(0.2)
