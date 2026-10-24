"""Hold-to-drive teleop (viewer.py): a velocity only while an arrow is down."""

import time

import pytest

viewer = pytest.importorskip("syncai_sim.viewer")


class FakeSim:
    def __init__(self):
        self.cmd = None

    def set_cmd(self, vx, vy, wz):
        self.cmd = (vx, vy, wz)


def test_keymap_held_drives_and_release_stops():
    t = viewer.Teleop((0, 0, 0, 0), log=lambda *a: None)
    sim = FakeSim()
    t.set_held({"up", "left"})
    t.apply(sim)
    assert sim.cmd == (t.vx_speed, 0.0, t.wz_speed)
    t.set_held({"down", "right"})
    t.apply(sim)
    assert sim.cmd == (-t.vx_speed, 0.0, -t.wz_speed)
    t.set_held({"up", "down"})  # opposite keys cancel
    t.apply(sim)
    assert sim.cmd == (0.0, 0.0, 0.0)
    t.set_held(set())  # released
    t.apply(sim)
    assert sim.cmd == (0.0, 0.0, 0.0)


def test_without_keymap_a_press_lapses():
    t = viewer.Teleop((0, 0, 0, 0), log=lambda *a: None)
    sim = FakeSim()
    t.key(viewer.KEY_UP)
    t.apply(sim)
    assert sim.cmd[0] == pytest.approx(t.vx_speed)
    assert t.held(time.monotonic() + viewer.PRESS_HOLD_S + 0.01) == set()
    t.key(viewer.KEY_UP)
    t.key(viewer.KEY_SPACE)  # space drops the pending presses
    t.apply(sim)
    assert sim.cmd == (0.0, 0.0, 0.0)


def test_speed_keys_scale_and_clamp():
    t = viewer.Teleop((0, 0, 0, 0), log=lambda *a: None)
    v0 = t.vx_speed
    t.key(viewer.KEY_EQUAL)
    assert t.vx_speed == pytest.approx(v0 * 1.25)
    for _ in range(20):
        t.key(viewer.KEY_EQUAL)
    assert t.vx_speed == 0.8
    for _ in range(40):
        t.key(viewer.KEY_MINUS)
    assert t.vx_speed == 0.1


def test_without_drive_only_teleport():
    # The session (no --teleop): arrows and motion keys are the nav stack's.
    t = viewer.Teleop((1.0, 2.0, 0.0, 0.5), log=lambda *a: None, drive=False)
    sim = FakeSim()
    sim.teleported = None
    sim.teleport = lambda x, y, yaw, z: setattr(sim, "teleported", (x, y, yaw, z))
    t.set_held({"up"})
    t.key(viewer.KEY_UP)
    t.key(viewer.KEY_P)
    t.key(viewer.KEY_T)
    t.apply(sim)
    assert sim.cmd is None
    assert sim.teleported == (1.0, 2.0, 0.5, 0.0)
