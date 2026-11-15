"""Teleop (teleop.py): a held command from the browser's panel, a stop once,
the teleport as a closure under the lock."""

from syncai_sim.viewer.teleop import Teleop


class FakeSim:
    def __init__(self):
        self.cmds = []
        self.teleported = None

    def set_cmd(self, vx, vy, wz):
        self.cmds.append((vx, vy, wz))

    def teleport(self, x, y, yaw, z):
        self.teleported = (x, y, yaw, z)


def test_command_is_held_and_the_stop_sent_once():
    t = Teleop((0, 0, 0, 0), log=lambda *a: None)
    sim = FakeSim()
    t.apply(sim)
    assert sim.cmds == []  # nothing touched: the nav stack's cmd_vel stands
    t.set_cmd(0.4, 0.1, -0.2)
    t.apply(sim)
    t.apply(sim)
    assert sim.cmds == [(0.4, 0.1, -0.2)] * 2  # every loop while moving
    t.stop()
    t.apply(sim)
    t.apply(sim)
    assert sim.cmds == [(0.4, 0.1, -0.2)] * 2 + [(0.0, 0.0, 0.0)]  # one stop, then nothing


def test_teleport_runs_under_the_lock():
    t = Teleop((1, 2, 0, 0.5), log=lambda *a: None)
    sim = FakeSim()
    t.teleport_home()
    assert sim.teleported is None  # queued, not run
    t.apply(sim)
    assert sim.teleported == (1, 2, 0.5, 0)

