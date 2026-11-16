"""Teleport (teleport.py): the panel's teleport home, queued and run under the lock."""

from syncai_sim.viewer.teleport import Teleport


class FakeSim:
    teleported = None

    def teleport(self, x, y, yaw, z):
        self.teleported = (x, y, yaw, z)


def test_teleport_runs_under_the_lock_once():
    t = Teleport((1, 2, 0, 0.5))
    sim = FakeSim()
    t.home()
    assert sim.teleported is None  # queued, not run
    t.apply(sim)
    assert sim.teleported == (1, 2, 0.5, 0)
    sim.teleported = None
    t.apply(sim)
    assert sim.teleported is None  # run once
