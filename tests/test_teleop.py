"""Teleop (teleop.py): a held command from the browser's panel, a stop once,
the keys as closures under the lock, and the session's teleport-only mode."""

from syncai_sim.teleop import Teleop


class FakeDriver:
    policy = 0

    def set_motion_key(self, key):
        return f"ok {key}"

    def set_policy_mode(self, mode):
        FakeDriver.policy = mode
        return "ok"


class FakeSim:
    def __init__(self):
        self.cmds = []
        self.driver = FakeDriver()
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
    t.set_cmd(0.4, -0.2)
    t.apply(sim)
    t.apply(sim)
    assert sim.cmds == [(0.4, 0.0, -0.2)] * 2  # every loop while moving
    t.stop()
    t.apply(sim)
    t.apply(sim)
    assert sim.cmds == [(0.4, 0.0, -0.2)] * 2 + [(0.0, 0.0, 0.0)]  # one stop, then nothing


def test_keys_run_under_the_lock_in_order():
    logged = []
    t = Teleop((0, 0, 0, 0), log=logged.append)
    sim = FakeSim()
    t.motion_key("1")
    t.toggle_policy()
    t.toggle_policy()
    t.teleport_home()
    assert sim.teleported is None  # queued, not run
    t.apply(sim)
    assert logged[0] == "[teleop] key 1 -> ok 1"
    assert FakeDriver.policy == 0 and "policy -> 1" in logged[1] and "policy -> 0" in logged[2]
    assert sim.teleported == (0, 0, 0, 0)


def test_without_drive_only_the_teleport():
    # The session (no --teleop): the command and the keys are the nav stack's.
    t = Teleop((1.0, 2.0, 0.0, 0.5), log=lambda *a: None, drive=False)
    sim = FakeSim()
    t.set_cmd(0.5, 0.0)
    t.motion_key("5")
    t.toggle_policy()
    t.teleport_home()
    t.apply(sim)
    assert sim.cmds == []
    assert sim.teleported == (1.0, 2.0, 0.5, 0.0)
