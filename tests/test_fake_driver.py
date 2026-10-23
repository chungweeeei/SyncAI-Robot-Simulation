from syncai_sim.fake_driver import LOCOMOTION, MPC, STAND, FakeDriver


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_key_table_and_unknown_key():
    clk = Clock()
    d = FakeDriver(switch_delay=1.0, clock=clk)
    assert d.motion_state == LOCOMOTION and d.policy == 0
    ok, msg = d.set_motion_key("x")
    assert not ok and "Unknown motion key" in msg
    ok, msg = d.set_motion_key("5")
    assert ok and msg == "Motion key sent"
    assert d.update() is False and d.motion_state == LOCOMOTION  # not yet
    clk.t += 0.99
    assert d.update() is False
    clk.t += 0.02
    assert d.update() is True and d.motion_state == MPC
    assert d.mode_array() == [0, MPC]
    assert d.motion_allowed()


def test_stand_gates_motion_and_policy_switch_is_delayed():
    clk = Clock()
    d = FakeDriver(switch_delay=0.5, clock=clk)
    d.set_motion_key("0")
    d.set_policy_mode(1)
    clk.t += 0.5
    assert d.update() is True
    assert d.motion_state == STAND and not d.motion_allowed()
    assert d.policy == 1 and d.mode_array() == [1, STAND]
    assert d.update() is False  # nothing pending now


def test_latest_request_wins():
    clk = Clock()
    d = FakeDriver(switch_delay=1.0, clock=clk)
    d.set_motion_key("5")
    clk.t += 0.5
    d.set_motion_key("1")  # re-armed: MPC never shows
    clk.t += 0.6
    assert d.update() is False and d.motion_state == LOCOMOTION
    clk.t += 0.5
    d.update()
    assert d.motion_state == LOCOMOTION
