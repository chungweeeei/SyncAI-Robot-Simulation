"""The main loop's step schedule (envs/pacing.py), against the wall clock it
paces to and the MID360's clock that reads it."""

import pytest

from syncai_sim.envs.pacing import Pacer
from syncai_sim.sensor.livox_link import DeviceClock

DT = 0.002


def test_a_late_loop_repays_the_whole_debt_in_one_burst():
    pacer = Pacer(DT, 0.1, start=0.0)
    # 50 ms late: 26 steps, and the next step is due after now, not before.
    assert pacer.steps(0.05) == 26
    assert pacer.sleep_s(0.05) > 0
    assert pacer.resyncs == 0


def test_a_loop_on_time_takes_one_step_and_one_ahead_still_takes_one():
    pacer = Pacer(DT, 0.1, start=0.0)
    assert pacer.steps(0.0) == 1
    assert pacer.steps(0.0) == 1  # ahead of the schedule: never zero
    assert pacer.sleep_s(0.0) == pytest.approx(2 * DT)


def test_a_debt_past_one_burst_is_dropped_not_carried():
    pacer = Pacer(DT, 0.1, start=0.0)
    assert pacer.steps(0.55) == 1
    assert pacer.resyncs == 1
    # Nothing is left owed: the next iteration on time takes one step.
    assert pacer.steps(0.55 + DT) == 1


def run(stall_s: float, max_catch_up_s: float = 0.1, loop_cost_s: float = 0.0005):
    """The main loop for 20 s, with one stall at 10 s, feeding the MID360's
    clock the way __main__ does: steps, then `DeviceClock.update`. Returns
    the last stamp less the wall, in seconds, and every stamp read."""
    wall = [0]
    clock = DeviceClock(wall_ns=lambda: wall[0])
    pacer = Pacer(DT, max_catch_up_s, start=0.0)
    sim_t, now, stalled, stamps = 0.0, 0.0, False, []
    while now < 20.0:
        if not stalled and now >= 10.0:
            now += stall_s
            stalled = True
        sim_t += pacer.steps(now) * DT
        wall[0] = int(now * 1e9)
        clock.update(sim_t)
        stamps.append(clock.ns(sim_t))
        now = max(now + loop_cost_s, now + pacer.sleep_s(now))
    return (stamps[-1] - wall[0]) / 1e9, stamps


@pytest.mark.parametrize("stall_s", [0.05, 0.2, 0.55, 1.0])
def test_a_stall_leaves_the_device_clock_on_the_wall(stall_s):
    # Measured 2026-11-05 with the debt carried: a 0.55 s stall left every
    # stamp 0.45 s in the future, past the robot's 100 ms of TF.
    lead, stamps = run(stall_s)
    assert abs(lead) < 0.03
    assert all(b > a for a, b in zip(stamps, stamps[1:]))  # never back


def test_a_stall_in_the_udp_profile_leaves_it_on_the_wall_too():
    lead, _ = run(0.3, max_catch_up_s=0.008)
    assert abs(lead) < 0.03
