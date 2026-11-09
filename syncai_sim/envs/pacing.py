"""Fixed physics steps paced to the wall clock: how many to take now.

A late loop catches up in a burst of at most `max_catch_up_s` of sim time;
what one burst cannot repay is dropped, not carried, and the schedule starts
again from now (a resync). This is MuJoCo's own `simulate` loop (main.cc
PhysicsLoop: step while the sim lags the CPU within the refresh budget, and
past `syncMisalign` reset the sync times and take a single step).

Dropping is what keeps the sim to one tie between sim time and the wall.
The MID360's clock (sensor/livox_link.py DeviceClock) is a second one: it
moves its offset forward when the sim has fallen behind, and never back.
When this loop also carried the debt, a stall was absorbed twice — the clock
moved forward by the lag, then the loop repaid it in bursts — and every
stamp after it read in the future by the stall less one burst, for good.
Measured 2026-11-05: a ~0.55 s stall left the clouds 0.45 s ahead of the
host, past lio_bridge's 100 ms of TF, and the robot's local costmap dropped
every cloud from then on. Here a burst always repays the whole debt before
the clock reads the sim's time, and a debt it cannot is never repaid.
"""

from __future__ import annotations


class Pacer:
    """The step schedule: `steps(now)` before stepping, `sleep_s(now)` after."""

    def __init__(self, dt: float, max_catch_up_s: float, start: float):
        self.dt = dt
        self.max_steps = max(1, int(round(max_catch_up_s / dt)))
        self.next_t = start
        self.resyncs = 0

    def steps(self, now: float) -> int:
        """Steps to take at wall time `now` (perf_counter seconds), at least
        one; the schedule moves past them."""
        owed = int((now - self.next_t) / self.dt) + 1
        if owed > self.max_steps:
            # More than one burst behind: drop it and start again from now.
            self.next_t = now
            self.resyncs += 1
            owed = 1
        n = max(1, owed)
        self.next_t += n * self.dt
        return n

    def sleep_s(self, now: float) -> float:
        """How long until the next step is due; <= 0 when it already is."""
        return self.next_t - now
