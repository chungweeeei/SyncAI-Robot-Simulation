"""What the sim answers in place of `syncai_driver_manager`.

The real driver_manager is a UDP bridge to the gait controller: `set_motion_key`
maps a key to a `MODE x` datagram, `set_policy_mode` to `MODE <n>`, both return
success at once (the send is fire-and-forget), and the controller's telemetry is
echoed on `mode` as `Int32MultiArray [policy, motion_state]` — only when a
telemetry datagram arrives, never periodically. The sim keeps the key table and
the two services' semantics and adds the one thing a test needs: a settable
delay between the request and the reported state, `switch_delay`, because that
is what the stack waits on.

Two consumers, two coupling constraints (see the workspace's stair_mode.cpp and
wait_for_motion_state.cpp):

* `StairMode` sends policy 1 before a flight and does NOT wait for `mode` to
  report it — it settles `stair_policy_settle_time` (3 s) and climbs. So
  `switch_delay` must stay below 3 s for a correct controller to pass; 4 s is
  the "slow driver" negative test. It also restores the policy to whatever
  `mode` reported before the switch (`flat_policy_mode: -1`), so the sim must
  have published a valid non-stair policy before the first stair: the node
  republishes `mode` at 2 Hz for that.
* `WaitForMotionState` (the dock trees) compares `mode` data[1] with the key it
  sent and times out after 5 s, so a dock test with the default 1 s delay sees
  the switch and one with 6 s does not.

The controller's own boot value is 8 (UNKNOWN) and `move_3d.xml` never sends a
motion key, so the sim boots in Locomotion (1) by default — "the operator stood
the robot up" — or nothing would ever move.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# syncai_common/msg/RobotLowLevelMode.msg, data[1]
STAND, LOCOMOTION, LIE_DOWN, DAMPING, ESTOP, MPC, UNKNOWN = 0, 1, 2, 3, 4, 5, 8
# driver_manager's protocol.rs motion_key_command: key -> the state the controller reports
MOTION_KEYS: dict[str, int] = {"0": STAND, "1": LOCOMOTION, "2": LIE_DOWN, "3": DAMPING,
                               "4": ESTOP, "5": MPC}
# data[0]
POLICY_NAMES = {0: "PPO", 1: "HIMLOCO", 2: "CHAMP", 3: "ISSAC"}


@dataclass
class FakeDriver:
    switch_delay: float = 1.0
    motion_state: int = LOCOMOTION
    policy: int = 0
    clock: callable = field(default=time.monotonic, repr=False)
    _pending_motion: tuple[float, int] | None = field(default=None, repr=False)
    _pending_policy: tuple[float, int] | None = field(default=None, repr=False)

    # -- the two services -------------------------------------------------------
    def set_motion_key(self, key: str) -> tuple[bool, str]:
        key = key.strip()
        if key not in MOTION_KEYS:
            return False, f"Unknown motion key '{key}'"
        self._pending_motion = (self.clock() + self.switch_delay, MOTION_KEYS[key])
        return True, "Emergency stop sent" if key == "4" else "Motion key sent"

    def set_policy_mode(self, mode: int) -> tuple[bool, str]:
        self._pending_policy = (self.clock() + self.switch_delay, int(mode))
        return True, "Policy updated"

    # -- the telemetry side -----------------------------------------------------
    def update(self) -> bool:
        """Apply requests whose delay has elapsed; True if the reported state changed."""
        now = self.clock()
        changed = False
        if self._pending_motion and now >= self._pending_motion[0]:
            changed |= self.motion_state != self._pending_motion[1]
            self.motion_state = self._pending_motion[1]
            self._pending_motion = None
        if self._pending_policy and now >= self._pending_policy[0]:
            changed |= self.policy != self._pending_policy[1]
            self.policy = self._pending_policy[1]
            self._pending_policy = None
        return changed

    def mode_array(self) -> list[int]:
        return [int(self.policy), int(self.motion_state)]

    def motion_allowed(self) -> bool:
        return self.motion_state in (LOCOMOTION, MPC)
