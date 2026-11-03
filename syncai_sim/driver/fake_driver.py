"""The gait controller's state machine, stood in for when the sim walks its own
policies: motion keys and policy numbers in, the state it is in out.

`set_motion_key` takes the key table driver_manager uses ("0".."5": stand,
locomotion, lie down, damping, e-stop, MPC), `set_policy_mode` a policy
number; both succeed at once, and the reported state follows after
`switch_delay`, the controller's response time — what anything waiting on
the state (the nav stack's StairMode settles 3 s without waiting) sees.
`mode_array()` is the [policy, motion_state] pair the controller's telemetry
carries (driver/controller_link.py sends it). The viewer's panel
(viewer/teleop.py) and the ASCII link drive it.

The controller's own boot value is 8 (UNKNOWN); the sim boots in Locomotion
(1) by default — "the operator stood the robot up" — or nothing would move
until a key came.
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
