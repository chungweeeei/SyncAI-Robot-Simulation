"""The panel's driving, forwarded to the controller process.

The joints are a controller's (locomotion/hardware_link.py), and so is its
state machine: the motion keys and policy numbers the viewer's panel or a
bench asks for go to it as the ASCII lines the nav stack's driver_manager
sends — `MODE Z/C/X/R/M`, `MODE <n>`, `ESTOP`, `AXES vx vy wz`, on its command
port (50051) — and what it is in is whatever its telemetry says, which
nothing here listens to by default (`motion_state` stays UNKNOWN until a
bench that reads the telemetry writes it).

Until 2026-11-13 a stand-in state machine (fake_driver.py) had this surface
for the sim's own policies, and the sim could answer as the controller on
the same ASCII link (controller_link.py); both went with the policies.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field

# syncai_common/msg/RobotLowLevelMode.msg, data[1]: the state the controller reports
STAND, LOCOMOTION, LIE_DOWN, DAMPING, ESTOP, MPC, UNKNOWN = 0, 1, 2, 3, 4, 5, 8
# driver_manager's protocol.rs motion_key_command: key -> that state
MOTION_KEYS: dict[str, int] = {"0": STAND, "1": LOCOMOTION, "2": LIE_DOWN, "3": DAMPING,
                               "4": ESTOP, "5": MPC}
# protocol.rs motion_key_command: the motion key -> the MODE character ("4" is ESTOP).
KEY_CHARS = {"0": "Z", "1": "C", "2": "X", "3": "R", "5": "M"}


@dataclass
class UdpDriver:
    host: str = "127.0.0.1"
    command_port: int = 50051  # the controller's command port
    motion_state: int = UNKNOWN
    policy: int = 0
    log: callable = field(default=print, repr=False)
    sent: int = field(default=0, repr=False)
    _sock: socket.socket | None = field(default=None, repr=False)

    def __post_init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # -- the two services, as ASCII to the controller ------------------------------
    def set_motion_key(self, key: str) -> tuple[bool, str]:
        key = key.strip()
        if key not in MOTION_KEYS:
            return False, f"Unknown motion key '{key}'"
        self.send("ESTOP" if key == "4" else f"MODE {KEY_CHARS[key]}")
        return True, "Emergency stop sent" if key == "4" else "Motion key sent"

    def set_policy_mode(self, mode: int) -> tuple[bool, str]:
        """`policy` becomes the number sent: nothing reads the controller's
        telemetry by default, and the panel's toggle flips what it last asked for."""
        self.policy = int(mode)
        self.send(f"MODE {int(mode)}")
        return True, "Policy updated"

    def send_axes(self, vx: float, vy: float, wz: float) -> None:
        self.send(f"AXES {vx:.6f} {vy:.6f} {wz:.6f}")

    def send(self, line: str) -> None:
        """One command line (`RESET_SAFETY`, `GAIT ...` and the rest included)."""
        try:
            self._sock.sendto((line + "\n").encode("ascii"), (self.host, self.command_port))
            self.sent += 1
        except OSError as err:
            if not isinstance(err, ConnectionRefusedError):
                self.log(f"[driver] command send failed: {err}")

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
