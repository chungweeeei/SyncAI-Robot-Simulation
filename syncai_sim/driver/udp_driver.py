"""The panel's driving, forwarded to the controller process.

The joints are a controller's (locomotion/hardware_link.py), and so is its
state machine: the viewer's sliders go to it as the `AXES vx vy wz` line the
nav stack's driver_manager sends, on its command port (50051), and a bench
sends any other line as given (`send`). What it is in is whatever its
telemetry says, which nothing here listens to by default (`motion_state`
stays UNKNOWN until a bench that reads the telemetry writes it).

Until 2026-11-15 the panel also sent the motion keys (`MODE Z/C/X/R/M`,
`ESTOP`) and the policy toggle (`MODE <n>`); they went while the
controller's low-level motor control is rewritten, its modes with it.
Until 2026-11-13 a stand-in state machine (fake_driver.py) had this surface
for the sim's own policies, and the sim could answer as the controller on
the same ASCII link (controller_link.py); both went with the policies.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field

# syncai_common/msg/RobotLowLevelMode.msg, data[1]: the state no telemetry has reported
UNKNOWN = 8


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

    # -- as ASCII to the controller ------------------------------------------------
    def send_axes(self, vx: float, vy: float, wz: float) -> None:
        self.send(f"AXES {vx:.6f} {vy:.6f} {wz:.6f}")

    def send(self, line: str) -> None:
        """One command line, as given (a bench's `MODE ...`, `RESET_SAFETY` and the rest)."""
        try:
            self._sock.sendto((line + "\n").encode("ascii"), (self.host, self.command_port))
            self.sent += 1
        except OSError as err:
            if not isinstance(err, ConnectionRefusedError):
                self.log(f"[driver] command send failed: {err}")

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
