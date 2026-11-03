"""The driver's surface when a real controller holds the joints (`locomotion: udp`).

FakeDriver stands in for the controller's state machine when the sim walks
its own policies. With the joints handed to a controller process over the
hardware link (locomotion/hardware_link.py), the state machine is that
process's: the motion keys and policy numbers the viewer's panel or a bench
asks for go to it as the ASCII lines the nav stack's driver_manager sends —
`MODE Z/C/X/R/M`, `MODE <n>`, `ESTOP`, `AXES vx vy wz`, on its command port
(50051) — and what it is in is whatever its telemetry says, which nothing
here listens to by default (`motion_state` stays UNKNOWN until a bench that
reads the telemetry writes it). The surface is FakeDriver's so that every
caller — the panel's teleop, gait_bench, status() — is unchanged.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field

from syncai_sim.driver.controller_link import MODE_CHARS
from syncai_sim.driver.fake_driver import LOCOMOTION, MOTION_KEYS, MPC, UNKNOWN

# driver_manager's protocol.rs motion_key_command: the motion key -> the MODE character.
KEY_CHARS = {key: char for char, key in MODE_CHARS.items()}


@dataclass
class UdpDriver:
    host: str = "127.0.0.1"
    command_port: int = 50051  # the controller's command port
    switch_delay: float = 0.0  # FakeDriver's knob; the controller takes the time it takes
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

    # -- FakeDriver's surface ------------------------------------------------------------
    def update(self) -> bool:
        return False

    def mode_array(self) -> list[int]:
        return [int(self.policy), int(self.motion_state)]

    def motion_allowed(self) -> bool:
        """Unknown counts as allowed: the controller decides, not the sim."""
        return self.motion_state in (LOCOMOTION, MPC, UNKNOWN)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
