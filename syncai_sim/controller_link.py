"""The sim as the gait controller, on driver_manager's UDP link.

On the robot, syncai_driver_manager (Rust, in the workspace) is a UDP bridge to
the gait controller: cmd_vel and the two services go out as ASCII commands,
and the controller's telemetry comes back as ASCII datagrams that become
`imu`, `motor_states`, `battery_state` and `mode`. The safety lock lives in
the driver. With the sim answering on that link, the real driver_manager —
and syncai_robot_state on top of it, and the backend on top of that — run
unchanged in the sim session (`driver_link: udp`); the sim's own services and
`mode` publisher (fake_driver.py) stand down so nothing is served twice.

The protocol, from the workspace's src/syncai_driver_manager/src/driver_manager_node/protocol.rs:

    commands in   AXES <vx> <vy> <wz> | MODE <char> | MODE <uint> | ESTOP
                  MODE Z/C/X/R/M = motion keys 0/1/2/3/5, ESTOP = key 4,
                  MODE <n> = policy n
    telemetry out IMU_RPY r p y  ACC ax ay az  OMEGA wx wy wz  JOINT_POS q0..q11
                  JOINT_VEL ..  JOINT_TAU ..  JOINT_TEMP ..  JOINT_ERR ..  MODE_STATE pol mot
                  BMS_V2 V A soc soh mode event1 thm0 thm1 thm2 internal cell0..cell7
                  (alone in its datagram)

AXES arrives already multiplied by the driver's velocity gains, as the real
controller receives it. The switch delay a key takes to show in MODE_STATE is
FakeDriver's (`mode_switch_delay`): the driver forwards at once, the controller
takes its time, and StairMode waits on that.
"""

from __future__ import annotations

import math
import socket
from dataclasses import dataclass

import numpy as np

from .fake_driver import FakeDriver

# protocol.rs motion_key_command, inverted: the MODE character -> the motion key.
MODE_CHARS = {"Z": "0", "C": "1", "X": "2", "R": "3", "M": "5"}


@dataclass
class LinkConfig:
    command_port: int = 50051  # driver_manager_sim.yaml command_target_port
    telemetry_host: str = "127.0.0.1"
    telemetry_port: int = 50010  # driver_manager_sim.yaml telemetry_recv_port
    telemetry_hz: float = 50.0
    bms_hz: float = 1.0
    battery_soc: float = 90.0  # % at start
    battery_drain_per_hour: float = 0.0  # % per hour of sim time; 0 = a pack that never empties


class Battery:
    """A pack that reads as BMS_V2 does. Eight cells (BMS_V2 carries cell0..7);
    each cell's voltage linear in charge between 3.3 and 4.2 V, the usual
    Li-ion empty / full. The current is a fixed draw, negative = discharging,
    so the driver reports DISCHARGING (protocol.rs takes the state from the
    sign). Nothing in the stack reads more than the percentage
    (syncai_robot_state's low-battery WARNING at 20 %); the rest is for display."""

    CELLS = 8
    CELL_EMPTY, CELL_FULL = 3.3, 4.2
    CURRENT = -2.0  # A
    TEMPERATURE = 30.0  # C, every thermistor

    def __init__(self, soc: float, drain_per_hour: float):
        self.soc = float(soc)
        self.drain_per_s = float(drain_per_hour) / 3600.0

    def advance(self, dt: float) -> None:
        self.soc = max(0.0, self.soc - self.drain_per_s * dt)

    def datagram(self) -> str:
        cell = self.CELL_EMPTY + (self.CELL_FULL - self.CELL_EMPTY) * self.soc / 100.0
        t = self.TEMPERATURE
        values = [cell * self.CELLS, self.CURRENT, self.soc, 100.0, 0, 0, t, t, t, t]
        values += [cell] * self.CELLS
        return "BMS_V2 " + " ".join(_fmt(v) for v in values)


class ControllerLink:
    def __init__(self, sim, driver: FakeDriver, cfg: LinkConfig | None = None, log=print):
        self.sim = sim
        self.driver = driver
        self.cfg = cfg or LinkConfig()
        self.log = log
        self.battery = Battery(self.cfg.battery_soc, self.cfg.battery_drain_per_hour)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind(("127.0.0.1", self.cfg.command_port))
        self.rx.setblocking(False)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (self.cfg.telemetry_host, self.cfg.telemetry_port)
        self._next_tel = 0.0
        self._next_bms = 0.0
        self._last_t: float | None = None
        self.commands = 0
        log(f"[link] gait controller on udp 127.0.0.1:{self.cfg.command_port}, telemetry to "
            f"{self.dest[0]}:{self.dest[1]} at {self.cfg.telemetry_hz:g} Hz")

    # -- commands from driver_manager ---------------------------------------------
    def poll(self) -> None:
        while True:
            try:
                data, _ = self.rx.recvfrom(1024)
            except BlockingIOError:
                return
            for line in data.decode("ascii", "replace").splitlines():
                self.handle(line)

    def handle(self, line: str) -> None:
        parts = line.split()
        if not parts:
            return
        self.commands += 1
        word = parts[0]
        if word == "AXES" and len(parts) == 4:
            try:
                vx, vy, wz = (float(p) for p in parts[1:])
            except ValueError:
                return self.log(f"[link] bad command {line!r}")
            if all(math.isfinite(v) for v in (vx, vy, wz)):
                self.sim.set_cmd(vx, vy, wz)
        elif word == "ESTOP":
            self.driver.set_motion_key("4")
        elif word == "MODE" and len(parts) == 2:
            arg = parts[1]
            if arg.isdigit():
                self.driver.set_policy_mode(int(arg))
            elif arg in MODE_CHARS:
                self.driver.set_motion_key(MODE_CHARS[arg])
            else:
                self.log(f"[link] unknown MODE {arg!r}")
        else:
            self.log(f"[link] unknown command {line!r}")

    # -- telemetry to driver_manager ------------------------------------------------
    def tick(self, t: float) -> None:
        """Send what is due at sim time t (call from the main loop)."""
        if self._last_t is not None:
            self.battery.advance(max(0.0, t - self._last_t))
        self._last_t = t
        if t >= self._next_tel:
            self._next_tel = t + 1.0 / self.cfg.telemetry_hz
            self._send(self.sections())
        if t >= self._next_bms:
            self._next_bms = t + 1.0 / self.cfg.bms_hz
            self._send(self.battery.datagram())

    def sections(self) -> str:
        tel = self.sim.controller_telemetry()
        pol, mot = self.driver.mode_array()
        out = ["IMU_RPY", *map(_fmt, tel["rpy"]), "ACC", *map(_fmt, tel["acc"]),
               "OMEGA", *map(_fmt, tel["omega"])]
        for key in ("JOINT_POS", "JOINT_VEL", "JOINT_TAU"):
            out += [key, *map(_fmt, tel[key])]
        # The sim has no motor temperatures or fault codes: report 0, which
        # the driver passes through as-is (and nothing judges yet: its
        # JOINT_TEMP overheat trigger is still a TODO there).
        out += ["JOINT_TEMP", *(["0"] * 12), "JOINT_ERR", *(["0"] * 12)]
        out += ["MODE_STATE", str(int(pol)), str(int(mot))]
        return " ".join(out)

    def _send(self, text: str) -> None:
        try:
            self.tx.sendto(text.encode("ascii"), self.dest)
        except OSError as err:  # nobody listening yet is fine; anything else is worth a line
            if not isinstance(err, ConnectionRefusedError):
                self.log(f"[link] telemetry send failed: {err}")

    def close(self) -> None:
        self.rx.close()
        self.tx.close()


def _fmt(v) -> str:
    return f"{float(v):.6g}" if isinstance(v, float | np.floating) else str(v)
