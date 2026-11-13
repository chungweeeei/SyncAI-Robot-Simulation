"""The hardware link: the twelve joints handed to a controller process over UDP.

The sim is the robot's hardware as a controller sees
it — motors that close a PD on the command they are sent, an IMU, joint
encoders, and (what hardware does not have) the base's ground truth and
foot contacts. The controller is another process: SyncAI-Robot-Control's
`--backend sim`, whose RL or LMA (MPC) state runs unchanged against this.
The protocol is this module's, and this docstring plus
tests/test_hardware_link.py are its specification.

Two datagram types, both little-endian float32 with no header:

    state   sim -> controller, one per physics step, to `state_host:state_port` (30010)
            "<d55f", 228 bytes:
              t            f64  seconds, strictly increasing, arbitrary origin (below)
              rpy[3]            roll, pitch, yaw: ZYX Euler angles of the base
              acc[3]            the body IMU's specific force, body frame (+9.81 up at rest)
              gyro[3]           the body IMU's angular rate, body frame
              q[12] dq[12]      joint position / velocity, FL FR HL HR x HipX HipY Knee
              tau[12]           the torque applied on each joint this step (the clamp included)
              base_pos[3]       base_link in the world (ground truth)
              base_linvel[3]    base_link's velocity in the WORLD frame (ground truth)
              contact[4]        FL FR HL HR, 1.0 where the foot touches something not the robot
            (sensor/proprioception.py builds the 55 floats; the packet is its record's bytes)

    command controller -> sim, to `command_port` (20001), "<60f", 240 bytes:
              kp[12]  q_des[12]  kd[12]  dq_des[12]  tau_ff[12]
            The motors then apply tau = kp (q_des - q) + kd (dq_des - dq) + tau_ff
            (actuator/motors.py) on every step until the next command; a shorter
            datagram is dropped and logged once.

Why each field is there. A whole-body controller wants the base's attitude,
rate and acceleration, the joints, and ground-truth position and velocity
while its own estimator is not trusted; the estimator wants `tau` because it
confirms a planned foot contact from the foot force it computes with the
Jacobian transpose of the joint torques, and the measured contacts to grade
that. The timestamp must strictly increase: a controller's whole-body stage
treats a repeated time as the same sample and returns its cached output, and
its filter uses the time difference as dt. So `t` is `epoch + sim_time` with
the epoch taken from the wall clock when the link is made: a sim that is
restarted, or that teleports, never hands a running controller a time it has
seen before (an f64 at 1.7e9 still resolves microseconds).

The command is sticky, as a motor driver's is — but a controller that dies
would leave its last torque on the joints for ever, so after `watchdog_s`
without a command the motors fall back to damping (kp 0, kd 2: the
controller's own JointDamping state) and the trip is logged once.

Timing. The controller paces itself on its own wall clock (1 kHz); the sim
paces its physics to the wall clock too and sends a state per step, so a
controller sees timestamps 1 ms apart at `physics_hz` 1000 (the
default; at 500 it sees 2 ms, fine, its loop just reads each state twice). The
sim catches up on a late loop in bursts; `SimConfig.max_catch_up_s` keeps a
burst short (8 ms) so that a controller is never handed a
quarter second of states in a few milliseconds while its command stays the
same.

`StandInController` is the other end for tests and the bench: it answers
every state with a PD to the default pose, which is enough to stand the
robot up through the link with no controller process at all.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

from syncai_sim.actuator.motors import DAMPING_KD, JointCommand
from syncai_sim.entity.robot import Q_DEFAULT
from syncai_sim.sensor.proprioception import N_FIELDS, Sensors

STATE_FMT = f"<d{N_FIELDS}f"
CMD_FMT = "<60f"
STATE_SIZE = struct.calcsize(STATE_FMT)  # 228
CMD_SIZE = struct.calcsize(CMD_FMT)  # 240


@dataclass
class HardwareLinkConfig:
    command_port: int = 20001  # the sim receives commands here
    state_host: str = "127.0.0.1"  # and sends states to this address
    state_port: int = 30010
    bind_host: str = "127.0.0.1"
    watchdog_s: float = 0.1  # no command for this long: damping


def unpack_command(data: bytes) -> JointCommand:
    """The 240-byte command datagram as a JointCommand (float64 copies)."""
    f = np.frombuffer(data, dtype="<f4", count=60).astype(np.float64).reshape(5, 12)
    return JointCommand(q_des=f[1], dq_des=f[3], kp=f[0], kd=f[2], tau_ff=f[4])


def pack_command(cmd: JointCommand) -> bytes:
    return np.concatenate([cmd.kp, cmd.q_des, cmd.kd, cmd.dq_des, cmd.tau_ff]).astype("<f4").tobytes()


class HardwareLink:
    def __init__(self, cfg: HardwareLinkConfig | None = None, log=print, epoch: float | None = None):
        self.cfg = cfg or HardwareLinkConfig()
        self.log = log
        self.epoch = time.time() if epoch is None else float(epoch)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind((self.cfg.bind_host, self.cfg.command_port))
        self.rx.setblocking(False)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (self.cfg.state_host, self.cfg.state_port)
        self._buf = bytearray(STATE_SIZE)
        self._cmd: JointCommand | None = None
        self._cmd_time: float = -np.inf  # sim time the last command arrived (poll's t)
        self._damping = JointCommand.damping(DAMPING_KD)
        self._tripped = False
        self.commands = 0
        self.states = 0
        self.short = 0
        self.watchdog_trips = 0
        log(f"[link] hardware link: commands on udp {self.cfg.bind_host}:{self.rx.getsockname()[1]}, "
            f"states to {self.dest[0]}:{self.dest[1]} ({STATE_SIZE} B each), watchdog "
            f"{self.cfg.watchdog_s * 1000:g} ms")

    @property
    def command_port(self) -> int:
        return int(self.rx.getsockname()[1])

    def time(self, sim_time: float) -> float:
        """The state timestamp for a sim time."""
        return self.epoch + sim_time

    # -- states out -----------------------------------------------------------------
    def send(self, sensors: Sensors) -> None:
        """The record `sensors.read()` just filled, as one state datagram."""
        struct.pack_into("<d", self._buf, 0, sensors.t)
        self._buf[8:] = sensors.f.tobytes()
        try:
            self.tx.sendto(self._buf, self.dest)
            self.states += 1
        except OSError as err:  # no controller yet is fine; anything else is worth a line
            if not isinstance(err, ConnectionRefusedError):
                self.log(f"[link] state send failed: {err}")

    # -- commands in ------------------------------------------------------------------
    def poll(self, t: float) -> JointCommand:
        """The command the motors act on at sim time t: the newest received,
        or damping after `watchdog_s` without one."""
        latest = None
        while True:
            try:
                data, _ = self.rx.recvfrom(4096)
            except BlockingIOError:
                break
            if len(data) < CMD_SIZE:
                self.short += 1
                if self.short == 1:
                    self.log(f"[link] a {len(data)}-byte command dropped (expected {CMD_SIZE}); "
                             f"counting the rest silently")
                continue
            latest = data
        if latest is not None:
            self._cmd = unpack_command(latest)
            self._cmd_time = t
            self.commands += 1
            if self._tripped:
                self.log("[link] controller back: commands again")
                self._tripped = False
        if self._cmd is None or t - self._cmd_time > self.cfg.watchdog_s:
            if self._cmd is not None and not self._tripped:
                self.watchdog_trips += 1
                self._tripped = True
                self.log(f"[link] no command for {self.cfg.watchdog_s * 1000:g} ms: damping")
            return self._damping
        return self._cmd

    def reset(self) -> None:
        """After a spawn / teleport: whatever the controller sent for the old
        pose is stale; damping until it speaks again."""
        self._cmd = None
        self._cmd_time = -np.inf
        self._tripped = False

    def stats(self) -> dict:
        return {"commands": self.commands, "states": self.states, "short": self.short,
                "watchdog_trips": self.watchdog_trips}

    def close(self) -> None:
        self.rx.close()
        self.tx.close()


class StandInController(threading.Thread):
    """The other end of the link with no controller process: a thread that
    answers every state datagram with a PD to the default pose (the
    controller's stand-up gains, kp 70 / kd 2) — or with `command` when one
    is given — so the robot stands through the link. Tests and
    scripts/controller_bench.py --fake-controller use it."""

    def __init__(self, state_port: int, command_port: int, host: str = "127.0.0.1",
                 command: JointCommand | None = None, kp: float = 70.0, kd: float = 2.0):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, state_port))
        self.sock.settimeout(0.05)
        self.dest = (host, command_port)
        self.command = command or JointCommand.pd(Q_DEFAULT, np.full(12, kp), np.full(12, kd))
        self._payload = pack_command(self.command)
        self.last_state: np.ndarray | None = None
        self.last_t: float = -np.inf
        self.received = 0
        self.sent = 0
        self._stopping = threading.Event()  # not `_stop`: Thread's own

    @property
    def state_port(self) -> int:
        return int(self.sock.getsockname()[1])

    def run(self) -> None:
        while not self._stopping.is_set():
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) < STATE_SIZE:
                continue
            self.received += 1
            self.last_t = struct.unpack_from("<d", data, 0)[0]
            self.last_state = np.frombuffer(data, dtype="<f4", count=N_FIELDS, offset=8).copy()
            try:
                self.sock.sendto(self._payload, self.dest)
                self.sent += 1
            except OSError:
                pass

    def stop(self) -> None:
        self._stopping.set()
        self.join(timeout=1.0)
        self.sock.close()
