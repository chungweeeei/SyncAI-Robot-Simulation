"""The CAN link: the twelve joints as J60 motors on four SocketCAN buses.

With `joint_link: can` the sim is the robot's motors as the controller's own
hardware layer sees them: four CAN interfaces, one per leg, three J60s on
each, speaking the J60's frames. A controller that drives the real robot
over SocketCAN drives the sim the same way, on vcan interfaces of the same
names (scripts/vcan_up.sh makes them). What a CAN bus does not carry on the
robot it does not carry here: the IMU (a serial port on the robot), the
base's truth and the foot contacts stay on the hardware link's state
datagram (locomotion/hardware_link.py), which is still sent every step.
Commands on the hardware link's UDP port are not read in this mode.

The wiring, as verified on the G2.3 (the robot's controller, g2_deploy_rs:
docs/real_robot/canbus_port_mapping.md and src/hardware/leg_can_mapping.rs):

    can1  HR (its RR)   joints 9..11     motor ids 2, 3, 4 = HipX, HipY, Knee
    can2  FR (its RF)   joints 3..5
    can3  FL (its LF)   joints 0..2
    can4  HL (its LR)   joints 6..8

A motor's position is its own: zero where it was homed (the leg folded
against its stops, `JOINT_HOME`), positive the way it is mounted
(`JOINT_SIGN`). So for joint i

    motor_pos = sign_i (q_i - home_i)      motor_vel = sign_i dq_i
    motor_tau = sign_i tau_i

and the controller's mapping back is the inverse. `SET_HOME` moves a motor's
zero to where the joint is now, as on the robot — after which a controller
that assumes the homed pose reads that joint wrong, as it would there.

Frames (classic CAN, 11-bit id, src/hardware/robot_can_control.rs there):
the id is `command << 5 | motor id`; the motor answers with the same id and
bit 4 set (`| 0x10`). Fields are unsigned integers over a fixed range,
truncated, little-endian:

    CONTROL (4), controller -> motor, 8 bytes
        position 16 bit +-40 rad | velocity 14 bit +-40 rad/s | kp 10 bit 0..1023
        | kd 8 bit 0..51 | torque 16 bit +-40 Nm
    CONTROL reply, motor -> controller, 8 bytes
        position 20 bit +-40 rad | velocity 20 bit +-40 rad/s | torque 16 bit
        +-40 Nm | temperature flag 1 bit | temperature 7 bit -20..200 C
    DISABLE (1), ENABLE (2), SET_HOME (6), ERROR_RESET (17): no data
    SET_CAN_TIMEOUT (9): 1 byte, ms.  SET_BANDWIDTH (10): u16
    GET_STATUS_WORD (23): reply 2 bytes, the error word (0 here, always)
    GET_CONFIG (24): reply 8 bytes, [3] the CAN timeout ms, [4:6] the bandwidth

A motor only reports when it is commanded: every CONTROL frame is answered
with one reply, so the controller's feedback rate is its command rate. The
quantisation is the protocol's and is kept: a position command resolves
1.2 mrad, and a zero velocity or torque command decodes to -2.4 mrad/s and
-0.6 mNm (the ranges have an even number of steps, and the packing
truncates).

What the motor does with it (actuator/motors.py closes the loop):

    disabled (as at power-up)      no torque
    enabled, commanded             tau = kp (p_des - p) + kd (v_des - v) + tq
    enabled, no CONTROL for the    damping (kp 0, kd 2) until the next one
      CAN timeout

Three things here are this module's assumptions, not read from a J60:
that a timed-out motor damps (the hardware link's watchdog does the same)
and recovers on the next command; that a timeout of 0 ms means none; and
what the acknowledgements of the commands without a documented reply carry
(one zero byte, as ERROR_RESET's "success"). The bandwidth is stored and
reported, not modelled (`torque_bw_hz` is the sim's torque filter). No
thermal model: the temperature is `TEMPERATURE_C`.

vcan has no bitrate, arbitration or errors: a frame arrives at once, always.
Two knobs put a bus's faults back, per bus, off by default: `delay_ms` holds
every CONTROL command that long before the motor acts on it (the reply is
not delayed), `drop_rate` loses that fraction of the frames the sim receives
(the command is not applied and nothing is answered). A bus taken down
(`ip link set can2 down`) is a leg lost: its three motors time out and damp
while the others go on.

`StandInCanController` is the other end for tests and the bench: it enables
the motors and commands a PD to the default pose at 1 kHz, which stands the
robot up through the buses with no controller process.
"""

from __future__ import annotations

import collections
import errno
import socket
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

from syncai_sim.actuator.motors import DAMPING_KD, JointCommand
from syncai_sim.entity.robot import Q_DEFAULT
from syncai_sim.sensor.proprioception import Sensors

# struct can_frame: can_id, can_dlc, 3 pad bytes, 8 data bytes.
FRAME_FMT = "=IB3x8s"
FRAME_SIZE = struct.calcsize(FRAME_FMT)  # 16
# can_id's flag bits (extended, remote, error): none of the J60's frames has one.
ID_FLAGS = 0xE0000000

DISABLE_MOTOR, ENABLE_MOTOR, CONTROL_MOTOR, SET_HOME = 1, 2, 4, 6
SET_CAN_TIMEOUT, SET_BANDWIDTH, ERROR_RESET = 9, 10, 17
GET_STATUS_WORD, GET_CONFIG = 23, 24
CMD_SHIFT = 5
REPLY_BIT = 0x10
MOTOR_ID_MASK = 0x0F

POSITION_RANGE = (-40.0, 40.0)  # rad
VELOCITY_RANGE = (-40.0, 40.0)  # rad/s
TORQUE_RANGE = (-40.0, 40.0)  # Nm
KP_RANGE = (0.0, 1023.0)
KD_RANGE = (0.0, 51.0)
TEMPERATURE_RANGE = (-20.0, 200.0)  # C
TEMPERATURE_C = 30.0  # reported for every motor: there is no thermal model

N_BUSES, PER_BUS = 4, 3
# The bus (index into the interfaces, can1..can4) and the slot on it of each
# joint, in the sim's joint order (entity/robot.py LEGS x JOINTS).
JOINT_BUS = (2, 2, 2, 1, 1, 1, 3, 3, 3, 0, 0, 0)
JOINT_SLOT = (0, 1, 2) * 4
# The motor's direction against the joint's (its K_SIGN_BUS x K_SIGN_RL).
JOINT_SIGN = np.array([1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1], dtype=float)
# The joint angles at the homed pose, where every motor reads 0 (its
# K_PRONE_BUS_V2): the leg folded back against its stops.
JOINT_HOME = np.array([-0.450, -3.495, 2.745, 0.450, -3.495, 2.745,
                       -0.450, -3.495, 2.745, 0.450, -3.495, 2.745])
# joint index of (bus, slot)
BUS_JOINT = [[JOINT_BUS.index(b) + s for s in range(PER_BUS)] for b in range(N_BUSES)]


def float_to_uint(x: float, lo: float, hi: float, bits: int) -> int:
    """The protocol's packing: clamped, scaled to 2^bits - 1 steps, truncated."""
    x = min(max(float(x), lo), hi)
    return int((x - lo) * (((1 << bits) - 1) / (hi - lo)))


def uint_to_float(u: int, lo: float, hi: float, bits: int) -> float:
    return u * (hi - lo) / ((1 << bits) - 1) + lo


def can_id(cmd: int, motor_id: int, reply: bool = False) -> int:
    return (cmd << CMD_SHIFT) | (REPLY_BIT if reply else 0) | motor_id


def pack_frame(cid: int, data: bytes = b"") -> bytes:
    return struct.pack(FRAME_FMT, cid, len(data), data.ljust(8, b"\0"))


def unpack_frame(frame: bytes) -> tuple[int, bytes]:
    cid, dlc, data = struct.unpack(FRAME_FMT, frame[:FRAME_SIZE])
    return cid, data[:dlc]


def pack_control(p: float, v: float, kp: float, kd: float, tq: float) -> bytes:
    """A CONTROL command's 8 bytes, motor frame."""
    p_u = float_to_uint(p, *POSITION_RANGE, 16)
    v_u = float_to_uint(v, *VELOCITY_RANGE, 14)
    kp_u = float_to_uint(kp, *KP_RANGE, 10)
    kd_u = float_to_uint(kd, *KD_RANGE, 8)
    t_u = float_to_uint(tq, *TORQUE_RANGE, 16)
    return bytes([p_u & 0xFF, p_u >> 8, v_u & 0xFF, ((v_u >> 8) & 0x3F) | ((kp_u & 0x03) << 6),
                  kp_u >> 2, kd_u, t_u & 0xFF, t_u >> 8])


def unpack_control(d: bytes) -> tuple[float, float, float, float, float]:
    """(p, v, kp, kd, tq) of a CONTROL command, as the motor decodes them."""
    p_u = d[0] | (d[1] << 8)
    v_u = d[2] | ((d[3] & 0x3F) << 8)
    kp_u = (d[3] >> 6) | (d[4] << 2)
    t_u = d[6] | (d[7] << 8)
    return (uint_to_float(p_u, *POSITION_RANGE, 16), uint_to_float(v_u, *VELOCITY_RANGE, 14),
            uint_to_float(kp_u, *KP_RANGE, 10), uint_to_float(d[5], *KD_RANGE, 8),
            uint_to_float(t_u, *TORQUE_RANGE, 16))


def pack_feedback(p: float, v: float, tq: float, temp_c: float = TEMPERATURE_C) -> bytes:
    """A CONTROL reply's 8 bytes, motor frame; the temperature flagged as the motor's."""
    raw = (float_to_uint(p, *POSITION_RANGE, 20)
           | (float_to_uint(v, *VELOCITY_RANGE, 20) << 20)
           | (float_to_uint(tq, *TORQUE_RANGE, 16) << 40)
           | (1 << 56)
           | (float_to_uint(temp_c, *TEMPERATURE_RANGE, 7) << 57))
    return raw.to_bytes(8, "little")


def unpack_feedback(d: bytes) -> tuple[float, float, float, float]:
    """(p, v, tq, temperature) of a CONTROL reply."""
    raw = int.from_bytes(d[:8], "little")
    return (uint_to_float(raw & 0xFFFFF, *POSITION_RANGE, 20),
            uint_to_float((raw >> 20) & 0xFFFFF, *VELOCITY_RANGE, 20),
            uint_to_float((raw >> 40) & 0xFFFF, *TORQUE_RANGE, 16),
            uint_to_float((raw >> 57) & 0x7F, *TEMPERATURE_RANGE, 7))


def open_bus(ifname: str) -> socket.socket:
    """A non-blocking raw CAN socket on `ifname`."""
    if not hasattr(socket, "AF_CAN"):
        raise OSError(errno.EAFNOSUPPORT, "this platform has no SocketCAN (Linux only)")
    s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        s.bind((ifname,))
    except OSError:
        s.close()
        raise
    s.setblocking(False)
    return s


@dataclass
class CanLinkConfig:
    interfaces: tuple[str, ...] = ("can1", "can2", "can3", "can4")
    motor_ids: tuple[int, ...] = (2, 3, 4)  # HipX, HipY, Knee on every bus
    # A motor's CAN timeout at power-up, until SET_CAN_TIMEOUT changes it (the
    # robot's controller sets 200 ms at start).
    timeout_ms: float = 100.0
    delay_ms: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)  # per bus, CONTROL commands held
    drop_rate: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)  # per bus, received frames lost


class CanLink:
    def __init__(self, cfg: CanLinkConfig | None = None, log=print, seed: int = 0,
                 socks: list | None = None):
        """`socks`: four sockets in place of the interfaces' (tests: anything
        with send / recv of 16-byte frames, e.g. AF_UNIX datagram pairs)."""
        self.cfg = cfg or CanLinkConfig()
        self.log = log
        c = self.cfg
        if len(c.interfaces) != N_BUSES or len(c.motor_ids) != PER_BUS:
            raise ValueError(f"the CAN link is {N_BUSES} buses of {PER_BUS} motors: "
                             f"{c.interfaces}, ids {c.motor_ids}")
        if len(c.delay_ms) != N_BUSES or len(c.drop_rate) != N_BUSES:
            raise ValueError(f"can_delay_ms / can_drop_rate take one value per bus ({N_BUSES})")
        if socks is None:
            socks = []
            try:
                for name in c.interfaces:
                    socks.append(open_bus(name))
            except OSError as err:
                for s in socks:
                    s.close()
                raise SystemExit(
                    f"joint_link can: cannot open {name}: {err}. The interfaces must exist where "
                    f"the sim runs (scripts/vcan_up.sh makes them; in the container that needs "
                    f"SIM_NETWORK_MODE=host)") from err
        for s in socks:
            s.setblocking(False)
        self.socks = socks
        self._slot = {mid: k for k, mid in enumerate(c.motor_ids)}
        self._rng = np.random.default_rng(seed)
        # The motors, in joint order.
        self.enabled = np.zeros(12, dtype=bool)
        self.zero = JOINT_HOME.copy()  # the joint angle at which each motor reads 0
        self.timeout_s = np.full(12, c.timeout_ms / 1000.0)
        self.bandwidth = np.zeros(12, dtype=int)
        self._has_cmd = np.zeros(12, dtype=bool)
        self._cmd_time = np.full(12, -np.inf)
        self._timed_out = np.zeros(12, dtype=bool)
        # The command each motor holds, joint frame (JointCommand's arrays).
        self._q_des, self._dq_des = np.zeros(12), np.zeros(12)
        self._kp, self._kd, self._tau_ff = np.zeros(12), np.zeros(12), np.zeros(12)
        self._held: list[collections.deque] = [collections.deque() for _ in range(N_BUSES)]
        self.frames = [0] * N_BUSES  # received, per bus
        self.commands = [0] * N_BUSES  # CONTROL commands applied, per bus
        self.dropped = [0] * N_BUSES
        self.replies = 0
        self.timeouts = 0
        self.tx_errors = 0
        self._tx_logged = False
        log(f"[can] J60 motors on {', '.join(c.interfaces)} (ids {list(c.motor_ids)} each), "
            f"CAN timeout {c.timeout_ms:g} ms; disabled until the controller enables them")

    # -- one physics step -------------------------------------------------------------
    def poll(self, t: float, sensors: Sensors) -> JointCommand:
        """Take the frames that came in, answer them from `sensors`' record (the
        joints after the last step), and return what the motors do at sim time t."""
        delay = self.cfg.delay_ms
        for b, sock in enumerate(self.socks):
            while True:
                try:
                    frame = sock.recv(FRAME_SIZE)
                except BlockingIOError:
                    break
                except OSError:  # the interface went down: nothing to read
                    break
                if len(frame) < FRAME_SIZE:
                    continue
                cid, data = unpack_frame(frame)
                if cid & ID_FLAGS or cid & REPLY_BIT:
                    continue
                slot = self._slot.get(cid & MOTOR_ID_MASK)
                if slot is None:
                    continue
                self.frames[b] += 1
                if self.cfg.drop_rate[b] > 0 and self._rng.random() < self.cfg.drop_rate[b]:
                    self.dropped[b] += 1
                    continue
                self._handle(b, BUS_JOINT[b][slot], cid, data, t, sensors, delay[b] / 1000.0)
            held = self._held[b]
            while held and held[0][0] <= t:
                _, i, cmd = held.popleft()
                self._take(b, i, cmd, t)
        return self._command(t)

    def _handle(self, b: int, i: int, cid: int, data: bytes, t: float, sensors: Sensors,
                delay_s: float) -> None:
        cmd = (cid >> CMD_SHIFT) & 0x3F
        mid = cid & MOTOR_ID_MASK
        s = JOINT_SIGN[i]
        if cmd == CONTROL_MOTOR:
            if len(data) < 8:
                return
            if delay_s > 0:
                self._held[b].append((t + delay_s, i, unpack_control(data)))
            else:
                self._take(b, i, unpack_control(data), t)
            reply = pack_feedback(s * (float(sensors.q[i]) - self.zero[i]),
                                  s * float(sensors.dq[i]), s * float(sensors.tau[i]))
        elif cmd == ENABLE_MOTOR:
            self.enabled[i] = True
            reply = b"\0"
        elif cmd == DISABLE_MOTOR:
            self.enabled[i] = False
            self._has_cmd[i] = False
            reply = b"\0"
        elif cmd == SET_HOME:
            self.zero[i] = float(sensors.q[i])
            self._has_cmd[i] = False  # its target was in the old frame
            reply = b"\0"
        elif cmd == ERROR_RESET:
            reply = b"\0"  # success: there is no error to clear
        elif cmd == SET_CAN_TIMEOUT:
            if data:
                self.timeout_s[i] = data[0] / 1000.0
            reply = b"\0"
        elif cmd == SET_BANDWIDTH:
            if len(data) >= 2:
                self.bandwidth[i] = data[0] | (data[1] << 8)
            reply = b"\0"
        elif cmd == GET_STATUS_WORD:
            reply = b"\0\0"
        elif cmd == GET_CONFIG:
            bw = int(self.bandwidth[i])
            reply = bytes([0, 0, 0, min(255, int(round(self.timeout_s[i] * 1000.0))),
                           bw & 0xFF, bw >> 8, 0, 0])
        else:
            return
        self._send(b, pack_frame(can_id(cmd, mid, reply=True), reply))

    def _take(self, b: int, i: int, cmd: tuple, t: float) -> None:
        """A CONTROL command reaches motor i: into the joint's frame."""
        p, v, kp, kd, tq = cmd
        s = JOINT_SIGN[i]
        self._q_des[i] = s * p + self.zero[i]
        self._dq_des[i] = s * v
        self._kp[i], self._kd[i] = kp, kd
        self._tau_ff[i] = s * tq
        self._has_cmd[i] = True
        self._cmd_time[i] = t
        self.commands[b] += 1

    def _command(self, t: float) -> JointCommand:
        live = self.enabled & self._has_cmd
        late = live & (self.timeout_s > 0) & (t - self._cmd_time > self.timeout_s)
        new = late & ~self._timed_out
        if new.any():
            self.timeouts += int(new.sum())
            self.log(f"[can] no command for the CAN timeout on joints {np.flatnonzero(new).tolist()}: "
                     f"damping")
        back = self._timed_out & ~late
        if back.any():
            self.log(f"[can] joints {np.flatnonzero(back).tolist()} commanded again")
        self._timed_out = late
        on = live & ~late
        return JointCommand(
            q_des=np.where(on, self._q_des, 0.0), dq_des=np.where(on, self._dq_des, 0.0),
            kp=np.where(on, self._kp, 0.0), kd=np.where(on, self._kd, np.where(late, DAMPING_KD, 0.0)),
            tau_ff=np.where(on, self._tau_ff, 0.0))

    def _send(self, b: int, frame: bytes) -> None:
        try:
            self.socks[b].send(frame)
            self.replies += 1
        except OSError as err:  # the bus is down or its queue is full: the reply is lost
            self.tx_errors += 1
            if not self._tx_logged:
                self._tx_logged = True
                self.log(f"[can] a reply on {self.cfg.interfaces[b]} was not sent ({err}); "
                         f"counting the rest silently")

    def reset(self) -> None:
        """After a spawn / teleport: what the controller commanded was for the
        old pose. The motors stay enabled and homed where they were (a
        teleport is not a power cycle)."""
        self._has_cmd[:] = False
        self._timed_out[:] = False
        for held in self._held:
            held.clear()

    def stats(self) -> dict:
        return {"frames": list(self.frames), "commands": list(self.commands),
                "replies": self.replies, "dropped": list(self.dropped),
                "enabled": int(self.enabled.sum()), "timeouts": self.timeouts,
                "tx_errors": self.tx_errors}

    def close(self) -> None:
        for s in self.socks:
            s.close()


class StandInCanController(threading.Thread):
    """The other end of the buses with no controller process: a thread that
    resets and enables the twelve motors, then sends each a CONTROL frame at
    `rate_hz` — a PD to the default pose (the controller's stand-up gains,
    kp 70 / kd 2) — and reads the replies back into joint angles. Tests and
    scripts/controller_bench.py --fake-controller use it; it is also the
    joint <-> motor mapping a controller has to do, in one place."""

    def __init__(self, interfaces=("can1", "can2", "can3", "can4"), motor_ids=(2, 3, 4),
                 q_des: np.ndarray = Q_DEFAULT, kp: float = 70.0, kd: float = 2.0,
                 rate_hz: float = 1000.0, socks: list | None = None):
        super().__init__(daemon=True)
        self.socks = socks if socks is not None else [open_bus(name) for name in interfaces]
        for s in self.socks:
            s.setblocking(False)
        self.motor_ids = tuple(motor_ids)
        self.q_des = np.asarray(q_des, float)
        self.kp, self.kd = float(kp), float(kd)
        self.period = 1.0 / rate_hz
        self.q = np.full(12, np.nan)  # the joints, as the replies report them
        self.received = 0
        self.sent = 0
        self._stopping = threading.Event()  # not `_stop`: Thread's own

    def _send(self, b: int, frame: bytes) -> None:
        try:
            self.socks[b].send(frame)
            self.sent += 1
        except OSError:
            pass

    def _read(self) -> None:
        for b, sock in enumerate(self.socks):
            while True:
                try:
                    frame = sock.recv(FRAME_SIZE)
                except OSError:
                    break
                cid, data = unpack_frame(frame)
                if not cid & REPLY_BIT or (cid >> CMD_SHIFT) & 0x3F != CONTROL_MOTOR:
                    continue
                if (cid & MOTOR_ID_MASK) not in self.motor_ids or len(data) < 8:
                    continue
                i = BUS_JOINT[b][self.motor_ids.index(cid & MOTOR_ID_MASK)]
                self.q[i] = JOINT_SIGN[i] * unpack_feedback(data)[0] + JOINT_HOME[i]
                self.received += 1

    def run(self) -> None:
        for b in range(N_BUSES):
            for mid in self.motor_ids:
                self._send(b, pack_frame(can_id(ERROR_RESET, mid)))
                self._send(b, pack_frame(can_id(ENABLE_MOTOR, mid)))
        frames = [[pack_frame(can_id(CONTROL_MOTOR, mid),
                              pack_control(JOINT_SIGN[i] * (self.q_des[i] - JOINT_HOME[i]), 0.0,
                                           self.kp, self.kd, 0.0))
                   for mid, i in zip(self.motor_ids, BUS_JOINT[b])] for b in range(N_BUSES)]
        next_t = time.perf_counter()
        while not self._stopping.is_set():
            for b in range(N_BUSES):
                for frame in frames[b]:
                    self._send(b, frame)
            self._read()
            next_t += self.period
            wait = next_t - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                next_t = time.perf_counter()  # late: do not burst to catch up

    def stop(self) -> None:
        self._stopping.set()
        self.join(timeout=1.0)
        for s in self.socks:
            s.close()
