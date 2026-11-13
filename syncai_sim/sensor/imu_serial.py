"""The body IMU on a serial port: the robot's Yesense, as its controller reads it.

On the robot the body IMU is not on a CAN bus and not in a datagram: it is a
Yesense unit on a UART (`/dev/ttyTHS1`, 115200 baud) that streams frames on
its own, and the controller's hardware layer parses them (g2_deploy_rs
src/hardware/imu.rs, which this module is byte-compatible with). With
`imu_link: serial` the sim is that unit: it opens a pseudo-terminal, links
`imu_serial_port` to its far end, and writes one frame every
1 / `imu_serial_hz` s of sim time from the body's record
(sensor/proprioception.py: `acc`, `gyro`, `rpy`, the IMU on base_link). The
state datagram of the hardware link carries the same three fields and still
goes out; a controller on the robot's own hardware layer reads this instead.

The frame (little-endian throughout):

    0x59 0x53 | TID u16 | LEN u8 | payload, LEN bytes | CK1 CK2

    payload: items of  ID u8 | LEN u8 | data
        0x10  acceleration   3 x i32, 1e-6 m/s^2      x, y, z (specific force:
                                                      +9.81 on z at rest)
        0x20  angular rate   3 x i32, 1e-6 deg/s      x, y, z
        0x40  Euler angles   3 x i32, 1e-6 deg        pitch, roll, yaw
    CK1, CK2: over TID, LEN and the payload,  CK1 += byte; CK2 += CK1  (mod 256)

so a frame is 49 bytes; TID counts frames and wraps. The controller reads
the three vectors as they are, with no remounting, so they are the sim's
body-frame values: the trunk's x forward, y left, z up, ZYX Euler angles.
Note the Euler item's order, pitch first: it is the unit's, and the
controller swaps it back.

What is not the unit's and is this module's choice:

* The rate. The controller does not configure the unit and nothing in its
  repository says what the robot's is set to; 200 Hz is the default here
  because it is the most 115200 baud carries of this frame (490 bits each,
  235 a second) at a round number. `imu_serial_hz` changes it.
* No noise, bias or latency: the values are the model's sensors'.

A pseudo-terminal has no baud rate and, unlike a UART, keeps what nobody
reads. So that a controller that opens the port late (or reads slowly) gets
the present and not a backlog, the queue is cleared whenever more than
`BACKLOG_S` of frames is waiting in it — a UART would have lost them too.

The pseudo-terminal is in the sim's container: a controller reads it from
there (`docker compose exec sim ...`), or `imu_serial_port` names a real
serial device both can open, which is then written at 115200 8N1 instead.
"""

from __future__ import annotations

import array
import fcntl
import math
import os
import stat
import struct
import termios
import tty
from dataclasses import dataclass

from syncai_sim.sensor.proprioception import Sensors

HEADER = b"\x59\x53"
ID_ACC, ID_GYRO, ID_EULER = 0x10, 0x20, 0x40
MICRO = 1e6
FRAME_SIZE = 5 + 3 * (2 + 12) + 2  # 49
BAUD = 115200
# Frames older than this are dropped from the pseudo-terminal's queue: several
# times the 10 ms the robot's controller's reader blocks for, so a reader
# that keeps up never loses one, and one that opens late is at the present
# within 50 ms.
BACKLOG_S = 0.05


def checksum(data: bytes) -> bytes:
    """The Yesense checksum's two bytes, in the order they are sent."""
    a = b = 0
    for byte in data:
        a = (a + byte) & 0xFF
        b = (b + a) & 0xFF
    return bytes((a, b))


def _micro(x: float) -> int:
    return int(round(x * MICRO))


def pack_frame(tid: int, acc, gyro, rpy) -> bytes:
    """One frame: `acc` m/s^2, `gyro` rad/s, `rpy` rad (roll, pitch, yaw), as
    the body's record has them."""
    deg = math.degrees
    payload = (struct.pack("<BB3i", ID_ACC, 12, *(_micro(float(v)) for v in acc))
               + struct.pack("<BB3i", ID_GYRO, 12, *(_micro(deg(float(v))) for v in gyro))
               + struct.pack("<BB3i", ID_EULER, 12, _micro(deg(float(rpy[1]))),
                             _micro(deg(float(rpy[0]))), _micro(deg(float(rpy[2])))))
    body = struct.pack("<HB", tid & 0xFFFF, len(payload)) + payload
    return HEADER + body + checksum(body)


@dataclass
class ImuSerialConfig:
    port: str = "/tmp/syncai_imu"  # the link to the pseudo-terminal, or a real serial device
    rate_hz: float = 200.0


class ImuSerial:
    def __init__(self, cfg: ImuSerialConfig | None = None, log=print):
        self.cfg = cfg or ImuSerialConfig()
        self.log = log
        c = self.cfg
        if c.rate_hz <= 0 or c.rate_hz * FRAME_SIZE * 10 > BAUD:
            raise ValueError(f"imu_serial_hz {c.rate_hz:g}: a {FRAME_SIZE}-byte frame at {BAUD} baud "
                             f"goes at most {BAUD // (FRAME_SIZE * 10)} times a second")
        self.period = 1.0 / c.rate_hz
        self._backlog = max(2, int(BACKLOG_S * c.rate_hz)) * FRAME_SIZE
        self._link: str | None = None
        self._slave: int | None = None
        try:
            mode = os.lstat(c.port).st_mode
        except FileNotFoundError:
            mode = None
        if mode is not None and stat.S_ISCHR(mode):
            # A real serial device: 115200 8N1, raw, as the unit's UART.
            self.fd = os.open(c.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            attrs = termios.tcgetattr(self.fd)
            attrs[4] = attrs[5] = termios.B115200
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            tty.setraw(self.fd)
            self.device = c.port
        else:
            if mode is not None and not stat.S_ISLNK(mode):
                raise SystemExit(f"imu_link serial: {c.port} exists and is neither a serial device "
                                 f"nor a link left by an earlier run")
            # The far end is kept open here: the pseudo-terminal then outlives
            # a controller that closes it, and its queue can be measured.
            self.fd, self._slave = os.openpty()
            tty.setraw(self._slave)  # no echo, no line editing: bytes are bytes
            os.set_blocking(self.fd, False)
            self.device = os.ttyname(self._slave)
            if mode is not None:
                os.unlink(c.port)
            os.symlink(self.device, c.port)
            self._link = c.port
        self._next = -math.inf
        self.tid = 0
        self.frames = 0
        self.cleared = 0  # times the unread queue was dropped
        self.lost = 0  # frames the port did not take
        log(f"[imu] the body IMU as a Yesense on {c.port}"
            + (f" -> {self.device}" if self._link else "")
            + f", {c.rate_hz:g} Hz ({BAUD} baud on the robot)")

    def tick(self, t: float, sensors: Sensors) -> None:
        """Called every physics step at sim time t: a frame when one is due."""
        if t < self._next:
            return
        # Keep the phase, unless the sim jumped (a start, a long stall).
        self._next = self._next + self.period if t - self._next < 2 * self.period else t + self.period
        if self._slave is not None and self._waiting() > self._backlog:
            termios.tcflush(self._slave, termios.TCIFLUSH)
            self.cleared += 1
        frame = pack_frame(self.tid, sensors.acc, sensors.gyro, sensors.rpy)
        self.tid = (self.tid + 1) & 0xFFFF
        try:
            os.write(self.fd, frame)
            self.frames += 1
        except (BlockingIOError, OSError):
            self.lost += 1

    def _waiting(self) -> int:
        """Bytes written and not yet read by whoever has the port open."""
        n = array.array("i", [0])
        fcntl.ioctl(self._slave, termios.FIONREAD, n)
        return n[0]

    def stats(self) -> dict:
        return {"frames": self.frames, "cleared": self.cleared, "lost": self.lost}

    def close(self) -> None:
        if self._link is not None:
            try:
                os.unlink(self._link)
            except OSError:
                pass
            self._link = None
        for fd in (self.fd, self._slave):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._slave = None
