"""sensor/imu_serial.py: the Yesense frame byte for byte, read back the way
the robot's controller reads it (g2_deploy_rs src/hardware/imu.rs, ported
below), and the pseudo-terminal: bytes arrive unaltered, at the rate, and a
port nobody reads does not hand its backlog to whoever opens it later."""

import math
import os
import struct
import termios
import tty
from types import SimpleNamespace

import numpy as np
import pytest

from syncai_sim.sensor.imu_serial import (FRAME_SIZE, ImuSerial, ImuSerialConfig, checksum,
                                          pack_frame)

QUIET = lambda *a: None  # noqa: E731


def record(acc=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    """The IMU's part of a Sensors record."""
    return SimpleNamespace(acc=np.array(acc, np.float32), gyro=np.array(gyro, np.float32),
                           rpy=np.array(rpy, np.float32))


def controller_parse(buffer: bytes) -> list[dict]:
    """The controller's reader: sync on 0x59 0x53, the length at [4], the
    checksum over [2:-2], then the items; one snapshot per good frame."""
    out, pos = [], 0
    while pos + 7 <= len(buffer):
        if buffer[pos] != 0x59 or buffer[pos + 1] != 0x53:
            pos += 1
            continue
        n = buffer[pos + 4] + 7
        if pos + n > len(buffer):
            break
        frame = buffer[pos:pos + n]
        a = b = 0
        for byte in frame[2:-2]:
            a = (a + byte) & 0xFF
            b = (b + a) & 0xFF
        if struct.unpack("<H", frame[-2:])[0] != (b << 8 | a):
            pos += 1
            continue
        snap, i = {"tid": struct.unpack("<H", frame[2:4])[0]}, 5
        while i + 2 <= n - 2:
            iid, length = frame[i], frame[i + 1]
            v = [x * 1e-6 for x in struct.unpack("<3i", frame[i + 2:i + 14])]
            if iid == 0x10:
                snap["acc"] = v
            elif iid == 0x20:
                snap["gyro"] = [math.radians(x) for x in v]
            elif iid == 0x40:
                snap["rpy"] = [math.radians(v[1]), math.radians(v[0]), math.radians(v[2])]
            i += 2 + length
        out.append(snap)
        pos += n
    return out


def port(tmp_path, **kw):
    path = str(tmp_path / "imu")
    return ImuSerial(ImuSerialConfig(port=path, **kw), log=QUIET), path


def open_port(path) -> int:
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    # Raw, as the controller's serial library sets it; TCSANOW so that the
    # test sees what was queued (a library that flushes on open sees less).
    tty.setraw(fd, termios.TCSANOW)
    return fd


def read_all(fd) -> bytes:
    data = b""
    while True:
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            return data
        if not chunk:
            return data
        data += chunk


def test_the_frame_byte_for_byte():
    f = pack_frame(0x0102, (1.0, -2.0, 9.81), (math.radians(10.0), 0.0, math.radians(-90.0)),
                   (math.radians(1.0), math.radians(2.0), math.radians(3.0)))
    assert len(f) == FRAME_SIZE == 49
    assert f[:5] == bytes([0x59, 0x53, 0x02, 0x01, 42])  # header, TID little-endian, payload length
    assert f[5:7] == bytes([0x10, 12]) and struct.unpack("<3i", f[7:19]) == (1000000, -2000000, 9810000)
    assert f[19:21] == bytes([0x20, 12]) and struct.unpack("<3i", f[21:33]) == (10000000, 0, -90000000)
    # Euler: pitch first, then roll, then yaw, in micro-degrees.
    assert f[33:35] == bytes([0x40, 12]) and struct.unpack("<3i", f[35:47]) == (2000000, 1000000, 3000000)
    assert f[47:] == checksum(f[2:47])


def test_the_checksum_is_the_units():
    # CK1 the byte sum, CK2 the sum of the running sums: 1, 2, 3 -> 6 and 1 + 3 + 6.
    assert checksum(bytes([1, 2, 3])) == bytes([6, 10])
    assert checksum(bytes([0xFF, 0x01])) == bytes([0x00, 0xFF])  # each wraps at 256


def test_the_controllers_reader_gets_the_bodys_values_back():
    rpy = (0.1, -0.2, 3.0)
    f = pack_frame(7, (0.3, -0.4, 9.7), (0.5, -0.6, 0.7), rpy)
    (snap,) = controller_parse(b"\x00\x59" + f)  # garbage before the header is skipped
    assert snap["tid"] == 7
    assert np.allclose(snap["acc"], (0.3, -0.4, 9.7), atol=1e-6)
    assert np.allclose(snap["gyro"], (0.5, -0.6, 0.7), atol=1e-7)
    assert np.allclose(snap["rpy"], rpy, atol=1e-7)  # roll, pitch, yaw again
    bad = bytearray(f)
    bad[10] ^= 1
    assert controller_parse(bytes(bad)) == []


def test_the_port_streams_at_the_rate_and_unaltered(tmp_path):
    imu, path = port(tmp_path, rate_hz=200.0)
    assert os.path.islink(path) and os.readlink(path) == imu.device
    fd = open_port(path)
    # Values whose bytes a cooked terminal would eat or rewrite: 0x0D, 0x0A,
    # 0x03, 0x11, 0x13, 0x7F all occur in these frames.
    rec = record(acc=(0.000013, 0.00001, 0.000003), gyro=(0.0, 0.0, 0.0),
                 rpy=(math.radians(0.000017), math.radians(0.000019), math.radians(0.000127)))
    expect = data = b""
    for step in range(1, 101):  # 100 ms at 1 kHz
        imu.tick(step * 0.001, rec)
        if step % 5 == 1:
            expect += pack_frame(step // 5, rec.acc, rec.gyro, rec.rpy)
        if step % 10 == 0:  # the controller's reader, every 10 ms
            data += read_all(fd)
    assert imu.stats() == {"frames": 20, "cleared": 0, "lost": 0}
    assert data == expect
    assert [s["tid"] for s in controller_parse(data)] == list(range(20))
    os.close(fd)
    imu.close()
    assert not os.path.lexists(path)


def test_a_port_nobody_reads_keeps_no_backlog(tmp_path):
    imu, path = port(tmp_path, rate_hz=200.0)
    for step in range(1, 5001):  # 5 s with no reader
        imu.tick(step * 0.001, record())
    assert imu.stats()["frames"] == 1000 and imu.stats()["cleared"] > 0 and imu.stats()["lost"] == 0
    fd = open_port(path)
    stale = controller_parse(read_all(fd))
    assert len(stale) <= 12  # 50 ms of frames and the one that tipped it, not five seconds
    assert stale[-1]["tid"] == 999
    imu.tick(5.005, record())
    assert [s["tid"] for s in controller_parse(read_all(fd))] == [1000]
    os.close(fd)
    imu.close()


def test_a_stale_link_is_replaced_and_a_file_is_not(tmp_path):
    first, path = port(tmp_path)
    os.close(first.fd)  # died without close(): the link stays
    second, _ = port(tmp_path)
    assert os.readlink(path) == second.device
    second.close()
    with open(path, "w") as f:
        f.write("not a port")
    with pytest.raises(SystemExit):
        ImuSerial(ImuSerialConfig(port=path), log=QUIET)


def test_a_rate_the_baud_cannot_carry_is_refused(tmp_path):
    with pytest.raises(ValueError):
        port(tmp_path, rate_hz=250.0)
