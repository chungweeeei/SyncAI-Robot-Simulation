"""sensor/bms_link.py: the Foxlink BMS's frames byte for byte (extended ids,
big-endian data), read back the way the robot's controller reads them
(g2_deploy_rs src/hardware/bms.rs), that it is silent until asked, and the
battery model behind the numbers. The bus is an AF_UNIX datagram pair, as in
tests/test_can_link.py; the last test uses a real interface and skips
without one."""

import os
import socket
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from syncai_sim.locomotion.can_link import FRAME_SIZE, pack_frame, unpack_frame
from syncai_sim.sensor.bms_link import CAN_EFF_FLAG, HEAT_W_PER_NM2, TABLE, Bms, BmsConfig

QUIET = lambda *a: None  # noqa: E731
EFF = 0x80000000


def body(tau=None, dq=None):
    return SimpleNamespace(tau=np.zeros(12) if tau is None else np.asarray(tau, float),
                           dq=np.zeros(12) if dq is None else np.asarray(dq, float))


def make(**kw):
    sim, ctrl = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    ctrl.setblocking(False)
    return Bms(BmsConfig(**kw), log=QUIET, sock=sim), ctrl


def drain(sock) -> list[tuple[int, bytes]]:
    out = []
    while True:
        try:
            out.append(unpack_frame(sock.recv(FRAME_SIZE)))
        except BlockingIOError:
            return out


def controller_read(frames) -> dict:
    """The controller's BmsState.apply: what it keeps of each frame."""
    s = {"cells": [None] * 14}
    for cid, d in frames:
        assert cid & EFF, "every frame of this BMS is an extended one"
        i = cid & 0x1FFFFFFF
        if 0x01 <= i <= 0x0E:
            s["cells"][i - 1] = struct.unpack(">H", d[:2])[0] / 1000.0
        elif i in (0x13, 0x14, 0x15, 0x18):
            s[f"temp_{i:02x}"] = struct.unpack(">h", d[:2])[0] / 10.0
        elif i == 0x1A:
            s["pack_voltage"] = struct.unpack(">H", d[:2])[0] / 1000.0
        elif i == 0x39:
            s["current"] = struct.unpack(">i", d[:4])[0] / 1000.0
        elif i == 0xCB:
            soc, soh = struct.unpack(">2H", d[:4])
            s["soc"], s["soh"] = soc / 10.0, soh
        elif i == 0xAB:
            s["hw_event"] = struct.unpack(">I", d[:4])[0]
        elif i == 0x8B:
            s["sw_event"] = struct.unpack(">I", d[:4])[0]
        elif i == 0x94:
            s["state"] = struct.unpack(">H", d[:2])[0]
    return s


def test_the_bms_says_nothing_until_it_is_asked():
    bms, ctrl = make()
    for step in range(1, 501):
        bms.tick(step * 0.001, body())
    assert drain(ctrl) == [] and bms.stats()["sent"] == 0


def test_an_item_asked_for_is_answered_with_its_id():
    bms, ctrl = make(soc=0.5)
    ctrl.send(pack_frame(0x1A | EFF))  # pack voltage
    ctrl.send(pack_frame(0x01 | EFF))  # cell 1
    ctrl.send(pack_frame(0x09 | EFF))  # cell 9 of an 8S pack
    ctrl.send(pack_frame(0x5B | EFF))  # an AFE register: not modelled
    ctrl.send(pack_frame(0x1A))  # a standard frame: not this BMS's format
    bms.tick(0.001, body())
    # 3.6 V a cell at half charge, eight of them.
    assert drain(ctrl) == [(0x1A | EFF, bytes([0x70, 0x80])),  # 28800 mV, high byte first
                           (0x01 | EFF, bytes([0x0E, 0x10])),  # 3600 mV
                           (0x09 | EFF, bytes([0x00, 0x00]))]


def test_0x87_starts_the_cycle_and_0x86_stops_it():
    bms, ctrl = make(soc=0.8, period_ms=100.0, idle_w=40.0)
    ctrl.send(pack_frame(0x87 | EFF))  # as the controller sends it: extended, no data
    for step in range(1, 251):
        bms.tick(step * 0.001, body())
    frames = drain(ctrl)
    assert len(frames) == 3 * len(TABLE)  # at 1, 101 and 201 ms
    assert [cid & 0x1FFFFFFF for cid, _ in frames[:len(TABLE)]] == list(TABLE)
    s = controller_read(frames)
    assert s["cells"][:8] == [3.96] * 8 and s["cells"][8:] == [0.0] * 6
    assert s["pack_voltage"] == 31.68
    assert s["current"] == pytest.approx(-40.0 / 31.68, abs=2e-3)  # the idle draw, discharging
    assert s["soc"] == 80.0 and s["soh"] == 100
    assert s["state"] == 0b001 and s["hw_event"] == 0 and s["sw_event"] == 0
    assert s["temp_13"] == s["temp_18"] == 30.0
    ctrl.send(pack_frame(0x86 | EFF))
    for step in range(251, 501):
        bms.tick(step * 0.001, body())
    assert drain(ctrl) == [] and not bms.stats()["looping"]


def test_the_frames_bytes_of_current_and_soc():
    bms, ctrl = make(soc=0.8, idle_w=31.68 * 2.5)  # 2.5 A at 31.68 V
    bms.tick(0.0, body())
    bms.tick(0.001, body())
    ctrl.send(pack_frame(0x39 | EFF))
    ctrl.send(pack_frame(0xCB | EFF))
    ctrl.send(pack_frame(0x94 | EFF))
    bms.tick(0.002, body())
    (_, cur), (_, soc), (_, state) = drain(ctrl)
    assert cur == struct.pack(">i", -2500)  # mA, signed, discharge negative
    assert soc == bytes([0x03, 0x20, 0x00, 0x64, 0x00, 0x00, 0x00, 0x00])  # 80.0 %, SOH 100 %
    assert state == bytes([0x00, 0x01])  # bit 0: discharging


def test_the_charge_goes_down_with_what_the_robot_draws():
    bms, _ = make(soc=1.0, capacity_ah=1.0, idle_w=0.0)
    tau = np.zeros(12)
    tau[2] = 10.0  # a knee holding 10 Nm, still: only the winding's heat
    bms.tick(0.0, body())
    bms.tick(1.0, body(tau=tau))
    assert bms.power == pytest.approx(100.0 * HEAT_W_PER_NM2[2]) and 40.0 < bms.power < 50.0
    assert bms.current == pytest.approx(-bms.power / 33.6)
    assert bms.soc == pytest.approx(1.0 - (bms.power / 33.6) / 3600.0)
    # A joint braked returns its mechanical power: -10 Nm x 5 rad/s against the heat.
    bms.tick(2.0, body(tau=-tau, dq=np.where(tau > 0, 5.0, 0.0)))
    assert bms.power == pytest.approx(100.0 * HEAT_W_PER_NM2[2] - 50.0)
    empty, _ = make(soc=0.0, idle_w=100.0)
    empty.tick(0.0, body())
    empty.tick(10.0, body())
    assert empty.soc == 0.0  # empty stays empty; nothing trips


def test_the_config_is_checked():
    with pytest.raises(ValueError):
        make(soc=1.5)
    assert CAN_EFF_FLAG == EFF


def test_on_a_real_can_interface():
    """With a (v)can interface up (scripts/vcan_up.sh; SYNCAI_TEST_BMS_IF
    names another): the same exchange over SocketCAN."""
    name = os.environ.get("SYNCAI_TEST_BMS_IF", "can0")
    if not hasattr(socket, "AF_CAN") or not os.path.exists(f"/sys/class/net/{name}"):
        pytest.skip(f"no CAN interface {name}")
    bms = Bms(BmsConfig(interface=name), log=QUIET)
    ctrl = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    ctrl.bind((name,))
    ctrl.settimeout(1.0)
    try:
        ctrl.send(pack_frame(0x87 | EFF))
        import time
        time.sleep(0.05)
        bms.tick(0.001, body())
        frames = [unpack_frame(ctrl.recv(FRAME_SIZE)) for _ in TABLE]
    finally:
        ctrl.close()
        bms.close()
    s = controller_read(frames)
    assert s["pack_voltage"] == 31.68 and s["soc"] == 80.0
