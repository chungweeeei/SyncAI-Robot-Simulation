"""locomotion/hardware_link.py: the datagrams byte for byte, the sticky
command and its watchdog, the timestamp, and the stand-in controller
driving the stand-in body through the link; driver/udp_driver.py's lines."""

import socket
import struct

import mujoco
import numpy as np

from syncai_sim.actuator.motors import DAMPING_KD, JointCommand
from syncai_sim.driver.udp_driver import UdpDriver
from syncai_sim.entity.robot import Q_DEFAULT
from syncai_sim.locomotion.hardware_link import (CMD_FMT, CMD_SIZE, STATE_FMT, STATE_SIZE,
                                                 HardwareLink, HardwareLinkConfig,
                                                 StandInController, pack_command,
                                                 unpack_command)
from syncai_sim.sensor.proprioception import N_FIELDS, SLICES

from tests.test_sensors import make as make_body
from tests.test_sensors import settle

QUIET = lambda *a: None  # noqa: E731


def peer():
    """A socket on an ephemeral port: the controller's end for the states."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(1.0)
    return s


def link(state_port: int, epoch=None, **kw) -> HardwareLink:
    cfg = HardwareLinkConfig(command_port=0, state_port=state_port, **kw)  # port 0: any free one
    return HardwareLink(cfg, log=QUIET, epoch=epoch)


def test_the_two_datagrams_are_228_and_240_bytes():
    assert STATE_SIZE == 8 + 4 * N_FIELDS == 228
    assert CMD_SIZE == 240
    assert struct.calcsize(STATE_FMT) == STATE_SIZE and struct.calcsize(CMD_FMT) == CMD_SIZE


def test_the_state_datagram_is_the_sensor_record_behind_a_timestamp():
    model, data, sensors, motors = make_body()
    settle(model, data, motors)
    sensors.read(3.25)
    rx = peer()
    lk = link(rx.getsockname()[1])
    lk.send(sensors)
    pkt, _ = rx.recvfrom(1024)
    assert len(pkt) == STATE_SIZE
    values = struct.unpack(STATE_FMT, pkt)
    assert values[0] == 3.25
    f = np.array(values[1:], dtype=np.float32)
    assert np.array_equal(f, sensors.f)
    # The fields by name, where the spec says they are.
    assert np.array_equal(f[SLICES["q"]], sensors.q) and np.array_equal(f[SLICES["contact"]], sensors.contact)
    assert np.all(f[SLICES["contact"]] == 1.0)
    assert lk.stats()["states"] == 1
    lk.close()
    rx.close()


def test_commands_latest_wins_and_short_ones_are_dropped():
    rx = peer()
    lk = link(rx.getsockname()[1])
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    first = JointCommand(np.full(12, 0.1), np.zeros(12), np.full(12, 10.0), np.ones(12), np.zeros(12))
    second = JointCommand(np.full(12, 0.2), np.full(12, 0.5), np.zeros(12), np.full(12, 3.0),
                          np.full(12, 4.0))
    dest = ("127.0.0.1", lk.command_port)
    tx.sendto(pack_command(first), dest)
    tx.sendto(b"\x00" * 100, dest)  # too short: dropped
    tx.sendto(pack_command(second), dest)
    _wait(lambda: _ready(lk.rx))
    cmd = lk.poll(0.0)
    assert np.allclose(cmd.q_des, 0.2) and np.allclose(cmd.dq_des, 0.5) and np.allclose(cmd.kp, 0.0)
    assert np.allclose(cmd.kd, 3.0) and np.allclose(cmd.tau_ff, 4.0)
    s = lk.stats()
    assert s["commands"] == 1 and s["short"] == 1  # one poll, the newest datagram, one dropped
    assert lk.poll(0.01) is cmd  # sticky between datagrams
    round_trip = unpack_command(pack_command(second))
    for name in ("q_des", "dq_des", "kp", "kd", "tau_ff"):
        assert np.allclose(getattr(round_trip, name), getattr(second, name))
    lk.close()
    tx.close()
    rx.close()


def test_without_a_command_the_motors_get_damping_and_the_watchdog_trips_once():
    rx = peer()
    lk = link(rx.getsockname()[1], watchdog_s=0.1)
    d = lk.poll(0.0)  # nothing ever received: damping, not a trip
    assert np.all(d.kp == 0) and np.allclose(d.kd, DAMPING_KD) and lk.stats()["watchdog_trips"] == 0
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.sendto(pack_command(JointCommand.pd(Q_DEFAULT, np.full(12, 70.0), np.full(12, 2.0))),
              ("127.0.0.1", lk.command_port))
    _wait(lambda: _ready(lk.rx))
    assert np.allclose(lk.poll(1.0).kp, 70.0)
    assert np.allclose(lk.poll(1.05).kp, 70.0)  # held within the watchdog
    late = lk.poll(1.2)
    assert np.all(late.kp == 0) and lk.stats()["watchdog_trips"] == 1
    assert np.all(lk.poll(1.3).kp == 0) and lk.stats()["watchdog_trips"] == 1  # once per trip
    tx.sendto(pack_command(JointCommand.pd(Q_DEFAULT, np.full(12, 70.0), np.full(12, 2.0))),
              ("127.0.0.1", lk.command_port))
    _wait(lambda: _ready(lk.rx))
    assert np.allclose(lk.poll(1.4).kp, 70.0)  # back
    lk.reset()
    assert np.all(lk.poll(1.41).kp == 0) and lk.stats()["watchdog_trips"] == 1  # stale after a teleport
    lk.close()
    tx.close()
    rx.close()


def test_timestamps_never_go_backwards_across_a_restart():
    rx = peer()
    first = link(rx.getsockname()[1], epoch=1000.0)
    t_last = first.time(12.5)  # a sim that ran 12.5 s
    first.close()
    second = link(rx.getsockname()[1])  # restarted: the epoch is the wall clock
    assert second.time(0.0) > t_last
    assert second.time(0.001) > second.time(0.0)
    second.close()
    rx.close()


def test_the_stand_in_controller_stands_the_body_through_the_link():
    model, data, sensors, motors = make_body()
    rx_port = peer()  # only to pick a free state port
    state_port = rx_port.getsockname()[1]
    rx_port.close()
    lk = link(state_port)
    ctrl = StandInController(state_port, lk.command_port)
    ctrl.start()
    try:
        q0 = motors.q()
        motors.set_joints(q0 + 0.3)  # knocked off the pose; the controller's PD should pull it back
        for i in range(int(1.0 / model.opt.timestep)):
            motors.apply(lk.poll(i * model.opt.timestep))
            mujoco.mj_step(model, data)
            sensors.read(lk.time((i + 1) * model.opt.timestep))
            lk.send(sensors)
        _wait(lambda: ctrl.received > 0)
        final = lk.poll(1.0)  # the controller's command is what the motors hold
    finally:
        ctrl.stop()
        lk.close()
    assert lk.stats()["commands"] > 0 and ctrl.received > 100
    assert ctrl.last_t > lk.epoch and ctrl.last_state.shape == (N_FIELDS,)
    assert np.abs(motors.q() - Q_DEFAULT).max() < 0.1  # pulled to the default pose
    assert np.allclose(final.kp, 70.0) and np.allclose(final.q_des, Q_DEFAULT)


def test_udp_driver_speaks_the_controllers_lines():
    rx = peer()
    drv = UdpDriver(command_port=rx.getsockname()[1], log=QUIET)
    assert drv.set_motion_key("0") == (True, "Motion key sent")
    assert drv.set_motion_key("5")[0] and drv.set_motion_key("4") == (True, "Emergency stop sent")
    assert drv.set_motion_key("9") == (False, "Unknown motion key '9'")
    assert drv.set_policy_mode(1) == (True, "Policy updated")
    drv.send_axes(0.5, 0.0, -0.25)
    drv.send("RESET_SAFETY")
    lines = [rx.recvfrom(256)[0].decode() for _ in range(6)]
    assert lines == ["MODE Z\n", "MODE M\n", "ESTOP\n", "MODE 1\n", "AXES 0.500000 0.000000 -0.250000\n",
                     "RESET_SAFETY\n"]
    assert drv.motion_state == 8 and drv.sent == 6  # UNKNOWN: nothing here reads the telemetry
    drv.close()
    rx.close()


def _ready(sock) -> bool:
    import select
    return bool(select.select([sock], [], [], 0.0)[0])


def _wait(cond, timeout=2.0):
    import time
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, "timed out"
        time.sleep(0.005)
