"""locomotion/can_link.py: the J60's frames byte for byte, the robot's wiring
(which joint is which motor on which bus, its sign and its homed zero), and
what a motor does when it is disabled, commanded, timed out, delayed or
dropped. The buses are AF_UNIX datagram pairs — the same send / recv of
16-byte frames a raw CAN socket has — so this runs without a CAN interface;
the last test uses real ones and skips without them."""

import os
import socket
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from syncai_sim.actuator.motors import DAMPING_KD
from syncai_sim.entity.robot import Q_DEFAULT
from syncai_sim.locomotion.can_link import (BUS_JOINT, CONTROL_MOTOR, DISABLE_MOTOR, ENABLE_MOTOR,
                                            FRAME_SIZE, GET_CONFIG, GET_STATUS_WORD, JOINT_HOME,
                                            JOINT_SIGN, SET_CAN_TIMEOUT, SET_HOME, CanLink,
                                            CanLinkConfig, StandInCanController, can_id,
                                            float_to_uint, open_bus, pack_control, pack_feedback,
                                            pack_frame, unpack_control, unpack_feedback,
                                            unpack_frame)

from tests.test_sensors import make as make_body

QUIET = lambda *a: None  # noqa: E731
IDS = (2, 3, 4)


def buses():
    """Four buses: (the sim's ends, the controller's ends)."""
    pairs = [socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM) for _ in range(4)]
    for _, ctrl in pairs:
        ctrl.settimeout(1.0)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def body(q=Q_DEFAULT, dq=None, tau=None):
    """The joints as a Sensors record has them."""
    return SimpleNamespace(q=np.asarray(q, float), dq=np.zeros(12) if dq is None else dq,
                           tau=np.zeros(12) if tau is None else tau)


def make(**kw):
    sim, ctrl = buses()
    return CanLink(CanLinkConfig(**kw), log=QUIET, socks=sim), ctrl


def control(i, q_des, kp=70.0, kd=2.0, dq_des=0.0, tau_ff=0.0) -> bytes:
    """The CONTROL frame a controller sends for joint i's target."""
    s = JOINT_SIGN[i]
    slot = BUS_JOINT[[i in js for js in BUS_JOINT].index(True)].index(i)
    return pack_frame(can_id(CONTROL_MOTOR, IDS[slot]),
                      pack_control(s * (q_des - JOINT_HOME[i]), s * dq_des, kp, kd, s * tau_ff))


def bus_of(i) -> int:
    return [i in js for js in BUS_JOINT].index(True)


def enable_all(ctrl):
    for sock in ctrl:
        for mid in IDS:
            sock.send(pack_frame(can_id(ENABLE_MOTOR, mid)))


def command_all(ctrl, q_des=Q_DEFAULT):
    for i in range(12):
        ctrl[bus_of(i)].send(control(i, q_des[i]))


def drain(sock) -> list[tuple[int, bytes]]:
    out = []
    sock.setblocking(False)
    while True:
        try:
            out.append(unpack_frame(sock.recv(FRAME_SIZE)))
        except BlockingIOError:
            return out


# -- the frames -------------------------------------------------------------------
def test_a_frame_is_struct_can_frame_and_the_reply_sets_bit_4():
    f = pack_frame(can_id(CONTROL_MOTOR, 3), bytes(range(8)))
    assert len(f) == FRAME_SIZE == 16
    assert f[:4] == (0x83).to_bytes(4, "little") and f[4] == 8 and f[8:] == bytes(range(8))
    assert unpack_frame(f) == (0x83, bytes(range(8)))
    assert unpack_frame(pack_frame(can_id(ENABLE_MOTOR, 2))) == (0x42, b"")
    # What the robot's bus check expects to see commanded, and their answers.
    assert [can_id(CONTROL_MOTOR, m) for m in IDS] == [0x82, 0x83, 0x84]
    assert [can_id(CONTROL_MOTOR, m, reply=True) for m in IDS] == [0x92, 0x93, 0x94]


def test_the_control_command_packs_as_the_controller_packs_it():
    # p 1.0 -> 33586 (16 bit), v -2.0 -> 7781 (14 bit), kp 70 (10 bit), kd 2.0 -> 10 (8 bit),
    # tq 5.0 -> 36863 (16 bit), laid out as robot_can_control.rs send_raw_cmd lays them.
    d = pack_control(1.0, -2.0, 70.0, 2.0, 5.0)
    assert d == bytes([0x32, 0x83, 0x65, 0x9E, 0x11, 0x0A, 0xFF, 0x8F])
    p, v, kp, kd, tq = unpack_control(d)
    assert abs(p - 1.0) < 80 / 65535 and abs(v + 2.0) < 80 / 16383 and abs(tq - 5.0) < 80 / 65535
    assert kp == 70.0 and kd == 2.0
    assert unpack_control(pack_control(99.0, 99.0, 5000.0, 99.0, -99.0)) == (40.0, 40.0, 1023.0, 51.0, -40.0)


def test_zero_does_not_survive_the_quantisation():
    """The ranges have an even number of steps and the packing truncates."""
    assert float_to_uint(0.0, -40.0, 40.0, 14) == 8191
    _, v, _, _, tq = unpack_control(pack_control(0.0, 0.0, 0.0, 0.0, 0.0))
    assert -0.0025 < v < -0.0024 and -0.00062 < tq < -0.00060


def test_the_feedback_packs_as_the_controller_reads_it():
    d = pack_feedback(2.845, -1.5, 3.25, 30.0)
    raw = int.from_bytes(d, "little")
    assert raw & 0xFFFFF == float_to_uint(2.845, -40, 40, 20)
    assert (raw >> 20) & 0xFFFFF == float_to_uint(-1.5, -40, 40, 20)
    assert (raw >> 40) & 0xFFFF == float_to_uint(3.25, -40, 40, 16)
    assert (raw >> 56) & 1 == 1 and (raw >> 57) & 0x7F == float_to_uint(30.0, -20, 200, 7)
    p, v, tq, temp = unpack_feedback(d)
    assert abs(p - 2.845) < 80 / 2**20 and abs(v + 1.5) < 80 / 2**20 and abs(tq - 3.25) < 80 / 65535
    assert abs(temp - 30.0) < 220 / 127


# -- the wiring ---------------------------------------------------------------------
def test_the_wiring_is_the_robots():
    # can1 HR, can2 FR, can3 FL, can4 HL, in the sim's joint order FL FR HL HR.
    assert BUS_JOINT == [[9, 10, 11], [3, 4, 5], [0, 1, 2], [6, 7, 8]]
    assert JOINT_SIGN.tolist() == [1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1]
    # Homed, the leg is folded back: every motor reads 0 there.
    assert JOINT_HOME.reshape(4, 3)[:, 1:].tolist() == [[-3.495, 2.745]] * 4
    assert JOINT_HOME[::3].tolist() == [-0.45, 0.45, -0.45, 0.45]


def test_a_frame_on_can1_id_3_is_the_hind_right_hip_y_and_nothing_else():
    lk, ctrl = make()
    enable_all(ctrl)
    ctrl[0].send(control(10, q_des=-0.9, kp=50.0, kd=1.0, dq_des=0.5, tau_ff=2.0))
    cmd = lk.poll(0.0, body())
    assert abs(cmd.q_des[10] + 0.9) < 80 / 65535 and cmd.kp[10] == 50.0 and cmd.kd[10] == 1.0
    assert abs(cmd.dq_des[10] - 0.5) < 80 / 16383 and abs(cmd.tau_ff[10] - 2.0) < 80 / 65535
    others = np.arange(12) != 10
    assert np.all(cmd.kp[others] == 0) and np.all(cmd.kd[others] == 0) and np.all(cmd.tau_ff[others] == 0)
    assert lk.stats()["commands"] == [1, 0, 0, 0] and lk.stats()["enabled"] == 12


def test_the_reply_is_the_joint_in_the_motors_frame():
    lk, ctrl = make()
    enable_all(ctrl)
    q = Q_DEFAULT.astype(float) + 0.1
    dq = np.linspace(-1, 1, 12)
    tau = np.linspace(-5, 5, 12)
    command_all(ctrl)
    lk.poll(0.0, body(q, dq, tau))
    for b in range(4):
        replies = [(cid, d) for cid, d in drain(ctrl[b]) if cid >> 5 == CONTROL_MOTOR]
        assert [cid for cid, _ in replies] == [0x92, 0x93, 0x94]
        for (_, d), i in zip(replies, BUS_JOINT[b]):
            p, v, tq, _ = unpack_feedback(d)
            # What the controller does with it: back into the joint's frame.
            assert abs(JOINT_SIGN[i] * p + JOINT_HOME[i] - q[i]) < 80 / 2**20
            assert abs(JOINT_SIGN[i] * v - dq[i]) < 80 / 2**20
            assert abs(JOINT_SIGN[i] * tq - tau[i]) < 80 / 65535
    assert lk.stats()["replies"] == 24  # 12 enables, 12 controls


# -- what a motor does ----------------------------------------------------------------
def test_a_motor_gives_no_torque_until_it_is_enabled():
    lk, ctrl = make()
    command_all(ctrl)
    cmd = lk.poll(0.0, body())
    assert not np.any(cmd.kp) and not np.any(cmd.kd) and not np.any(cmd.tau_ff)
    assert sum(len(drain(s)) for s in ctrl) == 12  # it still reports
    enable_all(ctrl)
    command_all(ctrl)
    cmd = lk.poll(0.001, body())
    assert np.all(cmd.kp == 70.0) and np.abs(cmd.q_des - Q_DEFAULT).max() < 80 / 65535
    ctrl[2].send(pack_frame(can_id(DISABLE_MOTOR, 4)))  # FL knee
    cmd = lk.poll(0.002, body())
    assert cmd.kp[2] == 0 and cmd.kd[2] == 0 and np.all(cmd.kp[np.arange(12) != 2] == 70.0)


def test_a_silent_bus_damps_its_leg_and_the_others_go_on():
    lk, ctrl = make(timeout_ms=100.0)
    enable_all(ctrl)
    command_all(ctrl)
    lk.poll(0.0, body())
    for step in range(1, 201):  # can2 (FR) goes silent
        for i in range(12):
            if bus_of(i) != 1:
                ctrl[bus_of(i)].send(control(i, Q_DEFAULT[i]))
        cmd = lk.poll(step * 0.001, body())
        if step == 100:
            assert np.all(cmd.kp == 70.0)  # not yet: exactly the timeout
    fr = np.isin(np.arange(12), BUS_JOINT[1])
    assert np.all(cmd.kp[fr] == 0) and np.all(cmd.kd[fr] == DAMPING_KD) and np.all(cmd.q_des[fr] == 0)
    assert np.all(cmd.kp[~fr] == 70.0) and lk.stats()["timeouts"] == 3
    command_all(ctrl)
    assert np.all(lk.poll(0.201, body()).kp == 70.0)  # commanded again


def test_set_can_timeout_and_get_config():
    lk, ctrl = make(timeout_ms=100.0)
    enable_all(ctrl)
    for sock in ctrl:
        for mid in IDS:
            sock.send(pack_frame(can_id(SET_CAN_TIMEOUT, mid), bytes([200])))
    command_all(ctrl)
    lk.poll(0.0, body())
    assert np.all(lk.poll(0.150, body()).kp == 70.0)  # past 100 ms, inside 200
    assert np.all(lk.poll(0.250, body()).kp == 0)
    for s in ctrl:
        drain(s)
    ctrl[3].send(pack_frame(can_id(GET_CONFIG, 2)))
    ctrl[3].send(pack_frame(can_id(GET_STATUS_WORD, 2)))
    lk.poll(0.251, body())
    (cid, cfg), (sid, status) = drain(ctrl[3])
    assert cid == can_id(GET_CONFIG, 2, reply=True) and len(cfg) == 8 and cfg[3] == 200
    assert sid == can_id(GET_STATUS_WORD, 2, reply=True) and status == b"\0\0"


def test_set_home_moves_the_motors_zero():
    lk, ctrl = make()
    enable_all(ctrl)
    q = Q_DEFAULT.astype(float)
    ctrl[2].send(pack_frame(can_id(SET_HOME, 3)))  # FL hip y, standing
    lk.poll(0.0, body(q))
    drain(ctrl[2])
    ctrl[2].send(control(1, q[1]))
    cmd = lk.poll(0.001, body(q))
    p = unpack_feedback(drain(ctrl[2])[0][1])[0]
    assert abs(p) < 80 / 2**20  # the motor reads 0 where it was homed
    # A controller that assumes the folded pose is now off by the difference.
    assert abs(cmd.q_des[1] - (q[1] + (q[1] - JOINT_HOME[1]))) < 80 / 65535


def test_a_delayed_bus_acts_late_and_answers_at_once():
    lk, ctrl = make(delay_ms=(0.0, 0.0, 3.0, 0.0))
    enable_all(ctrl)
    command_all(ctrl)
    cmd = lk.poll(0.0, body())
    fl = np.isin(np.arange(12), BUS_JOINT[2])
    assert np.all(cmd.kp[fl] == 0) and np.all(cmd.kp[~fl] == 70.0)
    assert len([1 for cid, _ in drain(ctrl[2]) if cid >> 5 == CONTROL_MOTOR]) == 3
    assert np.all(lk.poll(0.002, body()).kp[fl] == 0)
    assert np.all(lk.poll(0.003, body()).kp == 70.0)


def test_a_dropped_frame_is_neither_applied_nor_answered():
    lk, ctrl = make(drop_rate=(0.0, 1.0, 0.0, 0.0))
    enable_all(ctrl)
    command_all(ctrl)
    cmd = lk.poll(0.0, body())
    fr = np.isin(np.arange(12), BUS_JOINT[1])
    assert np.all(cmd.kp[fr] == 0) and np.all(cmd.kp[~fr] == 70.0)
    assert drain(ctrl[1]) == [] and lk.stats()["dropped"] == [0, 6, 0, 0]


def test_a_teleport_drops_the_commands_and_keeps_the_motors_enabled():
    lk, ctrl = make()
    enable_all(ctrl)
    command_all(ctrl)
    lk.poll(0.0, body())
    lk.reset()
    cmd = lk.poll(0.001, body())
    assert not np.any(cmd.kp) and not np.any(cmd.kd) and lk.stats()["enabled"] == 12
    command_all(ctrl)
    assert np.all(lk.poll(0.002, body()).kp == 70.0)


def test_frames_that_are_not_a_motors_are_ignored():
    lk, ctrl = make()
    ctrl[0].send(pack_frame(can_id(ENABLE_MOTOR, 7)))  # no such motor
    ctrl[0].send(pack_frame(can_id(CONTROL_MOTOR, 2, reply=True), bytes(8)))  # another motor's answer
    ctrl[0].send(pack_frame(can_id(ENABLE_MOTOR, 2) | 0x80000000))  # an extended id
    ctrl[0].send(pack_frame(can_id(CONTROL_MOTOR, 2), b"\1\2"))  # too short to be a command
    lk.poll(0.0, body())
    assert lk.stats()["enabled"] == 0 and lk.stats()["commands"] == [0, 0, 0, 0]
    assert drain(ctrl[0]) == []


def test_the_shape_of_the_link_is_checked():
    with pytest.raises(ValueError):
        CanLink(CanLinkConfig(interfaces=("can1", "can2")), log=QUIET, socks=[])
    with pytest.raises(ValueError):
        CanLink(CanLinkConfig(delay_ms=(1.0,)), log=QUIET, socks=[])


# -- end to end -------------------------------------------------------------------------
def _wait(cond, timeout=3.0):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < timeout:
        time.sleep(0.002)


def test_the_stand_in_controller_stands_the_body_through_the_buses():
    model, data, sensors, motors = make_body()
    sim, ctrl_socks = buses()
    # No CAN timeout: the physics below runs far faster than the stand-in's
    # wall-clock 1 kHz, and a sim-time timeout would damp the joints between
    # two of its frames.
    lk = CanLink(CanLinkConfig(timeout_ms=0.0), log=QUIET, socks=sim)
    ctrl = StandInCanController(socks=ctrl_socks)
    q0 = motors.q()
    motors.set_joints(q0 + 0.3)  # knocked off the pose; the controller's PD should pull it back
    sensors.read(0.0)
    ctrl.start()
    try:
        _wait(lambda: (lk.poll(0.0, sensors), sum(lk.stats()["commands"]) >= 12)[1])
        cmd = lk.poll(0.0, sensors)
        assert np.all(cmd.kp == 70.0) and np.abs(cmd.q_des - Q_DEFAULT).max() < 80 / 65535
        for step in range(int(1.0 / model.opt.timestep)):
            motors.apply(lk.poll(step * model.opt.timestep, sensors))
            mujoco.mj_step(model, data)
            sensors.read((step + 1) * model.opt.timestep)
        before = ctrl.received
        _wait(lambda: (lk.poll(1.0, sensors), ctrl.received > before + 48)[1])
    finally:
        ctrl.stop()
        lk.close()
    assert lk.stats()["enabled"] == 12 and lk.stats()["timeouts"] == 0
    assert np.abs(motors.q() - Q_DEFAULT).max() < 0.1  # pulled to the default pose
    assert np.abs(ctrl.q - motors.q()).max() < 5e-3  # and the controller read it back


def test_on_real_can_interfaces():
    """The same exchange on SocketCAN (vcan). Skips where the interfaces are
    not there: scripts/vcan_up.sh makes them, SYNCAI_TEST_CAN_IFS names
    others. Not while a sim with joint_link can runs on them: it answers too."""
    names = tuple(os.environ.get("SYNCAI_TEST_CAN_IFS", "can1 can2 can3 can4").split())
    try:
        ctrl = [open_bus(n) for n in names]
    except OSError as err:
        pytest.skip(f"no CAN interfaces {names}: {err}")
    lk = CanLink(CanLinkConfig(interfaces=names), log=QUIET)
    try:
        enable_all(ctrl)
        command_all(ctrl)
        _wait(lambda: (lk.poll(0.0, body()), sum(lk.stats()["commands"]) >= 12)[1])
        assert np.all(lk.poll(0.0, body()).kp == 70.0)
        _wait(lambda: False, timeout=0.05)
        for b in range(4):
            ids = sorted(cid for cid, _ in drain(ctrl[b]) if cid >> 5 == CONTROL_MOTOR)
            assert ids == [0x92, 0x93, 0x94]  # the answers, and not our own frames back
    finally:
        lk.close()
        for s in ctrl:
            s.close()
