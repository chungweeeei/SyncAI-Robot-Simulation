"""The sim's side of driver_manager's UDP link, against the layout protocol.rs
parses (the workspace's src/syncai_driver_manager/src/driver_manager_node/protocol.rs)."""

import socket

import numpy as np
import pytest

from syncai_sim.controller_link import ControllerLink, LinkConfig
from syncai_sim.fake_driver import ESTOP, LIE_DOWN, LOCOMOTION, FakeDriver


class FakeSim:
    def __init__(self):
        self.cmd = None

    def set_cmd(self, vx, vy, wz):
        self.cmd = (vx, vy, wz)

    def controller_telemetry(self):
        return {"rpy": (0.01, -0.02, 1.5), "acc": np.array([0.0, 0.0, 9.81]),
                "omega": np.zeros(3), "JOINT_POS": np.arange(12) * 0.1,
                "JOINT_VEL": np.zeros(12), "JOINT_TAU": np.ones(12)}


@pytest.fixture
def link():
    driver = FakeDriver(switch_delay=0.0, motion_state=LOCOMOTION, policy=0)
    lk = ControllerLink(FakeSim(), driver, LinkConfig(command_port=0, telemetry_port=1),
                        log=lambda *a: None)
    yield lk
    lk.close()


def sections(text):
    """keyword -> values, the way protocol.rs splits a datagram."""
    out, key = {}, None
    for tok in text.split():
        if tok.isalpha() or "_" in tok:
            key = tok
            out[key] = []
        else:
            out[key].append(float(tok))
    return out


def test_commands_map_to_the_controller(link):
    link.handle("AXES 0.560000 0.000000 -0.700000")
    assert link.sim.cmd == pytest.approx((0.56, 0.0, -0.7))
    link.handle("MODE X")  # motion key 2
    link.driver.update()
    assert link.driver.motion_state == LIE_DOWN
    link.handle("MODE 1")  # a number: the policy
    link.driver.update()
    assert link.driver.policy == 1
    link.handle("ESTOP")
    link.driver.update()
    assert link.driver.motion_state == ESTOP
    link.handle("AXES nan 0 0")  # rejected, the last command stands
    assert link.sim.cmd == pytest.approx((0.56, 0.0, -0.7))


def test_sections_have_every_field_protocol_rs_reads(link):
    s = sections(link.sections())
    assert [len(s[k]) for k in ("IMU_RPY", "ACC", "OMEGA")] == [3, 3, 3]
    for k in ("JOINT_POS", "JOINT_VEL", "JOINT_TAU", "JOINT_TEMP", "JOINT_ERR"):
        assert len(s[k]) == 12, k
    assert s["JOINT_POS"][3] == pytest.approx(0.3)
    assert s["MODE_STATE"] == [0, LOCOMOTION]


def test_bms_layout_and_drain(link):
    vals = link.battery.datagram().split()
    assert vals[0] == "BMS_V2" and len(vals) == 1 + 18  # BMS_FULL_VALUES
    v, a, soc = float(vals[1]), float(vals[2]), float(vals[3])
    assert soc == pytest.approx(90.0) and a < -0.1  # DISCHARGING by the current's sign
    assert v == pytest.approx(sum(float(c) for c in vals[11:19]), rel=1e-4)
    link.battery.drain_per_s = 1.0
    link.battery.advance(5.0)
    assert link.battery.soc == pytest.approx(85.0)


def test_commands_arrive_over_udp(link):
    port = link.rx.getsockname()[1]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(b"AXES 0.1 0.0 0.2\n", ("127.0.0.1", port))
    for _ in range(100):
        link.poll()
        if link.sim.cmd is not None:
            break
    assert link.sim.cmd == pytest.approx((0.1, 0.0, 0.2))
