"""legged.py's motor side on a stand-in twelve-joint body (no URDF, no ONNX):
the clamps of the two motor models, the two actuators, the command delay
and the torque filter."""

import math

import mujoco
import numpy as np
import pytest

from syncai_sim.fake_driver import DAMPING, ESTOP, LIE_DOWN, LOCOMOTION, MPC
from syncai_sim.legged import (JOINTS, LEGS, STAND_KD, STAND_KP, TORQUE_LIMIT, LeggedController,
                               j60_torque_limits)
from syncai_sim.policy import HimlocoRunner, MultiPolicy, PpoRunner
from syncai_sim.robot import J60_MAX_SPEED, J60_PEAK_TORQUE

from tests.test_policy import Recorder

STAND = next(v for v in range(20) if v not in (LOCOMOTION, MPC, LIE_DOWN, DAMPING, ESTOP))


def body_xml(actuators: bool, timestep: float = 0.002) -> str:
    legs = []
    for leg in LEGS:
        legs.append(f"""
      <body name="{leg}_HIP" pos="0.1 0.05 0">
        <joint name="{leg}_HipX_joint" axis="1 0 0"/>
        <geom type="capsule" size="0.02 0.03" mass="0.5"/>
        <body name="{leg}_THIGH" pos="0 0.05 0">
          <joint name="{leg}_HipY_joint" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.8"/>
          <body name="{leg}_SHANK" pos="0 0 -0.2">
            <joint name="{leg}_Knee_joint" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.015" mass="0.15"/>
          </body>
        </body>
      </body>""")
    acts = "".join(f'<position name="act_{leg}_{j}_joint" joint="{leg}_{j}_joint" kp="0" kv="0" '
                   f'forcerange="-100 100"/>' for leg in LEGS for j in JOINTS)
    return f"""
<mujoco>
  <option timestep="{timestep}" gravity="0 0 0"/>
  <worldbody>
    <body name="base_link" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.2 0.1 0.05" mass="5"/>{''.join(legs)}
    </body>
  </worldbody>
  {'<actuator>' + acts + '</actuator>' if actuators else ''}
</mujoco>"""


def make(actuators=False, **kw):
    """The body (with MuJoCo position actuators when `actuators`) and a
    controller driving them (`actuator` position then, qfrc otherwise)."""
    model = mujoco.MjModel.from_xml_string(body_xml(actuators))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    policies = MultiPolicy({0: PpoRunner(Recorder(np.zeros(12))),
                            1: HimlocoRunner(Recorder(np.zeros(12)))}, log=lambda *a: None)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    kw.setdefault("actuator", "position" if actuators else "qfrc")
    legs = LeggedController(model, data, base, policies, log=lambda *a: None, **kw)
    return model, data, legs


def test_j60_envelope():
    peak, wmax = np.array(J60_PEAK_TORQUE), np.array(J60_MAX_SPEED)
    lo, hi = j60_torque_limits(np.zeros(3), peak, wmax)
    assert np.allclose(hi, peak) and np.allclose(lo, -peak)
    lo, hi = j60_torque_limits(wmax, peak, wmax)  # at top speed: no more drive, full brake
    assert np.allclose(hi, 0.0) and np.allclose(lo, -peak)
    lo, hi = j60_torque_limits(-wmax / 2, peak, wmax)
    assert np.allclose(hi, peak) and np.allclose(lo, -peak / 2)


def test_qfrc_ideal_clamps_at_the_urdf_effort():
    model, data, legs = make(transition_s=1e-4)  # the ease is over by the second step
    legs.reset()
    legs.set_joints(legs.q() + 1.0)  # a radian off the stand pose: kp 70 wants 70 Nm
    legs.step(np.zeros(3), 0, STAND)
    legs.step(np.zeros(3), 0, STAND)  # the ease starts where the joints are; at the goal next step
    tau = data.qfrc_applied[legs.vadr]
    assert np.allclose(np.abs(tau), TORQUE_LIMIT)
    assert np.allclose(legs.tau(), tau)
    assert np.all(data.ctrl == 0)


def test_qfrc_j60_envelope_shrinks_with_joint_speed():
    model, data, legs = make(motor_model="j60", transition_s=1e-4)
    legs.reset()
    legs.set_joints(legs.q() + 1.0)
    data.qvel[legs.vadr] = 0.5 * np.array(J60_MAX_SPEED * 4)  # half of top speed, driving away
    legs.step(np.zeros(3), 0, STAND)
    legs.step(np.zeros(3), 0, STAND)
    tau = data.qfrc_applied[legs.vadr]
    # The error is positive (target below q): the PD pulls back, against the
    # motion, so the clamp is the full peak.
    assert np.allclose(np.abs(tau), np.array(J60_PEAK_TORQUE * 4))
    data.qvel[legs.vadr] = -0.5 * np.array(J60_MAX_SPEED * 4)  # now moving with the torque
    legs.step(np.zeros(3), 0, STAND)
    assert np.allclose(np.abs(data.qfrc_applied[legs.vadr]), 0.5 * np.array(J60_PEAK_TORQUE * 4))


def test_position_actuators_get_the_gains_the_target_and_the_range():
    model, data, legs = make(actuators=True, transition_s=1e-4)
    legs.reset()
    assert np.allclose(data.ctrl[legs.act], legs.q())
    legs.set_joints(legs.q() + 0.1)
    legs.step(np.zeros(3), 0, STAND)
    legs.step(np.zeros(3), 0, STAND)
    assert np.all(data.qfrc_applied[legs.vadr] == 0.0)
    assert np.allclose(model.actuator_gainprm[legs.act, 0], STAND_KP)
    assert np.allclose(model.actuator_biasprm[legs.act, 1], -STAND_KP)
    assert np.allclose(model.actuator_biasprm[legs.act, 2], -STAND_KD)
    assert np.allclose(model.actuator_forcerange[legs.act, 0], -TORQUE_LIMIT)
    assert np.allclose(model.actuator_forcerange[legs.act, 1], TORQUE_LIMIT)
    assert np.allclose(data.ctrl[legs.act], legs.target)
    mujoco.mj_forward(model, data)  # MuJoCo computes the force: kp (q* - q), 7 Nm, inside the range
    assert np.allclose(legs.tau(), -STAND_KP * 0.1, atol=1e-6)
    legs.step(np.zeros(3), 0, DAMPING)
    assert np.allclose(model.actuator_gainprm[legs.act, 0], 0.0)
    assert np.allclose(model.actuator_biasprm[legs.act, 2], -2.0)


def test_position_with_j60_writes_the_envelope_every_step():
    model, data, legs = make(actuators=True, motor_model="j60")
    legs.reset()
    data.qvel[legs.vadr] = np.array(J60_MAX_SPEED * 4)
    legs.step(np.zeros(3), 0, STAND)
    assert np.allclose(model.actuator_forcerange[legs.act, 1], 0.0)
    assert np.allclose(model.actuator_forcerange[legs.act, 0], -np.array(J60_PEAK_TORQUE * 4))


def test_command_delay_holds_the_old_target():
    def torques(delay_s):
        model, data, legs = make(cmd_delay_s=delay_s, transition_s=1e-4)
        legs.reset()
        out = []
        for _ in range(8):
            legs.step(np.zeros(3), 0, LIE_DOWN)  # the target jumps to the lying pose at step 2
            out.append(float(np.abs(data.qfrc_applied[legs.vadr]).max()))
        return out
    prompt = torques(0.0)
    assert prompt[0] == 0.0 and prompt[1] > 0.0
    late = torques(3 * 0.002)
    assert late[:4] == [0.0] * 4 and late[4] > 0.0
    assert math.isclose(late[4], prompt[1])


def test_torque_filter_is_first_order():
    model, data, legs = make(torque_bw_hz=20.0, transition_s=1e-4)
    legs.reset()
    legs.set_joints(legs.q() + 0.05)  # 3.5 Nm wanted: well inside the clamp
    legs.step(np.zeros(3), 0, STAND)
    legs.step(np.zeros(3), 0, STAND)
    alpha = 0.002 / (0.002 + 1.0 / (2 * math.pi * 20.0))
    assert np.allclose(data.qfrc_applied[legs.vadr], -alpha * STAND_KP * 0.05, atol=1e-9)


def test_refused_combinations():
    with pytest.raises(ValueError):
        make(motor_model="dc")
    with pytest.raises(ValueError):
        make(actuators=False, actuator="position")  # the model has no actuators
    with pytest.raises(ValueError):
        make(actuators=True, actuator="position", torque_bw_hz=10.0)
