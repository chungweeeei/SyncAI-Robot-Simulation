"""actuator/motors.py on a stand-in twelve-joint body (no URDF, no ONNX): the
clamps of the two motor models, the two actuators, the command delay, the
torque filter, and the two command terms the policies never send (a target
velocity, a feed-forward torque), driven through legged.py's states."""

import math

import mujoco
import numpy as np
import pytest

from syncai_sim.actuator.motors import TORQUE_LIMIT, JointCommand, Motors, j60_torque_limits
from syncai_sim.driver.fake_driver import DAMPING, ESTOP, LIE_DOWN, LOCOMOTION, MPC
from syncai_sim.entity.robot import J60_MAX_SPEED, J60_PEAK_TORQUE, JOINTS, LEGS
from syncai_sim.locomotion.legged import STAND_KD, STAND_KP, LeggedController
from syncai_sim.locomotion.policy import HimlocoRunner, MultiPolicy, PpoRunner

from tests.test_policy import Recorder

STAND = next(v for v in range(20) if v not in (LOCOMOTION, MPC, LIE_DOWN, DAMPING, ESTOP))


def body_xml(actuators: bool, timestep: float = 0.002, feet: bool = False) -> str:
    """A trunk on four three-joint legs; with `feet`, a sphere at the end of
    each shank named like the URDF's feet, a floor to stand on and gravity."""
    legs = []
    for i, leg in enumerate(LEGS):
        x = 0.1 if leg[0] == "F" else -0.1
        y = 0.05 if leg[1] == "L" else -0.05
        foot = (f'<body name="{leg}_FOOT" pos="0 0 -0.2"><geom type="sphere" size="0.02" mass="0.02"/>'
                f'</body>' if feet else "")
        legs.append(f"""
      <body name="{leg}_HIP" pos="{x} {y} 0">
        <joint name="{leg}_HipX_joint" axis="1 0 0"/>
        <geom type="capsule" size="0.02 0.03" mass="0.5"/>
        <body name="{leg}_THIGH" pos="0 {y} 0">
          <joint name="{leg}_HipY_joint" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.8"/>
          <body name="{leg}_SHANK" pos="0 0 -0.2">
            <joint name="{leg}_Knee_joint" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.015" mass="0.15"/>{foot}
          </body>
        </body>
      </body>""")
    acts = "".join(f'<position name="act_{leg}_{j}_joint" joint="{leg}_{j}_joint" kp="0" kv="0" '
                   f'forcerange="-100 100"/>' for leg in LEGS for j in JOINTS)
    floor = '<geom name="floor" type="plane" size="2 2 0.1"/>' if feet else ""
    gravity = "0 0 -9.81" if feet else "0 0 0"
    imu = '<site name="base_imu" size="0.01"/>' if feet else ""
    sensors = ('<sensor><gyro name="base_gyro" site="base_imu"/>'
               '<accelerometer name="base_acc" site="base_imu"/></sensor>' if feet else "")
    return f"""
<mujoco>
  <option timestep="{timestep}" gravity="{gravity}"/>
  <worldbody>{floor}
    <body name="base_link" pos="0 0 {0.42 if feet else 1}">
      <freejoint/>{imu}
      <geom name="trunk" type="box" size="0.2 0.1 0.05" mass="5"/>{''.join(legs)}
    </body>
  </worldbody>
  {'<actuator>' + acts + '</actuator>' if actuators else ''}{sensors}
</mujoco>"""


def make(actuators=False, transition_s=1.0, **kw):
    """The body (with MuJoCo position actuators when `actuators`), its motors
    (`actuator` position then, qfrc otherwise) and the state machine on top."""
    model = mujoco.MjModel.from_xml_string(body_xml(actuators))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    policies = MultiPolicy({0: PpoRunner(Recorder(np.zeros(12))),
                            1: HimlocoRunner(Recorder(np.zeros(12)))}, log=lambda *a: None)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    kw.setdefault("actuator", "position" if actuators else "qfrc")
    motors = Motors(model, data, **kw)
    legs = LeggedController(model, data, base, policies, motors, transition_s=transition_s,
                            log=lambda *a: None)
    return model, data, legs, motors


def tick(legs, motors, state, cmd=np.zeros(3), policy=0):
    """One physics step's worth of control: the state's command, applied."""
    motors.apply(legs.step(cmd, policy, state))


def test_j60_envelope():
    peak, wmax = np.array(J60_PEAK_TORQUE), np.array(J60_MAX_SPEED)
    lo, hi = j60_torque_limits(np.zeros(3), peak, wmax)
    assert np.allclose(hi, peak) and np.allclose(lo, -peak)
    lo, hi = j60_torque_limits(wmax, peak, wmax)  # at top speed: no more drive, full brake
    assert np.allclose(hi, 0.0) and np.allclose(lo, -peak)
    lo, hi = j60_torque_limits(-wmax / 2, peak, wmax)
    assert np.allclose(hi, peak) and np.allclose(lo, -peak / 2)


def test_qfrc_ideal_clamps_at_the_urdf_effort():
    model, data, legs, motors = make(transition_s=1e-4)  # the ease is over by the second step
    motors.reset()
    legs.reset()
    motors.set_joints(motors.q() + 1.0)  # a radian off the stand pose: kp 70 wants 70 Nm
    tick(legs, motors, STAND)
    tick(legs, motors, STAND)  # the ease starts where the joints are; at the goal next step
    tau = data.qfrc_applied[motors.vadr]
    assert np.allclose(np.abs(tau), TORQUE_LIMIT)
    assert np.allclose(motors.tau(), tau)
    assert np.all(data.ctrl == 0)


def test_qfrc_j60_envelope_shrinks_with_joint_speed():
    model, data, legs, motors = make(motor_model="j60", transition_s=1e-4)
    motors.reset()
    legs.reset()
    motors.set_joints(motors.q() + 1.0)
    data.qvel[motors.vadr] = 0.5 * np.array(J60_MAX_SPEED * 4)  # half of top speed, driving away
    tick(legs, motors, STAND)
    tick(legs, motors, STAND)
    tau = data.qfrc_applied[motors.vadr]
    # The error is positive (target below q): the PD pulls back, against the
    # motion, so the clamp is the full peak.
    assert np.allclose(np.abs(tau), np.array(J60_PEAK_TORQUE * 4))
    data.qvel[motors.vadr] = -0.5 * np.array(J60_MAX_SPEED * 4)  # now moving with the torque
    tick(legs, motors, STAND)
    assert np.allclose(np.abs(data.qfrc_applied[motors.vadr]), 0.5 * np.array(J60_PEAK_TORQUE * 4))


def test_position_actuators_get_the_gains_the_target_and_the_range():
    model, data, legs, motors = make(actuators=True, transition_s=1e-4)
    motors.reset()
    legs.reset()
    assert np.allclose(data.ctrl[motors.act], motors.q())
    motors.set_joints(motors.q() + 0.1)
    tick(legs, motors, STAND)
    tick(legs, motors, STAND)
    assert np.all(data.qfrc_applied[motors.vadr] == 0.0)
    assert np.allclose(model.actuator_gainprm[motors.act, 0], STAND_KP)
    assert np.allclose(model.actuator_biasprm[motors.act, 1], -STAND_KP)
    assert np.allclose(model.actuator_biasprm[motors.act, 2], -STAND_KD)
    assert np.allclose(model.actuator_forcerange[motors.act, 0], -TORQUE_LIMIT)
    assert np.allclose(model.actuator_forcerange[motors.act, 1], TORQUE_LIMIT)
    assert np.allclose(data.ctrl[motors.act], legs.target)
    mujoco.mj_forward(model, data)  # MuJoCo computes the force: kp (q* - q), 7 Nm, inside the range
    assert np.allclose(motors.tau(), -STAND_KP * 0.1, atol=1e-6)
    tick(legs, motors, DAMPING)
    assert np.allclose(model.actuator_gainprm[motors.act, 0], 0.0)
    assert np.allclose(model.actuator_biasprm[motors.act, 2], -2.0)


def test_position_with_j60_writes_the_envelope_every_step():
    model, data, legs, motors = make(actuators=True, motor_model="j60")
    motors.reset()
    legs.reset()
    data.qvel[motors.vadr] = np.array(J60_MAX_SPEED * 4)
    tick(legs, motors, STAND)
    assert np.allclose(model.actuator_forcerange[motors.act, 1], 0.0)
    assert np.allclose(model.actuator_forcerange[motors.act, 0], -np.array(J60_PEAK_TORQUE * 4))


def test_command_delay_holds_the_old_target():
    def torques(delay_s):
        model, data, legs, motors = make(cmd_delay_s=delay_s, transition_s=1e-4)
        motors.reset()
        legs.reset()
        out = []
        for _ in range(8):
            tick(legs, motors, LIE_DOWN)  # the target jumps to the lying pose at step 2
            out.append(float(np.abs(data.qfrc_applied[motors.vadr]).max()))
        return out
    prompt = torques(0.0)
    assert prompt[0] == 0.0 and prompt[1] > 0.0
    late = torques(3 * 0.002)
    assert late[:4] == [0.0] * 4 and late[4] > 0.0
    assert math.isclose(late[4], prompt[1])


def test_torque_filter_is_first_order():
    model, data, legs, motors = make(torque_bw_hz=20.0, transition_s=1e-4)
    motors.reset()
    legs.reset()
    motors.set_joints(motors.q() + 0.05)  # 3.5 Nm wanted: well inside the clamp
    tick(legs, motors, STAND)
    tick(legs, motors, STAND)
    alpha = 0.002 / (0.002 + 1.0 / (2 * math.pi * 20.0))
    assert np.allclose(data.qfrc_applied[motors.vadr], -alpha * STAND_KP * 0.05, atol=1e-9)


def test_velocity_target_and_feedforward_reach_the_joints():
    """What an MPC's whole-body controller sends: kp 0, kd on a velocity
    target, a feed-forward torque — tau = kd (dq_des - dq) + tau_ff, clamped."""
    model, data, legs, motors = make()
    motors.reset()
    cmd = JointCommand(q_des=np.zeros(12), dq_des=np.full(12, 2.0), kp=np.zeros(12),
                       kd=np.full(12, 3.0), tau_ff=np.full(12, 5.0))
    motors.apply(cmd)
    assert np.allclose(data.qfrc_applied[motors.vadr], 3.0 * 2.0 + 5.0)  # dq is 0
    big = JointCommand(q_des=np.zeros(12), dq_des=np.zeros(12), kp=np.zeros(12),
                       kd=np.zeros(12), tau_ff=np.full(12, 100.0))
    motors.apply(big)
    assert np.allclose(data.qfrc_applied[motors.vadr], TORQUE_LIMIT)  # the clamp holds for tau_ff too
    assert motors.last is big


def test_position_actuators_refuse_what_they_cannot_express():
    model, data, legs, motors = make(actuators=True)
    motors.reset()
    motors.apply(JointCommand.pd(motors.q(), np.full(12, 10.0), np.ones(12)))  # fine
    with pytest.raises(ValueError, match="qfrc"):
        motors.apply(JointCommand(np.zeros(12), np.zeros(12), np.zeros(12), np.ones(12),
                                  np.full(12, 1.0)))
    with pytest.raises(ValueError, match="qfrc"):
        motors.apply(JointCommand(np.zeros(12), np.ones(12), np.zeros(12), np.ones(12),
                                  np.zeros(12)))


def test_damping_command_is_kd_only():
    d = JointCommand.damping(2.0)
    assert np.all(d.kp == 0) and np.allclose(d.kd, 2.0) and np.all(d.tau_ff == 0)


def test_refused_combinations():
    with pytest.raises(ValueError):
        make(motor_model="dc")
    with pytest.raises(ValueError):
        make(actuators=False, actuator="position")  # the model has no actuators
    with pytest.raises(ValueError):
        make(actuators=True, actuator="position", torque_bw_hz=10.0)
