"""actuator/motors.py on a stand-in twelve-joint body (no URDF): the clamps of
the two motor models, the command delay, the torque filter, and the two
command terms only a whole-body controller sends (a target velocity, a
feed-forward torque)."""

import math

import mujoco
import numpy as np
import pytest

from syncai_sim.actuator.motors import TORQUE_LIMIT, JointCommand, Motors, j60_torque_limits
from syncai_sim.entity.robot import J60_MAX_SPEED, J60_PEAK_TORQUE, LEGS, Q_DEFAULT

# The robot's controller's stand-up gains (what the stand-in sends too).
KP, KD = 70.0, 2.0


def body_xml(timestep: float = 0.002, feet: bool = False) -> str:
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
  {sensors}
</mujoco>"""


def make(**kw):
    """The body and its motors."""
    model = mujoco.MjModel.from_xml_string(body_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, Motors(model, data, **kw)


def stand(q_des=Q_DEFAULT) -> JointCommand:
    return JointCommand.pd(q_des, np.full(12, KP), np.full(12, KD))


def test_j60_envelope():
    peak, wmax = np.array(J60_PEAK_TORQUE), np.array(J60_MAX_SPEED)
    lo, hi = j60_torque_limits(np.zeros(3), peak, wmax)
    assert np.allclose(hi, peak) and np.allclose(lo, -peak)
    lo, hi = j60_torque_limits(wmax, peak, wmax)  # at top speed: no more drive, full brake
    assert np.allclose(hi, 0.0) and np.allclose(lo, -peak)
    lo, hi = j60_torque_limits(-wmax / 2, peak, wmax)
    assert np.allclose(hi, peak) and np.allclose(lo, -peak / 2)


def test_ideal_clamps_at_the_urdf_effort():
    model, data, motors = make()
    motors.reset()
    motors.set_joints(motors.q() + 1.0)  # a radian off the stand pose: kp 70 wants 70 Nm
    motors.apply(stand())
    tau = data.qfrc_applied[motors.vadr]
    assert np.allclose(np.abs(tau), TORQUE_LIMIT)
    assert np.allclose(motors.tau(), tau)


def test_j60_envelope_shrinks_with_joint_speed():
    model, data, motors = make(motor_model="j60")
    motors.reset()
    motors.set_joints(motors.q() + 1.0)
    data.qvel[motors.vadr] = 0.5 * np.array(J60_MAX_SPEED * 4)  # half of top speed, driving away
    motors.apply(stand())
    tau = data.qfrc_applied[motors.vadr]
    # The error is positive (target below q): the PD pulls back, against the
    # motion, so the clamp is the full peak.
    assert np.allclose(np.abs(tau), np.array(J60_PEAK_TORQUE * 4))
    data.qvel[motors.vadr] = -0.5 * np.array(J60_MAX_SPEED * 4)  # now moving with the torque
    motors.apply(stand())
    assert np.allclose(np.abs(data.qfrc_applied[motors.vadr]), 0.5 * np.array(J60_PEAK_TORQUE * 4))


def test_command_delay_holds_the_old_target():
    def torques(delay_s):
        model, data, motors = make(cmd_delay_s=delay_s)
        motors.reset()
        out = []
        for i in range(8):
            # Held where the joints are for two steps, then a target 0.3 rad off.
            motors.apply(stand(Q_DEFAULT if i < 2 else Q_DEFAULT + 0.3))
            out.append(float(np.abs(data.qfrc_applied[motors.vadr]).max()))
        return out
    prompt = torques(0.0)
    assert prompt[:2] == [0.0, 0.0] and prompt[2] > 0.0
    late = torques(3 * 0.002)
    assert late[:5] == [0.0] * 5 and late[5] > 0.0
    assert math.isclose(late[5], prompt[2])


def test_torque_filter_is_first_order():
    model, data, motors = make(torque_bw_hz=20.0)
    motors.reset()
    motors.set_joints(motors.q() + 0.05)  # 3.5 Nm wanted: well inside the clamp
    motors.apply(stand())
    alpha = 0.002 / (0.002 + 1.0 / (2 * math.pi * 20.0))
    # The filter starts at the first torque, then moves alpha of the way.
    first = data.qfrc_applied[motors.vadr].copy()
    assert np.allclose(first, -KP * 0.05, atol=1e-9)
    motors.set_joints(Q_DEFAULT)  # now nothing is wanted
    motors.apply(stand())
    assert np.allclose(data.qfrc_applied[motors.vadr], (1 - alpha) * first, atol=1e-9)


def test_velocity_target_and_feedforward_reach_the_joints():
    """What an MPC's whole-body controller sends: kp 0, kd on a velocity
    target, a feed-forward torque — tau = kd (dq_des - dq) + tau_ff, clamped."""
    model, data, motors = make()
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


def test_damping_command_is_kd_only():
    d = JointCommand.damping(2.0)
    assert np.all(d.kp == 0) and np.allclose(d.kd, 2.0) and np.all(d.tau_ff == 0)


def test_refused_combinations():
    with pytest.raises(ValueError):
        make(motor_model="dc")
