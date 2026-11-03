"""sensor/proprioception.py on the stand-in body with feet, a floor and a body
IMU: the record's layout and frames, the contact flags' semantics, the
posture and trunk queries."""

import math

import mujoco
import numpy as np

from syncai_sim.actuator.motors import JointCommand, Motors
from syncai_sim.entity.robot import LEGS
from syncai_sim.scene.scene import SceneInfo
from syncai_sim.sensor.proprioception import N_FIELDS, SLICES, Sensors, rpy_of_quat

from tests.test_motors import body_xml


def make(xml_tweak=lambda s: s):
    model = mujoco.MjModel.from_xml_string(xml_tweak(body_xml(False, feet=True)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "trunk")
    feet = []
    for leg in LEGS:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_FOOT")
        feet.append(int(model.body_geomadr[b]))
    info = SceneInfo(base_body=base, mocap_body=-1, mocap_id=-1, lidar_body=base, imu_body=base,
                     trunk_geoms=[trunk], prop_bodies={}, prop_mocap_ids={},
                     terrain_geoms=np.array([floor]), foot_geoms=feet)
    motors = Motors(model, data)
    return model, data, Sensors(model, data, info, motors), motors


def settle(model, data, motors, seconds=1.0):
    """Hold the joints where they are with the stand gains while gravity sets
    the body on its feet."""
    q0 = motors.q()
    for _ in range(int(seconds / model.opt.timestep)):
        motors.apply(JointCommand.pd(q0, np.full(12, 70.0), np.full(12, 2.0)))
        mujoco.mj_step(model, data)


def test_at_rest_the_record_reads_gravity_and_four_feet():
    model, data, sensors, motors = make()
    settle(model, data, motors)
    f = sensors.read(1.0)
    assert f.shape == (N_FIELDS,) and f.dtype == np.float32 and sensors.t == 1.0
    assert np.allclose(sensors.rpy, 0.0, atol=0.05)
    assert np.allclose(sensors.acc, [0.0, 0.0, 9.81], atol=0.5)  # specific force: +g up at rest
    assert np.allclose(sensors.gyro, 0.0, atol=0.1)
    assert np.allclose(sensors.base_linvel, 0.0, atol=0.05)
    assert np.all(sensors.contact == 1.0), sensors.contact
    assert np.allclose(sensors.q, data.qpos[motors.qadr]) and np.allclose(sensors.dq, data.qvel[motors.vadr])
    assert np.allclose(sensors.tau, motors.tau())
    assert np.allclose(sensors.base_pos, data.xpos[sensors.base_body])
    # The views are the record: the packet is f's bytes in this order.
    for name, s in SLICES.items():
        assert np.shares_memory(getattr(sensors, name), f[s])
    assert not sensors.upset() and not sensors.trunk_hit()


def test_lifted_feet_report_no_contact_and_a_grounded_trunk_is_a_hit():
    model, data, sensors, motors = make()
    settle(model, data, motors)
    data.qpos[2] += 0.3  # in the air
    mujoco.mj_forward(model, data)
    sensors.read(2.0)
    assert np.all(sensors.contact == 0.0) and not sensors.trunk_hit()
    data.qpos[2] = 0.04  # the trunk's 5 cm half-height into the floor
    mujoco.mj_forward(model, data)
    assert sensors.trunk_hit()


def test_a_foot_against_the_robots_own_body_is_not_a_contact():
    # A sphere on base_link where the FL foot rests: same root body, so the
    # pair collides in MuJoCo but is not ground to the sensor.
    tweak = lambda s: s.replace('<geom name="trunk"',  # noqa: E731
                                '<geom name="self" type="sphere" size="0.03" pos="0.1 0.1 -0.4"/>'
                                '<geom name="trunk"')
    model, data, sensors, motors = make(tweak)
    data.qpos[2] += 0.3  # off the floor: the only contact left is foot vs self
    mujoco.mj_forward(model, data)
    assert data.ncon >= 1
    sensors.read(0.5)
    assert np.all(sensors.contact == 0.0), sensors.contact


def test_rpy_is_zyx_and_the_linear_velocity_is_world_frame():
    assert np.allclose(rpy_of_quat(math.cos(0.25), 0.0, math.sin(0.25), 0.0), (0.0, 0.5, 0.0))
    model, data, sensors, motors = make()
    yaw, roll = math.radians(90.0), math.radians(10.0)
    q_yaw = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    q_roll = np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0])
    q = np.empty(4)
    mujoco.mju_mulQuat(q, q_yaw, q_roll)  # yaw about world z, then roll about the body x
    data.qpos[2] += 0.5
    data.qpos[3:7] = q
    data.qvel[0:3] = (1.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    sensors.read(0.1)
    assert np.allclose(sensors.rpy, (roll, 0.0, yaw), atol=1e-6)
    assert np.allclose(sensors.quat_wxyz, q)
    assert np.allclose(sensors.base_linvel, (1.0, 0.0, 0.0))  # not rotated into the body
    r, p = sensors.roll_pitch()
    assert math.isclose(r, roll, abs_tol=1e-6) and math.isclose(p, 0.0, abs_tol=1e-6)
    data.qpos[3:7] = (math.cos(0.4), 0.0, math.sin(0.4), 0.0)  # pitch 0.8 rad > 45 deg
    mujoco.mj_forward(model, data)
    assert sensors.upset()
