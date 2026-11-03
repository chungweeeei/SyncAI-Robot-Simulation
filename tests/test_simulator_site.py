"""The whole simulator on a real site: the workspace's map/dp (its octomap,
map.pcd and mapping track), the robot's URDF, the policies in policy/. Skips
when any of them is missing, so it runs in the container and nowhere else.

The caches are the repository's own (cache/dp), so after the first build
each Simulator is up in a second or two. dp's first fitted flight is the
stairs these tests walk: 8 risers of 0.174 m on 0.240 m treads, 1.4 m up.
"""

import math
import os
import socket

import numpy as np
import pytest

from syncai_sim.envs.simulator import SimConfig, Simulator
from syncai_sim.world.build import default_cache_dir

WS = os.environ.get("ROBOT_WS_DIR", "/home/syncrobotic/robot_ws")
MAP = os.path.join(WS, "map", os.environ.get("SYNCAI_TEST_MAP", "dp"))
URDF = os.path.join(WS, "src", "syncai_bringup", "description", "G23.urdf")
POLICY_DIR = os.environ.get("SYNCAI_POLICY_DIR",
                            os.path.join(os.path.dirname(__file__), "..", "policy"))
PPO = os.path.join(POLICY_DIR, "ppo", "policy7.onnx")
HIMLOCO = os.path.join(POLICY_DIR, "isaacsim", "policy_0812.onnx")
QUIET = lambda *a: None  # noqa: E731


def site_or_skip():
    pytest.importorskip("mujoco")
    if not (os.path.isfile(os.path.join(MAP, "octomap.bt")) and os.path.isfile(URDF)):
        pytest.skip(f"no site: {MAP}/octomap.bt or {URDF} missing (the workspace mount)")


def make(locomotion="policy", **kw):
    site_or_skip()
    if locomotion == "policy":
        pytest.importorskip("onnxruntime")
        if not (os.path.isfile(PPO) and os.path.isfile(HIMLOCO)):
            pytest.skip(f"policies not under {POLICY_DIR} (policy/README.md)")
    cfg = SimConfig(map_dir=MAP, cache_dir=default_cache_dir(MAP), urdf_path=URDF,
                    locomotion=locomotion, physics_hz=kw.pop("physics_hz", 500.0),
                    ppo_policy=PPO, himloco_policy=HIMLOCO, **kw)
    cfg.lidar.points_per_scan = 2000
    return Simulator(cfg, log=QUIET)


def drive(sim, vx, seconds):
    for _ in range(int(round(seconds / sim.model.opt.timestep))):
        sim.set_cmd(vx, 0.0, 0.0)
        sim.step()


def before_flight(sim, k=0, before=1.0):
    """(x, y, z, yaw) `before` m before fitted flight k, facing up it
    (scripts/gait_bench.py flight_pose)."""
    if sim.surface is None or len(sim.surface.flights) <= k:
        pytest.skip("this map has no fitted flight to climb")
    f = sim.surface.flights[k]
    along = f.sign * (f.u0 - before)
    x, y = (along, f.across) if f.axis == 0 else (f.across, along)
    yaw = math.atan2(f.sign if f.axis == 1 else 0.0, f.sign if f.axis == 0 else 0.0)
    return x, y, f.bottom, yaw


def free_ports(n):
    probes = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(n)]
    for s in probes:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in probes]
    for s in probes:
        s.close()
    return ports


def test_the_robot_stands_and_the_lidar_scans():
    sim = make()
    sim.spawn(0.0, 0.0, 0.0, 0.0)
    drive(sim, 0.0, 2.0)
    st = sim.status()
    assert st["falls"] == 0 and st["collisions"] == 0, st
    xyzi = sim.scan()
    assert xyzi.shape[1] == 4 and xyzi.shape[0] > 500
    # The walking body stands its lidar ~0.44 m over the floor (README).
    T = sim.T_map_imu()
    assert 0.35 < T[2, 3] - sim.state.z_ground < 0.55


def test_walking_body_knobs_compile_into_the_model():
    """The vendor's motor model, trunk and contact, MuJoCo actuators and the
    implicit integrator: each lands in the compiled model and the robot
    still stands on them."""
    import mujoco

    from syncai_sim.entity.robot import J60_ARMATURE, J60_FRICTIONLOSS
    sim = make(motor_model="j60", trunk_inertial="vendor", foot_solref=(0.005, 1.0),
               foot_friction=(1.0, 0.01, 0.01), actuator="position", integrator="implicitfast",
               cmd_delay_ms=4.0)
    m = sim.model
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert m.nu == 12 and sim.motors.act is not None
    assert math.isclose(m.body_mass[sim.info.base_body], 9.368, abs_tol=1e-6)
    knee = m.joint("FL_Knee_joint")
    assert math.isclose(m.dof_armature[knee.dofadr[0]], J60_ARMATURE[2], rel_tol=1e-6)
    assert math.isclose(m.dof_frictionloss[knee.dofadr[0]], J60_FRICTIONLOSS[2], rel_tol=1e-6)
    assert np.allclose(m.geom(int(sim.info.terrain_geoms[0])).solref, [0.005, 1.0])
    foot = int(sim.info.foot_geoms[0])
    assert foot >= 0 and np.allclose(m.geom_friction[foot], [1.0, 0.01, 0.01])
    sim.spawn(0.0, 0.0, 0.0, 0.0)
    drive(sim, 0.0, 2.0)
    assert sim.status()["falls"] == 0
    assert np.abs(sim.motors.tau()).max() > 1.0
    assert sim.data.xpos[sim.info.base_body][2] - sim.state.z_ground > 0.2


def test_himloco_climbs_the_first_flight():
    """The stair policy up dp's first flight at 0.5 m/s (gait_bench's numbers,
    2026-11-03: 1.0 m climbed in 10 s, no fall, two runs of two)."""
    sim = make()
    x, y, z, yaw = before_flight(sim)
    sim.driver.policy = 1
    sim.spawn(x, y, yaw, z)
    z0 = sim.state.z_ground
    drive(sim, 0.0, 1.0)
    drive(sim, 0.5, 10.0)
    st = sim.status()
    assert st["falls"] == 0, st
    assert sim.state.z_ground > z0 + 0.8, st


def test_udp_locomotion_stands_through_the_hardware_link():
    """The joints handed to a controller process: here the stand-in that
    answers every state with a PD to the default pose. No ONNX file is
    needed, every physics step sends one state, the body stands."""
    from syncai_sim.locomotion.hardware_link import StandInController
    site_or_skip()
    with pytest.raises(ValueError, match="qfrc"):
        make("udp", actuator="position")
    cmd_port, state_port = free_ports(2)
    sim = make("udp", physics_hz=1000.0, link_command_port=cmd_port, link_state_port=state_port,
               max_catch_up_s=0.008, controller_command_port=free_ports(1)[0])
    assert sim.legs is None and sim.link is not None and sim.model.opt.timestep == 0.001
    sim.spawn(0.0, 0.0, 0.0, 0.0)
    ctrl = StandInController(state_port, cmd_port)
    ctrl.start()
    try:
        drive(sim, 0.0, 2.0)
    finally:
        ctrl.stop()
        sim.close()
    st = sim.status()
    assert st["falls"] == 0, st
    assert st["link"]["states"] == 2000 and st["link"]["commands"] > 1000, st
    assert st["link"]["watchdog_trips"] == 0
    assert ctrl.last_t > sim.link.epoch + 1.9 and ctrl.last_state[51:55].sum() >= 3  # feet down
    assert sim.data.xpos[sim.info.base_body][2] - sim.state.z_ground > 0.25, st
    assert np.abs(sim.motors.q() - sim.motors.last.q_des).max() < 0.15  # held at the pose
