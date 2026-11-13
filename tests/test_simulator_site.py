"""The whole simulator on a real site: the workspace's map/dp (its octomap,
map.pcd and mapping track) and the robot's URDF, the joints held by the
hardware link's stand-in controller. Skips when the site is missing, so it
runs in the container and nowhere else.

The caches are the repository's own (cache/dp), so after the first build
each Simulator is up in a second or two.
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
QUIET = lambda *a: None  # noqa: E731


def site_or_skip():
    pytest.importorskip("mujoco")
    if not (os.path.isfile(os.path.join(MAP, "octomap.bt")) and os.path.isfile(URDF)):
        pytest.skip(f"no site: {MAP}/octomap.bt or {URDF} missing (the workspace mount)")


def make(**kw):
    """A sim on free ports (a sim of `docker compose up` may hold the defaults)."""
    site_or_skip()
    cmd_port, state_port, ctrl_port = free_ports(3)
    cfg = SimConfig(map_dir=MAP, cache_dir=default_cache_dir(MAP), urdf_path=URDF,
                    link_command_port=cmd_port, link_state_port=state_port,
                    controller_command_port=ctrl_port, **kw)
    cfg.lidar.points_per_scan = 2000
    return Simulator(cfg, log=QUIET)


def run(sim, seconds, controller=True):
    """Step `seconds` of sim time, the stand-in controller on the link (or nothing)."""
    from syncai_sim.locomotion.hardware_link import StandInController
    ctrl = StandInController(sim.cfg.link_state_port, sim.cfg.link_command_port) if controller else None
    if ctrl is not None:
        ctrl.start()
    try:
        for _ in range(int(round(seconds / sim.model.opt.timestep))):
            sim.step()
    finally:
        if ctrl is not None:
            ctrl.stop()
    return ctrl


def free_ports(n):
    probes = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(n)]
    for s in probes:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in probes]
    for s in probes:
        s.close()
    return ports


def test_the_robot_stands_through_the_link_and_the_lidar_scans():
    """Every physics step sends one state and the stand-in's PD holds the
    body up at the default pose; the lidar scans the site from it."""
    sim = make()
    assert sim.model.opt.timestep == 0.001  # the default physics_hz, 1 kHz
    sim.spawn(0.0, 0.0, 0.0, 0.0)
    try:
        ctrl = run(sim, 2.0)
    finally:
        sim.close()
    st = sim.status()
    assert st["falls"] == 0 and st["collisions"] == 0, st
    assert st["link"]["states"] == 2000 and st["link"]["commands"] > 1000, st
    assert st["link"]["watchdog_trips"] == 0
    assert ctrl.last_t > sim.link.epoch + 1.9 and ctrl.last_state[51:55].sum() >= 3  # feet down
    assert np.abs(sim.motors.q() - sim.motors.last.q_des).max() < 0.15  # held at the pose
    xyzi = sim.scan()
    assert xyzi.shape[1] == 4 and xyzi.shape[0] > 500
    # The standing body holds its lidar ~0.44 m over the floor (README).
    T = sim.T_map_imu()
    assert 0.35 < T[2, 3] - sim.state.z_ground < 0.55


def test_without_a_controller_the_joints_are_damped():
    """Nothing on the link: the motors get the damping command (kp 0) from
    the first step, and the body sinks instead of standing."""
    sim = make()
    sim.spawn(0.0, 0.0, 0.0, 0.0)
    h0 = sim.data.xpos[sim.info.base_body][2] - sim.state.z_ground
    try:
        run(sim, 1.0, controller=False)
    finally:
        sim.close()
    assert np.all(sim.motors.last.kp == 0) and np.all(sim.motors.last.kd > 0)
    assert sim.status()["link"]["commands"] == 0
    assert sim.data.xpos[sim.info.base_body][2] - sim.state.z_ground < h0 - 0.05


def test_body_knobs_compile_into_the_model():
    """The alternatives to the vendor's body — the URDF's motor clamp and
    trunk, MuJoCo's contact — and the implicit integrator: each lands in the
    compiled model and the robot still stands on them."""
    import mujoco

    from syncai_sim.entity.robot import JOINT_ARMATURE
    sim = make(motor_model="ideal", trunk_inertial="urdf", foot_solref=(0.02, 1.0),
               foot_friction=(1.0, 0.005, 0.0001), integrator="implicitfast", cmd_delay_ms=4.0)
    m = sim.model
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert m.nu == 0  # the torque is a generalized force, never an actuator
    assert not math.isclose(m.body_mass[sim.info.base_body], 9.368, abs_tol=1e-3)
    knee = m.joint("FL_Knee_joint")
    assert math.isclose(m.dof_armature[knee.dofadr[0]], JOINT_ARMATURE, rel_tol=1e-6)
    assert m.dof_frictionloss[knee.dofadr[0]] == 0.0
    assert np.allclose(m.geom(int(sim.info.terrain_geoms[0])).solref, [0.02, 1.0])
    foot = int(sim.info.foot_geoms[0])
    assert foot >= 0 and np.allclose(m.geom_friction[foot], [1.0, 0.005, 0.0001])
    sim.spawn(0.0, 0.0, 0.0, 0.0)
    try:
        run(sim, 2.0)
    finally:
        sim.close()
    assert sim.status()["falls"] == 0
    assert np.abs(sim.motors.tau()).max() > 1.0
    assert sim.data.xpos[sim.info.base_body][2] - sim.state.z_ground > 0.2


def test_vendor_body_is_the_default():
    """The defaults are the vendor's: the J60, its trunk, its contact."""
    from syncai_sim.entity.robot import J60_ARMATURE, J60_FRICTIONLOSS
    sim = make()
    sim.close()
    m = sim.model
    assert math.isclose(m.body_mass[sim.info.base_body], 9.368, abs_tol=1e-6)
    knee = m.joint("FL_Knee_joint")
    assert math.isclose(m.dof_armature[knee.dofadr[0]], J60_ARMATURE[2], rel_tol=1e-6)
    assert math.isclose(m.dof_frictionloss[knee.dofadr[0]], J60_FRICTIONLOSS[2], rel_tol=1e-6)
    assert np.allclose(m.geom(int(sim.info.terrain_geoms[0])).solref, [0.005, 1.0])
