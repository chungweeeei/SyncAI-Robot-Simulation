"""End to end on the planner CLI's synthetic `stairs` scene: spawn, walk into
the flight without the stair policy (refused), with it (climbed). Needs the
workspace's octomap_planner_3d_cli (built in install/) and MuJoCo, so it runs
in the container and skips elsewhere."""

import math
import os
import shutil
import subprocess

import numpy as np
import pytest

from syncai_sim.kinematics import KinematicsConfig
from syncai_sim.simulator import SimConfig, Simulator

WS = os.environ.get("ROBOT_WS_DIR", "/home/syncrobotic/robot_ws")
URDF = os.path.join(WS, "src", "syncai_bringup", "description", "G23.urdf")


@pytest.fixture(scope="module")
def stairs_map(tmp_path_factory):
    # The CLI is a workspace binary that needs the ROS environment (its message
    # libraries are on LD_LIBRARY_PATH only after `source install/setup.bash`),
    # so it is run through `ros2 run`; without that environment the test skips.
    if shutil.which("ros2") is None or not os.path.isfile(URDF):
        pytest.skip("ROS environment / workspace install not sourced")
    pytest.importorskip("mujoco")
    d = tmp_path_factory.mktemp("stairs")
    r = subprocess.run(["ros2", "run", "syncai_global_planner", "octomap_planner_3d_cli",
                        "--synthetic", "stairs", "--write-bt", str(d / "octomap.bt")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        pytest.skip(f"octomap_planner_3d_cli not runnable: {r.stderr.strip()[:200]}")
    return str(d)


def make_sim(stairs_map, tmp_path, **kin):
    cfg = SimConfig(map_dir=stairs_map, cache_dir=str(tmp_path / "cache"), urdf_path=URDF,
                    kin=KinematicsConfig(**kin), initial_policy=0)
    cfg.lidar.points_per_scan = 5000
    return Simulator(cfg, log=lambda *a: None)


def drive(sim, vx, seconds):
    for _ in range(int(seconds / sim.model.opt.timestep)):
        sim.set_cmd(vx, 0.0, 0.0)
        sim.step()


def test_scene_builds_and_scans(stairs_map, tmp_path):
    sim = make_sim(stairs_map, tmp_path)
    lo, hi = sim.grid.bounds_m()
    sim.spawn((lo[0] + hi[0]) / 2 - 3.0, (lo[1] + hi[1]) / 2, 0.0, 0.0)
    xyzi = sim.scan()
    # The synthetic scene is a floor and a flight with no walls: most rays miss.
    assert xyzi.shape[1] == 4 and xyzi.shape[0] > 100
    # Returns are in the body frame: the floor is seen below the sensor only
    # where the -7 deg edge reaches it (~3.9 m out) before the scene ends.
    assert xyzi[:, 2].min() < -0.35
    T = sim.T_map_imu()
    assert abs(T[2, 3] - (sim.state.z_ground + 0.481)) < 1e-3


def test_stairs_need_the_policy(stairs_map, tmp_path):
    sim = make_sim(stairs_map, tmp_path)
    lo, hi = sim.grid.bounds_m()
    y = (lo[1] + hi[1]) / 2
    # The scene's flight rises along +x from the lower floor; find where.
    xs = np.arange(lo[0] + 0.5, hi[0] - 0.5, 0.1)
    g = np.array([sim.grid.ground_under(x, y, 0.0, 5.0, 5.0) for x in xs])
    g = np.where(np.isnan(g), -1e9, g)
    low = float(np.min(g[g > -1e8]))
    first_riser = xs[np.argmax(g > low + 0.1)]
    sim.spawn(first_riser - 2.0, y, 0.0, low)
    z0 = sim.state.z_ground
    drive(sim, 0.5, 8.0)
    st = sim.status()
    assert st["refused"] > 0, st
    assert sim.state.z_ground < z0 + 0.2  # at most the first tread
    # Now the stair policy: climbs.
    sim.driver.set_policy_mode(1)
    sim.driver.switch_delay = 0.0
    sim.driver.update()
    sim.driver.policy = 1
    drive(sim, 0.5, 10.0)
    assert sim.state.z_ground > z0 + 0.6, sim.status()
    assert sim.status()["collisions"] == 0


POLICY_DIR = os.environ.get("SYNCAI_POLICY_DIR",
                            os.path.join(os.path.dirname(__file__), "..", "policy"))


def make_walking_sim(stairs_map, tmp_path, **body):
    pytest.importorskip("onnxruntime")
    ppo = os.path.join(POLICY_DIR, "ppo", "policy7.onnx")
    him = os.path.join(POLICY_DIR, "isaacsim", "policy_0812.onnx")
    if not (os.path.isfile(ppo) and os.path.isfile(him)):
        pytest.skip(f"policies not under {POLICY_DIR} (policy/README.md)")
    cfg = SimConfig(map_dir=stairs_map, cache_dir=str(tmp_path / "cache"), urdf_path=URDF,
                    locomotion="policy", physics_hz=500.0, ppo_policy=ppo, himloco_policy=him,
                    **body)
    cfg.lidar.points_per_scan = 2000
    return Simulator(cfg, log=lambda *a: None)


def test_walking_body_knobs_compile_into_the_model(stairs_map, tmp_path):
    """The vendor's motor model, trunk and contact, MuJoCo actuators and the
    implicit integrator: each lands in the compiled model and the robot
    still stands and walks on the flat."""
    import mujoco
    from syncai_sim.robot import J60_ARMATURE, J60_FRICTIONLOSS
    sim = make_walking_sim(stairs_map, tmp_path, motor_model="j60", trunk_inertial="vendor",
                           foot_solref=(0.005, 1.0), foot_friction=(1.0, 0.01, 0.01),
                           actuator="position", integrator="implicitfast", cmd_delay_ms=4.0)
    m = sim.model
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert m.nu == 12 and sim.legs.act is not None
    assert math.isclose(m.body_mass[sim.info.base_body], 9.368, abs_tol=1e-6)
    knee = m.joint("FL_Knee_joint")
    assert math.isclose(m.dof_armature[knee.dofadr[0]], J60_ARMATURE[2], rel_tol=1e-6)
    assert math.isclose(m.dof_frictionloss[knee.dofadr[0]], J60_FRICTIONLOSS[2], rel_tol=1e-6)
    foot = m.geom("FL_FOOT_collision") if "FL_FOOT_collision" in [m.geom(i).name for i in range(m.ngeom)] \
        else None
    terrain = m.geom(int(sim.info.terrain_geoms[0]))
    assert np.allclose(terrain.solref, [0.005, 1.0])
    if foot is not None:
        assert np.allclose(foot.solref, [0.005, 1.0]) and np.allclose(foot.friction, [1.0, 0.01, 0.01])
    first, y, low = foot_of_flight(sim)
    sim.spawn(first - 3.0, y, 0.0, low)
    drive(sim, 0.3, 4.0)
    st = sim.status()
    assert st["falls"] == 0, st
    # The actuators carry it: torque on the joints, the trunk up on its legs, and
    # it went forward (how fast is scripts/gait_bench.py's business).
    assert np.abs(sim.legs.tau()).max() > 1.0
    assert sim.data.xpos[sim.info.base_body][2] - sim.state.z_ground > 0.2, st
    assert sim.state.x > first - 3.0 + 0.2, st


def foot_of_flight(sim):
    lo, hi = sim.grid.bounds_m()
    y = (lo[1] + hi[1]) / 2
    xs = np.arange(lo[0] + 0.5, hi[0] - 0.5, 0.1)
    g = np.array([sim.grid.ground_under(x, y, 0.0, 5.0, 5.0) for x in xs])
    g = np.where(np.isnan(g), -1e9, g)
    low = float(np.min(g[g > -1e8]))
    return xs[np.argmax(g > low + 0.1)], y, low


@pytest.mark.parametrize("policy", [0, 1])
def test_walking_flight_needs_himloco(stairs_map, tmp_path, policy):
    """The policies themselves decide: HimLoco climbs the 1 m flight, PPO trips
    at the first riser and the posture check puts the driver in Damping."""
    sim = make_walking_sim(stairs_map, tmp_path)
    first, y, low = foot_of_flight(sim)
    sim.driver.policy = policy
    sim.spawn(first - 1.0, y, 0.0, low)
    z0 = sim.state.z_ground
    drive(sim, 0.4, 12.0)
    st = sim.status()
    if policy == 1:
        assert sim.state.z_ground > z0 + 0.9, st
        assert st["falls"] == 0 and st["collisions"] == 0, st
    else:
        assert sim.state.z_ground < z0 + 0.25, st
