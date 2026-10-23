"""End to end on the planner CLI's synthetic `stairs` scene: spawn, walk into
the flight without the stair policy (refused), with it (climbed). Needs the
workspace's octomap_planner_3d_cli (built in install/) and MuJoCo, so it runs
in the container and skips elsewhere."""

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
