# CLAUDE.md

Guidance for Claude Code in this repository. `README.md` is the user-facing
document (what the sim is, how to run it, the decisions); this file is the
short list of what you must know to edit it correctly.

## What this is

A MuJoCo simulation of the SyncAI G23 in a real mapped site, so the navigation
stack of `chungweeeei/SyncAI-Robot-Workspace` (a sibling checkout, mounted into
the container) runs end to end with no robot. The G23 walks on its own ONNX
policies (`policy.py`, a port of `g2_deploy_rs`'s runners; `legged.py` drives
the joints) on a patch of the voxel grid (`terrain_patch.py`); `locomotion:
kinematic` keeps the older rigid body carried by rules (`kinematics.py`).
The two ONNX files live in `policy/`, gitignored: copied from g2_deploy_rs,
where they are not versioned either (`policy/README.md`); never commit them.
Python throughout (decided
2026-10-23); one container (`docker-compose.yml`), the workspace mounted at
`/home/syncrobotic/robot_ws`, this repo at `/home/syncrobotic/sim`.

## Rules that shape edits

1. **The workspace is read, never written.** The INI is bind-mounted over its
   `config/system.ini`, robot_state_publisher comes from `launch/rsp.launch.py`
   here, the session spec is `config/sessions/start_sim.yaml` here. If the sim
   needs something from the workspace (a CLI flag, a CI stage), that is a
   change in that repository, said so in the commit message.
2. **The voxel grid is the truth; MuJoCo is the picture and the props.** Lidar
   rays, the ground under the robot and whether the trunk is blocked are
   answered by `voxel_grid.py` (numba) on the voxels of `octomap.bt`. MuJoCo
   gets a visual mesh (`contype 0`), the large merged boxes as collision
   plates for props, and — walking — a pool of voxel boxes moved under the
   feet (`terrain_patch.py`), never the grid whole. Do not route sensing through `mj_multiRay` or collide the
   trunk with the world mesh: the first scans every geom per ray, the second is
   a convex hull. The measurements that decided this are in README
   ("Decisions worth knowing").
3. **Every ROS-facing value mirrors a consumer in the workspace**, and
   `config/params/sim_params.yaml` names which. Change both sides together.
   The ones with teeth: `mode_switch_delay` < StairMode's 3 s settle; `mode`
   reliable and republished; `pointlio/lio_odom` is what lio_bridge reads (not
   TF); `initialpose` moves `map -> pointlio_odom`, never the body; goals carry
   the floor's ground z.
4. **Wall clock.** `use_sim_time` is false everywhere in the stack and the sim
   paces fixed 2 ms steps to real time. Never publish a stamp that is not `now`.
5. **Comments explain why**, as in the workspace, and record measurements and
   rejected alternatives. A bare number without its reason is out of place.
6. Python: `ruff.toml` (the workspace's rule set, py310, 100 columns). Tests:
   `pytest tests` in the container; `test_simulator_synthetic.py` needs the
   ROS environment sourced (it runs the planner CLI through `ros2 run`).

## Where things happen

| Need | Place |
|---|---|
| map -> world (parse, mesh, plates, cache) | `world.py`, `btree.py`, `greedy_mesh.py` |
| Foxglove (map clouds, joint_states, bridge) | `map_cloud.py`, `config/foxglove/bridge.yaml`, `scripts/foxglove.sh`; mesh URIs in `launch/rsp.launch.py` |
| the viewer's look (tiles, intensity shading, the Poisson and TSDF surfaces, colours, storeys) | `site_tiles.py` (10 m tiles as GLB, a texel per voxel face), `voxel_intensity.py`, `pcd_mesh.py`, `tsdf_mesh.py`, `visual.py`, `storeys.py` (picture only, never sensing); the site, storey and layer switches are the panel in `web_view.py` |
| ray cast / ground / blocked | `voxel_grid.py` |
| the policies (port of g2_deploy_rs) | `policy.py` (`tests/test_policy.py`; names the Rust source of every constant) |
| joints per motion state, PD, the motor clamp, falls | `legged.py` (`tests/test_legged_motor.py`); the body's physics knobs (`motor_model`, `trunk_inertial`, contact, `actuator`, `integrator`, delays) are `RobotConfig` in `robot.py`, their A/B `scripts/gait_bench.py` |
| the feet's ground (voxels -> boxes) | `terrain_patch.py` (numba, `tests/test_terrain_patch.py`) |
| the feet's floor heights, flights fitted as staircases | `surface_heights.py`, `stairs.py`; stairs the track never climbed: `config/stair_walks/<map>.yaml`; slopes across voxel layers: `config/ramps/<map>.yaml` |
| the step rule, gates, collisions (kinematic) | `kinematics.py` (pure Python, `tests/test_kinematics.py`) |
| URDF -> MJCF, mocap weld, standing pose | `robot.py` (`DEFAULT_STANDING` was solved by FK, feet 0.372 m under the hips) |
| MjSpec assembly, bitmasks, groups, the MJB scene cache | `scene.py` (`scene_info` reads a loaded model back by name; `scene_key` lists what the cache depends on; the site's picture is not in the model) |
| the lidar pattern, prop rays | `lidar.py`, `prop_rays.py`; the recorded MID360 bank `config/lidar/mid360_pattern.npz` (built by `scripts/lidar_pattern.py` through `livox_bag.py`, which reads a rosbag2 sqlite without ROS) |
| the gait controller on driver_manager's UDP link (the session) | `controller_link.py` (`tests/test_controller_link.py`), `config/params/driver_manager_sim.yaml` |
| services and `mode` (driver_link: fake) | `fake_driver.py` |
| one physics tick, scan, spawn, teleport | `simulator.py` |
| publishers, subscribers, the main loop | `ros_node.py`; CLI in `__main__.py` (sets `OPENBLAS_NUM_THREADS=1` before numpy); the viewer (viser: the site as GLB tiles once, poses at 20 Hz, drawn by the host's browser; nothing renders in the container) in `web_view.py`, its panel's driving in `teleop.py` |
| the session, the test driver | `scripts/run_session.sh`, `scripts/nav_test.py` |

## Running and testing (inside the container)

```bash
docker compose up -d && docker compose exec sim bash
source /opt/ros/humble/setup.bash && source /opt/ros2_rust_underlay/install/setup.bash && source install/setup.bash
/home/syncrobotic/sim/scripts/run_session.sh         # logs: log/stack/sim01/sim/<pane>/current; the picture: http://localhost:8080
python3 /home/syncrobotic/sim/scripts/nav_test.py --goal 3 0 0
python3 -m pytest /home/syncrobotic/sim/tests
```

Do not run `docker compose build` or the session on your own after an edit;
leave that to the user unless asked — the session starts the workspace's whole
nav stack.
