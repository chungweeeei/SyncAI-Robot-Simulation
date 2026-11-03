# CLAUDE.md

Guidance for Claude Code in this repository. `README.md` is the user-facing
document (what the sim is, how to run it, the decisions); this file is the
short list of what you must know to edit it correctly.

## What this is

A MuJoCo simulation of the SyncAI G23 in a real mapped site, for two things:

* **the nav stack end to end** — `chungweeeei/SyncAI-Robot-Workspace` (a
  sibling checkout, mounted into the container) runs with no robot while the
  G23 walks on its own ONNX policies in this process (`locomotion: policy`:
  `locomotion/policy.py` + `locomotion/legged.py`) on a patch of the voxel grid
  (`terrains/terrain_patch.py`);
* **a controller's hardware** — `locomotion: udp` hands the joints to a
  controller process over the hardware link (`locomotion/hardware_link.py`),
  ROS-free: the sim talks UDP only there. The consumer is
  SyncAI-Robot-Control's `--backend sim`; this is where its MPC (LMA) is
  studied.

`locomotion: kinematic` keeps the older rigid body carried by rules
(`locomotion/kinematics.py`). The layers and one tick of each mode are drawn
in `docs/architecture.md`. The ONNX files live in `policy/`, gitignored
(`policy/README.md`); never commit them. Python throughout (decided
2026-10-23); one container (`docker-compose.yml`), the workspace mounted at
`/home/syncrobotic/robot_ws`, this repo at `/home/syncrobotic/sim`.

## Rules that shape edits

1. **The workspace is read, never written.** The INI is bind-mounted over its
   `config/system.ini`, robot_state_publisher comes from `launch/rsp.launch.py`
   here, the session spec is `config/sessions/start_sim.yaml` here. If the sim
   needs something from the workspace (a CLI flag, a CI stage), that is a
   change in that repository, said so in the commit message.
2. **The voxel grid is the truth; MuJoCo is the robot, the props and the feet's
   floor; the browser is the picture.** Lidar rays, the ground under the robot
   and whether the trunk is blocked are answered by `world/voxel_grid.py`
   (numba) on the voxels of `octomap.bt`. MuJoCo gets the large merged boxes
   as collision plates for props and — walking — a pool of voxel boxes moved
   under the feet (`terrains/terrain_patch.py`), never the grid whole; the
   site's picture is not in the model at all (`visuals/site_tiles.py` sends it
   to the browser as GLB tiles). Do not route sensing through `mj_multiRay` or
   put the site's mesh back in the MJCF: the first scans every geom per ray,
   the second cost 628 MB of MJB for BVHs nothing used. The measurements that
   decided this are in README ("Decisions worth knowing").
3. **Every ROS-facing value mirrors a consumer in the workspace**, and
   `config/params/sim_params.yaml` names which. Change both sides together.
   The ones with teeth: `mode_switch_delay` < StairMode's 3 s settle; `mode`
   reliable and republished; `pointlio/lio_odom` is what lio_bridge reads (not
   TF); `initialpose` moves `map -> pointlio_odom`, never the body; goals carry
   the floor's ground z.
4. **Wall clock.** `use_sim_time` is false everywhere in the stack and the sim
   paces fixed physics steps to real time (2 ms; 1 ms in the udp profile).
   Never publish a ROS stamp that is not `now`. The hardware link's `t` is
   the exception by design: `epoch + sim_time`, strictly increasing across
   restarts and teleports (a controller treats a repeated time as the same
   sample).
5. **The hardware link is a protocol with a consumer.** Its spec is the
   docstring of `locomotion/hardware_link.py` plus `tests/test_hardware_link.py`
   (228-byte state out, 240-byte command in, field order, frames, the
   watchdog); a change there is a change in SyncAI-Robot-Control's `--backend
   sim` too, said so in the commit message. Keep the udp mode ROS-free:
   `ros/ros_node.py` knows nothing of it, and `python3 -m syncai_sim` refuses
   it without `--no-ros`.
6. **Layers stay layers.** One subpackage per layer; every `__init__.py` is a
   docstring and re-exports nothing (import by full path). The picture
   (`visuals/`, `viewer/`) is never read by anything that senses or stands;
   a backend returns a `JointCommand` and only `actuator/motors.py` writes
   the joints; only `sensor/` reads the body for others.
7. **Comments explain why**, as in the workspace, and record measurements and
   rejected alternatives. A bare number without its reason is out of place.
8. Python: `ruff.toml` (the workspace's rule set, py310, 100 columns). Tests:
   `pytest tests` in the container; `test_simulator_synthetic.py` needs the
   ROS environment sourced (it runs the planner CLI through `ros2 run`).

## Where things happen

Paths are under `syncai_sim/`; `docs/architecture.md` draws the layers.

| Need | Place |
|---|---|
| map -> voxel world (parse, mesh, plates, cache); ray cast / ground / blocked | `world/`: `build.py`, `btree.py`, `greedy_mesh.py`; `voxel_grid.py` (numba); `storeys.py` |
| the feet's floor heights, flights fitted as staircases; the voxels -> boxes under the feet | `terrains/`: `surface_heights.py`, `stairs.py`, `terrain_patch.py` (numba, `tests/test_terrain_patch.py`); stairs the track never climbed: `config/stair_walks/<map>.yaml`; slopes across voxel layers: `config/ramps/<map>.yaml` |
| URDF -> MJCF, mocap weld, standing pose, the joint order (`LEGS` x `JOINTS`) and default pose (`Q_DEFAULT`), the body's physics knobs (`motor_model`, `trunk_inertial`, contact, `actuator`) | `entity/robot.py` (`RobotConfig`; `DEFAULT_STANDING` was solved by FK, feet 0.372 m under the hips); their A/B `scripts/gait_bench.py` |
| MjSpec assembly, bitmasks, groups, the MJB scene cache | `scene/scene.py` (`scene_info` reads a loaded model back by name; `scene_key` lists what the cache depends on; the site's picture is not in the model) |
| the body's measurements (the body IMU, joints, the base's truth, foot contacts), posture, the trunk's contact; the lidar's IMU | `sensor/proprioception.py` (`Sensors`, one float32 record per step; `LidarImu`; `tests/test_sensors.py`) |
| the lidar pattern, prop rays | `sensor/lidar.py`, `sensor/prop_rays.py`; the recorded MID360 bank `config/lidar/mid360_pattern.npz` (built by `scripts/lidar_pattern.py` through `sensor/livox_bag.py`, which reads a rosbag2 sqlite without ROS) |
| the motors: the PD closed every step, the motor clamp, the command delay and torque filter, `JointCommand` | `actuator/motors.py` (`tests/test_motors.py`) |
| the policies (ported from the robot's controller) | `locomotion/policy.py` (`tests/test_policy.py`; names the source of every constant) |
| the motion states (stand / lie / damping / the policy at 50 Hz), falls | `locomotion/legged.py`: each step returns a `JointCommand` for the motors; the `integrator` and delays are `SimConfig` in `envs/simulator.py` |
| the step rule, gates, collisions (kinematic) | `locomotion/kinematics.py` (pure Python, `tests/test_kinematics.py`) |
| the hardware link: the body's record out, a controller process's command in (`locomotion: udp`); the stand-in controller | `locomotion/hardware_link.py` (its docstring is the protocol's spec; `tests/test_hardware_link.py`); `config/params/controller_profile.yaml`; the bench `scripts/controller_bench.py` |
| the gait controller on driver_manager's UDP link (the session); services and `mode` (driver_link: fake); the same surface forwarding to a controller process (udp) | `driver/controller_link.py` (`tests/test_controller_link.py`), `config/params/driver_manager_sim.yaml`; `driver/fake_driver.py`; `driver/udp_driver.py` |
| one physics tick, scan, spawn, teleport | `envs/simulator.py` |
| the viewer (viser: the site as GLB tiles once, poses at 20 Hz, drawn by the host's browser; nothing renders in the container), its panel's driving | `viewer/web_view.py`, `viewer/teleop.py` |
| the viewer's look (tiles, intensity shading, the Poisson and TSDF surfaces, colours) | `visuals/`: `site_tiles.py` (10 m tiles as GLB, a texel per voxel face), `voxel_intensity.py`, `pcd_mesh.py`, `tsdf_mesh.py`, `visual.py` (picture only, never sensing); the site, storey and layer switches are the panel in `viewer/web_view.py` |
| Foxglove (map clouds, joint_states, bridge) | `visuals/map_cloud.py`, `config/foxglove/bridge.yaml`, `scripts/foxglove.sh`; mesh URIs in `launch/rsp.launch.py` |
| publishers, subscribers, the main loop | `ros/ros_node.py`; CLI in `__main__.py` (sets `OPENBLAS_NUM_THREADS=1` before numpy); params in `config.py` |
| the session, the test driver | `scripts/run_session.sh`, `scripts/nav_test.py` |

## Running and testing (inside the container)

```bash
docker compose up -d && docker compose exec sim bash   # up starts the sim + viewer (scripts/sim_entry.sh; SIM_AUTOSTART=0 before the session)
source /opt/ros/humble/setup.bash && source /opt/ros2_rust_underlay/install/setup.bash && source install/setup.bash
/home/syncrobotic/sim/scripts/run_session.sh         # logs: log/stack/sim01/sim/<pane>/current; the picture: http://localhost:8080
python3 /home/syncrobotic/sim/scripts/nav_test.py --goal 3 0 0
python3 -m pytest /home/syncrobotic/sim/tests
# the sim as a controller's hardware (locomotion udp, ROS-free; docs/architecture.md):
python3 /home/syncrobotic/sim/scripts/controller_bench.py --profile /home/syncrobotic/sim/config/params/controller_profile.yaml --fake-controller --seconds 5
```

`test_simulator_synthetic.py::test_stairs_need_the_policy` fails on `dev`
too (2026-11-03, the kinematic step rule walks into the synthetic flight): a
red run with only that test is not a regression of your change.

Do not run `docker compose build` or the session on your own after an edit;
leave that to the user unless asked — the session starts the workspace's whole
nav stack.
