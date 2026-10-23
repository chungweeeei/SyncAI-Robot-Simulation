# SyncAI-Robot-Simulation

A MuJoCo simulation of the SyncAI robot (the G23 quadruped) in a **real, mapped
site**, built so the navigation stack in
[SyncAI-Robot-Workspace](https://github.com/chungweeeei/SyncAI-Robot-Workspace)
— above all its 3D branch: the `Octo3D` planner, the `TerrainLayer`, stair
mode and the `*_3d.xml` behaviour trees — can run **end to end without a
robot**. One container holds the workspace's own nav stack and the simulator,
on a DDS domain of its own; the viewer is served to a browser.

What the simulator is, in one paragraph: the site's `octomap.bt` (written by
`syncai_mapping`'s `clean_map --octomap 1`) becomes a voxel world; a rigid G23
in a standing pose is carried through it kinematically on `cmd_vel`, following
the ground and climbing stairs only when the stair policy has been requested;
a Livox MID360 is ray-cast against the voxels at 10 Hz; and one ROS 2 node
publishes exactly what the Livox driver, Point-LIO, the localizer and
`driver_manager` publish on the robot, so the planner, costmaps, controller,
task_runner and lio_bridge run **unchanged, from the workspace's own
`install/`**.

What it is not: a legged-locomotion simulation. There is no gait policy here
(the robot's HIMLOCO / MPC controller is a black box on the real robot), so the
legs are decoration and the body is placed, not balanced. The physics engine
contributes contacts with movable props, the GUI, and a place to put a policy
if one ever becomes available.

## Layout

```
Dockerfile, docker-compose.yml     one container: FROM the workspace's syncai-robot-base + MuJoCo + numba + Xvnc/noVNC
.env.example                       ROBOT_WS (the workspace checkout to mount) and the display
config/instances/sim01.ini         the instance INI bind-mounted over the workspace's config/system.ini
config/cyclonedds.xml              the robot's DDS config with Domain Id="any" (ROS_DOMAIN_ID 42 in compose)
config/params/sim_params.yaml      every sim_node parameter, with the stack value each one mirrors
config/sessions/start_sim.yaml     the session spec (start_nav.yaml with the hardware replaced)
launch/rsp.launch.py               robot_state_publisher alone (bringup.launch.py would start the lidar)
scripts/run_session.sh             runs a spec as process groups with logs, no byobu (the CI smoke test's loop)
scripts/vnc.sh                     the container's display: Xvnc :1 + noVNC on 6080
scripts/nav_test.py                sends a NavigateToGoal and reports collisions / result
syncai_sim/                        the Python package (python3 -m syncai_sim)
  btree.py                           OctoMap .bt reader (no octomap dependency)
  greedy_mesh.py                     voxels -> boxes
  world.py                           octomap.bt -> VoxelGrid + visual mesh + collision plates, cached
  voxel_grid.py                      the numba ray caster, ground query, blocking test
  robot.py                           the workspace URDF -> rigid MJCF with a mocap handle
  scene.py                           robot + site + props in one MjSpec
  kinematics.py                      the base: ground following, the step rule, blocking, gates
  lidar.py, prop_rays.py             the MID360 pattern; analytic rays for props
  fake_driver.py                     set_motion_key / set_policy_mode / mode
  simulator.py                       the ROS-free core (step, scan, imu, spawn, teleport)
  ros_node.py, viewer.py, __main__.py the node, the viewer with teleop, the CLI
tests/                             pytest; ROS-free except test_simulator_synthetic (needs the ROS env + workspace install)
docs/findings.md                   what the sim found about the navigation stack
cache/                             per-map build products (gitignored)
```

## Setup

Prerequisites on the host: Docker (Desktop on a Mac; the image is linux/arm64
like the robot's) and a checkout of the workspace **that has been built** —
the sim runs whatever is in its `install/`. Build it with that repo's
`docker-compose.build.yml`; the image here derives from the same
`syncai-robot-base`, so the artefacts load unchanged. The map to simulate must
have an `octomap.bt` (`map/<name>/`, see the workspace's `syncai_mapping`).

```bash
cp .env.example .env            # ROBOT_WS = path to the workspace checkout (default: ../SyncAI-Robot-Workspace-octomap-planner-3d)
docker compose build            # once; FROM syncai-robot-base:latest, which the workspace's compose builds
docker compose up -d
docker compose exec sim bash    # cwd is the workspace root, like every pane on the robot
```

`config/instances/sim01.ini` picks the map (`[map] name`) and the spawn pose
(`[initial_pose]`; z is the ground height, a hint for the floor). It is a
single-file bind mount: edit it in place, do not rename over it.

## Running

Inside the container (every command below is run there, from the workspace
root, after sourcing the ROS environment):

```bash
source /opt/ros/humble/setup.bash && source /opt/ros2_rust_underlay/install/setup.bash && source install/setup.bash
/home/syncrobotic/sim/scripts/vnc.sh               # the display; then open http://localhost:6080/vnc.html?autoconnect=1&resize=remote on the host (the picture follows the browser window)
/home/syncrobotic/sim/scripts/run_session.sh       # the whole stack + the sim, logs under log/stack/sim01/sim/<pane>/current
python3 /home/syncrobotic/sim/scripts/nav_test.py --goal 3.0 0.0 0.0     # send the robot 3 m ahead
/home/syncrobotic/sim/scripts/run_session.sh --stop
```

The first run on a map builds its cache (dp: ~12 s — parsing 1.6 M occupied
leaves, merging faces into a 3.5 M-triangle viewer mesh, 18 k collision
plates, a 133 MB voxel grid); later runs load it in a second. The viewer
follows the robot; keys `0`–`5` toggle geom groups (0 the site, 1 the robot,
2 props, 3 collision shapes, 4 the mocap handle).

The simulator alone, without the stack:

```bash
python3 -m syncai_sim --no-ros --teleop            # drive with the arrow keys in the viewer window (press h for help)
python3 -m syncai_sim --headless --seconds 30      # as a ROS node, no window (CI)
python3 -m syncai_sim.world map/dp                 # (re)build a map's cache and print its stats
python3 -m pytest /home/syncrobotic/sim/tests      # unit tests; the synthetic-stairs one needs the workspace install
```

Parameters are in `config/params/sim_params.yaml` and can be overridden the
ROS way: `python3 -m syncai_sim --ros-args --params-file ... -p mode_switch_delay:=4.0`.

## What the stack sees

| On the robot | Here | Notes |
|---|---|---|
| Livox driver `livox/lidar` + `livox/imu` | `livox/imu` only (gyro + gravity) | the cloud goes out as Point-LIO's output directly (below); a raw `CustomMsg` mode that lets the real Point-LIO run is the planned second phase |
| Point-LIO `pointlio/body_cloud`, `pointlio/lio_odom`, TF `pointlio_odom -> pointlio_body` | the same names, frame `<id>/pointlio_body` = the URDF's `lidar_top_imu`, cloud rendered at the end-of-scan pose (= motion-compensated) | `lio_bridge` reads the **topic** `lio_odom`, not TF |
| localizer TF `map -> pointlio_odom`, `initialpose` | the same; `initialpose` teleports the robot and moves `map -> pointlio_odom` so the odom chain stays continuous (covariance[14] != 0 means the z is meant, as in the localizer) | the real localizer's correction has the same shape |
| `driver_manager` `set_motion_key`, `set_policy_mode`, `mode` | the same services and `mode` (`[policy, motion_state]`, reliable, 2 Hz + on change), with a settable `mode_switch_delay` | nothing in the nav stack reads `imu`/`battery_state`/`motor_states`, so they are not published |
| — | `sim/ground_truth` (Odometry in `map`), `sim/status` (DiagnosticArray) | for test scripts |

The frames, rates and QoS were read off the consumers in the workspace; the
params file names the value each setting mirrors. Three couplings to know:

* **`mode_switch_delay` must stay below `StairMode`'s `stair_policy_settle_time`
  (3 s).** The controller sends policy 1 before a flight and does *not* wait
  for `mode` to report it — it settles 3 s and climbs. 1 s (default) lets a
  correct controller pass; 4 s is the "slow driver" negative test. It also
  restores the policy `mode` reported *before* the switch, so the sim boots
  reporting policy 0 and republishes `mode` at 2 Hz.
* **The robot boots in Locomotion (1).** The real controller boots in 8
  (UNKNOWN) and `move_3d.xml` never sends a motion key; a faithful boot would
  never move. `initial_motion_state` is the knob.
* **Goals carry a z.** `Octo3D`'s `goal_z_mode` is `request` and the
  controller's goal check has `goal_z_tolerance` 0.5, so a goal needs the
  ground height of its floor; `nav_test.py` uses the robot's current one
  unless `--goal-z` is given.

## The world, the robot, the rules

**World.** `octomap.bt` is parsed directly (`btree.py`; the format is two
bytes per inner node) into a dense boolean voxel grid, after two cleanups
that make the voxel world agree with what the mapping robot physically
passed through: one-column posts 0.2 m or more above every neighbour are
clipped (the mapper's own feet), and whatever stands inside the body volume
the mapper swept along `poses.txt` is erased (`track_clear_radius_m`; on dp a
one-voxel-thick 0.3 m "fence" across the only corridor to the stairs, which
`TerrainLayer` had made lethal — see `docs/findings.md`). That grid is the
truth for the lidar, the ground and blocking — all three are numba kernels in
`voxel_grid.py`, and 20 k rays take about 1.3 ms. Lidar returns on surface
voxels are placed at the voxel's mid-height (the real floor is somewhere
inside the voxel), which keeps the 0.1 m steps of a mapped floor from reading
as dynamic obstacles. MuJoCo never sees the voxels:
it gets a *visual* copy (the exposed faces, greedy-merged into rectangles,
drawn with `contype 0`) and the few large merged boxes as **collision plates**
so props have floors to stand on. This split is deliberate: MuJoCo collides a
mesh as its convex hull, `mj_multiRay` scans every geom per ray (18 k plates
would cost hundreds of milliseconds a scan) and even against the site as one
mesh its BVH needed 325 ms for 20 k rays; against the voxels the same 20 k
rays are 1.3 ms. Numbers for dp are in `cache/dp/world.json`.

**Robot.** The workspace's `G23.urdf` is loaded by MuJoCo itself with two
compiler switches injected (`fusestatic=false` so `base_link` can get a free
joint, `discardvisual=false` to keep the STL visuals), then edited: the twelve
leg joints are removed and baked at a standing angle, the leg and foot
collision primitives are turned off (a carried statue must not catch a riser),
the trunk's two boxes collide with props only, and a mocap body is welded to
`base_link`. `kinematics.py` moves the mocap; `lidar_top` and `lidar_top_imu`
stay as bodies so the sensor extrinsics are the URDF's, read, not copied. The
body stands with `lidar_top_imu` 0.481 m over the ground — the planner's and
controllers' `lidar_height` — and pitches with the ground, so the planner's
known pitch error on stairs (`stair_links.cpp`) is present.

**The step rule** (`kinematics.py`). One step of at most `max_step_height`
(0.17) is a kerb and is walked in any policy; a second step within
`stair_tread_window` (0.45 m) of travel is a staircase and needs policy 1, as
does any single step taller than that; a drop deeper than `max_drop` (0.3) is
never walked; a descending step in policy 1 needs `vx < 0`
(`require_backward_descent`) — the controllers walk stairs down backwards and
this is the only way the sim can tell. The planner CLI's synthetic stairs have
0.17 m risers, exactly the limit, which is why single-step height alone cannot
decide. A blocked move (wall, too-tall riser, a voxel in the trunk's height
band) leaves the base where it is and counts a **collision**; `nav_test.py`
fails on that count.

**Props.** `props_json` in the params lists boxes and cylinders: a free body is
pushable (it stands on the plates, is pushed by the trunk and seen by the
lidar); a body with `waypoints` walks between them as a kinematic "person" —
food for the planner's dynamic overlay and TerrainLayer's `obstacle_memory`.

## Logs and what to look for

`log/stack/sim01/sim/<pane>/current` under the workspace:

| pane | a healthy line |
|---|---|
| `sim_node` | `[sim] ready in … s`, `[sim] spawned at …`, then `realtime x1.00  scan ~20 ms` every 10 s |
| `lio_bridge` | `localization bridged: map -> sim01/odom = …` |
| `planner` | `surface map: … columns`, `N stair links from … keyframes` |
| `controller` | `terrain_layer: 3D map … -- layer on` |
| `task_runner` | `3D map (planner_server/Octo3D/map_available) -- goals run the *_3d.xml trees` |
| `map_server`, `keepout` | the "3D-only map … starting idle" warnings |

`ros2 topic hz /sim01/pointlio/body_cloud` is ~10, `/sim01/odom` ~20;
`ros2 run tf2_ros tf2_echo map sim01/pointlio_body` shows z = ground + 0.481
and a pitch of 0.25 rad on flat ground (the lidar mount).

## Negative tests

```bash
# a driver too slow for StairMode: the robot stops at the first riser, refused_steps grows
python3 -m syncai_sim --ros-args --params-file ... -p mode_switch_delay:=4.0
# descent must be backwards: a goal downstairs is refused unless the controller reverses
#   (require_backward_descent is on by default)
# a crate on the path: add it to props_json and watch the overlay / is_path_valid replan
```

## Decisions worth knowing

* **One container, not a Mac-side simulator with a bridge.** The sim shares a
  DDS domain with the stack it tests; that stack is started here from the
  mounted workspace on domain 42 with loopback-only DDS, so nothing can be
  heard outside the container and a live robot01 is never disturbed.
* **The display is Xvnc + noVNC, not X11 to XQuartz.** XQuartz was tried: it
  offers only indirect GLX (OpenGL 1.4), below what MuJoCo's renderer needs.
  Rendering is Mesa's llvmpipe; the full dp mesh is ~180 ms a frame at
  1280x720, the default 3 m height band (`visual_max_height_above_column_floor_m`)
  ~120 ms. The lidar and the collision queries never depend on it.
* **The workspace is not modified.** robot_state_publisher is started by this
  repo's launch file, the INI is bind-mounted, the specs are this repo's. The
  two things the sim would like from the workspace — a `--write-poses` for the
  planner CLI's synthetic scenes (so they have stair links) and a CI stage —
  are separate changes there.
* **No `use_sim_time`.** The whole stack runs on the wall clock and so does the
  sim (fixed 5 ms physics steps paced to real time, catching up to 0.1 s at
  once).

## Findings

What the sim has shown about the stack so far — the planner and `TerrainLayer`
disagreeing about thin artefacts, FollowPath retries that get an empty plan —
is written up in `docs/findings.md`, separated from what was the sim's own
fault.

## Status

Phase 1 (this): ground-truth sensing mode, the full session, the test driver.
Verified on dp: a 3 m flat goal succeeds (4.8 s, zero collisions); the 49.5 m
route up the stair link climbs two flights and takes the first landing's
corners with the real controller's stair mode, zero collisions, and stops at
the next landing's turn (finding 4).
Phase 2: a raw mode publishing `livox/lidar` as `CustomMsg` with per-point
times so the real Point-LIO and localizer run; a TSDF mesh for looks.
Phase 3 (workspace side): planner CLI `--write-poses`, a CI smoke stage.
