# SyncAI-Robot-Simulation

A MuJoCo simulation of the SyncAI robot (the G23 quadruped) in a **real, mapped
site**, built so the navigation stack in
[SyncAI-Robot-Workspace](https://github.com/chungweeeei/SyncAI-Robot-Workspace)
— above all its 3D branch: the `Octo3D` planner, the `TerrainLayer`, stair
mode and the `*_3d.xml` behaviour trees — can run **end to end without a
robot**. One container holds the workspace's own nav stack and the simulator,
on a DDS domain of its own; the viewer is served to a browser.

What the simulator is, in one paragraph: the site's `octomap.bt` (written by
`syncai_mapping`'s `clean_map --octomap 1`) becomes a voxel world; the G23
walks through it on `cmd_vel` with its own ONNX locomotion policies — PPO for
flat ground, HimLoco when the stair policy has been requested — driving twelve
PD joints the way the robot's Rust controller (`g2_deploy_rs`) does;
a Livox MID360 is ray-cast against the voxels at 10 Hz, firing the scan
pattern recorded from the real one; and one ROS 2 node
publishes exactly what the Livox driver, Point-LIO, the localizer and
`driver_manager` publish on the robot, so the planner, costmaps, controller,
task_runner and lio_bridge run **unchanged, from the workspace's own
`install/`**.

Two locomotions, `locomotion` in the params: **policy** (the default, above)
and **kinematic** — the robot as a rigid statue carried along the ground by
rules (`kinematics.py`), which is what the sim was before the policies were
available and stays for runs that want the step rule's verdicts instead of a
gait's. What neither is: a simulation of the MPC controller (policy 2 and
motion state MPC walk on PPO).

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
  pcd_mesh.py                        map.pcd -> a smooth viewer surface (Poisson, open3d), cached
  visual.py                          the viewer's look: colour by height, flat-shaded voxel faces, sky
  storeys.py                         the site's storeys from poses.txt (the viewer's 1F / 2F switch)
  map_cloud.py                       map.pcd thinned per storey for Foxglove (sim/map_cloud/storey<N>)
config/foxglove/bridge.yaml        foxglove_bridge's settings (port 8765, the .STL asset allowlist)
scripts/foxglove.sh                foxglove_bridge (+ robot_state_publisher) for a sim run by hand
  voxel_grid.py                      the numba ray caster, ground query, blocking test
  robot.py                           the workspace URDF -> MJCF: articulated, or rigid with a mocap handle
  scene.py                           robot + site (in 10 m tiles) + props (+ the terrain pool) in one MjSpec; the compiled model cached as MJB
  policy.py                          g2_deploy_rs's PPO / HimLoco runners, ported (onnxruntime)
  legged.py                          motion state -> policy / stand / lie / damping, PD, the fall check
  terrain_patch.py                   the voxels around the robot as collision boxes for its feet
  kinematics.py                      the kinematic base: ground following, the step rule, blocking, gates
  lidar.py, prop_rays.py             the MID360: its recorded pattern fired frame by frame, rays from where it was when each fired; analytic rays for props
  livox_bag.py                       CustomMsg frames straight out of a rosbag2 sqlite (no ROS) -> the pattern bank
config/lidar/mid360_pattern.npz    that bank, from the workspace's record/dp (config/lidar/README.md)
scripts/lidar_pattern.py           rebuilds it from any MID360 recording
  fake_driver.py                     set_motion_key / set_policy_mode / mode
  simulator.py                       the ROS-free core (step, scan, imu, spawn, teleport)
  ros_node.py, viewer.py, __main__.py the node, the viewer (a GLFW window of our own: frame cap, tiles within reach) with teleop, the CLI
policy/                            the G23's ONNX policies, gitignored: copy them in (policy/README.md)
tests/                             pytest; ROS-free except test_simulator_synthetic (needs the ROS env + workspace install)
docs/findings.md                   what the sim found about the navigation stack
cache/                             per-map build products (gitignored): the world, the compiled scene (scene-*.mjb, 628 MB on dp), the decimated robot meshes (robot/)
```

## Setup

Prerequisites on the host: Docker (Desktop on a Mac; the image is linux/arm64
like the robot's) and a checkout of the workspace **that has been built** —
the sim runs whatever is in its `install/`. Build it with that repo's
`docker-compose.build.yml`; the image here derives from the same
`syncai-robot-base`, so the artefacts load unchanged. The map to simulate must
have an `octomap.bt` (`map/<name>/`, see the workspace's `syncai_mapping`).

```bash
cp .env.example .env            # ROBOT_WS = path to the workspace checkout (default: ../SyncAI-Robot-Workspace)
docker compose build            # once; FROM syncai-robot-base:latest, which the workspace's compose builds
docker compose up -d
docker compose exec sim bash    # cwd is the workspace root, like every pane on the robot
```

**Isolated or integrated.** By default the container is isolated: a bridge
network, ROS_DOMAIN_ID 42, this repo's `cyclonedds.xml` and `sim01.ini`. To
run `syncai_backend` and the frontend against the sim, `.env` switches it to
the robot's own topology — `SIM_NETWORK_MODE=host`, `SIM_ROS_DOMAIN_ID=1`, the
workspace's `config/cyclonedds.xml` and an instance file of the workspace's
(`SIM_INSTANCE_INI`, e.g. `config/instances/robot01.ini`) — the same three the
backend's compose uses, so it finds the sim as it finds robot01. Both DDS
configs are loopback-only, so nothing reaches the network in either mode; the
integrated one must never run on a robot's Jetson, where robot01 shares that
loopback on domain 1.

`SIM_INSTANCE_INI` (default `config/instances/sim01.ini`) picks the map (`[map] name`) and the spawn pose
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
plates, a 133 MB voxel grid) and compiles the scene (~5 s; cached as an
MJB, so later starts take about a second). The viewer follows the robot:
left-drag orbits, scroll zooms, `9` cycles the storeys, keys `0 2 3 4`
toggle geom groups (0 the site, 2 props, 3 collision shapes, 4 the mocap
handle), `h` lists the teleop keys. In the session (no `--teleop`: the nav
stack owns `cmd_vel`) only `t` and `h` answer: `t` puts the robot back on the
spawn, standing.

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
| `driver_manager` and `robot_state` | **the real ones** in the session: the sim is the gait controller at the far end of driver_manager's UDP link (`driver_link: udp`, `controller_link.py`: AXES / MODE / ESTOP in; IMU, joints, MODE_STATE and BMS_V2 telemetry out), so `battery_state`, `motor_states`, `mode`, `safety_locked` / `set_safety_lock` and RobotState come from the workspace's own code; the sim answers sys_manager's `get_mode` (AUTO) | `config/params/driver_manager_sim.yaml` is the driver's params file with the controller on loopback; `mode_switch_delay` is the controller's response time. `driver_link: fake` (the default outside the session) keeps the sim's own services and `mode` |
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
`TerrainLayer` had made lethal — see `docs/findings.md`). Enclosed floor holes
of up to 25 columns — floor the lidar never saw, 3965 on dp — are filled at
their rim's height (`floor_hole_max_cells`). That grid is the
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

**Lidar** (`lidar_pattern`, `lidar.py`). The scan fires the directions the
robot's own MID360 fired: twenty consecutive frames of the workspace's
`record/dp` (`livox_ros_driver2` CustomMsg, read out of the rosbag2 sqlite by
`livox_bag.py` with no ROS), ~16 k samples a frame that returned, cycled
frame after frame, so the clouds have the sensor's rosette — four lasers
sweeping 1.2° a sample, dense along the sweep and sparse across it, never the
same spots in two scans — instead of the uniform random directions of before
(`random` keeps those). Every sample has its firing time (its place in the
frame: the recording's per-point `offset_time` turned out not to be one,
`livox_bag.build_pattern` says why), and each ray is cast from where the
lidar was at that time during the 100 ms scan (`rays_along_motion`; the
simulator keeps the lidar pose of every physics step of the window): at
0.4 m/s the sensor moves 4 cm during a scan, turning at 1 rad/s 5.7°, and
the pattern smears over the world as it does on the robot. The cloud is then
expressed in the end-of-scan body frame — motion-compensated, which is what
Point-LIO's `body_cloud` is. The 20 k rays cost the same 1.3 ms.
`scripts/lidar_pattern.py` rebuilds `config/lidar/mid360_pattern.npz` from
any few seconds of MID360 recording (the pattern is the sensor's, not the
site's).

**Walking** (`locomotion: policy`). `policy.py` is a line-by-line port of the
runners in `g2_deploy_rs` (`src/policy_runner.rs`, `src/himloco_policy_runner.rs`:
observation layout and scales, joint order, action scaling and clamps, PD
gains) running the files its `main.rs` loads by default — `ppo/policy7.onnx`
for `set_policy_mode 0`, `isaacsim/policy_0812.onnx` for 1, PPO for anything
else — at 50 Hz, with the PD closed on every 2 ms physics step and clipped to
the URDF's joint efforts (`legged.py`). The motion state picks the controller
state: Locomotion / MPC walk the policy, Stand and Lie Down ease to a pose,
Damping and E-stop leave only kd. The posture check is the controller's
(|roll| > 30°, |pitch| > 45°): a fall puts the driver in Damping, as the real
controller locks itself; a teleport or `initialpose` stands the robot up again.
The feet stand on `terrain_patch.py`: ~300-700 voxel boxes around the robot,
moved as it walks, with the map's surface-only artefacts closed first (hollow
stair treads, one-voxel slots across a flight, one-column holes — each with
the run that found it in the module), and each column's top box stands at
the floor's measured height inside its voxel (`surface_heights.py`: the median
of `map.pcd` there, then an edge-preserving 5 x 5 median over the same surface,
which also fills the 43 % of floor voxels with no point inside) — the 0.1 m
grid turns dp's tilted floor into a strip of random blocks every ~13 m, where
PPO fell 4 times out of 4, and the per-voxel medians alone were still rough
(8.7 % of neighbouring columns more than 2 cm apart). Smoothed, dp's open floor
has its largest step between neighbours at ~1 cm; risers and kerbs keep their
edges. A voxel with no point inside takes its own layer's measured neighbours
first (a tread filled into the octomap otherwise took the tread under it), a
13 x 13 window when the 5 x 5 has none, and its voxel's middle when neither
has (its top is 5 cm high on average: on dp's last step to 2F that made a lip
PPO fell on). Last, steps between two mapping passes that drifted apart in
height (dp's stairwell sits ~0.13 m under the rest) are ramped wherever the
mapping track walked level over them: a 0.16 m "step" onto 2F that the real
robot never meets, where PPO fell twice at 0.7 m/s, became a 1 m ramp. And
every flight the mapping track climbed is fitted as a staircase of equal
risers and treads between the landings it joins: per voxel, a riser across a
column made a half step of the column (dp's flight 3 read 2.56 / 2.61 / 2.69
/ 2.91 ... 3.06 / 3.17 / 3.23). That did not decide the falls: HimLoco fell
on dp's stairwell in 7 of 20 climbs before the fit and 3 of 10 after
(2026-10-25), at the same 45 deg pitch limit g2_deploy_rs's rl_control.rs
has, on a 33 deg flight. A stair the track never climbed but the planner
detects and routes over (its `stair_detect`, 2026-10-25) gets the same fit
from a stand-in walk in `config/stair_walks/<map>.yaml` (dp: the middle
stair). A floor that slopes across voxel layers keeps a lip at each (dp's
1F hump east of the middle stair: 9 and 6 cm, where PPO fell 5 times in 19
crossings); `config/ramps/<map>.yaml` boxes such places, and their tops
become a 0.7 m box mean. Faults
in the octomap itself (unseen treads, two-level landings, a shut door) are
fixed in the map: the workspace's `docs/octomap-repair.md`. With those and
these, 10 of 10 alternating 1F <-> 2F navigations on dp (2026-10-24). On flat stretches
of the mapping track the world build also cuts one-voxel-thick ridges (the
corridor fence of finding 1 had kept 0.2-0.3 m; the robot fell on it both
ways). There is no step rule in this mode:
whether a flight is climbed is up to the policy. Measured (2026-10-23):
PPO 0.4 m/s at a 0.5 command, 1.0 rad/s yaw; HimLoco climbs the planner CLI's
synthetic flight and dp's first flight to the landing (1.2 m); PPO falls at
the first riser. A collision is the trunk touching the site.

The walking body is lower than the statue: `lidar_top_imu` rides **~0.44 m**
over the ground in PPO's gait (0.444 standing at its default pose on dp),
where the planner's and controllers' `lidar_height` is 0.481 — the value the
kinematic body is built to. One of the two is not the robot; worth measuring
on the real G23 before trusting the planner's ground estimate in this mode.

**The picture** (`visual_mesh`, `--visual-mesh`). By default the viewer draws
the octomap's voxels; `pointcloud` draws instead a smooth surface reconstructed
from the map's `map.pcd` (`pcd_mesh.py`:
points kept only on the cleaned voxels, 4 cm, screened Poisson, 1.5 M
triangles; dp builds in ~3.5 min once, then loads from the cache; build it
ahead with `python3 -m syncai_sim.pcd_mesh map/dp`). `voxels` draws the
cubes the sim actually senses and stands on. Both are coloured by height —
floors and treads along a ramp, walls a pale copy, undersides grey — and lit
by two oblique suns (`visual.py`). The site is cut into 10 m tiles and the
viewer draws only those within 30 m of the robot on the storeys it shows
(`SiteView`), at most 10 frames a second; the robot's own STLs are drawn
decimated (274 k → 29 k triangles, `robot.py`). Only the picture: the feet
stand on the measured floor (`surface_heights.py`), the voxels are drawn at
their tops, so the robot can look a few centimetres off the drawn floor.

**Foxglove** (the GPU view, on the host). The MuJoCo viewer renders in
software; for a detailed picture of the site and the stack, `foxglove_bridge`
serves everything to Foxglove (desktop app or app.foxglove.dev) at
`ws://localhost:8765`: the session starts it (window `foxglove`); with the sim
run by hand as a ROS node (`python3 -m syncai_sim --teleop`, not `--no-ros`),
`scripts/foxglove.sh` starts it, robot_state_publisher and lio_bridge (which
joins the sim's TF to the URDF's, as on the robot). The sim adds for it
the site's point cloud per storey (`sim/map_cloud/storey1`, `storey2`:
`map.pcd` on the cleaned voxels, 6 cm, with intensity; latched; dp 3.7 M
points), and the legs' `joint_states`; `robot_description` carries its meshes
as `package://` URIs so the bridge serves them. In Foxglove: Open connection
→ Foxglove WebSocket → `ws://localhost:8765`, then import
`config/foxglove/layout-<robot_id>.json` (layout menu → Import from file;
`scripts/foxglove.sh` writes it from `layout.template.json` for the INI's
robot_id): a 3D panel following `<robot_id>/base_link` with both map clouds (shaded by intensity), the
robot model, the live scan, and `sim/status` / `mode` beside it; the stack's
plans and costmaps are one toggle away in the panel's topic list.

**Storeys.** `storeys.py` finds them in `poses.txt` (where the mapping robot
spent most of its keyframes; dp: 1F at −1.0…−0.2 m, 2F at 4.4…5.6 m, split at
4.18 m, the flight and its landings with 1F). Each storey is drawn up to
`visual_max_height_above_column_floor_m` over its own floor, so an upper
floor is drawn whole, and the viewer shows by default the robot's storey and
those below it (the floor above never hides the robot; walking up brings the
next one in). Key `9` cycles auto / 1F / 2F / … / all.

**Robot.** The workspace's `G23.urdf` is loaded by MuJoCo itself with two
compiler switches injected (`fusestatic=false` so `base_link` can get a free
joint, `discardvisual=false` to keep the STL visuals). Walking, it keeps its
joints (armature 0.01), every leg primitive collides with the terrain patch
and the trunk with the patch and the props, and a gyro and accelerometer sit
on `lidar_top_imu`. Kinematic, it is edited: the twelve
leg joints are removed and baked at a standing angle, the leg and foot
collision primitives are turned off (a carried statue must not catch a riser),
the trunk's two boxes collide with props only, and a mocap body is welded to
`base_link`. `kinematics.py` moves the mocap; `lidar_top` and `lidar_top_imu`
stay as bodies so the sensor extrinsics are the URDF's, read, not copied. The
body stands with `lidar_top_imu` 0.481 m over the ground — the planner's and
controllers' `lidar_height` — and pitches with the ground, so the planner's
known pitch error on stairs (`stair_links.cpp`) is present.

**The step rule** (`kinematics.py`, kinematic mode only). One step of at most `max_step_height`
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
* **The viewer is a GLFW window of our own, not `launch_passive`.** On
  llvmpipe the passive viewer paced itself on a vsync Xvnc does not have, so
  it rendered flat out (7-11 fps for ~3 cores, moving or not), its `sync()`
  copied the model and the 18 MB MjData 60 times a second (12-15 ms each,
  most of the main thread), every letter was one of its display toggles, and
  its window and held keys were only reachable through libX11 by ctypes.
  `viewer.py` is one thread with a 10 fps cap, state-only copies at 20 Hz
  (0.3 ms), `glfw.get_key`, and the site drawn by tile within 30 m.
* **Where the CPU went (2026-10-24).** The node in the session used ~7 cores;
  three sinks, none of them physics (a walking step is 0.23 ms): numpy's
  OpenBLAS workers spin-waiting between the 50 Hz 4x4 TF inversions (350 %
  for the headless node; `OPENBLAS_NUM_THREADS=1`, set in `__main__.py`:
  47 %), the viewer above, and `PointCloud2.data` set from `bytes`, which the
  generated setter validates byte by byte (13.7 ms of a 20 ms scan publish;
  `array('B')`: 0.03 ms). After: ~1.2 cores, scan 3.7 ms, `Simulator()` in
  1.1 s from the caches.
* **The workspace is not modified.** robot_state_publisher is started by this
  repo's launch file, the INI is bind-mounted, the specs are this repo's. The
  two things the sim would like from the workspace — a `--write-poses` for the
  planner CLI's synthetic scenes (so they have stair links) and a CI stage —
  are separate changes there.
* **No `use_sim_time`.** The whole stack runs on the wall clock and so does the
  sim (fixed 2 ms physics steps paced to real time, catching up to 0.1 s at
  once). A walking step costs ~0.25 ms on dp, policy inference included.
* **The policies are ported, not the controller run.** `g2_deploy_rs` has a
  MuJoCo mode of its own (UDP to a Python script), but running it here would
  need a Rust toolchain in the image and its UDP bridge between the stack and
  the sim; the runners are ~150 lines of Python against the same ONNX files.
  The price: a change to a runner in `g2_deploy_rs` must be mirrored in
  `policy.py` (it names the source files).

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
