# SyncAI-Robot-Simulation

A MuJoCo simulation of the SyncAI robot (the G23 quadruped) in a **real, mapped
site**, built so the navigation stack in
[SyncAI-Robot-Workspace](https://github.com/chungweeeei/SyncAI-Robot-Workspace)
— above all its 3D branch: the `Octo3D` planner, the `TerrainLayer`, stair
mode and the `move.xml` / `dock.xml` behaviour trees — can run **end to end without a
robot**. One container holds the workspace's own nav stack and the simulator,
on a DDS domain of its own; the picture is served to a browser.

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
`install/`**. The picture is drawn by the host's browser (`web_view.py`):
nothing renders in the container.

Two locomotions, `locomotion` in the params: **policy** (the default, above)
and **kinematic** — the robot as a rigid statue carried along the ground by
rules (`kinematics.py`), which is what the sim was before the policies were
available and stays for runs that want the step rule's verdicts instead of a
gait's. What neither is: a simulation of the MPC controller (policy 2 and
motion state MPC walk on PPO).

## Layout

```
Dockerfile, docker-compose.yml     one container: FROM the workspace's syncai-robot-base + MuJoCo + numba + viser; no display, no GPU
.env.example                       ROBOT_WS (the workspace checkout to mount), the viewer's port, the network mode
config/instances/sim01.ini         the instance INI bind-mounted over the workspace's config/system.ini
config/cyclonedds.xml              the robot's DDS config with Domain Id="any" (ROS_DOMAIN_ID 42 in compose)
config/params/sim_params.yaml      every sim_node parameter, with the stack value each one mirrors
config/sessions/start_sim.yaml     the session spec (start_nav.yaml with the hardware replaced)
launch/rsp.launch.py               robot_state_publisher alone (bringup.launch.py would start the lidar)
scripts/run_session.sh             runs a spec as process groups with logs, no byobu (the CI smoke test's loop)
scripts/nav_test.py                sends a NavigateToGoal and reports collisions / result
syncai_sim/                        the Python package (python3 -m syncai_sim)
  btree.py                           OctoMap .bt reader (no octomap dependency)
  greedy_mesh.py                     voxels -> boxes
  world.py                           octomap.bt -> VoxelGrid + visual mesh (as rectangles) + collision plates, cached
  pcd_mesh.py                        map.pcd -> a smooth viewer surface (Poisson, open3d), cached
  tsdf_mesh.py                       the keyframes' patches + poses.txt -> a TSDF viewer surface (open3d), thin structure kept, cached
  visual.py                          the viewer's look: colour by height, shaded by the lidar's intensity; or the photo
  voxel_intensity.py                 map.pcd -> the median intensity of each voxel, cached
  voxel_colour.py                    the mapping run's bag -> the colour the camera saw at each voxel (config/colour/<map>.npz, scripts/colour_map.py)
  bag_camera.py                      the bag's images, camera_info and /tf_static, read without ROS
  site_tiles.py                      the site for the browser: 10 m tiles, every voxel face a texel of its intensity or its colour, as GLB, cached
  storeys.py                         the site's storeys from poses.txt (the viewer's 1F / 2F switch)
  map_cloud.py                       map.pcd thinned per storey for Foxglove (sim/map_cloud/storey<N>)
config/foxglove/bridge.yaml        foxglove_bridge's settings (port 8765, the .STL asset allowlist)
scripts/foxglove.sh                foxglove_bridge (+ robot_state_publisher) for a sim run by hand
  voxel_grid.py                      the numba ray caster, ground query, blocking test
  robot.py                           the workspace URDF -> MJCF: articulated, or rigid with a mocap handle
  scene.py                           robot + collision plates + props (+ the terrain pool) in one MjSpec, no picture in it; the compiled model cached as MJB
  policy.py                          g2_deploy_rs's PPO / HimLoco runners, ported (onnxruntime)
  legged.py                          motion state -> policy / stand / lie / damping, PD (qfrc or MuJoCo actuators), the motor clamp, the fall check
scripts/gait_bench.py              the walking body against a scripted command, ROS-free: the A/B of the body's physics knobs
  terrain_patch.py                   the voxels around the robot as collision boxes for its feet
  kinematics.py                      the kinematic base: ground following, the step rule, blocking, gates
  lidar.py, prop_rays.py             the MID360: its recorded pattern fired frame by frame, rays from where it was when each fired; analytic rays for props
  livox_bag.py                       CustomMsg frames straight out of a rosbag2 sqlite (no ROS) -> the pattern bank
config/lidar/mid360_pattern.npz    that bank, from the workspace's record/dp (config/lidar/README.md)
scripts/lidar_pattern.py           rebuilds it from any MID360 recording
  fake_driver.py                     set_motion_key / set_policy_mode / mode
  simulator.py                       the ROS-free core (step, scan, imu, spawn, teleport)
  ros_node.py, __main__.py           the node and the CLI
  web_view.py, teleop.py             the viewer: the site and the robot sent to the browser once, poses streamed, drawn by the host's GPU; its panel's driving
policy/                            the G23's ONNX policies, gitignored: copy them in (policy/README.md)
tests/                             pytest; ROS-free except test_simulator_synthetic (needs the ROS env + workspace install)
docs/findings.md                   what the sim found about the navigation stack
cache/                             per-map build products (gitignored): the world, the compiled scene (scene-*.mjb, 20 MB on dp), the voxel intensities, the browser's tiles (140 MB), the decimated robot meshes (robot/)
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
(`SIM_INSTANCE_INI`, e.g. `config/instances/robot.ini`) — the same three the
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
/home/syncrobotic/sim/scripts/run_session.sh       # the whole stack + the sim, logs under log/stack/sim01/sim/<pane>/current; the picture at http://localhost:8080 on the host
python3 /home/syncrobotic/sim/scripts/nav_test.py --goal 3.0 0.0 0.0     # send the robot 3 m ahead
/home/syncrobotic/sim/scripts/run_session.sh --stop
```

The first run on a map builds its cache (dp: ~15 s — parsing 1.6 M occupied
leaves, merging faces into a 2.5 M-triangle viewer mesh, 19 k collision
plates, a 133 MB voxel grid) and compiles the scene (~1 s, 20 MB as an
MJB); the browser's tiles are built behind the running sim the first time
(dp: the voxel intensities in 4 s, the 179 textured tiles in 9 s) and cached
too, so later starts take about a second and the picture is up in two.

**The viewer** (`web_view.py`) is a page the host's browser draws:
`http://localhost:8080` (compose maps the container's `web_view_port`;
`SIM_WEB_VIEW_PORT` moves it; with a host network the sim listens on 8080
itself). The site's tiles and the robot's meshes go over once (dp: 1078
tiles, 2.5 M triangles, in a second), then only poses at 20 Hz, so it draws
at the browser's own frame rate with shadows, on the host's GPU; nothing
renders in the container. Left-drag orbits, scroll zooms; the panel has the
storey switch (auto: the robot's storey and those below it, so the floor
above never hides the robot), follow-robot (the camera's look-at follows,
the offset the mouse gave it stays), the last lidar scan, the terrain boxes
and fitted stairs under the feet, the lidar map per storey, the status, and
the teleport back to the spawn — wanted after a fall in the session too.
With `--teleop` it also has the driving sliders and the motion keys
(`teleop.py`; in the session the nav stack owns `cmd_vel`, so there are
none). `--headless` serves no viewer.

The simulator alone, without the stack:

```bash
python3 -m syncai_sim --no-ros --teleop            # drive from the panel at http://localhost:8080
python3 -m syncai_sim --headless --seconds 30      # as a ROS node, no viewer (CI)
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
  (UNKNOWN) and `move.xml` never sends a motion key; a faithful boot would
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

**The body's physics, as knobs** (`motor_model`, `trunk_inertial`,
`foot_solref`, `foot_friction`, `actuator`, `integrator`, `cmd_delay_ms`,
`torque_bw_hz` in the params; `robot.py`, `legged.py`). The sim's body was
idealised — a flat torque clamp at the URDF's effort, armature 0.01, MuJoCo's
default contact, the PD recomputed every 2 ms on exact joint states — and
finding 5 of `docs/findings.md` (HimLoco tipping over on dp's stairwell in a
third of the climbs, at pitch swings the posture check does not forgive)
could not say whether that was the sim's doing. `g2_deploy_rs` ships its own
MuJoCo simulation of the G23 (`mujoco_simulation_G23.py`), and what it
models the robot with is now available here, each as a parameter whose
default is the sim as it was: the J60 motor's torque-speed envelope, rotor
inertia, dry friction and viscous damping (`j60`); the vendor MJCF's torso
(`vendor`: 9.368 kg against the URDF's 8.03, and 2.5x the pitch inertia —
which of the two is the robot is to be weighed); its contact parameters
(`[0.005, 1]`, `[1, 0.01, 0.01]`); MuJoCo position actuators under the
implicitfast integrator in place of the hand-computed torque; a command delay
and a torque bandwidth, off like there. `scripts/gait_bench.py` is the
ROS-free A/B: it spawns the robot before a fitted flight (or past it, to walk
down backwards), drives a scripted command for N jittered runs and reports
what finding 5 asks the robot for — falls, the pitch's mean / 5th / 95th
percentile / extremes, roll, stalls, speed, heading drift — so a knob's
effect and the robot's numbers can be read side by side. First matrix
(2026-10-26, dp's first flight, HimLoco at 0.5 m/s for 10 s then 2 s still,
up from 1 m before it and down backwards from 0.5 m past it, ten runs each
with 3 cm / 2° of start jitter): no configuration fell in any of its 140
runs, and the pitch barely moved between them — one flight on its own does
not reproduce finding 5, whose falls came one flight in ten through the
stack. What did move is the stalling under the 0.5 m/s command:

| configuration | up: climbed in 10 s / stalls per run / pitch p5, min | down: speed / pitch p5, min |
|---|---|---|
| as it was | 0.79 m / 3.1 / −27, −31° | 0.33 m/s / −31, −38° |
| `motor_model j60` | 0.97 m / 2.7 / −26, −30° | 0.29 / −30, −37° |
| `trunk_inertial vendor` | 0.71 m / 4.4 / −26, −30° | 0.35 / −27, −31° |
| j60 + vendor contact | 0.89 m / 6.8 / −25, −28° | 0.29 / −30, −33° |
| + vendor trunk | 0.79 m / 7.0 / −26, −29° | 0.33 / −28, −31° |
| + position actuators, implicitfast | 0.82 m / 7.6 / −26, −28° | 0.32 / −29, −32° |
| + 1 kHz physics | 0.61 m / 11.4 / −25, −27° | 0.32 / −27, −30° |

The stiffer contact and the finer step make the robot stall more under the
same command, not fall; the vendor's heavier trunk changes the pitch
envelope by a degree. The defaults stay as they were until the stack's own
1F ↔ 2F batches say otherwise.

The walking body is lower than the statue: `lidar_top_imu` rides **~0.44 m**
over the ground in PPO's gait (0.444 standing at its default pose on dp),
where the planner's and controllers' `lidar_height` is 0.481 — the value the
kinematic body is built to. One of the two is not the robot; worth measuring
on the real G23 before trusting the planner's ground estimate in this mode.

**The picture** (`visual_mesh`, `--visual-mesh`, the panel's Site switch;
`site_tiles.py`). Three sites, each built once and cached: `voxels` (the
default) is the octomap's cubes, what the sim actually senses and stands on;
`pointcloud` is a smooth surface reconstructed from the map's `map.pcd`
(`pcd_mesh.py`: points kept only on the cleaned voxels, 4 cm, screened
Poisson, 1.5 M triangles; dp ~3.5 min; it blurs thin structure and bubbles
where the cloud is sparse); `tsdf` is a truncated signed distance field
integrated from the mapping run's own keyframe patches and poses
(`tsdf_mesh.py`: each patch rendered to six cube-face depth images and
integrated with open3d at 7 cm, free space carved along every ray, so
handrails, door frames and tread edges keep their shape and unseen places
stay open; built in 35 m blocks, dp 6 min and 2.5 GB at most, 2 M triangles; `python3 -m syncai_sim.tsdf_mesh map/dp` builds it
ahead). All three are coloured by
height — floors and treads along a ramp, walls a pale copy, undersides grey
(`visual.py`) — and shaded by the lidar's intensity, which is what makes a
lidar map legible: `voxel_intensity.py` keeps the median intensity of
`map.pcd`'s points in each voxel (a voxel with none takes its neighbours';
dp: 1.675 M of 1.683 M occupied voxels shaded), and the ramp's colour is
scaled by it, dark for a matt floor's 15-30, up to a third brighter for a
marking's 90-150 — floor lines, door frames, stair nosings, signs show.
The voxel mesh's faces are merged rectangles (a floor slab is one), so the
shading goes on as a texture: each 10 m tile of each storey is one GLB with
an atlas in which every rectangle has its own patch of texels, one per voxel
face (dp: 179 tiles, 2.5 M triangles, 140 MB, built in 9 s once, cached).
The surfaces' vertices are a few centimetres apart, so there the intensity
goes on the vertices.

**The photo look** (`visual_look`, `--visual-look`, the panel's Look
switch). The same tiles painted with the colour the mapping run's camera
saw at each voxel, from `config/colour/<map>.npz` (`voxel_colour.py`): the
run's bag gives the scans, the images, the camera's intrinsics and its
mount (`bag_camera.py`, no ROS), and Point-LIO replayed from the same bag
gives the pose at every instant (`scripts/bag_odom.py`); every return is
placed in the map at its firing time, moved into the camera at the nearest
frame's time, projected (equidistant: the G23's lens is a fisheye) and, if
nothing nearer sits in its pixel cell, takes the pixel; each voxel keeps the
per-channel median over every observation, so passers-by and odd exposures
are outvoted. `scripts/colour_map.py` builds the file (outdoor_1027: 5240
scans and 17 M observations in 48 s, 77 % of the occupied voxels coloured,
9 MB; `config/colour/README.md` records the bags). Only
what the camera looked at gets a colour — one forward fisheye, so the
robot's back, its sides and the ceilings stay grey (a neutral shaded by
the intensity, `visual.shade_photo`) — and the sun, shadows and white
balance are as the camera had them; 10 cm voxels keep the texture blocky.
A picture, as the ramp is: nothing the lidar, the feet or the planner see
reads it. None of
this is in the MuJoCo model: the MJB holds the robot, the
plates, the props and the terrain pool, 20 MB that compile in a second. The
robot's own STLs go over decimated (274 k → 29 k triangles, `robot.py`). Only
the picture: the feet stand on the measured floor (`surface_heights.py`),
the voxels are drawn at their tops, so the robot can look a few centimetres
off the drawn floor.

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
next one in). The panel's Storeys switch: auto / 1F / 2F / … / all.

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
| `controller` | `TerrainLayer terrain_layer: cloud …/pointlio/body_cloud …` (the layer always runs now) |
| `task_runner` | `process started`, then quiet until a goal arrives (no per-goal `*_3d.xml` swap any more) |

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
* **Nothing renders in the container; the browser draws the picture.**
  `web_view.py` (viser: a websocket server and a bundled three.js client,
  2026-10-26) sends the site's tiles and the robot's meshes once and then
  only poses, and the host's GPU draws them at the browser's own rate with
  shadows. The road here, each step measured: X11 to XQuartz fails at GLX
  context creation (indirect GLX, OpenGL 1.4, below what MuJoCo's renderer
  needs); an Xvnc display of the container's own with Mesa's llvmpipe behind
  it and noVNC in front drew the full dp mesh at ~180 ms a frame (1280x720),
  the 3 m band at ~120 ms; MuJoCo's `launch_passive` on it paced itself on a
  vsync Xvnc does not have, rendered flat out at 7-11 fps for ~3 cores, and
  copied the 18 MB MjData 60 times a second (12-15 ms each); a GLFW window of
  our own with a 10 fps cap, state-only copies at 20 Hz (0.3 ms) and the site
  drawn by tile within 30 m was the best llvmpipe allowed. MuJoCo's own ways
  out were looked at: its Filament renderer (3.17 has it) needs a GPU the
  container does not have, and MuJoCo Studio's web mode ships without its
  `headless_ui` module in the 3.13–3.17 wheels (issue #3580). With the
  browser drawing, the display, the window and the VNC packages went
  (2026-10-26); `libgl1` stays because open3d's wheel links it to import.
  The site's mesh left the MuJoCo model the same day: it was only ever there
  to be rendered, cost 628 MB of MJB and 5 s of compile for its BVHs, and
  the browser takes its tiles from the world's own cache (`site_tiles.py`).
* **Where the CPU went (2026-10-24).** The node in the session used ~7 cores;
  three sinks, none of them physics (a walking step is 0.23 ms): numpy's
  OpenBLAS workers spin-waiting between the 50 Hz 4x4 TF inversions (350 %
  for the headless node; `OPENBLAS_NUM_THREADS=1`, set in `__main__.py`:
  47 %), the GLFW viewer of the time (above), and `PointCloud2.data` set from `bytes`, which the
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
