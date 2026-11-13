# SyncAI-Robot-Simulation

A MuJoCo simulation of the SyncAI robot (the G23 quadruped) on a **real,
mapped site**, built for the robot's controllers: the sim is the body, the
site and the sensors, and it talks to the outside over **UDP only** — no ROS
in the container, the image or the code (removed 2026-11-03).

What the simulator is, in one paragraph: the site's `octomap.bt` (written by
`syncai_mapping`'s `clean_map --octomap 1`) becomes a voxel world; the G23,
from the workspace's own URDF, stands on a patch of those voxels on twelve
motor-driven joints whose commands come from a controller process at the
other end of a UDP link; a Livox MID360 is ray-cast against the voxels at
10 Hz, firing the scan pattern recorded from the real one. The picture is
drawn by the host's browser (`viewer/web_view.py`): nothing renders in the
container.

**The sim is the robot's hardware and nothing more** (2026-11-13). It sends
out what the robot's sensors measure and applies the joint commands it is
sent; it decides no motion of its own. Turning cmd_vel into joint commands
— RL or MPC — is the controller's job (SyncAI-Robot-Control), and so is the
controller's state machine. Until 2026-11-13 the sim also walked itself (the
robot's ONNX policies in-process, or a rigid statue carried by rules) and
could answer as the gait controller on its ASCII link; git history has them.

The package is cut into one subpackage per layer so each is visible on its
own (`docs/architecture.md` draws them): the voxel truth (`world/`), the
feet's floor (`terrains/`), the robot and the MuJoCo scene (`entity/`,
`scene/`), what the body measures and what drives its joints (`sensor/`,
`actuator/`), the controller's link to the joints (`locomotion/`), the
panel's driving forwarded to the controller (`driver/`), and the picture
(`viewer/`, `visuals/`).

The UDP links, the only ways in and out:

* **The hardware link** (`locomotion/hardware_link.py`): every physics step
  the body's measurements go out (30010) and the joint command comes back
  (20001), so the controller's RL or its MPC (LMA) decides and the sim is
  its hardware. The consumer is SyncAI-Robot-Control's `--backend sim`
  (not written yet); with nothing on the link the motors are in damping and
  the robot lies down.
* **The panel's driving** (`driver/udp_driver.py`): the viewer's motion keys,
  policy toggle and sliders go to the controller's command port (50051) as
  the ASCII lines driver_manager sends (`MODE`, `ESTOP`, `AXES`). The nav
  stack's driver_manager talks to the controller directly, not to the sim.
* **The sim as the robot's lidar** (`lidar_link: livox`): the sim is a
  MID360 / MID360s on the Livox protocol (`sensor/livox_link.py`), so the
  workspace's own livox_ros_driver2 finds it and reads its points and IMU,
  and Point-LIO, the localizer and the planner run on the sim — the robot's
  pose in the map is the sim's (section "The sim as the robot's lidar").

## Layout

```
Dockerfile, docker-compose.yml     one container: Ubuntu 22.04 + MuJoCo + numba + viser; no ROS, no display, no GPU
.env.example                       ROBOT_WS (the workspace checkout: maps + URDF), the viewer's port, the network mode, what starts
config/instances/sim01.ini         the instance INI (the map, the spawn), bind-mounted over the workspace's config/system.ini
config/params/sim_params.yaml      every sim parameter with why it is what it is; always loaded (a plain YAML mapping)
scripts/sim_entry.sh               the container's command: the sim + its viewer at start (SIM_AUTOSTART, SIM_ARGS), then stays up
syncai_sim/                        the Python package (python3 -m syncai_sim): one subpackage per layer (docs/architecture.md)
  __main__.py, config.py             the CLI and the wall-clock main loop (the UDP links polled and ticked); the params and the INI
  world/                             the site as the sim senses it: octomap.bt -> VoxelGrid, the truth for rays, ground, blocking
    btree.py                           OctoMap .bt reader (no octomap dependency)
    greedy_mesh.py                     voxels -> boxes
    voxel_grid.py                      the numba ray caster, ground query, blocking test
    build.py                           octomap.bt -> VoxelGrid + the exposed faces as rectangles + collision plates, cached (python3 -m syncai_sim.world)
    storeys.py                         the site's storeys from poses.txt (the viewer's 1F / 2F switch)
  terrains/                          what the walking feet stand on (collision only; the lidar reads world/)
    surface_heights.py                 the floor's measured heights, flights fitted as staircases
    stairs.py                          the fitted flights as fixed boxes under the feet
    terrain_patch.py                   the voxels around the robot as collision boxes for its feet
  entity/robot.py                    the workspace URDF -> MJCF: twelve joints, two IMUs; the body's physics knobs
  scene/scene.py                     robot + collision plates + props (+ the terrain pool) in one MjSpec, no picture in it; cached as MJB
  sensor/                            what the robot measures, never what is drawn
    lidar.py, prop_rays.py             the MID360: its recorded pattern fired frame by frame, rays from where it was when each fired; analytic rays for props
    livox_bag.py                       CustomMsg frames straight out of a rosbag2 sqlite file -> the pattern bank
    proprioception.py                  the body IMU, joints q/dq/tau, the base's truth, foot contacts: one record per step
  actuator/motors.py                 the motors: JointCommand (q_des, dq_des, kp, kd, tau_ff), the PD closed every step, the motor clamp, delays
  locomotion/hardware_link.py        the body's record out and the controller's command in, one UDP datagram each per step (the protocol's spec); the stand-in controller
  driver/udp_driver.py               the panel's motion keys, policy number and AXES, forwarded to the controller process
  envs/simulator.py, placement.py    the core: step, scan, spawn, teleport; where the base is put down
  viewer/web_view.py, teleop.py      the browser viewer: the site and the robot sent once, poses streamed, drawn by the host's GPU; its panel's driving
  visuals/                           the site's picture (nothing the lidar, the feet or the planner see reads it)
    visual.py                          the viewer's look: colour by height, shaded by the lidar's intensity; or the photo
    site_tiles.py                      the site for the browser: 10 m tiles, every voxel face a texel of its intensity or its colour, as GLB, cached
    pcd_mesh.py                        map.pcd -> a smooth viewer surface (Poisson, open3d), cached
    tsdf_mesh.py                       the keyframes' patches + poses.txt -> a TSDF viewer surface (open3d), thin structure kept, cached
    voxel_intensity.py                 map.pcd -> the median intensity of each voxel, cached
    voxel_colour.py                    the mapping run's bag -> the colour the camera saw at each voxel (config/colour/<map>.npz, scripts/colour_map.py)
    bag_camera.py                      the bag's images, camera_info and /tf_static, read from its sqlite file
    map_cloud.py                       map.pcd thinned per storey: the viewer's lidar map layer
scripts/controller_bench.py        the sim as a controller's hardware, headless: MODE / AXES to the controller, its MODE_STATE back, the link's counters
docs/architecture.md               the layers, one physics tick, what a controller needs from the sim and why
config/lidar/mid360_pattern.npz    the MID360's recorded pattern bank, from the workspace's record/dp (config/lidar/README.md)
scripts/lidar_pattern.py           rebuilds it from any MID360 recording
tests/                             pytest; test_simulator_site runs the whole sim on the workspace's map/dp (skips without it)
cache/                             per-map build products (gitignored): the world, the compiled scene (scene-*.mjb, 20 MB on dp), the voxel intensities, the browser's tiles (140 MB), the decimated robot meshes (robot/)
```

## Setup

Prerequisites on the host: Docker (Desktop or OrbStack on a Mac; the image is
linux/arm64 like the robot's) and a checkout of the workspace for its maps
(`map/<name>/`, each with an `octomap.bt`, see the workspace's
`syncai_mapping`) and the robot's URDF and meshes (`src/syncai_bringup`).
Nothing in the workspace is built or run.

```bash
cp .env.example .env            # ROBOT_WS = path to the workspace checkout (default: ../SyncAI-Robot-Workspace)
docker compose build            # Ubuntu 22.04 + the Python packages in requirements.txt
docker compose up -d            # the sim + its browser viewer start with it: http://localhost:8080
docker compose logs -f sim      # the sim's output
docker compose exec sim bash    # cwd is the workspace root
```

**What starts with the container** (`scripts/sim_entry.sh`, two `.env`
knobs). By default `python3 -m syncai_sim --teleop`: the controller's
hardware, the viewer's panel driving the controller. `SIM_ARGS` replaces the
arguments (the robot's lidar too: `--teleop -p lidar_link:=livox`);
`SIM_AUTOSTART=0` starts nothing. The container stays up whatever the sim does.

**The network.** By default a bridge network: the viewer on 8080 and the
sim's UDP input (20001, a controller's joint commands) are mapped to the
host. The sim's outputs go where the params say (`link_state_host` and
`controller_command_ip`, 127.0.0.1 by default; from a bridge network to a
program on the host, `host.docker.internal`). `SIM_NETWORK_MODE=host` puts the
sim's sockets on the host's network instead, which is the simplest for a
controller process on the same machine (127.0.0.1 both ways).

`SIM_INSTANCE_INI` (default `config/instances/sim01.ini`) picks the map (`[map]
name`) and the spawn pose (`[initial_pose]`; z is the ground height, a hint
for the floor). It is a single-file bind mount: edit it in place, do not
rename over it.

## Running

Inside the container (`docker compose exec sim bash`; the sim that came up
with the container already holds the viewer's port and the UDP ports, so for
another one by hand bring the container up with `SIM_AUTOSTART=0` first):

```bash
python3 -m syncai_sim --teleop                                   # drive from the panel at http://localhost:8080
python3 -m syncai_sim --headless --seconds 30                    # no viewer
python3 -m syncai_sim.world map/dp                               # (re)build a map's cache and print its stats
python3 -m pytest /home/syncrobotic/sim/tests                    # the tests; test_simulator_site needs map/dp
```

The first run on a map builds its cache (dp: ~15 s — parsing 1.6 M occupied
leaves, merging faces into a 2.5 M-triangle viewer mesh, 19 k collision
plates, a 133 MB voxel grid) and compiles the scene (~1 s, 20 MB as an
MJB); the browser's tiles are built behind the running sim the first time
(dp: the voxel intensities in 4 s, the 179 textured tiles in 9 s) and cached
too, so later starts take about a second and the picture is up in two.

**Settings.** `config.PARAM_DEFAULTS`, then `config/params/sim_params.yaml`
(always; it documents every key), then each `--profile` in order, then `-p
key:=value`, then the flags (`--map`, `--visual-mesh`, `--visual-look`).

**The viewer** (`viewer/web_view.py`) is a page the host's browser draws:
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
the teleport back to the spawn. With `--teleop` it also has the driving
sliders and the motion keys (`viewer/teleop.py`), which go to the
controller's command port. `--headless` serves no viewer.

**The controller on the link.** Every physics step sends the body's
measurements to a controller process and applies the joint command it sends
back (`locomotion/hardware_link.py` is the protocol; `docs/architecture.md`
says what each field is for). The defaults set the link up — its ports,
1 kHz, the body the way a controller's own development simulation models it:

```bash
python3 /home/syncrobotic/sim/scripts/controller_bench.py --fake-controller --seconds 5   # the link end to end with the stand-in controller
python3 /home/syncrobotic/sim/scripts/controller_bench.py --send "MODE Z" --send "MODE M" --gap 3 --axes 0.5 0 0 --seconds 15 --listen   # a controller process on the link
```

With nothing on the link the motors are in damping and the robot lies down;
the panel's motion keys and sliders go to the controller's command port
(50051). The controller process itself (SyncAI-Robot-Control's `--backend
sim`) is the next step, not in this repository. Until it exists, the nav
stack has no way to walk the sim: driver_manager talks to the controller,
and the sim no longer plays one (the recipe of before, `-p
driver_link:=udp`, is gone with it).

### The sim as the robot's lidar

With `-p lidar_link:=livox` the sim answers the Livox driver as the robot's
lidar does: discovery on 56000, commands on 56100, points from 56300 and the
IMU from 56400, every packet in the SDK's own format (`sensor/livox_link.py`
documents the protocol and where each rule is in the vendored SDK). The
workspace's nav session runs unchanged against it — bringup's driver,
Point-LIO, the localizer, lio_bridge — and the robot's pose in the map, the
planner and anything built on them (the console's route preview, a goal)
follow the sim's body. Measured on dp, 2026-11-05: IMU at 199 Hz,
Point-LIO at 9.8 Hz, `map -> base_link` within 4 cm of the sim's pose, a
route planned to a waypoint 16 m off.

The lidar's IP and model are the instance INI's `[sensor.lidar]` (the ones
bringup writes into the driver's JSON; `livox_lidar_ip` /
`livox_lidar_type` override them). The model matters: the JSON is keyed by
it, so a MID360 (dev_type 9) answering a driver configured for a MID360s
(35) is ignored without a word. Two addresses must exist on the host the
sim and the robot container share (both on host networking): the lidar's
IP, which the sim binds, and the driver's `host_ip` (bringup.yaml,
192.168.1.103), which the SDK binds and whose packets the SDK refuses to
take as a device's. On a Mac (OrbStack) neither is there; add them to the
VM's loopback, then restart the robot container so its driver can bind:

```bash
docker run --rm --net host --privileged alpine ip addr add 192.168.1.103/32 dev lo
docker run --rm --net host --privileged alpine ip addr add 192.168.1.172/32 dev lo
docker restart robot01
SIM_ARGS="--teleop -p lidar_link:=livox" docker compose up -d
```

`[livox] found by the driver` and `[livox] configured: ...` in the sim's log
say the driver has taken it; the status line counts the packets. Time is
the device's own (PTP, `time_type` 1), the host's epoch clock tied to sim
time, because Point-LIO stamps the pose with it and lio_bridge stamps its
TF with the host's clock; a sim that falls behind the wall moves it forward
(`reanchors` in the status line), never back. The localizer starts from
its own last pose or the INI's `[initial_pose]`, which is where the sim
spawns the robot, so the two agree at the start.

## The world, the robot, the rules

**World.** `octomap.bt` is parsed directly (`world/btree.py`; the format is two
bytes per inner node) into a dense boolean voxel grid, after two cleanups
that make the voxel world agree with what the mapping robot physically
passed through: one-column posts 0.2 m or more above every neighbour are
clipped (the mapper's own feet), and whatever stands inside the body volume
the mapper swept along `poses.txt` is erased (`track_clear_radius_m`; on dp a
one-voxel-thick 0.3 m "fence" across the only corridor to the stairs, which
the nav stack's `TerrainLayer` had made lethal). Enclosed floor holes
of up to 25 columns — floor the lidar never saw, 3965 on dp — are filled at
their rim's height (`floor_hole_max_cells`). That grid is the
truth for the lidar, the ground and blocking — all three are numba kernels in
`world/voxel_grid.py`, and 20 k rays take about 1.3 ms. Lidar returns on surface
voxels are placed at the voxel's mid-height (the real floor is somewhere
inside the voxel), which keeps the 0.1 m steps of a mapped floor from reading
as dynamic obstacles. MuJoCo never sees the voxels:
the exposed faces, greedy-merged into rectangles, are the browser's picture
(`visuals/site_tiles.py`), and MuJoCo gets only the few large merged boxes as
**collision plates** so props have floors to stand on. This split is deliberate: MuJoCo collides a
mesh as its convex hull, `mj_multiRay` scans every geom per ray (18 k plates
would cost hundreds of milliseconds a scan) and even against the site as one
mesh its BVH needed 325 ms for 20 k rays; against the voxels the same 20 k
rays are 1.3 ms. Numbers for dp are in `cache/dp/world.json`.

**Lidar** (`lidar_pattern`, `sensor/lidar.py`). The scan fires the directions the
robot's own MID360 fired: twenty consecutive frames of the workspace's
`record/dp` (`livox_ros_driver2` CustomMsg, read out of the rosbag2 sqlite by
`sensor/livox_bag.py` with no ROS), ~16 k samples a frame that returned, cycled
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

**The feet's floor.** The joints' torque is the controller's command,
closed as a PD on every 1 ms physics step and clamped to the motor model
(`actuator/motors.py`). The posture check is the controller's (|roll| > 30°,
|pitch| > 45°): the sim counts a fall and logs it, and the controller locks
itself as on the robot; a teleport stands the robot up again. The fall
numbers below were measured with the robot's own policies in this process
(PPO for flat ground, HimLoco for stairs; removed 2026-11-13), which is
what shaped the floor.
The feet stand on `terrains/terrain_patch.py`: ~300-700 voxel boxes around the robot,
moved as it walks, with the map's surface-only artefacts closed first (hollow
stair treads, one-voxel slots across a flight, one-column holes — each with
the run that found it in the module), and each column's top box stands at
the floor's measured height inside its voxel (`terrains/surface_heights.py`: the median
of `map.pcd` there, then an edge-preserving 5 x 5 median over the same surface,
which also fills the 43 % of floor voxels with no point inside) — the 0.1 m
grid turns dp's tilted floor into a strip of random blocks every ~13 m, where
PPO fell 4 times out of 4, and the per-voxel medians alone were still rough
(8.7 % of neighbouring columns more than 2 cm apart). Smoothed, dp's open floor
has its largest step between neighbours at ~1 cm; risers and kerbs keep their
edges. A voxel with no point inside takes the mean of its measured
neighbours a layer up, level and down (until 2026-11-01 its own layer's
first, for dp's filled treads, which the flight fit now covers; on a long
slope that kept a lip at every layer, and PPO fell on outdoor_1027's road
from Home to A twice), a 13 x 13 window when the 5 x 5 has none, and its
voxel's middle when neither has (its top is 5 cm high on average: on dp's last step to 2F that made a lip
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
corridor fence above had kept 0.2-0.3 m; the robot fell on it both
ways). There is no step rule: whether a flight is climbed is up to the
controller. Measured with the in-process policies (2026-10-23): PPO 0.4 m/s
at a 0.5 command, 1.0 rad/s yaw; HimLoco climbed the planner CLI's synthetic
flight and dp's first flight to the landing (1.2 m); PPO fell at the first
riser. A collision is the trunk touching the site.

**The body's physics, as knobs** (`motor_model`, `trunk_inertial`,
`foot_solref`, `foot_friction`, `integrator`, `cmd_delay_ms`,
`torque_bw_hz` in the params; `entity/robot.py`, `actuator/motors.py`). The sim's body was
idealised — a flat torque clamp at the URDF's effort, armature 0.01, MuJoCo's
default contact, the PD recomputed every 2 ms on exact joint states — and
when HimLoco tipped over on dp's stairwell in a third of the climbs (2026-10-25,
at pitch swings the posture check does not forgive), nothing could say
whether that was the sim's doing. `g2_deploy_rs` ships its own
MuJoCo simulation of the G23 (`mujoco_simulation_G23.py`), and what it
models the robot with is now available here, each as a parameter whose
default is the sim as it was: the J60 motor's torque-speed envelope, rotor
inertia, dry friction and viscous damping (`j60`); the vendor MJCF's torso
(`vendor`: 9.368 kg against the URDF's 8.03, and 2.5x the pitch inertia —
which of the two is the robot is to be weighed); its contact parameters
(`[0.005, 1]`, `[1, 0.01, 0.01]`); MuJoCo position actuators under the
implicitfast integrator in place of the hand-computed torque (removed
2026-11-13: an MPC's command carries a velocity target and a feed-forward
torque, which a position actuator cannot apply); a command delay and a
torque bandwidth, off like there. `scripts/gait_bench.py` was the headless
A/B (removed with the policies it drove; git history has it): it spawned
the robot before a fitted flight (or past it, to walk down backwards), drove
a scripted command for N jittered runs and reported what to compare with
the robot on the same stairs — falls, the pitch's mean / 5th / 95th
percentile / extremes, roll, stalls, speed, heading drift — so a knob's
effect and the robot's numbers can be read side by side. First matrix
(2026-10-26, dp's first flight, HimLoco at 0.5 m/s for 10 s then 2 s still,
up from 1 m before it and down backwards from 0.5 m past it, ten runs each
with 3 cm / 2° of start jitter): no configuration fell in any of its 140
runs, and the pitch barely moved between them — one flight on its own does
not reproduce those falls, which came one flight in ten through the
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
envelope by a degree. Since 2026-11-13 the vendor's body (`j60`, `vendor`,
its contact) at 1 kHz is the default — what the controller-hardware profile
had set since 2026-11-03 — so a controller sees here what its own
development simulation shows it; the alternatives stay as knobs.

The body stands lower than the planner thinks: `lidar_top_imu` rides
**~0.44 m** over the ground (0.444 standing at the default pose on dp),
where the planner's and controllers' `lidar_height` is 0.481. One of the two
is not the robot; worth measuring on the real G23 before trusting the
planner's ground estimate against the sim.

**The picture** (`visual_mesh`, `--visual-mesh`, the panel's Site switch;
`visuals/site_tiles.py`). Three sites, each built once and cached: `voxels` (the
default) is the octomap's cubes, what the sim actually senses and stands on;
`pointcloud` is a smooth surface reconstructed from the map's `map.pcd`
(`visuals/pcd_mesh.py`: points kept only on the cleaned voxels, 4 cm, screened
Poisson, 1.5 M triangles; dp ~3.5 min; it blurs thin structure and bubbles
where the cloud is sparse); `tsdf` is a truncated signed distance field
integrated from the mapping run's own keyframe patches and poses
(`visuals/tsdf_mesh.py`: each patch rendered to six cube-face depth images and
integrated with open3d at 7 cm, free space carved along every ray, so
handrails, door frames and tread edges keep their shape and unseen places
stay open; built in 35 m blocks, dp 6 min and 2.5 GB at most, 2 M triangles; `python3 -m syncai_sim.visuals.tsdf_mesh map/dp` builds it
ahead). All three are coloured by
height — floors and treads along a ramp, walls a pale copy, undersides grey
(`visuals/visual.py`) — and shaded by the lidar's intensity, which is what makes a
lidar map legible: `visuals/voxel_intensity.py` keeps the median intensity of
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
saw at each voxel, from `config/colour/<map>.npz` (`visuals/voxel_colour.py`): the
run's bag gives the scans, the images, the camera's intrinsics and its
mount (`visuals/bag_camera.py`, no ROS), and Point-LIO replayed from the same bag
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
robot's own STLs go over decimated (274 k → 29 k triangles, `entity/robot.py`). Only
the picture: the feet stand on the measured floor (`terrains/surface_heights.py`),
the voxels are drawn at their tops, so the robot can look a few centimetres
off the drawn floor.

**Storeys.** `world/storeys.py` finds them in `poses.txt` (where the mapping robot
spent most of its keyframes; dp: 1F at −1.0…−0.2 m, 2F at 4.4…5.6 m, split at
4.18 m, the flight and its landings with 1F). Each storey is drawn up to
`visual_max_height_above_column_floor_m` over its own floor, so an upper
floor is drawn whole, and the viewer shows by default the robot's storey and
those below it (the floor above never hides the robot; walking up brings the
next one in). The panel's Storeys switch: auto / 1F / 2F / … / all.

**Robot.** The workspace's `G23.urdf` is loaded by MuJoCo itself with two
compiler switches injected (`fusestatic=false` so `base_link` can get a free
joint, `discardvisual=false` to keep the STL visuals). It keeps its twelve
joints (the motor model's armature), every leg primitive collides with the
terrain patch and the trunk with the patch and the props, and two IMUs sit
on `base_link` (the body's) and `lidar_top_imu` (the MID360's own).
`lidar_top` and `lidar_top_imu` stay as bodies so the sensor extrinsics are
the URDF's, read, not copied. Until 2026-11-13 a second, rigid form of it
existed for the kinematic mode: the leg joints baked at a standing angle, a
mocap body welded to `base_link` and moved by a step rule.

**Props.** `props_json` in the params lists boxes and cylinders: a free body is
pushable (it stands on the plates, is pushed by the trunk and seen by the
lidar); a body with `waypoints` walks between them as a mocap "person" —
food for the planner's dynamic overlay and TerrainLayer's `obstacle_memory`.

## Decisions worth knowing

* **UDP only, no ROS (2026-11-03).** Until then the sim was a ROS 2 node in
  the workspace's ROS image, publishing what the Livox driver, Point-LIO,
  the localizer and driver_manager publish so the nav stack could run end to
  end. The work moved to the controllers (the MPC above all), whose
  interfaces are UDP, so the ROS face went — the node, the session, the
  launch file, Foxglove, the DDS configs, the 5.4 GB ROS base image (the
  image is now 1.1 GB of Ubuntu and pip). git history has the ROS node.
* **Nothing renders in the container; the browser draws the picture.**
  `viewer/web_view.py` (viser: a websocket server and a bundled three.js client,
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
  the browser takes its tiles from the world's own cache (`visuals/site_tiles.py`).
* **Where the CPU went (2026-10-24).** The sim then used ~7 cores; the sinks
  were none of them physics (a walking step is 0.23 ms): numpy's OpenBLAS
  workers spin-waiting between small matrix calls (350 % headless;
  `OPENBLAS_NUM_THREADS=1`, set in `__main__.py`: 47 %) and the GLFW viewer
  of the time (above). After: ~1.2 cores, scan 3.7 ms, `Simulator()` in
  1.1 s from the caches.
* **The workspace is read, not modified.** The sim takes its maps and the
  robot's URDF; the instance INI is bind-mounted over its `config/system.ini`.
* **Wall clock.** The sim runs fixed physics steps paced to real time (1 ms),
  catching up a late loop in bursts of at most 8 ms and dropping what one
  burst cannot repay, as MuJoCo's `simulate` does (`envs/pacing.py`: a
  carried debt put the MID360's stamps 0.45 s ahead).
* **The sim is hardware; the controller is run (2026-11-03, 2026-11-13).**
  Until 2026-11-03 the robot's policies were ported into the sim (~150
  lines of Python against the same ONNX files). The MPC changed that: it is
  C++ behind the controller's FFI and cannot be ported, so the sim grew a
  hardware link over which a controller process commands the joints and
  reads the body. On 2026-11-13 the port went, with the kinematic statue and
  the sim's impersonation of the gait controller on the ASCII link: turning
  cmd_vel into joint commands is the controller's, wherever it runs, and a
  second copy of it here only had to be kept in step with the robot's. The
  price is that nothing walks the sim until a controller is on the link, and
  that the link's spec lives here and the controller's `--backend sim` must
  follow it.

## Status

* **Done.** The voxel world and its cleanups, the fitted stairs and ramps, the MID360's recorded pattern, the
  browser viewer with the voxel / Poisson / TSDF sites and the photo look.
  The layered package and the hardware link (2026-11-03): the sim as a
  controller's hardware, checked end to end with the stand-in controller (dp,
  5 s at 1 kHz: ~5000 states, ~4800 commands, 0 falls). ROS removed the same
  day: UDP is the only way out. The in-process policies, the kinematic
  statue and the ASCII controller link removed (2026-11-13): the sim is the
  hardware only.
* **Next.** The controller process on the link: SyncAI-Robot-Control's
  `--backend sim` against `locomotion/hardware_link.py`, in a container
  beside the sim's, then `scripts/controller_bench.py --send "MODE Z" --send
  "MODE M"` against its LMA.
