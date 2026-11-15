# SyncAI-Robot-Simulation

A MuJoCo simulation of the SyncAI robot (the G23 quadruped) on a **real,
mapped site**, built for the robot's controllers: the sim is the body, the
site and the sensors, and it talks to the outside over **UDP, SocketCAN and
a serial port only** — no ROS
in the container, the image or the code (removed 2026-11-03).

What the simulator is, in one paragraph: the site's `octomap.bt` (written by
`syncai_mapping`'s `clean_map --octomap 1`) becomes a voxel world; the G23,
from the workspace's own URDF, stands on a patch of those voxels on twelve
motor-driven joints whose commands come from the robot's controller over
the robot's four CAN buses; a Livox MID360 is ray-cast against the voxels at
10 Hz, firing the scan pattern recorded from the real one. The picture is
drawn by the host's browser (`viewer/web_view.py`): nothing renders in the
container.

**The sim is the robot's hardware and nothing more** (2026-11-13). It sends
out what the robot's sensors measure and applies the joint commands it is
sent; it decides no motion of its own. Turning cmd_vel into joint commands
— RL or MPC — is the controller's job (g2-device-control, the process the
robot itself runs), and so is the controller's state machine. Until 2026-11-13 the sim also walked itself (the
robot's ONNX policies in-process, or a rigid statue carried by rules) and
could answer as the gait controller on its ASCII link; git history has them.

The package is cut into one subpackage per layer so each is visible on its
own (`docs/architecture.md` draws them): the voxel truth (`world/`), the
feet's floor (`terrains/`), the robot and the MuJoCo scene (`entity/`,
`scene/`), what the body measures and what drives its joints (`sensor/`,
`actuator/`), the controller's link to the joints (`locomotion/`), the
panel's driving forwarded to the controller (`driver/`), and the picture
(`viewer/`, `visuals/`).

The links, the only ways in and out. Started with no argument the sim is
every one of them as the robot has it (2026-11-15: the robot's links are the
defaults, where each was a `-p` before), so the robot's controller and the
workspace's nav stack connect to it unchanged:

* **The joints on CAN** (`joint_link: can`, `locomotion/can_link.py`): the
  twelve joints are J60 motors on four SocketCAN buses, one per leg, as on
  the robot — `can1..can4`, motor ids 2 / 3 / 4, the J60's frames, each
  motor's own direction and homed zero — so a controller's real hardware
  layer drives the sim as it drives the robot (section "The joints on CAN").
* **The IMU on a serial port, the BMS on CAN** (`imu_link: serial`,
  `sensor/imu_serial.py`; `bms_link: can`, `sensor/bms_link.py`): the rest
  of what that hardware layer reads on the robot — the body IMU as its
  Yesense unit streaming on a serial port, the battery's BMS answering on
  `can0` (section "The IMU and the BMS").
* **The panel's driving** (`driver/udp_driver.py`): the viewer's sliders go
  to the controller's command port (50051) as the `AXES` line driver_manager
  sends. Its motion keys and policy toggle (`MODE`, `ESTOP`) were removed
  2026-11-15, while the controller's low-level motor control is rewritten. The nav
  stack's driver_manager talks to the controller directly, not to the sim.
* **The sim as the robot's lidar** (`lidar_link: livox`): the sim is a
  MID360 / MID360s on the Livox protocol (`sensor/livox_link.py`), so the
  workspace's own livox_ros_driver2 finds it and reads its points and IMU,
  and Point-LIO, the localizer and the planner run on the sim — the robot's
  pose in the map is the sim's (section "The sim as the robot's lidar").
* **The state datagram** (`locomotion/hardware_link.py`): every physics step
  the body's whole record goes out on UDP (30010), the base's truth and the
  foot contacts with it, which are on no bus: what a bench measures the
  controller against. With `-p joint_link:=udp` the same link also takes
  the joint commands (20001) in place of the CAN buses; no controller
  speaks that yet (the stand-in of `scripts/controller_bench.py` does).

## Layout

```
Dockerfile, docker-compose.yml     one container on the host's network: Ubuntu 22.04 + MuJoCo + numba + viser; no ROS, no display, no GPU
docker-compose.controller.yml      laid over the robot's controller's own compose file: what differs for it when its hardware is this sim
.env.example                       ROBOT_WS (the workspace checkout: maps + URDF), the instance INI, what starts
config/instances/sim01.ini         the instance INI (the map, the spawn, the lidar's model and IP), bind-mounted over the workspace's config/system.ini
config/params/sim_params.yaml      every sim parameter with why it is what it is; always loaded (a plain YAML mapping)
config/params/scenes/              a profile per made test site (scripts/make_scenes.py): its map and spawn
config/ramps/, stair_walks/, stair_shapes/   per map: sloping floors, stairs the mapping track never climbed, stairs measured on site (terrains/surface_heights.py)
scripts/sim_entry.sh               the container's command: the sim + its viewer at start (SIM_AUTOSTART, SIM_ARGS), then stays up
syncai_sim/                        the Python package (python3 -m syncai_sim): one subpackage per layer (docs/architecture.md)
  __main__.py, config.py             the wall-clock main loop (the links polled and ticked) and its four flags; the params and the INI
  world/                             the site as the sim senses it: octomap.bt -> VoxelGrid, the truth for rays, ground, blocking
    btree.py                           OctoMap .bt reader, and the writer the made test sites use (no octomap dependency)
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
    imu_serial.py                      imu_link serial: the body IMU as the robot's Yesense on a pseudo-terminal (the frame; the spec)
    bms_link.py                        bms_link can: the pack's Foxlink BMS on can0 (the items, the cycle; the spec) and the small battery behind it
  actuator/motors.py                 the motors: JointCommand (q_des, dq_des, kp, kd, tau_ff), the PD closed every step, the motor clamp, delays
  locomotion/can_link.py             joint_link can: the joints as J60 motors on four SocketCAN buses (the frames, the robot's wiring; the spec); its stand-in controller
  locomotion/hardware_link.py        the body's record out on UDP every step; with joint_link udp a controller's command in as well (the protocol's spec); the stand-in controller
  driver/udp_driver.py               the panel's AXES, forwarded to the controller process
  envs/simulator.py, placement.py    the core: step, scan, spawn, teleport; where the base is put down
  viewer/web_view.py, teleop.py      the browser viewer: the site and the robot sent once, poses streamed, drawn by the host's GPU; its panel's driving
  visuals/                           the site's picture (nothing the lidar, the feet or the planner see reads it)
    visual.py                          the viewer's look: colour by height, shaded by the lidar's intensity; or the photo
    site_tiles.py                      the site for the browser: 10 m tiles, every voxel face a texel of its intensity or its colour, as GLB, cached
    pcd_mesh.py                        map.pcd -> a smooth viewer surface (Poisson, open3d), cached
    voxel_intensity.py                 map.pcd -> the median intensity of each voxel, cached
    voxel_colour.py                    the mapping run's bag -> the colour the camera saw at each voxel (config/colour/<map>.npz, scripts/colour_map.py)
    bag_camera.py                      the bag's images, camera_info and /tf_static, read from its sqlite file
    map_cloud.py                       map.pcd thinned per storey: the viewer's lidar map layer
scripts/spin_sleep.sh, .c          OrbStack only: the LD_PRELOAD shim that makes the controller's sleeps end on time (its 1 kHz)
scripts/vcan_up.sh                 the vcan interfaces of joint_link can and bms_link can (can0..can4), made on the host (a privileged container on a Mac)
scripts/controller_bench.py        the sim headless with a stand-in controller or beside a real one: MODE / AXES to it, its MODE_STATE back, falls, posture, the links' counters
docs/architecture.md               the layers, one physics tick, what a controller needs from the sim and why
config/lidar/mid360_pattern.npz    the MID360's recorded pattern bank, from the workspace's record/dp (config/lidar/README.md)
scripts/lidar_pattern.py           rebuilds it from any MID360 recording
scripts/make_scenes.py             test sites made, not mapped (flat, slope, stairs15) into scenes/ (gitignored), their config files and profiles
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
scripts/vcan_up.sh              # the robot's CAN interfaces, can0..can4 (again after a restart of the machine / Docker's VM)
docker compose up -d            # the sim + its browser viewer start with it: http://localhost:8080
docker compose logs -f sim      # the sim's output
docker compose exec sim bash    # cwd is the workspace root
```

**What the host must have**, because the sim is the robot's devices on the
host's own network: the five vcan interfaces (`scripts/vcan_up.sh`), and
the lidar's IP as one of its addresses (`[sensor.lidar]` of the instance
INI, 192.168.1.172; section "The sim as the robot's lidar" adds it on a
Mac). Without the first the sim stops at start and says which interface;
`-p lidar_link:=none` runs without the second, for a run with no nav stack.

**What starts with the container** (`scripts/sim_entry.sh`, two `.env`
knobs). By default `python3 -m syncai_sim`, no argument: the robot.
`SIM_ARGS` gives it arguments (a test site: `--profile
/home/syncrobotic/sim/config/params/scenes/flat.yaml`); `SIM_AUTOSTART=0`
starts nothing. The container stays up whatever the sim does.

**The network** is the host's (`network_mode: host`): a vcan interface is
seen in no other network namespace, the lidar's IP is the host's, and the
controller and the nav stack reach the sim's ports on 127.0.0.1. The viewer
listens on 8080 (`web_view_port`). Until 2026-11-15 a bridge network was
the default and `docker-compose.can.yml` switched to the host's; nothing
the robot's software connects to worked on the bridge, so the file is gone.

`SIM_INSTANCE_INI` (default `config/instances/sim01.ini`) picks the map (`[map]
name`), the spawn pose (`[initial_pose]`; z is the ground height, a hint
for the floor) and the lidar the sim answers as (`[sensor.lidar]`). It is a
single-file bind mount: edit it in place, do not rename over it.

## Running

Inside the container (`docker compose exec sim bash`; the sim that came up
with the container already holds the viewer's port, the UDP ports and the
CAN buses, so for another one by hand bring the container up with
`SIM_AUTOSTART=0` first):

```bash
python3 -m syncai_sim                                            # the robot; the panel at http://localhost:8080
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
key:=value`. Everything that can be set is a parameter; the only flags are
`--headless` and `--seconds`, how a run is hosted. (`--map`,
`--visual-mesh`, `--visual-look` and `--system-config` each repeated a
parameter or `$SYNCAI_SYSTEM_INI` and were used nowhere; `--teleop` was
passed on every run. Removed 2026-11-15.)

**The viewer** (`viewer/web_view.py`) is a page the host's browser draws:
`http://localhost:8080` (`web_view_port`). The site's tiles and the robot's meshes go over once (dp: 1078
tiles, 2.5 M triangles, in a second), then only poses at 20 Hz, so it draws
at the browser's own frame rate with shadows, on the host's GPU; nothing
renders in the container. Left-drag orbits, scroll zooms; the panel has the
storey switch (auto: the robot's storey and those below it, so the floor
above never hides the robot), follow-robot (the camera's look-at follows,
the offset the mouse gave it stays), the last lidar scan, the terrain boxes
and fitted stairs under the feet, the lidar map per storey, the status, and
the teleport back to the spawn, and the driving sliders
(`viewer/teleop.py`), which go to the controller's command port
(50051) and are sent only while touched, so a navigation goal and the panel
drive together. `--headless` serves no viewer.

**A controller on the joints.** With none the motors are never enabled and
the robot lies down. The robot's own controller is the section "The robot's
controller in its own container"; the bench runs the sim headless with a
stand-in or beside a controller, and reports falls, posture and the links'
counters (`docs/architecture.md` says what each field of the state datagram
is for):

```bash
python3 /home/syncrobotic/sim/scripts/controller_bench.py --fake-controller --seconds 5   # the stand-in on the four CAN buses
python3 /home/syncrobotic/sim/scripts/controller_bench.py -p joint_link:=udp -p imu_link:=none -p bms_link:=none --fake-controller --seconds 5   # on the UDP link: no CAN interface needed
python3 /home/syncrobotic/sim/scripts/controller_bench.py --send "MODE Z" --axes 0.5 0 0 --seconds 15   # a controller process running beside it
```

### The joints on CAN

On the robot the controller reaches its motors over four CAN buses, one per
leg, three J60s on each, and commands them at 1 kHz. The sim is those
motors (`joint_link: can`, the default) (`locomotion/can_link.py` is the
specification): it answers on `can1..can4` with the J60's frames, so the
same hardware layer that drives the robot drives the sim, with what that
brings — the protocol's quantisation (a position command resolves 1.2 mrad;
a zero velocity command decodes to −2.4 mrad/s), feedback only when
commanded, motors that give no torque until they are enabled, a CAN timeout
per motor that disables it until its fault is cleared, and a leg that is
lost alone when its bus is.

| bus | leg | joints | motor ids |
|---|---|---|---|
| `can1` | HR | 9..11 | 2, 3, 4 = HipX, HipY, Knee |
| `can2` | FR | 3..5 | |
| `can3` | FL | 0..2 | |
| `can4` | HL | 6..8 | |

A motor's position is its own: zero at the homed pose (the leg folded
against its stops) and positive the way it is mounted, so `motor = sign x
(joint − home)`; the table of signs and homes is in the module, and
`StandInCanController` there is the whole mapping in one place. The IMU is
not on a CAN bus (a serial port on the robot: the next section), nor are
the base's truth or the foot contacts: the hardware link's state datagram
(30010) is still sent every step and carries them.

The interfaces are virtual (vcan) and must exist on the host the sim and
the controller share, both on the host's network:

```bash
scripts/vcan_up.sh                         # on the host: can0..can4 (a privileged container on a Mac); again after a VM restart
docker compose up -d
scripts/vcan_up.sh down                    # remove them
```

**The robot's controller on these buses always runs with
`G2_SKIP_HOMING=1`.** The sim's motors power up homed; the controller's
start-up homing drives each joint to its stop and sets the zero there, and
the model's stops are the URDF's limits, not the folded leg's: with the
homing run every HipY read 0.82 rad off (2026-11-14; `can_link.py` has the
numbers).

Checked on OrbStack (2026-11-13, kernel 7.2.12: vcan built in, no vxcan,
hence the host network): the stand-in stands the robot on dp through the
four buses, 0 falls in 5 s, ~2100 CONTROL frames a second per bus (the
in-process stand-in reaches ~700 Hz a motor of its 1 kHz), each
answered. `ip link set can2 down` on the host is a leg lost: the FR motors
time out and disable themselves, the robot falls onto that leg, and they
stay limp when the bus is back until the controller clears the fault and
enables them again (ERROR_RESET, ENABLE — the pair the robot's controller
always sends).

vcan has no bitrate, arbitration or error frames — every frame arrives, at
once. `can_delay_ms` and `can_drop_rate` (one value per bus) put a bus's
latency and losses back. Not modelled: the order of the frames inside one
1 ms step, bus load, the motor's bandwidth setting and its temperature.
The frames, their replies and the timeout are the J60 manual's (V1.0.7, in
g2_deploy_rs `docs/datasheets`): the first version here guessed that a
timed-out motor damps and takes the next command, and the manual says it is
disabled. Four things remain assumptions (the timeout runs from the first
CONTROL after ENABLE, not from ENABLE: the controller's own start-up takes
longer than its timeout to reach the first motor's first command; a timeout
of 0 is none; ENABLE fails while a fault is latched; SET_HOME's reply, a
command the manual does not list): the module's docstring lists them. The
status word and the current limit go out big-endian, as the robot's
controller reads them; every other field is little-endian.

### The IMU and the BMS

The robot's controller reads two more devices beside its motors, neither on
the legs' buses, and the sim is both:

* **`imu_link: serial`** — the body IMU is a Yesense unit on a UART
  (`/dev/ttyTHS1`, 115200 baud) that streams 49-byte frames of acceleration,
  angular rate and Euler angles. The sim opens a pseudo-terminal, links
  `imu_serial_port` (`run/syncai_imu` of this checkout) to it and writes
  the same frames from the body's record, 200 a second
  (`sensor/imu_serial.py` is the specification, byte-compatible with the
  controller's reader). The reader runs in the sim's container or in one
  that shares the host's `/dev/pts` and `run/` with it, as the controller's
  does (below); a real serial device named by `imu_serial_port` is written
  instead. A port nobody reads keeps no backlog: more than 50 ms of
  unread frames is dropped, as a UART would have lost them.
* **`bms_link: can`** — the battery's BMS (Foxlink Jeff-8S3P) is alone on
  `can0`, extended frames, and silent until asked: a frame with an item's
  id is answered with that item, `0x87` starts the cycle of all of them
  (cell voltages, temperatures, pack voltage, current, state, SOC / SOH)
  and `0x86` stops it (`sensor/bms_link.py`, from the vendor's protocol
  document). Behind the numbers is a small battery: the charge falls with
  the power the robot draws — an idle 40 W plus each joint's mechanical
  power and winding heat, the latter from the J60 manual's resistance and
  torque constants — so standing on dp reads about 4 A at 31.7 V.

Checked on OrbStack (2026-11-13) with a second process in the container
reading as the controller does while the stand-in stood the robot through
the buses: 208 IMU frames in a second, gravity on z and the trunk level;
the BMS's 27 items every 100 ms after `0x87`, 31.68 V, −3.6 A, 80.0 %.
Assumed, because nothing at hand says: the IMU's rate (200 Hz; 115200 baud
carries at most 235 of this frame), the BMS cycle's period (100 ms), the
pack's capacity (15 Ah) and the idle draw. No noise or bias on the IMU, no
voltage sag, no thermal model, and the BMS never trips. Both are read by
the robot's controller itself since 2026-11-14 (next section).

### The robot's controller in its own container

The robot's controller (g2-device-control) has one mode, the robot's: it
does not know a simulator from the robot, and its repository says nothing
about this one. It runs from its own, unchanged compose file against the
sim's links; what the sim makes different for it is one file here,
`docker-compose.controller.yml`, laid over the controller's.

```bash
scripts/vcan_up.sh                                                         # on the host
docker compose up -d                                                       # the sim
docker compose -f ../g2-device-control/docker-compose.yml -f docker-compose.controller.yml up -d   # from this directory, after the sim is up
```

Two compose files, then, and why not one: the controller as a service of
`docker-compose.yml` would have to `extends` its own compose file, which
compose reads whenever it reads the project, so the sim would not start
without a g2-device-control checkout beside it. (A third file,
`docker-compose.can.yml`, set the sim's side up until 2026-11-15; that is
now what `docker-compose.yml` and the defaults are.)

What is settled between the two containers:

* **The IMU's pseudo-terminal across containers.** A container has its own
  devpts, so the sim's `/dev/pts/N` exists in no other. Both containers
  mount the host's `/dev/pts`, and the link is written in `run/` of this
  checkout (`imu_serial_port`), which the controller's container mounts at
  `/run/syncai` (`G2_IMU_PORT=/run/syncai/syncai_imu`). Docker's `devices:`,
  which the controller's file uses for the robot's UART, cannot pass a
  pseudo-terminal through (its node outside a devpts does not open), so it
  is reset.
* **No homing** (`G2_SKIP_HOMING=1`, above).
* **The nav stack's addresses.** driver_manager sends its commands to
  192.168.1.120:50051 and binds 192.168.1.103:50010 for the telemetry
  (driver_manager_params.yaml), so the controller binds 0.0.0.0:50051 —
  the panel's 127.0.0.1:50051 arrives too — and sends its telemetry and the
  BMS's to 192.168.1.103:50010. Both addresses must exist on the host, as
  the lidar's must (below: on a Mac, on the VM's loopback).
* **The controller's 1 kHz on OrbStack.** Its VM wakes a sleeper on a 3 ms
  tick, so the controller's loop ran at 334 Hz and, since it counts time in
  ticks, its policy at a third of its rate: 0.5 m in 8 s. Nothing set from
  outside changes the tick (timer slack, `select`, SCHED_FIFO, busy cores:
  measured); spinning does. `scripts/spin_sleep.sh` builds an `LD_PRELOAD`
  shim into `run/` that spins the last 3.5 ms of every sleep, and
  `G2_SPIN_SLEEP=/run/syncai/spin_sleep.so` before the controller's
  `docker compose` loads it: 997-999 Hz, for about 1.3 cores. Not needed,
  and not measured, on a Linux host.

Start the sim first: the controller configures and enables the motors once,
at its start. By its code a sim restarted under a running controller has
motors nobody enables again (it never asks for the status word); restart
the controller with the sim. Not tried.

Checked on OrbStack (2026-11-14) with a program placed as driver_manager is
(telemetry on 192.168.1.103:50010, commands to 192.168.1.120:50051) and the
shim loaded: `MODE Z` to standing in RL in 2.0 s, 2.77 m in 8 s of
`AXES 0.4 0 0`, lie down, 0 falls, no CAN timeout, `BMS_V2` ten times a
second. The controller's IMU against the body's record over those 8 s
(1615 samples, 202 a second): roll / pitch / yaw 0.0002 rad apart on
average, the gyro 0.011 rad/s, the accelerometer 0.22 m/s^2 (footfalls,
where 1 ms of lag is that much), best fit at 1 ms of lag. driver_manager
itself was not run, nor the controller's MPC (`MODE M`).

### The sim as the robot's lidar

The sim answers the Livox driver as the robot's lidar does (`lidar_link:
livox`, the default): discovery on 56000, commands on 56100, points from 56300 and the
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
docker compose up -d
```

`[livox] found by the driver` and `[livox] configured: ...` in the sim's log
say the driver has taken it; the status line counts the packets. Time is
the device's own (PTP, `time_type` 1), the host's epoch clock tied to sim
time, because Point-LIO stamps the pose with it and lio_bridge stamps its
TF with the host's clock; a sim that falls behind the wall moves it forward
(`reanchors` in the status line), never back. The localizer starts from
its own last pose or the INI's `[initial_pose]`, which is where the sim
spawns the robot, so the two agree at the start.

### Test sites: level ground, ramps, a 15 cm stair, slippery floors, a course

`scripts/make_scenes.py` writes five sites as a mapped one is written
(`octomap.bt`, `map.pcd`, `poses.txt`) into `scenes/<name>/`, generated and
gitignored, and the files a mapped site keeps here (`config/ramps/<name>.yaml`,
`config/friction/<name>.yaml`, `config/stair_walks/stairs15.yaml`,
`config/stair_shapes/stairs15.yaml`) and
a profile per site (`config/params/scenes/<name>.yaml`: the map, the spawn).
The sim reads them like any map, so nothing in it knows they are made.

| site | what | spawn |
|---|---|---|
| `flat` | 20 x 8 m of level floor, walls round it, four pillars to the sides | (0, 0) facing +x |
| `slope` | four lanes side by side, 5 / 10 / 15 / 20 deg (y 3.6 / 1.2 / -1.2 / -3.6) up to a 0.5 m platform 2 m long and down again; walls between them | (0, 1.2), the 10 deg lane ahead |
| `stairs15` | six 0.15 m risers on 0.30 m treads (x 3.0 .. 4.5) up to a 0.9 m platform 3 m long, six down (x 7.5 .. 9.0), 1.6 m wide between walls | (0, 0) facing the stair |
| `friction` | level floor, four lanes side by side (y 3 / 1 / -1 / -3), each with a strip of sliding friction 0.6 / 0.4 / 0.2 / 0.1 from x 3 to 9 | (0, 1), the 0.4 lane ahead |
| `course` | one corridor along x: friction 0.4 (x 2 .. 5), friction 0.2 (x 5 .. 8), a 0.5 m hill at x 9.5 (10 deg up, 15 deg down), a second at x 17.5 (10 deg both ways) whose way up has friction 0.4 | (0, 0) facing +x |

A floor's friction is a file of boxes per map (`config/friction/<map>.yaml`,
`terrains/friction.py`): inside one, the terrain boxes under the feet take its
sliding friction and decide the contact's (MuJoCo would otherwise take the
larger of floor and foot, and the feet's is 1.0); elsewhere, and on every map
without such a file, nothing changes. Another lane: `-p spawn:=[0,-3,0,0]`.

```bash
python3 /home/syncrobotic/sim/scripts/make_scenes.py           # in the container, once (or after editing it)
python3 -m syncai_sim --profile /home/syncrobotic/sim/config/params/scenes/stairs15.yaml
SIM_ARGS="--profile /home/syncrobotic/sim/config/params/scenes/slope.yaml" docker compose up -d
```

Checked 2026-11-15: the stair fits
as six risers of 0.150 m on 0.300 m treads (residual 0) and is built as
boxes, 1.6 m wide; the feet's floor down each lane's middle is 0.3-0.4 cm
off the design on average, at worst 0.7 / 1.2 / 1.8 / 2.2 cm on 5 / 10 /
15 / 20 deg, at the kinks, which the ramps' smoothing rounds.
A ramp is still 0.1 m columns under the feet (`terrains/terrain_patch.py`):
risers of 0.1 tan(angle), 0.9 cm at 5 deg and 3.6 cm at 20 deg. Only the
sim knows these sites: the workspace's localizer has no map of them, so
drive with the panel or AXES, not with goals.

### The whole stack around it

The sim, the robot's controller, the workspace's robot container (the Livox
driver, Point-LIO, the localizer, driver_manager, the planner), the backend
and the frontend ran together on one Mac (OrbStack, 2026-11-14/15), each
from its own repository's compose file, none of them changed for the sim:
goals sent through the backend were walked on dp. driver_manager read the
controller's telemetry at 209.5 Hz with no parse warning, the lidar arrived
at 9-9.5 Hz and Point-LIO ran at 9.2 Hz, the backend showed the battery
(79 %), no motor error and a valid localization.

The order matters, and so does what restarts with what:

1. `scripts/vcan_up.sh` and three addresses on the host, again after a
   restart of the machine or the VM: the lidar's and the driver's (above)
   and the one driver_manager sends its commands to, added the same way
   (`ip addr add 192.168.1.120/32 dev lo`).
2. The sim (`docker compose up -d`), then the controller: it configures and
   enables the motors once, at its start.
3. The robot container, the backend, the frontend.

A sim restarted alone leaves the others on a body that is gone: restart the
controller with it, and Point-LIO (the backend's `POST
/api/v1/robot/restart`), whose odometry otherwise diverges, since the sim's
time and pose start over.

What was measured there, so nobody measures it twice:

* **The sim runs at x0.95 of real time on OrbStack** with the lidar on (the
  status line's `x`; `resyncs` ~6 a second, the Livox clock's `reanchors`
  ~2.5): the VM's 3 ms timer tick plus a 6-7 ms scan exceed the 8 ms a late
  loop may catch up. Not fixed; the controller's shim (`scripts/spin_sleep`)
  would be the same cure, not tried on the sim.
* **The body walks at 0.82 of the commanded AXES** forwards under the
  controller's RL policy (PPO), which is what the real robot does (0.78-0.82)
  and why driver_manager scales its commands by 1.40: the scale is right on
  the sim as it is, not to be set to 1 here.
* **Turning in place drifts**, and that, not the navigation, is why the
  robot ends away from a goal that ends in a turn. While it turns the body
  keeps a steady velocity in its own frame — turning left 0.11-0.12 m/s
  forwards and 0.07-0.08 to the right, turning right 0.06-0.07 forwards and
  0.05 to the right — so it walks a circle: 0.32-0.38 m displaced after
  180 deg to the left, 0.15-0.21 m to the right, more at a slower turn (the
  circle's radius is v / w). Left turns show the diagonal pairs unevenly
  loaded (FL / HR on the ground 0.71-0.78 of the time, FR / HL 0.35-0.42).
  Standing still it does not drift at all, and stopping from 0.3 m/s it
  slides 0.10 m in 0.64 s, so waiting before the turn would not help. The
  nav stack's goal checker is `stateful`: once inside the 0.10 m tolerance
  it only checks the yaw, which hides the drift. HimLoco was no better. A
  matter for the policy's training; not checked on the real robot.

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
`visuals/site_tiles.py`). Two sites, each built once and cached: `voxels` (the
default) is the octomap's cubes, what the sim actually senses and stands on;
`pointcloud` is a smooth surface reconstructed from the map's `map.pcd`
(`visuals/pcd_mesh.py`: points kept only on the cleaned voxels, 4 cm, screened
Poisson, 1.5 M triangles; dp ~3.5 min; it blurs thin structure and bubbles
where the cloud is sparse). A third, `tsdf` (a TSDF surface integrated
from the mapping run's keyframe patches with open3d: thin structure kept,
dp 6 min and 2.5 GB to build), was removed 2026-11-15, unused. Both are
coloured by
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

* **SocketCAN beside UDP (2026-11-13).** "UDP only" was the rule while the
  controller's only face to the sim was its `--backend sim`. The robot's
  motors are on CAN, and a sim that is the motors on CAN lets the
  controller's real hardware layer be what is tested — its packing, its
  mapping, its timeouts — instead of a second path that only the sim
  exercises. So CAN was let in, as the robot's protocol and nothing else;
  ROS and DDS stay out. The serial port followed the same day for the same
  reason: the robot's IMU is on one, and its controller's hardware layer
  does not start without it.
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
  price is that nothing walks the sim until a controller is on its joints.
* **The robot's links are the defaults; one compose file (2026-11-15).**
  Each of the robot's links arrived as an option beside the UDP link (`-p
  joint_link:=can` and so on, a second compose file for the host's network),
  and a run as the robot took a line of arguments and two `-f`. Once the
  robot's own controller drove the sim unchanged, the robot was the only
  thing anyone started: the links became the defaults, the container is
  always on the host's network, and the flags that repeated a parameter
  went. The controller's g2-device-control keeps one mode, the robot's;
  what the sim makes different for it is `docker-compose.controller.yml`
  here, never a switch there. The UDP command link stays as `joint_link:
  udp` for runs without CAN interfaces (the stand-in, a throwaway
  container).

## Status

* **Done.** The voxel world and its cleanups, the fitted stairs and ramps, the MID360's recorded pattern, the
  browser viewer with the voxel / Poisson sites and the photo look.
  The layered package and the hardware link (2026-11-03): the sim as a
  controller's hardware, checked end to end with the stand-in controller (dp,
  5 s at 1 kHz: ~5000 states, ~4800 commands, 0 falls). ROS removed the same
  day: UDP is the only way out. The in-process policies, the kinematic
  statue and the ASCII controller link removed (2026-11-13): the sim is the
  hardware only. The joints as J60 motors on four SocketCAN buses
  (2026-11-13, `joint_link: can`), checked with a stand-in on vcan; the
  body IMU on a serial port and the BMS on `can0` (`imu_link: serial`,
  `bms_link: can`). The robot's controller on all of them from a container
  of its own (2026-11-14), and with the workspace's nav stack, the backend
  and the frontend around it. The robot's links as the defaults, one compose
  file for the sim (2026-11-15). Test sites made by script (2026-11-15).
* **Next.** The controller's MPC (`MODE M`) on the sim, not run yet.
