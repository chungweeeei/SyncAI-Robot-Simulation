# CLAUDE.md

Guidance for Claude Code in this repository. `README.md` is the user-facing
document (what the sim is, how to run it, the decisions); this file is the
short list of what you must know to edit it correctly.

## What this is

A MuJoCo simulation of the SyncAI G23 on a real mapped site, talking to the
outside over **UDP, SocketCAN and a serial port only** (no ROS anywhere, removed 2026-11-03). **The sim is
the robot's hardware and nothing more** (2026-11-13): it sends out what the
sensors measure and applies the joint commands it is sent. Turning cmd_vel
into joint commands (RL or MPC) and the controller's state machine belong to
the controller process, never here.

* **the sim is a controller's hardware**: the joints are handed to a
  controller process over the binary hardware link
  (`locomotion/hardware_link.py`: the body's record out to 30010, joint
  commands in on 20001). The consumer is SyncAI-Robot-Control's
  `--backend sim` (not written yet); with nothing on the link the motors are
  in damping and the robot lies down. The viewer panel's keys and AXES go to
  the controller's command port, 50051 (`driver/udp_driver.py`).
* **the sim is the robot's motors on CAN** (`joint_link: can`): the twelve
  joints as J60 motors on four SocketCAN buses, one per leg, with the
  robot's frames, ids, signs and homed zero (`locomotion/can_link.py`), so a
  controller's real SocketCAN hardware layer drives the sim. vcan interfaces
  (`scripts/vcan_up.sh`), host network only. The state datagram still goes
  out: the IMU, the truth and the contacts are on no CAN bus.
* **the sim is the robot's IMU and BMS** (`imu_link: serial`, `bms_link:
  can`): the body IMU as the robot's Yesense streaming on a serial port (a
  pseudo-terminal, `sensor/imu_serial.py`) and the pack's Foxlink BMS
  answering on `can0` (`sensor/bms_link.py`), the two devices the
  controller's hardware layer reads beside its motors.
* **the sim is the robot's lidar** (`lidar_link: livox`): a MID360 / MID360s
  on the Livox protocol (`sensor/livox_link.py`), read by the workspace's own
  livox_ros_driver2, so Point-LIO, the localizer and the planner run on it.
  Keep it byte-compatible with the vendored Livox-SDK2.

Removed 2026-11-13, do not bring back: the in-process ONNX policies
(`locomotion: policy`), the kinematic statue (`locomotion: kinematic`), the
sim answering as the gait controller on the ASCII link (`driver_link: udp`).
The layers and one tick are drawn in `docs/architecture.md`. Python
throughout; one container (`docker-compose.yml`, plain Ubuntu 22.04), the
sim and its browser viewer start with it (`scripts/sim_entry.sh`). The
workspace (`chungweeeei/SyncAI-Robot-Workspace`, a sibling checkout) is
mounted at `/home/syncrobotic/robot_ws` for its maps and the robot's URDF
only; this repo at `/home/syncrobotic/sim`.

## Rules that shape edits

1. **The workspace is read, never written.** The sim takes the maps
   (`map/<name>/`) and the URDF and meshes (`src/syncai_bringup`) from it;
   the instance INI is bind-mounted over its `config/system.ini`. If the sim
   needs something there, that is a change in that repository.
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
3. **UDP, SocketCAN and the IMU's serial port are the only ways out, and
   each link is a protocol with a consumer.**
   The hardware link's spec is the docstring of `locomotion/hardware_link.py`
   plus `tests/test_hardware_link.py` (228-byte state out, 240-byte command
   in, field order, frames, the watchdog): a change there is a change in
   SyncAI-Robot-Control's `--backend sim` too, said so in the commit message.
   The panel's lines to the controller (`driver/udp_driver.py`) are the
   robot's controller protocol as driver_manager speaks it: keep them
   byte-compatible. The CAN link (`locomotion/can_link.py`, 2026-11-13) is
   the J60's protocol and the robot's wiring as its controller speaks them:
   keep it byte-compatible with the robot, never with the sim's convenience
   (`tests/test_can_link.py` pins the bytes and the mapping); where the
   J60's manual (g2_deploy_rs `docs/datasheets/J60.pdf`) and a guess differ,
   the manual wins. The same holds for the IMU's Yesense frames
   (`sensor/imu_serial.py`, read by the controller's `hardware/imu.rs`) and
   the BMS's items (`sensor/bms_link.py`, the vendor's protocol document
   beside the J60's). Do not add ROS, DDS or any other transport back.
4. **Wall clock.** The sim paces fixed physics steps to real time (1 ms) and
   catches up in short bursts (`max_catch_up_s`, 8 ms),
   dropping what one burst cannot repay (`envs/pacing.py`: carrying it put
   the MID360's stamps in the future). The
   hardware link's `t` is `epoch + sim_time`, strictly increasing across
   restarts and teleports (a controller treats a repeated time as the same
   sample).
5. **Layers stay layers.** One subpackage per layer; every `__init__.py` is a
   docstring and re-exports nothing (import by full path). The picture
   (`visuals/`, `viewer/`) is never read by anything that senses or stands;
   the link returns a `JointCommand` and only `actuator/motors.py` writes
   the joints; only `sensor/` reads the body for others.
6. **Comments explain why**, and record measurements and rejected
   alternatives. A bare number without its reason is out of place.
7. Python: `ruff.toml` (py310, 100 columns; `uvx ruff check syncai_sim tests
   scripts` on the host). Tests: `python3 -m pytest /home/syncrobotic/sim/tests`
   in the container; `tests/test_simulator_site.py` runs the whole sim on the
   workspace's `map/dp` and skips without it.

## Where things happen

Paths are under `syncai_sim/`; `docs/architecture.md` draws the layers.

| Need | Place |
|---|---|
| map -> voxel world (parse, mesh, plates, cache); ray cast / ground / blocked | `world/`: `build.py`, `btree.py`, `greedy_mesh.py`; `voxel_grid.py` (numba); `storeys.py` |
| the feet's floor heights, flights fitted as staircases; the voxels -> boxes under the feet | `terrains/`: `surface_heights.py`, `stairs.py`, `terrain_patch.py` (numba, `tests/test_terrain_patch.py`); stairs the track never climbed: `config/stair_walks/<map>.yaml`; slopes across voxel layers: `config/ramps/<map>.yaml` |
| URDF -> MJCF, the joint order (`LEGS` x `JOINTS`) and default pose (`Q_DEFAULT`), the body's physics knobs (`motor_model`, `trunk_inertial`, contact) | `entity/robot.py` (`RobotConfig`; the defaults in `config/params/sim_params.yaml` are the vendor's) |
| MjSpec assembly, bitmasks, groups, the MJB scene cache | `scene/scene.py` (`scene_info` reads a loaded model back by name; `scene_key` lists what the cache depends on; the site's picture is not in the model) |
| the body's measurements (the body IMU, joints, the base's truth, foot contacts), posture, the trunk's contact | `sensor/proprioception.py` (`Sensors`, one float32 record per step; `tests/test_sensors.py`) |
| the sim as the robot's MID360 on the Livox protocol (`lidar_link: livox`): discovery, commands, point and IMU packets, the device clock | `sensor/livox_link.py` (its docstring is the protocol's spec, with the vendored SDK's file for each rule; `tests/test_livox_link.py` pins the SDK's own bytes); the raw scan it sends is `Simulator.last_raw_scan`, the IMU `Simulator.lidar_imu` |
| the lidar pattern, prop rays | `sensor/lidar.py`, `sensor/prop_rays.py`; the recorded MID360 bank `config/lidar/mid360_pattern.npz` (built by `scripts/lidar_pattern.py` through `sensor/livox_bag.py`, which reads a rosbag2 sqlite file directly) |
| the motors: the PD closed every step (a generalized force, never a MuJoCo actuator), the motor clamp, the command delay and torque filter, `JointCommand` | `actuator/motors.py` (`tests/test_motors.py`) |
| the hardware link: the body's record out, a controller process's command in, the watchdog; the stand-in controller | `locomotion/hardware_link.py` (its docstring is the protocol's spec; `tests/test_hardware_link.py`); the bench `scripts/controller_bench.py` |
| the joints as J60 motors on four SocketCAN buses (`joint_link: can`): the frames, the wiring (bus, id, sign, homed zero per joint), enable / timeout (the motor disables itself, fault latched) / set-home, per-bus delay and drop; the stand-in | `locomotion/can_link.py` (its docstring is the spec, and names its four assumptions; `tests/test_can_link.py`); the interfaces `scripts/vcan_up.sh` |
| the body IMU as the robot's Yesense on a serial port (`imu_link: serial`): the frame, the rate, the pseudo-terminal and its backlog | `sensor/imu_serial.py` (its docstring is the spec; `tests/test_imu_serial.py` ports the controller's reader) |
| the pack's BMS on `can0` (`bms_link: can`): the items, request / cycle, the battery model (charge against the joints' power) | `sensor/bms_link.py` (its docstring is the spec; `tests/test_bms_link.py`) |
| the panel's motion keys, policy number and AXES forwarded to the controller | `driver/udp_driver.py` |
| one physics tick, scan, spawn, teleport, falls counted; where the base is put down | `envs/simulator.py`; `envs/placement.py` |
| the main loop, the links polled and ticked, the settings (defaults, `config/params/sim_params.yaml`, `--profile`, `-p`) | `__main__.py` (sets `OPENBLAS_NUM_THREADS=1` before numpy); `config.py` |
| the viewer (viser: the site as GLB tiles once, poses at 20 Hz, drawn by the host's browser; nothing renders in the container), its panel's driving | `viewer/web_view.py`, `viewer/teleop.py` |
| the viewer's look (tiles, intensity shading, the Poisson and TSDF surfaces, colours, the lidar map layer) | `visuals/`: `site_tiles.py` (10 m tiles as GLB, a texel per voxel face), `voxel_intensity.py`, `pcd_mesh.py`, `tsdf_mesh.py`, `visual.py`, `map_cloud.py` (picture only, never sensing); the site, storey and layer switches are the panel in `viewer/web_view.py` |
| the container's command | `scripts/sim_entry.sh` (`SIM_AUTOSTART`, `SIM_ARGS`) |

## Running and testing (inside the container)

```bash
docker compose up -d                 # the sim + its viewer start: http://localhost:8080 (scripts/sim_entry.sh)
docker compose logs -f sim
docker compose exec sim bash
python3 -m pytest /home/syncrobotic/sim/tests
python3 -m syncai_sim --headless --seconds 30     # no controller on the link: the motors in damping
# the link end to end with the stand-in controller (docs/architecture.md):
python3 /home/syncrobotic/sim/scripts/controller_bench.py --fake-controller --seconds 5
# the same over four CAN buses (on the host first: scripts/vcan_up.sh; SIM_NETWORK_MODE=host):
python3 /home/syncrobotic/sim/scripts/controller_bench.py -p joint_link:=can --fake-controller --seconds 5
# the robot's controller in its own container (README, "The robot's controller in its own container"):
#   docker compose -f docker-compose.yml -f docker-compose.can.yml up -d
#   docker compose -f ../g2-device-control/docker-compose.yml -f docker-compose.controller.yml up -d
# (its compose file unchanged: what the sim makes different is docker-compose.controller.yml's, here)
# with the IMU on /tmp/syncai_imu and the BMS on can0 as well:
python3 /home/syncrobotic/sim/scripts/controller_bench.py -p joint_link:=can -p imu_link:=serial -p bms_link:=can --fake-controller --seconds 5
```

`docker compose up` already runs a sim on the viewer's port and the UDP
ports; start another by hand with `SIM_AUTOSTART=0` (or `--headless` and other
ports), or the two clash. Do not run `docker compose build` on your own after
an edit; leave that to the user unless asked.
