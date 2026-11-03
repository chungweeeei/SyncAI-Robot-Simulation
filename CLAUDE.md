# CLAUDE.md

Guidance for Claude Code in this repository. `README.md` is the user-facing
document (what the sim is, how to run it, the decisions); this file is the
short list of what you must know to edit it correctly.

## What this is

A MuJoCo simulation of the SyncAI G23 on a real mapped site, talking to the
outside over **UDP only** (no ROS anywhere, removed 2026-11-03):

* **the sim walks on its own policies** (`locomotion: policy`, the default):
  the robot's ONNX policies in this process (`locomotion/policy.py`,
  `locomotion/legged.py`) on a patch of the voxel grid
  (`terrains/terrain_patch.py`). With `driver_link: udp` the sim also answers
  as the gait controller on its ASCII UDP link (`driver/controller_link.py`:
  AXES / MODE in on 50051, telemetry out to 50010).
* **the sim is a controller's hardware** (`locomotion: udp`): the joints are
  handed to a controller process over the binary hardware link
  (`locomotion/hardware_link.py`: the body's record out to 30010, joint
  commands in on 20001). The consumer is SyncAI-Robot-Control's
  `--backend sim`; this is where its MPC (LMA) is studied.

`locomotion: kinematic` keeps the older rigid body carried by rules
(`locomotion/kinematics.py`). The layers and one tick of each mode are drawn
in `docs/architecture.md`. The ONNX files live in `policy/`, gitignored
(`policy/README.md`); never commit them. Python throughout; one container
(`docker-compose.yml`, plain Ubuntu 22.04), the sim and its browser viewer
start with it (`scripts/sim_entry.sh`). The workspace
(`chungweeeei/SyncAI-Robot-Workspace`, a sibling checkout) is mounted at
`/home/syncrobotic/robot_ws` for its maps and the robot's URDF only; this
repo at `/home/syncrobotic/sim`.

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
3. **UDP is the only way out, and each link is a protocol with a consumer.**
   The hardware link's spec is the docstring of `locomotion/hardware_link.py`
   plus `tests/test_hardware_link.py` (228-byte state out, 240-byte command
   in, field order, frames, the watchdog): a change there is a change in
   SyncAI-Robot-Control's `--backend sim` too, said so in the commit message.
   The ASCII link (`driver/controller_link.py`) is the robot's controller
   protocol as driver_manager speaks it: keep it byte-compatible. Do not add
   ROS, DDS or any other transport back.
4. **Wall clock.** The sim paces fixed physics steps to real time (2 ms; 1 ms
   in the udp profile) and catches up in short bursts (`max_catch_up_s`). The
   hardware link's `t` is `epoch + sim_time`, strictly increasing across
   restarts and teleports (a controller treats a repeated time as the same
   sample).
5. **Layers stay layers.** One subpackage per layer; every `__init__.py` is a
   docstring and re-exports nothing (import by full path). The picture
   (`visuals/`, `viewer/`) is never read by anything that senses or stands;
   a backend returns a `JointCommand` and only `actuator/motors.py` writes
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
| URDF -> MJCF, mocap weld, standing pose, the joint order (`LEGS` x `JOINTS`) and default pose (`Q_DEFAULT`), the body's physics knobs (`motor_model`, `trunk_inertial`, contact, `actuator`) | `entity/robot.py` (`RobotConfig`; `DEFAULT_STANDING` was solved by FK, feet 0.372 m under the hips); their A/B `scripts/gait_bench.py` |
| MjSpec assembly, bitmasks, groups, the MJB scene cache | `scene/scene.py` (`scene_info` reads a loaded model back by name; `scene_key` lists what the cache depends on; the site's picture is not in the model) |
| the body's measurements (the body IMU, joints, the base's truth, foot contacts), posture, the trunk's contact | `sensor/proprioception.py` (`Sensors`, one float32 record per step; `tests/test_sensors.py`) |
| the lidar pattern, prop rays | `sensor/lidar.py`, `sensor/prop_rays.py`; the recorded MID360 bank `config/lidar/mid360_pattern.npz` (built by `scripts/lidar_pattern.py` through `sensor/livox_bag.py`, which reads a rosbag2 sqlite file directly) |
| the motors: the PD closed every step, the motor clamp, the command delay and torque filter, `JointCommand` | `actuator/motors.py` (`tests/test_motors.py`) |
| the policies (ported from the robot's controller) | `locomotion/policy.py` (`tests/test_policy.py`; names the source of every constant) |
| the motion states (stand / lie / damping / the policy at 50 Hz), falls | `locomotion/legged.py`: each step returns a `JointCommand` for the motors; the `integrator` and delays are `SimConfig` in `envs/simulator.py` |
| the step rule, gates, collisions (kinematic) | `locomotion/kinematics.py` (pure Python, `tests/test_kinematics.py`) |
| the hardware link: the body's record out, a controller process's command in (`locomotion: udp`); the stand-in controller | `locomotion/hardware_link.py` (its docstring is the protocol's spec; `tests/test_hardware_link.py`); `config/params/controller_profile.yaml`; the bench `scripts/controller_bench.py` |
| the sim as the gait controller on the ASCII UDP link (`driver_link: udp`); the stand-in state machine; the same surface forwarding to a controller process (`locomotion: udp`) | `driver/controller_link.py` (`tests/test_controller_link.py`); `driver/fake_driver.py`; `driver/udp_driver.py` |
| one physics tick, scan, spawn, teleport | `envs/simulator.py` |
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
python3 -m syncai_sim --headless --seconds 30 -p driver_link:=udp     # the sim as the gait controller on UDP
# the sim as a controller's hardware (locomotion udp; docs/architecture.md):
python3 /home/syncrobotic/sim/scripts/controller_bench.py --profile /home/syncrobotic/sim/config/params/controller_profile.yaml --fake-controller --seconds 5
```

`docker compose up` already runs a sim on the viewer's port and the UDP
ports; start another by hand with `SIM_AUTOSTART=0` (or `--headless` and other
ports), or the two clash. Do not run `docker compose build` on your own after
an edit; leave that to the user unless asked.
