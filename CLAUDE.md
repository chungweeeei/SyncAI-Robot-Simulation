# CLAUDE.md

Guidance for Claude Code in this repository. `README.md` is the user-facing
document (what the sim is, how to run it, the decisions); this file is the
short list of what you must know to edit it correctly.

## What this is

A MuJoCo simulation of the SyncAI G23 in a real mapped site, so the navigation
stack of `chungweeeei/SyncAI-Robot-Workspace` (a sibling checkout, mounted into
the container) runs end to end with no robot. It is **not** a legged
simulation: there is no gait policy, the robot is a rigid body carried
kinematically (`syncai_sim/kinematics.py`). Python throughout (decided
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
   gets a visual mesh (`contype 0`) and the large merged boxes as collision
   plates for props. Do not route sensing through `mj_multiRay` or collide the
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
   paces fixed 5 ms steps to real time. Never publish a stamp that is not `now`.
5. **Comments explain why**, as in the workspace, and record measurements and
   rejected alternatives. A bare number without its reason is out of place.
6. Python: `ruff.toml` (the workspace's rule set, py310, 100 columns). Tests:
   `pytest tests` in the container; `test_simulator_synthetic.py` needs the
   ROS environment sourced (it runs the planner CLI through `ros2 run`).

## Where things happen

| Need | Place |
|---|---|
| map -> world (parse, mesh, plates, cache) | `world.py`, `btree.py`, `greedy_mesh.py` |
| ray cast / ground / blocked | `voxel_grid.py` |
| the step rule, gates, collisions | `kinematics.py` (pure Python, `tests/test_kinematics.py`) |
| URDF -> MJCF, mocap weld, standing pose | `robot.py` (`DEFAULT_STANDING` was solved by FK, feet 0.372 m under the hips) |
| MjSpec assembly, bitmasks, groups | `scene.py` |
| the lidar pattern, prop rays | `lidar.py`, `prop_rays.py` |
| services and `mode` | `fake_driver.py` |
| one physics tick, scan, spawn, teleport | `simulator.py` |
| publishers, subscribers, the main loop | `ros_node.py`; CLI in `__main__.py`; viewer + teleop in `viewer.py` |
| the session, the display, the test driver | `scripts/run_session.sh`, `scripts/vnc.sh`, `scripts/nav_test.py` |

## Running and testing (inside the container)

```bash
docker compose up -d && docker compose exec sim bash
source /opt/ros/humble/setup.bash && source /opt/ros2_rust_underlay/install/setup.bash && source install/setup.bash
/home/syncrobotic/sim/scripts/vnc.sh                 # http://localhost:6080/vnc.html?autoconnect=1&resize=remote
/home/syncrobotic/sim/scripts/run_session.sh         # logs: log/stack/sim01/sim/<pane>/current
python3 /home/syncrobotic/sim/scripts/nav_test.py --goal 3 0 0
python3 -m pytest /home/syncrobotic/sim/tests
```

Do not run `docker compose build` or the session on your own after an edit;
leave that to the user unless asked — the session starts the workspace's whole
nav stack.
