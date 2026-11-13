# The layers

What the sim is made of, one subpackage per layer, and what crosses between
them. The rule that organises it: **the voxel grid is the truth, MuJoCo is the
robot, the props and the feet's floor, the browser is the picture, and the
joints are commanded by a controller process** at the other end of a UDP
link, whose RL or MPC decides. The sim is the robot's hardware: it decides
no motion of its own (since 2026-11-13; before, it could walk the robot's
ONNX policies in-process or carry a rigid statue by rules).

```
   map/<name>/octomap.bt, map.pcd, poses.txt, the mapping bag
          │
          ▼
   world/        octomap -> VoxelGrid: the TRUTH for lidar rays, the ground
   (truth)       under the robot, whether the trunk is blocked (numba)
          │
          ├──────────────► visuals/   tiles, the Poisson / TSDF surfaces, the
          │                (picture)  intensity shading, the photo look — drawn
          │                           by the host's browser (viewer/), read by
          │                           nothing that senses or stands
          ▼
   terrains/     the feet's floor: measured heights, flights fitted as
                 staircases, a pool of voxel boxes moved under the robot
          │
          ▼
   scene/        one MjSpec: the robot (entity/), the collision plates for
                 props, the props, the terrain pool — cached as MJB; the site's
                 picture is NOT in it (contype / conaffinity table in scene.py)
          │
          ▼
   ┌─────────────────────────── MuJoCo ────────────────────────────┐
   │   mj_step on 1 ms steps, paced to the wall clock              │
   └───────────┬───────────────────────────────────────┬───────────┘
               │ reads                                 ▲ writes
               ▼                                       │
   sensor/                                       actuator/
     proprioception.py  the body IMU, q/dq/tau,    motors.py  JointCommand
                        the base's truth, foot               (q_des dq_des kp
                        contacts: one float32                kd tau_ff) ->
                        record per step; posture;            tau = kp(q_des-q)
                        the trunk's contact                  + kd(dq_des-dq)
     lidar.py           the MID360's pattern cast            + tau_ff, clamped
                        on world/ (never mj_multiRay)        (J60 envelope or
               │                                             the URDF effort)
               │                                       ▲
               ▼                                       │ JointCommand
   locomotion/hardware_link.py   every step's record out, the controller's
                     command in — a controller PROCESS (SyncAI-Robot-Control
                     --backend sim) runs its RL or LMA against this
               │
   locomotion/can_link.py   `joint_link: can`: the same joints as J60 motors on
                     four SocketCAN buses, one per leg, as on the robot — the
                     controller's REAL hardware layer runs against this; the
                     record above still goes out (IMU, truth, contacts)
               │
   driver/udp_driver.py   the panel's motion keys, policy number and AXES,
                     forwarded to the controller's command port as ASCII
               │
   envs/simulator.py   the composition root: one tick, scan, spawn, teleport
               │
       ┌───────┴────────┐
   __main__.py       viewer/web_view.py + teleop.py
   (the main loop:   (the browser; the panel's keys and sliders)
    wall clock, the
    UDP links)
```

## One physics tick

```
terrain patch moved if due
-> JointCommand = link.poll(sim_time)       the controller's newest, or damping
                or can.poll(sim_time, sensors)   joint_link can: the frames that came
                                                 in are answered, each motor's command
                                                 (or none, or damping) returned
-> motors.apply(JointCommand)
-> props walked -> mj_step -> sim_time += dt
-> sensors.read(epoch + sim_time); link.send(sensors)
-> the lidar pose recorded -> state read back (posture, collisions, falls)
```

The 55-float record is built in place and sent as one datagram per step.
cmd_vel never reaches the body: the panel's AXES go to the controller
(driver/udp_driver.py), which turns them into joint commands.

Everything the sim tells the outside goes over UDP or, for the joints with
`joint_link: can`, SocketCAN (removed 2026-11-03: the ROS node, the nav
session and the image's ROS base).

## What a controller needs from the sim, and why

The hardware link (`locomotion/hardware_link.py`, its docstring is the
specification; `tests/test_hardware_link.py` checks it byte for byte) sends
one state datagram per physics step and takes a command datagram back.

| field | why it is there |
|---|---|
| `t` (f64, strictly increasing) | a whole-body controller treats a repeated time as the same sample and returns its cached output; its filter takes the time difference as dt. `epoch + sim_time`, the epoch from the wall clock, so a restart or a teleport never hands a running controller a time it has seen |
| `rpy`, `gyro` | the base's attitude (ZYX) and body-frame rate: every policy's observation, the MPC's base state |
| `acc` | the body IMU's specific force (+9.81 up at rest): what a Kalman filter fuses for base velocity. The body's IMU, on `base_link`, not the lidar's 0.4 m up (its accelerometer carries the lever arm) |
| `q`, `dq` | the joints, FL FR HL HR x HipX HipY Knee |
| `tau` | the torque actually applied this step (clamp included): a filter confirms a planned foot contact from the foot force it computes with the Jacobian transpose of the joint torques |
| `base_pos`, `base_linvel` (world) | ground truth: fed straight to an NMPC while its own estimate is not trusted, or used to grade that estimate. World frame, as the free joint's translational velocity is |
| `contact[4]` | 1.0 where a foot's sphere touches a geom of another root body (the floor, a stair, a prop): contact sensing as a controller has it, not a force threshold |

| command | what the motors do with it |
|---|---|
| `kp, q_des, kd, dq_des, tau_ff` | `tau = kp (q_des − q) + kd (dq_des − dq) + tau_ff`, every step until the next command, through the motor model's clamp. A whole-body controller sends kp 0, kd ≈ 3, tau_ff the torque; stand / lie / RL send kp, kd, q_des |

Two things the link does that a motor driver would not: a **watchdog**
(100 ms without a command → damping, kp 0 / kd 2, logged once), so a dead
controller does not leave its last torque on the joints; and a **short
catch-up burst** (`max_catch_up_s`, 8 ms), so a late loop never
hands the controller a quarter second of states while its command stays
the same.

## Running it

```bash
# the sim as a controller's hardware: nothing on the link, the motors in damping
python3 -m syncai_sim --teleop

# the link end to end with no controller process: the stand-in's PD stands the robot
python3 /home/syncrobotic/sim/scripts/controller_bench.py --fake-controller --seconds 5

# a controller process on the link (its --backend sim, ports 20001 / 30010 / 50051):
python3 /home/syncrobotic/sim/scripts/controller_bench.py \
    --send "MODE Z" --send "MODE M" --gap 3 --axes 0.5 0 0 --seconds 15 --listen
```

The defaults (config/params/sim_params.yaml) set the body the way a
controller's own development simulation models it (the J60 envelope, the
vendor trunk and contact, 1 kHz, the hand-computed torque) so that what a
controller sees here is what it would see there; every value says why.

## What is next

The controller process itself: SyncAI-Robot-Control's `--backend sim` against
this specification, in a container of its own beside the sim's (Ubuntu 22.04,
Rust, the C++ MPC stack), and `controller_bench.py --send "MODE M"` against
its LMA. Not in this repository yet.
