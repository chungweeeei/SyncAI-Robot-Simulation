# The layers

What the sim is made of, one subpackage per layer, and what crosses between
them. The rule that organises it: **the voxel grid is the truth, MuJoCo is the
robot, the props and the feet's floor, the browser is the picture, and the
joints are commanded by whichever locomotion backend is chosen** — the
kinematic rules, the in-process RL policies, or a controller process at the
other end of a UDP link, whose RL or MPC then decides.

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
   │   mj_step on 1-2 ms steps, paced to the wall clock            │
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
   locomotion/   who decides the joint command         │
     kinematics.py   `kinematic`: the statue carried by rules (moves the mocap;
                     no joints, no motors)
     legged.py       `policy`: the controller's motion states + the ONNX
       policy.py     runners at 50 Hz, in this process
     hardware_link.py `udp`: every step's record out, the controller's
                     command in — a controller PROCESS (SyncAI-Robot-Control
                     --backend sim) runs its RL or LMA against this
               │
   driver/     the nav stack's side of the gait controller
     fake_driver.py      the controller's state machine stood in for (motion
                         keys, policy mode, the switch delay) — policy / kinematic
     udp_driver.py       the same surface forwarding MODE / ESTOP / AXES to the
                         controller process — udp
     controller_link.py  the sim AS the controller on driver_manager's ASCII
                         link (the nav session, `driver_link: udp`)
               │
   envs/simulator.py   the composition root: one tick, scan, spawn, teleport
               │
       ┌───────┴────────┐
   ros/ros_node.py   viewer/web_view.py + teleop.py
   (the nav session) (the browser; the panel's keys and sliders)
```

## One physics tick, walking

```
driver.update() -> cmd_vel (clamped) -> terrain patch moved if due
-> JointCommand = legs.step(cmd_vel, policy, motion_state)        policy
               or link.poll(sim_time)                              udp
-> motors.apply(JointCommand)                                      every step
-> props walked -> mj_step -> sim_time += dt
-> [udp] sensors.read(epoch + sim_time); link.send(sensors)        every step
-> the lidar pose recorded -> state read back (posture, collisions, falls)
```

The RL path pays nothing for the record or the packet: its runner reads
MjData itself at 50 Hz, and the posture check and the trunk's contact test
are separate queries. The udp path builds the 55-float record in place and
sends it as one datagram per step.

## The three locomotions

| `locomotion` | the joints | the driver | who walks | runs under |
|---|---|---|---|---|
| `kinematic` | none: a rigid statue on a mocap weld, placed by the step rule | `FakeDriver` | `locomotion/kinematics.py` | ROS, `--no-ros` |
| `policy` | twelve PD joints on the terrain patch | `FakeDriver` | the ONNX policies in this process (`legged.py`, `policy.py`) | ROS (the nav session), `--no-ros`, `scripts/gait_bench.py` |
| `udp` | the same, commanded over the hardware link | `UdpDriver` (forwards keys and AXES to the controller) | a controller process: its RL or its MPC | `--no-ros` only, `scripts/controller_bench.py` |

`udp` is ROS-free on purpose: in that mode the sim talks UDP only, and the
nav stack is not what is under study. `python3 -m syncai_sim` refuses it
without `--no-ros`; `ros/ros_node.py` knows nothing of it.

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
catch-up burst** (`max_catch_up_s`, 8 ms in the profile), so a late loop never
hands the controller a quarter second of states while its command stays
the same.

## Running it

```bash
# the RL path, as before
python3 -m syncai_sim --no-ros --teleop

# the sim as a controller's hardware: nothing on the link, the motors in damping
python3 -m syncai_sim --no-ros --teleop --profile /home/syncrobotic/sim/config/params/controller_profile.yaml

# the link end to end with no controller process: the stand-in's PD stands the robot
python3 /home/syncrobotic/sim/scripts/controller_bench.py \
    --profile /home/syncrobotic/sim/config/params/controller_profile.yaml --fake-controller --seconds 5

# a controller process on the link (its --backend sim, ports 20001 / 30010 / 50051):
python3 /home/syncrobotic/sim/scripts/controller_bench.py \
    --profile /home/syncrobotic/sim/config/params/controller_profile.yaml \
    --send "MODE Z" --send "MODE M" --gap 3 --axes 0.5 0 0 --seconds 15 --listen
```

`config/params/controller_profile.yaml` sets the body the way a controller's
own development simulation models it (the J60 envelope, the vendor trunk and
contact, 1 kHz, the hand-computed torque) so that what a controller sees here
is what it would see there; every value says why.

## What is next

The controller process itself: SyncAI-Robot-Control's `--backend sim` against
this specification, in a container of its own beside the sim's (Ubuntu 22.04,
Rust, the C++ MPC stack), and `controller_bench.py --send "MODE M"` against
its LMA. Not in this repository yet.
