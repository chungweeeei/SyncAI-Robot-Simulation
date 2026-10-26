# config/lidar — the MID360's scan pattern, from the robot

`mid360_pattern.npz` is the bank of directions the sim's lidar fires when
`lidar_pattern: mid360` (the default; `syncai_sim/lidar.py`): twenty
consecutive frames of the real Livox MID360 on the G23, each sample with the
time it fired, so the sim's scans have the sensor's rosette — four lasers
sweeping 1.2° a sample, dense along the sweep, sparse across it, never the
same spots twice — and the motion smear of a 100 ms scan (each ray is cast
from where the lidar was at its firing time).

Built from the workspace's `record/dp` (the mapping run of dp, robot01,
`/robot01/livox/lidar` as livox_ros_driver2 CustomMsg; frames 100–119, i.e.
10 s into the recording) with

```bash
python3 scripts/lidar_pattern.py --bag /home/syncrobotic/robot_ws/record/dp \
    --frames 20 --start 100 --out config/lidar/mid360_pattern.npz
```

`syncai_sim/livox_bag.py` reads the rosbag2 sqlite files directly (no ROS),
keeps the samples that returned and fills no-return gaps of up to three
samples along a laser's sweep (the recording had 19.5 % no-return samples, in
runs of 2–3; what the lidar did not return is not a direction the sim should
fire, so longer runs are dropped). The recording itself is not in this
repository; the file is committed so a checkout runs without it. Any
MID360 recording of a few seconds will do to rebuild it — the pattern is the
sensor's, not the site's.
