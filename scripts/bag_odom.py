#!/usr/bin/env python3
"""Log Point-LIO's odometry while a bag replays, for scripts/colour_map.py:
`stamp x y z qw qx qy qz` per line of nav_msgs/Odometry on
/<robot_id>/pointlio/lio_odom — the body (the MID360's IMU, pointlio_body)
in the odometry frame, which is the map's own frame when the bag is the
map's mapping run (outdoor_1027's replay met its keyframes within 5 cm).

    python3 scripts/bag_odom.py record/<bag>/odom.txt [--robot-id robot01]

The replay is the workspace's docs/localizer-offline-replay.md, replay A
without the localizer, on a DDS domain of its own so that nothing reaches a
live stack. Point-LIO and the player in the workspace's throwaway build
container, this logger in the sim container (both on the host's network):

    # host, workspace root
    docker compose -f docker-compose.build.yml run --rm --entrypoint bash \\
        -e ROS_DOMAIN_ID=42 build
    # inside it
    source /opt/ros/humble/setup.bash && source /opt/ros2_rust_underlay/install/setup.bash \\
        && source install/setup.bash
    ros2 launch syncai_pointlio pointlio.launch.py system_config:=$PWD/record/<bag>/replay.ini &
    # host, sim repo
    docker exec -e ROS_DOMAIN_ID=42 syncai-sim bash -lc 'source /opt/ros/humble/setup.bash && \\
        source /opt/ros2_rust_underlay/install/setup.bash && cd /home/syncrobotic/robot_ws && \\
        python3 /home/syncrobotic/sim/scripts/bag_odom.py record/<bag>/odom.txt' &
    # back in the build container
    ros2 bag play record/<bag> --topics /robot01/livox/lidar /robot01/livox/imu

replay.ini names the robot_id and the map: `[system] robot_id = robot01`,
`[map] name = <map>` and `pcd = /home/syncrobotic/robot_ws/map/<map>/map.pcd`
(an absolute path: the doc says why). Play at 1x and to the end — a 9 min bag
takes 9 min; stop the logger with Ctrl-C afterwards. A bag that starts in
motion tilts the whole replay (Point-LIO's gravity init): the mapping run
stood still first, as it should.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="the file to write (overwritten)")
    ap.add_argument("--robot-id", default="robot01")
    a = ap.parse_args(argv)

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data

    rclpy.init()
    node = rclpy.create_node("bag_odom")
    f = open(a.out, "w", buffering=1)
    f.write("# stamp x y z qw qx qy qz (pointlio/lio_odom: the body in the odometry frame)\n")
    n = 0

    def cb(m: Odometry) -> None:
        nonlocal n
        p, q = m.pose.pose.position, m.pose.pose.orientation
        f.write(f"{m.header.stamp.sec + m.header.stamp.nanosec * 1e-9:.4f} "
                f"{p.x:.4f} {p.y:.4f} {p.z:.4f} {q.w:.6f} {q.x:.6f} {q.y:.6f} {q.z:.6f}\n")
        n += 1
        if n % 600 == 0:
            print(f"[bag_odom] {n} samples, at ({p.x:.1f}, {p.y:.1f}, {p.z:.1f})", flush=True)

    node.create_subscription(Odometry, f"/{a.robot_id}/pointlio/lio_odom", cb,
                             qos_profile_sensor_data)
    print(f"[bag_odom] logging /{a.robot_id}/pointlio/lio_odom to {a.out}; Ctrl-C when the bag "
          f"has played out", file=sys.stderr, flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    f.close()
    print(f"[bag_odom] {n} samples written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
