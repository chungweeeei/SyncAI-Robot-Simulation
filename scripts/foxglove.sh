#!/usr/bin/env bash
# Foxglove without the whole session: robot_state_publisher and lio_bridge (each
# only if none is up) and foxglove_bridge, for a sim started by hand as a ROS node:
#
#   python3 -m syncai_sim --teleop      # in one shell (not --no-ros: Foxglove reads topics)
#   /home/syncrobotic/sim/scripts/foxglove.sh
#
# Then, on the host, Foxglove (desktop app or app.foxglove.dev) -> Open
# connection -> Foxglove WebSocket -> ws://localhost:8765. The bridge (3.6,
# the apt package) speaks only the foxglove.sdk.v1 subprotocol; a client
# offering just the older foxglove.websocket.v1 is refused with HTTP 400, so a
# client built on the old protocol (Lichtblick, older Foxglove Studio) may not
# connect. The session (run_session.sh) starts the bridge itself.
#
# Why lio_bridge: the sim's TF is map -> pointlio_odom -> pointlio_body and
# robot_state_publisher's is base_link -> ... -> lidar_top_imu; lio_bridge is
# what joins them (map -> <id>/odom -> base_link), as on the robot. Without it
# Foxglove has two TF trees and cannot place the robot model on the map
# (tf2_echo map sim01/base_link: "not part of the same tree", 2026-10-24).
set -euo pipefail
cd /home/syncrobotic/robot_ws
robot_id=$(python3 -c "import sys; sys.path.insert(0, '/home/syncrobotic/sim'); from syncai_sim.config import read_ini; print(read_ini().robot_id)")
# The layout to import in Foxglove, for this robot_id: the template's topics
# are absolute (/<robot_id>/...), so it is written per id, beside the template
# (gitignored), visible on the host through the repo's bind mount.
layout=/home/syncrobotic/sim/config/foxglove/layout-${robot_id}.json
sed "s#{robot_id}#${robot_id}#g" /home/syncrobotic/sim/config/foxglove/layout.template.json > "${layout}"
echo "Foxglove layout for ${robot_id}: config/foxglove/layout-${robot_id}.json (layout menu -> Import from file)"
pids=()
nodes=$(ros2 node list 2>/dev/null || true)
if ! grep -q "/${robot_id}/robot_state_publisher" <<<"${nodes}"; then
    ros2 launch /home/syncrobotic/sim/launch/rsp.launch.py > /tmp/sim-rsp.log 2>&1 &
    pids+=($!)
    echo "robot_state_publisher started (log /tmp/sim-rsp.log)"
fi
if ! grep -q "/${robot_id}/lio_bridge_node" <<<"${nodes}"; then
    ros2 launch syncai_lio_bridge lio_bridge.launch.py > /tmp/sim-lio_bridge.log 2>&1 &
    pids+=($!)
    echo "lio_bridge started (log /tmp/sim-lio_bridge.log)"
fi
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
echo "foxglove_bridge on ws://localhost:8765 (Ctrl-C to stop)"
ros2 run foxglove_bridge foxglove_bridge --ros-args \
    --params-file /home/syncrobotic/sim/config/foxglove/bridge.yaml
