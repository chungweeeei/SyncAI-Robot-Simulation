#!/usr/bin/env python3
"""Send the simulated robot somewhere and report how it went.

    scripts/nav_test.py --goal X Y YAW [--bt move.xml] [--timeout 180]
    scripts/nav_test.py --goal X Y YAW --expect-fail      # a negative test

Runs inside the sim container with the ROS environment sourced and the session
up (scripts/run_session.sh). It:

  1. waits for planner_server/Octo3D/map_available (latched) — the surface map
     is built in a background thread and a goal before it fails at once;
  2. reads sim/ground_truth for the robot's current ground z and sends a
     NavigateToGoal with the goal's z set to the ground under the goal, found
     by asking the sim (sim/status carries z_ground; for a goal on another
     floor pass --goal-z) — Octo3D's goal_z_mode is `request` and the
     controller's goal check has goal_z_tolerance 0.5;
  3. watches sim/status while the action runs and prints collisions,
     refused steps and the policy / motion state as they change;
  4. exits 0 on SUCCEEDED with zero collisions (or, with --expect-fail, on a
     failed navigation), 1 otherwise, printing error_code / error_msg.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from syncai_common.action import NavigateToGoal  # noqa: E402  (workspace install sourced)


class NavTest(Node):
    def __init__(self, robot_id: str):
        super().__init__("nav_test", namespace=f"/{robot_id}")
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.map_available = None
        self.status = {}
        self.gt = None
        self.create_subscription(Bool, "planner_server/Octo3D/map_available",
                                 self._on_map, latched)
        self.create_subscription(DiagnosticArray, "sim/status", self._on_status, 10)
        self.create_subscription(Odometry, "sim/ground_truth", self._on_gt, 10)
        self.client = ActionClient(self, NavigateToGoal, "navigate_to_goal")
        self.feedback_msg = ""

    def _on_map(self, msg):
        self.map_available = msg.data

    def _on_status(self, msg):
        for s in msg.status:
            if s.name == "sim":
                self.status = {kv.key: json.loads(kv.value) for kv in s.values}

    def _on_gt(self, msg):
        self.gt = msg

    def _on_feedback(self, fb):
        m = fb.feedback.status_msg
        if m != self.feedback_msg:
            self.feedback_msg = m
            self.get_logger().info(f"feedback: {m}")

    def spin_until(self, pred, timeout, what):
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if pred():
                return True
        self.get_logger().error(f"timed out waiting for {what}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal", nargs=3, type=float, metavar=("X", "Y", "YAW"), required=True)
    ap.add_argument("--goal-z", type=float, default=None, help="ground z at the goal (default: the robot's)")
    ap.add_argument("--bt", default="", help="behavior tree (empty = move.xml; the sim map picks *_3d)")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--expect-fail", action="store_true")
    ap.add_argument("--robot-id", default=None, help="default: [system] robot_id of config/system.ini")
    a = ap.parse_args()

    robot_id = a.robot_id
    if robot_id is None:
        import configparser
        c = configparser.ConfigParser()
        c.read("config/system.ini")
        robot_id = c.get("system", "robot_id", fallback="sim01").strip()

    rclpy.init()
    n = NavTest(robot_id)
    try:
        if not n.spin_until(lambda: n.map_available is not None, 30, "map_available"):
            return 1
        n.get_logger().info(f"map_available = {n.map_available}")
        if not n.spin_until(lambda: n.gt is not None and n.status, 10, "sim/ground_truth + status"):
            return 1
        if not n.client.wait_for_server(timeout_sec=20.0):
            n.get_logger().error("navigate_to_goal action server not up")
            return 1

        goal = NavigateToGoal.Goal()
        goal.behavior_tree = a.bt
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = n.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = a.goal[0], a.goal[1]
        pose.pose.position.z = a.goal_z if a.goal_z is not None else float(n.status.get("z_ground", 0.0))
        pose.pose.orientation.z = math.sin(a.goal[2] / 2)
        pose.pose.orientation.w = math.cos(a.goal[2] / 2)
        goal.pose = pose
        start = dict(n.status)
        n.get_logger().info(f"goal ({a.goal[0]:.2f}, {a.goal[1]:.2f}, yaw {a.goal[2]:.2f}) z {pose.pose.position.z:.2f}; "
                            f"robot at ({start.get('x')}, {start.get('y')}) z {start.get('z_ground')}")
        t0 = time.time()
        send = n.client.send_goal_async(goal, feedback_callback=n._on_feedback)
        if not n.spin_until(lambda: send.done(), 10, "goal acceptance"):
            return 1
        handle = send.result()
        if not handle.accepted:
            n.get_logger().error("goal rejected")
            return 1
        result_future = handle.get_result_async()
        last = {}
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(n, timeout_sec=0.2)
            cur = {k: n.status.get(k) for k in ("policy", "motion_state", "on_stairs", "collisions",
                                                "refused", "prop_collisions")}
            if cur != last:
                n.get_logger().info(f"sim: {cur} z={n.status.get('z_ground')}")
                last = cur
            if time.time() - t0 > a.timeout:
                n.get_logger().error("navigation timeout; cancelling")
                handle.cancel_goal_async()
                n.spin_until(lambda: result_future.done(), 10, "cancel")
                break
        res = result_future.result()
        status = res.status if res else -1
        r = res.result if res else None
        elapsed = time.time() - t0
        collisions = int(n.status.get("collisions", 0)) - int(start.get("collisions", 0))
        refused = int(n.status.get("refused", 0)) - int(start.get("refused", 0))
        succeeded = status == 4  # GoalStatus.STATUS_SUCCEEDED
        print(json.dumps({
            "succeeded": succeeded, "action_status": status,
            "error_code": getattr(r, "error_code", None), "error_msg": getattr(r, "error_msg", None),
            "elapsed_s": round(elapsed, 1), "collisions": collisions, "refused_steps": refused,
            "final": {k: n.status.get(k) for k in ("x", "y", "z_ground", "policy", "motion_state")},
        }, indent=2))
        if a.expect_fail:
            return 0 if not succeeded else 1
        return 0 if succeeded and collisions == 0 else 1
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
