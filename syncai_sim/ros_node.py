"""The ROS 2 face of the simulator: what the nav stack used to get from the
Livox driver, Point-LIO, the localizer and driver_manager.

Published (relative names, so they land under /<robot_id>/ like the real ones):

    pointlio/body_cloud   PointCloud2 xyz+intensity, frame <id>/pointlio_body,
                          10 Hz, rendered at the end-of-scan pose = already
                          motion-compensated, which is what Point-LIO publishes
    pointlio/lio_odom     Odometry, frame <id>/pointlio_odom -> <id>/pointlio_body,
                          body-frame twist; lio_bridge reads THIS, not TF
    /tf                   map -> <id>/pointlio_odom (the localizer's edge) and
                          <id>/pointlio_odom -> <id>/pointlio_body (Point-LIO's),
                          both 50 Hz, stamped like the cloud, published before it
    livox/imu             Imu, gyro + gravity in g; lio_bridge takes angular_velocity.z
    mode                  Int32MultiArray [policy, motion_state], reliable, 2 Hz and
                          on change (the real driver only echoes telemetry)
    sim/ground_truth      Odometry in map of base_link, for test scripts
    sim/status            DiagnosticArray of the simulator's counters

Served: set_motion_key, set_policy_mode (syncai_common). Subscribed: cmd_vel,
and initialpose (the localizer's topic) as a teleport — covariance[14] != 0
means z is meant, as in localizer_node.cpp.

With `driver_link: udp` (the session) the driver is the workspace's real
syncai_driver_manager and the sim is its gait controller (controller_link.py):
the sim then serves neither set_motion_key / set_policy_mode nor `mode`, and
does not read cmd_vel — the driver does, and its safety lock drops it — and
it answers sys_manager's `get_mode` (AUTO, the nav session) for
syncai_robot_state.

Threading: the physics loop runs in the main thread at wall-clock pace (the
whole stack runs `use_sim_time: false`, so every stamp is `now`); an rclpy
executor spins in a daemon thread and its callbacks write into the simulator
under one lock.
"""

from __future__ import annotations

import array
import json
import math
import threading
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState, PointCloud2, PointField
from std_msgs.msg import Int32MultiArray
from tf2_ros import TransformBroadcaster

from .config import PARAM_DEFAULTS, IniSettings, build_sim_config  # noqa: F401
from .simulator import Simulator

try:
    from syncai_common.msg import RobotMode
    from syncai_common.srv import GetMode, SetMotionKey, SetPolicyMode
except ImportError as err:  # the workspace install is not sourced
    raise SystemExit(
        "syncai_common is not importable: source the workspace's install/setup.bash "
        f"(and build it) before starting the sim ({err})"
    )


def _mat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation -> (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _yaw_of(q) -> float:
    x, y, z, w = q.x, q.y, q.z, q.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class SimNode(Node):
    def __init__(self, ini: IniSettings, sim: Simulator, params: dict, frames: dict):
        super().__init__("sim_node", namespace=f"/{ini.robot_id}")
        self.ini = ini
        self.sim = sim
        self.p = params
        self.f = frames
        self.lock = threading.Lock()
        reliable10 = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.pub_cloud = self.create_publisher(PointCloud2, "pointlio/body_cloud", reliable10)
        self.pub_odom = self.create_publisher(Odometry, "pointlio/lio_odom", reliable10)
        self.pub_imu = self.create_publisher(Imu, "livox/imu", reliable10)
        self.link = None
        if params["driver_link"] == "udp":
            from .controller_link import ControllerLink, LinkConfig
            self.link = ControllerLink(sim, sim.driver, LinkConfig(
                command_port=params["controller_command_port"],
                telemetry_port=params["controller_telemetry_port"],
                telemetry_hz=params["controller_telemetry_hz"],
                battery_soc=params["battery_soc"],
                battery_drain_per_hour=params["battery_drain_per_hour"]),
                log=lambda m: self.get_logger().info(m))
        elif params["driver_link"] != "fake":
            raise SystemExit(f"driver_link {params['driver_link']!r}: fake | udp")
        else:
            self.pub_mode = self.create_publisher(Int32MultiArray, "mode", reliable10)
        self.pub_gt = self.create_publisher(Odometry, "sim/ground_truth", reliable10)
        self.pub_status = self.create_publisher(DiagnosticArray, "sim/status", reliable10)
        # robot_state_publisher (launch/rsp.launch.py, same namespace) turns
        # these into the legs' TF; nothing in the nav stack reads them.
        self.pub_joints = self.create_publisher(JointState, "joint_states", reliable10)
        self.tf = TransformBroadcaster(self)
        self._map_pubs = []  # kept alive: latched map clouds (publish_map_clouds)

        self.create_subscription(PoseWithCovarianceStamped, "initialpose", self._on_initialpose, 10)
        if self.link is None:
            self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 10)
            self.create_service(SetMotionKey, "set_motion_key", self._on_motion_key)
            self.create_service(SetPolicyMode, "set_policy_mode", self._on_policy_mode)
        else:
            # sys_manager's NodeManager answers this on the robot with the
            # running session; the sim session is the nav session's stand-in.
            self.create_service(GetMode, "get_mode", self._on_get_mode)

        self._last_pose: tuple[float, np.ndarray] | None = None  # (stamp s, T_map_imu)
        self._cloud_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.get_logger().info(
            f"sim_node up as /{ini.robot_id}/sim_node; frames {frames['body']} / {frames['odom']}; "
            f"map {sim.cfg.map_dir}")

    # -- callbacks (executor thread) ------------------------------------------
    def _on_cmd_vel(self, msg: Twist) -> None:
        with self.lock:
            self.sim.set_cmd(msg.linear.x, msg.linear.y, msg.angular.z)

    def _on_initialpose(self, msg: PoseWithCovarianceStamped) -> None:
        p = msg.pose.pose
        z_hint = p.position.z if msg.pose.covariance[14] != 0.0 else None
        yaw = _yaw_of(p.orientation)
        try:
            with self.lock:
                self.sim.teleport(p.position.x, p.position.y, yaw, z_hint)
        except ValueError as err:
            self.get_logger().error(f"initialpose refused: {err}")

    def _on_get_mode(self, req, res):
        res.success, res.message = True, "simulation (start_sim)"
        res.mode, res.session = RobotMode.AUTO, "syncai-sim"
        return res

    def _on_motion_key(self, req, res):
        with self.lock:
            res.success, res.message = self.sim.driver.set_motion_key(req.key)
        self.get_logger().info(f"set_motion_key '{req.key}': {res.message}")
        if res.success:
            self._publish_mode()
        return res

    def _on_policy_mode(self, req, res):
        with self.lock:
            res.success, res.message = self.sim.driver.set_policy_mode(int(req.mode))
        self.get_logger().info(f"set_policy_mode {req.mode}: {res.message} "
                               f"(reported in {self.sim.driver.switch_delay:.1f} s)")
        return res

    # -- publishers (main loop) --------------------------------------------------
    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _tf_msg(self, stamp, parent, child, T: np.ndarray) -> TransformStamped:
        m = TransformStamped()
        m.header.stamp = stamp
        m.header.frame_id = parent
        m.child_frame_id = child
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (
            float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        x, y, z, w = _mat_to_quat(T[:3, :3])
        m.transform.rotation.x, m.transform.rotation.y = x, y
        m.transform.rotation.z, m.transform.rotation.w = z, w
        return m

    def publish_pose(self, stamp=None) -> None:
        """TF x2 + lio_odom + ground truth, all with one stamp."""
        stamp = stamp or self._stamp()
        with self.lock:
            T_mo = self.sim.T_map_odom.copy()
            T_mi = self.sim.T_map_imu()
            T_mb = self.sim.T_map_base()
        T_oi = np.linalg.inv(T_mo) @ T_mi
        self.tf.sendTransform([
            self._tf_msg(stamp, self.f["map"], self.f["odom"], T_mo),
            self._tf_msg(stamp, self.f["odom"], self.f["body"], T_oi),
        ])

        now_s = stamp.sec + stamp.nanosec * 1e-9
        v_body = np.zeros(3)
        if self._last_pose is not None and now_s > self._last_pose[0]:
            dt = now_s - self._last_pose[0]
            v_world = (T_mi[:3, 3] - self._last_pose[1][:3, 3]) / dt
            v_body = T_mi[:3, :3].T @ v_world
        self._last_pose = (now_s, T_mi)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.f["odom"]
        odom.child_frame_id = self.f["body"]
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = (
            float(T_oi[0, 3]), float(T_oi[1, 3]), float(T_oi[2, 3]))
        x, y, z, w = _mat_to_quat(T_oi[:3, :3])
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y = x, y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = z, w
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = (
            float(v_body[0]), float(v_body[1]), float(v_body[2]))
        gyro, _ = self.sim.imu_sample()
        odom.twist.twist.angular.z = float(gyro[2])
        self.pub_odom.publish(odom)

        gt = Odometry()
        gt.header.stamp = stamp
        gt.header.frame_id = self.f["map"]
        gt.child_frame_id = self.f["base"]
        gt.pose.pose.position.x, gt.pose.pose.position.y, gt.pose.pose.position.z = (
            float(T_mb[0, 3]), float(T_mb[1, 3]), float(T_mb[2, 3]))
        x, y, z, w = _mat_to_quat(T_mb[:3, :3])
        gt.pose.pose.orientation.x, gt.pose.pose.orientation.y = x, y
        gt.pose.pose.orientation.z, gt.pose.pose.orientation.w = z, w
        gt.twist.twist.linear.x, gt.twist.twist.linear.y = float(v_body[0]), float(v_body[1])
        gt.twist.twist.angular.z = float(gyro[2])
        self.pub_gt.publish(gt)

    def publish_scan(self) -> int:
        """Render and publish one cloud; the pose at the same stamp goes first."""
        stamp = self._stamp()
        self.publish_pose(stamp)
        with self.lock:
            xyzi = self.sim.scan()
        msg = self._cloud_msg(stamp, self.f["body"], xyzi)
        self.pub_cloud.publish(msg)
        return msg.width

    def _cloud_msg(self, stamp, frame_id: str, xyzi: np.ndarray) -> PointCloud2:
        """A PointCloud2 of (N, 4) float32 x y z intensity."""
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = int(xyzi.shape[0])
        msg.fields = self._cloud_fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * msg.width
        msg.is_dense = True
        # array('B'), not bytes: the generated setter checks a bytes value
        # element by element in Python (13.7 ms for a 311 KB scan, so the
        # 3 ms scan published in 20 ms; ~1.5 s for a 35 MB map cloud) and
        # takes an array('B') as is (0.03 ms; 2026-10-24).
        msg.data = array.array("B", np.ascontiguousarray(xyzi, dtype=np.float32).tobytes())
        return msg

    def publish_joints(self) -> None:
        with self.lock:
            names, q = self.sim.joint_states()
        m = JointState()
        m.header.stamp = self._stamp()
        m.name = names
        m.position = [float(v) for v in q]
        self.pub_joints.publish(m)

    def publish_map_clouds(self, clouds: dict[int, np.ndarray]) -> None:
        """The site's point cloud per storey (map_cloud.py), once, latched,
        on sim/map_cloud/storey<N> in the map frame: for Foxglove."""
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for k, xyzi in sorted(clouds.items()):
            pub = self.create_publisher(PointCloud2, f"sim/map_cloud/storey{k + 1}", latched)
            pub.publish(self._cloud_msg(self._stamp(), self.f["map"], xyzi))
            self._map_pubs.append(pub)
        self.get_logger().info(
            f"map clouds latched: {[f'storey{k + 1} {c.shape[0]} pts' for k, c in sorted(clouds.items())]}")

    def publish_imu(self) -> None:
        with self.lock:
            gyro, accel = self.sim.imu_sample()
        m = Imu()
        m.header.stamp = self._stamp()
        m.header.frame_id = self.f["imu"]
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, gyro)
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, accel)
        m.orientation_covariance[0] = -1.0  # no orientation, as the Livox driver
        self.pub_imu.publish(m)

    def _publish_mode(self) -> None:
        m = Int32MultiArray()
        m.data = self.sim.driver.mode_array()
        self.pub_mode.publish(m)

    def publish_mode(self) -> None:
        if self.link is None:  # with the udp link, driver_manager publishes mode
            self._publish_mode()

    def publish_status(self) -> None:
        with self.lock:
            st = self.sim.status()
        # A refusal that persists is the thing to read in the log: say where and why.
        if st["last_event"] not in ("moved", "idle", "gated") and st["last_event"] != getattr(
                self, "_last_logged_event", None):
            self._last_logged_event = st["last_event"]
            self.get_logger().warn(
                f"base {st['last_event']} at ({st['x']:.2f}, {st['y']:.2f}) yaw {st['yaw']:.2f} "
                f"z_ground {st['z_ground']:.2f}: next ground {st['next_ground']} (dz {st['last_dz']}), "
                f"policy {st['policy']}, on_stairs {st['on_stairs']}")
        elif st["last_event"] in ("moved", "idle"):
            self._last_logged_event = None
        da = DiagnosticArray()
        da.header.stamp = self._stamp()
        ds = DiagnosticStatus()
        ds.name = "sim"
        ds.hardware_id = self.ini.robot_id
        ds.level = DiagnosticStatus.OK if st["collisions"] == 0 else DiagnosticStatus.WARN
        ds.message = st["last_event"]
        ds.values = [KeyValue(key=k, value=json.dumps(v)) for k, v in st.items()]
        da.status.append(ds)
        self.pub_status.publish(da)


def make_frames(robot_id: str) -> dict:
    return {"map": "map", "odom": f"{robot_id}/pointlio_odom", "body": f"{robot_id}/pointlio_body",
            "base": f"{robot_id}/base_link", "imu": f"{robot_id}/lidar_top_imu"}


class Spinner:
    """rclpy executor in a daemon thread."""

    def __init__(self, node: Node):
        self.exec = rclpy.executors.SingleThreadedExecutor()
        self.exec.add_node(node)
        self.thread = threading.Thread(target=self.exec.spin, daemon=True, name="rclpy-spin")

    def start(self):
        self.thread.start()

    def stop(self):
        # shutdown() returns while the thread may still be inside rmw_wait;
        # destroying the node or the context under it ended the process with
        # "terminate called without an active exception" (a joinable thread
        # in rmw_cyclonedds, exit 134) in 6 of 8 --seconds runs; joined, 0 of
        # 12 (2026-10-24).
        self.exec.shutdown(timeout_sec=1.0)
        self.thread.join(timeout=3.0)


class ParamNode(Node):
    """A throwaway node that only declares the parameters, so the params file
    can be read before the (slow) simulator is built."""

    def __init__(self, robot_id: str):
        super().__init__("sim_node", namespace=f"/{robot_id}")
        for k, v in PARAM_DEFAULTS.items():
            self.declare_parameter(k, v)

    def values(self) -> dict:
        return {k: self.get_parameter(k).value for k in PARAM_DEFAULTS}


def run_loop(node: SimNode, viewer_factory, teleop, show_scan: bool, stop_after: float | None,
             log) -> None:
    """The main loop: fixed-step physics at wall-clock pace, publications by rate."""
    sim = node.sim
    p = node.p
    dt = sim.model.opt.timestep
    periods = {
        "pose": 1.0 / p["pose_publish_hz"], "imu": 1.0 / p["imu_publish_hz"],
        "scan": 1.0 / p["scan_rate_hz"], "mode": 1.0 / p["mode_publish_rate_hz"],
        "status": 1.0 / p["status_publish_hz"],
    }
    due = {k: 0.0 for k in periods}
    max_catch_up = max(1, int(round(0.1 / dt)))
    t_start = time.perf_counter()
    next_t = t_start
    steps_total = 0
    last_report = t_start
    scan_ms = 0.0
    n = 0
    viewer = viewer_factory() if viewer_factory else None
    if viewer is not None:
        periods["view"] = 1.0 / viewer.rate_hz
        due["view"] = 0.0
    try:
        while rclpy.ok():
            now = time.perf_counter()
            if stop_after is not None and now - t_start > stop_after:
                break
            if viewer is not None and not viewer.is_running():
                log("[sim] viewer closed; exiting")
                break
            behind = int((now - next_t) / dt) + 1
            steps = max(1, min(behind, max_catch_up))  # catch up, but never more than 0.1 s at once
            with node.lock:
                if node.link is not None:
                    node.link.poll()
                if teleop is not None:
                    teleop.apply(sim)
                for _ in range(steps):
                    sim.step()
                if node.link is not None:
                    node.link.tick(sim.sim_time)
            steps_total += steps
            next_t += steps * dt
            t = sim.sim_time
            if t >= due["pose"]:
                due["pose"] = t + periods["pose"]
                node.publish_pose()
            if t >= due["imu"]:
                due["imu"] = t + periods["imu"]
                node.publish_imu()
                node.publish_joints()  # same 50 Hz: legs for robot_state_publisher
            if t >= due["scan"]:
                due["scan"] = t + periods["scan"]
                due["pose"] = t + periods["pose"]  # publish_scan sends the pose itself
                t0 = time.perf_counter()
                n = node.publish_scan()
                scan_ms = (time.perf_counter() - t0) * 1000
                if viewer is not None and show_scan:
                    viewer.show_scan(sim)
            if t >= due["mode"]:
                due["mode"] = t + periods["mode"]
                node.publish_mode()
            if t >= due["status"]:
                due["status"] = t + periods["status"]
                node.publish_status()
            if viewer is not None and now >= due["view"]:
                due["view"] = now + periods["view"]
                viewer.sync(sim)
            if now - last_report > 10.0:
                last_report = now
                rt = sim.sim_time / (now - t_start)
                log(f"[sim] t={sim.sim_time:7.1f} s  realtime x{rt:.2f}  scan {scan_ms:.1f} ms "
                    f"({n} pts)  {sim.status()['last_event']}  z={sim.state.z_ground:.2f}")
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
    finally:
        if viewer is not None:
            viewer.close()
        if node.link is not None:
            node.link.close()
