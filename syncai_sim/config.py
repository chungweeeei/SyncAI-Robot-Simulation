"""Where the sim's settings come from: the instance INI and the params YAML.

The INI is the workspace's `config/system.ini` — on the robot the instance file
bind-mounted over it, here `config/instances/sim01.ini` the same way — read the
way the launch files read it: `[system] robot_id` names the namespace and the
frame prefix, `[map] pcd` locates the map directory (its `octomap.bt` is the
world), `[initial_pose]` is the spawn. Resolution of a relative `pcd` follows
planner_server.launch.py: two directories above the INI is the workspace root.
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the real import stays local to keep INI reading free of MuJoCo / numba
    from .simulator import SimConfig

DEFAULT_INI = "config/system.ini"  # relative: every pane runs in the workspace root


@dataclass
class IniSettings:
    path: str
    workspace_root: str
    robot_id: str
    map_dir: str  # "" when the INI names no map
    initial_pose: tuple[float, float, float, float] | None  # x y z yaw


def read_ini(path: str | None = None) -> IniSettings:
    path = path or os.environ.get("SYNCAI_SYSTEM_INI") or DEFAULT_INI
    path = os.path.abspath(path)
    workspace_root = os.path.dirname(os.path.dirname(path))
    c = configparser.ConfigParser()
    found = c.read(path)
    robot_id = c.get("system", "robot_id", fallback="").strip() if found else ""
    if not robot_id:
        robot_id = "default_robot"
    pcd = c.get("map", "pcd", fallback="").strip() if found else ""
    if pcd and not os.path.isabs(pcd):
        pcd = os.path.join(workspace_root, pcd)
    map_dir = os.path.dirname(pcd) if pcd else ""
    pose = None
    if found and c.has_section("initial_pose"):
        try:
            pose = tuple(c.getfloat("initial_pose", k, fallback=0.0) for k in ("x", "y", "z", "yaw"))
        except ValueError:
            pose = None
    return IniSettings(path=path, workspace_root=workspace_root, robot_id=robot_id,
                       map_dir=map_dir, initial_pose=pose)


def build_sim_config(ini: IniSettings, p: dict) -> SimConfig:
    """ROS params (already read into a dict) + the INI -> SimConfig.

    Imports are local so that reading an INI needs neither MuJoCo nor numba."""
    import json

    from .kinematics import KinematicsConfig
    from .lidar import LidarConfig
    from .scene import PropSpec
    from .simulator import SimConfig
    from .world import default_cache_dir

    map_dir = p["map_dir"] or ini.map_dir
    if not map_dir:
        raise SystemExit("no map: set [map] pcd in the INI or the map_dir parameter")
    props = [PropSpec(**d) for d in json.loads(p["props_json"] or "[]")]
    for pr in props:
        pr.size = tuple(pr.size)
        pr.pos = tuple(pr.pos)
        pr.waypoints = [tuple(w) for w in pr.waypoints]
    return SimConfig(
        map_dir=map_dir,
        cache_dir=p["cache_dir"] or default_cache_dir(map_dir),
        urdf_path=os.path.join(ini.workspace_root, "src", "syncai_bringup", "description", "G23.urdf"),
        world=world_options(p),
        kin=KinematicsConfig(
            max_step_height=p["max_step_height"], stair_tread_window=p["stair_tread_window"],
            max_drop=p["max_drop"], require_backward_descent=p["require_backward_descent"],
            stair_policy=p["stair_policy"], robot_radius=p["robot_radius"],
            body_clearance=p["body_clearance"], min_obstacle_height=p["min_obstacle_height"],
            walk_vx_min=p["walk_vx_min"], in_place_wz_min=p["in_place_wz_min"]),
        lidar=LidarConfig(
            points_per_scan=p["points_per_scan"], scan_rate_hz=p["scan_rate_hz"],
            fov_lower_deg=p["fov_lower_deg"], fov_upper_deg=p["fov_upper_deg"],
            min_range=p["lidar_min_range"], max_range=p["lidar_max_range"],
            range_noise_sigma=p["range_noise_sigma"], pattern=p["lidar_pattern"],
            pattern_file=p["lidar_pattern_file"]),
        lidar_height=p["lidar_height"], physics_hz=p["physics_hz"],
        switch_delay=p["mode_switch_delay"], initial_motion_state=p["initial_motion_state"],
        initial_policy=p["initial_policy"], props=props,
        locomotion=p["locomotion"], ppo_policy=p["ppo_policy"],
        himloco_policy=p["himloco_policy"], visual_mesh=p["visual_mesh"],
        visual_look=p["visual_look"], stairs=p["stairs"],
        motor_model=p["motor_model"], trunk_inertial=p["trunk_inertial"],
        foot_solref=tuple(p["foot_solref"]), foot_friction=tuple(p["foot_friction"]),
        actuator=p["actuator"], integrator=p["integrator"],
        cmd_delay_ms=p["cmd_delay_ms"], torque_bw_hz=p["torque_bw_hz"],
    )


def world_options(p: dict):
    """The params' WorldOptions (shared with `python3 -m syncai_sim.pcd_mesh`,
    whose cache key must match the sim's world)."""
    from .world import WorldOptions
    return WorldOptions(
        visual_max_height_above_column_floor_m=p["visual_max_height_above_column_floor_m"],
        plate_min_voxels=p["plate_min_voxels"],
        clip_spikes_above_neighbours_m=p["clip_spikes_above_neighbours_m"],
        track_clear_radius_m=p["track_clear_radius_m"],
        track_floor_fill_radius_m=p["track_floor_fill_radius_m"],
        floor_hole_max_cells=p["floor_hole_max_cells"],
        planner_hole_fill_passes=p["planner_hole_fill_passes"])


PARAM_DEFAULTS: dict = {
    "map_dir": "", "cache_dir": "",
    "visual_max_height_above_column_floor_m": 3.0, "plate_min_voxels": 16,
    "clip_spikes_above_neighbours_m": 0.2, "track_clear_radius_m": 0.30, "track_floor_fill_radius_m": 0.30,
    "floor_hole_max_cells": 25, "planner_hole_fill_passes": 2,
    "visual_mesh": "voxels", "visual_look": "ramp", "publish_map_cloud": True,
    "web_view_port": 8080,
    "driver_link": "fake", "controller_command_port": 50051, "controller_telemetry_port": 50010,
    "controller_telemetry_hz": 50.0, "battery_soc": 90.0, "battery_drain_per_hour": 0.0,
    "use_ini_initial_pose": True, "spawn": [0.0, 0.0, 0.0, 0.0],
    "lidar_height": 0.481, "robot_radius": 0.25, "body_clearance": 0.6,
    "min_obstacle_height": 0.1, "max_step_height": 0.17, "stair_tread_window": 0.45,
    "max_drop": 0.3, "require_backward_descent": True, "stair_policy": 1,
    "walk_vx_min": 0.0, "in_place_wz_min": 0.0,
    "points_per_scan": 20000, "scan_rate_hz": 10.0, "fov_lower_deg": -7.0,
    "fov_upper_deg": 52.0, "lidar_min_range": 0.1, "lidar_max_range": 40.0,
    "range_noise_sigma": 0.01, "lidar_pattern": "mid360", "lidar_pattern_file": "",
    "mode_switch_delay": 1.0, "initial_motion_state": 1, "initial_policy": 0,
    "mode_publish_rate_hz": 2.0,
    "locomotion": "policy", "stairs": "fitted",
    "motor_model": "ideal", "trunk_inertial": "urdf", "foot_solref": [0.02, 1.0],
    "foot_friction": [1.0, 0.005, 0.0001], "actuator": "qfrc", "integrator": "euler",
    "cmd_delay_ms": 0.0, "torque_bw_hz": 0.0,
    "ppo_policy": "/home/syncrobotic/sim/policy/ppo/policy7.onnx",
    "himloco_policy": "/home/syncrobotic/sim/policy/isaacsim/policy_0812.onnx",
    "physics_hz": 500.0, "pose_publish_hz": 50.0, "imu_publish_hz": 50.0,
    "status_publish_hz": 2.0,
    "props_json": "[]",
}
