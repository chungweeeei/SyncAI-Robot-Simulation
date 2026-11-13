"""Where the sim's settings come from: the instance INI and the params YAML.

The INI is the workspace's `config/system.ini` (docker-compose.yml mounts
`config/instances/sim01.ini` over it): `[map] pcd` locates the map directory
(its `octomap.bt` is the world), `[initial_pose]` is the spawn, and
`[system] robot_id` names the run. A relative `pcd` is resolved the way the
workspace resolves it: two directories above the INI is the workspace root.
The params are PARAM_DEFAULTS, overridden by the YAML profiles
(`load_profile`) and `key:=value` strings (`apply_overrides`).
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the real import stays local to keep INI reading free of MuJoCo / numba
    from syncai_sim.envs.simulator import SimConfig

DEFAULT_INI = "config/system.ini"  # relative: every pane runs in the workspace root


@dataclass
class IniSettings:
    path: str
    workspace_root: str
    robot_id: str
    map_dir: str  # "" when the INI names no map
    initial_pose: tuple[float, float, float, float] | None  # x y z yaw
    # [sensor.lidar]: the robot's lidar as its driver is configured for it,
    # which the sim plays with `lidar_link: livox` (sensor/livox_link.py).
    lidar_ip: str = ""
    lidar_type: str = ""  # mid360 | mid360s; "" = none named


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
    lidar_ip = c.get("sensor.lidar", "ip", fallback="").strip() if found else ""
    lidar_type = c.get("sensor.lidar", "type", fallback="").strip().lower() if found else ""
    return IniSettings(path=path, workspace_root=workspace_root, robot_id=robot_id,
                       map_dir=map_dir, initial_pose=pose, lidar_ip=lidar_ip,
                       lidar_type=lidar_type)


def build_sim_config(ini: IniSettings, p: dict) -> SimConfig:
    """The params (a dict, PARAM_DEFAULTS and its overrides) + the INI -> SimConfig.

    Imports are local so that reading an INI needs neither MuJoCo nor numba."""
    import json

    from syncai_sim.sensor.lidar import LidarConfig
    from syncai_sim.scene.scene import PropSpec
    from syncai_sim.envs.simulator import SimConfig
    from syncai_sim.world.build import default_cache_dir

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
        lidar=LidarConfig(
            points_per_scan=p["points_per_scan"], scan_rate_hz=p["scan_rate_hz"],
            fov_lower_deg=p["fov_lower_deg"], fov_upper_deg=p["fov_upper_deg"],
            min_range=p["lidar_min_range"], max_range=p["lidar_max_range"],
            range_noise_sigma=p["range_noise_sigma"], pattern=p["lidar_pattern"],
            pattern_file=p["lidar_pattern_file"]),
        lidar_height=p["lidar_height"], physics_hz=p["physics_hz"], props=props,
        visual_mesh=p["visual_mesh"],
        visual_look=p["visual_look"], stairs=p["stairs"],
        motor_model=p["motor_model"], trunk_inertial=p["trunk_inertial"],
        foot_solref=tuple(p["foot_solref"]), foot_friction=tuple(p["foot_friction"]),
        integrator=p["integrator"],
        cmd_delay_ms=p["cmd_delay_ms"], torque_bw_hz=p["torque_bw_hz"],
        link_command_port=int(p["link_command_port"]), link_state_host=p["link_state_host"],
        link_state_port=int(p["link_state_port"]), link_watchdog_ms=p["link_watchdog_ms"],
        controller_command_ip=p["controller_command_ip"],
        controller_command_port=int(p["controller_command_port"]),
        max_catch_up_s=p["max_catch_up_s"],
    )


def load_profile(path: str) -> dict:
    """A params YAML — a plain mapping of PARAM_DEFAULTS keys, as
    config/params/sim_params.yaml is written — as
    a dict of overrides. A key PARAM_DEFAULTS does not have is an error: a
    typo would otherwise be a setting silently ignored."""
    import yaml

    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise SystemExit(f"{path}: expected a mapping of parameters")
    unknown = sorted(k for k in doc if k not in PARAM_DEFAULTS)
    if unknown:
        raise SystemExit(f"{path}: not sim parameters: {', '.join(unknown)}")
    return dict(doc)


def apply_overrides(params: dict, overrides: list[str]) -> dict:
    """`key:=value` strings onto `params`, each parsed as YAML and typed like
    its default."""
    import yaml

    for item in overrides:
        if ":=" not in item:
            raise SystemExit(f"-p {item!r}: expected key:=value")
        key, raw = item.split(":=", 1)
        if key not in PARAM_DEFAULTS:
            raise SystemExit(f"-p {key}: not a sim parameter (config.PARAM_DEFAULTS)")
        value = yaml.safe_load(raw)
        default = PARAM_DEFAULTS[key]
        if isinstance(default, bool):
            value = bool(value)
        elif isinstance(default, float) and isinstance(value, int):
            value = float(value)
        params[key] = value
    return params


def world_options(p: dict):
    """The params' WorldOptions (shared with `python3 -m syncai_sim.visuals.pcd_mesh`,
    whose cache key must match the sim's world)."""
    from syncai_sim.world.build import WorldOptions
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
    "clip_spikes_above_neighbours_m": 0.1, "track_clear_radius_m": 0.30, "track_floor_fill_radius_m": 0.30,
    "floor_hole_max_cells": 25, "planner_hole_fill_passes": 2,
    "visual_mesh": "voxels", "visual_look": "ramp",
    "web_view_port": 8080,
    "controller_command_ip": "127.0.0.1", "controller_command_port": 50051,
    "lidar_link": "none", "livox_lidar_ip": "", "livox_lidar_type": "",
    "livox_points_per_packet": 96, "livox_imu_hz": 200.0,
    "use_ini_initial_pose": True, "spawn": [0.0, 0.0, 0.0, 0.0],
    "lidar_height": 0.481,
    "points_per_scan": 20000, "scan_rate_hz": 10.0, "fov_lower_deg": -7.0,
    "fov_upper_deg": 52.0, "lidar_min_range": 0.1, "lidar_max_range": 40.0,
    "range_noise_sigma": 0.01, "lidar_pattern": "mid360", "lidar_pattern_file": "",
    "stairs": "fitted",
    "motor_model": "j60", "trunk_inertial": "vendor", "foot_solref": [0.005, 1.0],
    "foot_friction": [1.0, 0.01, 0.01], "integrator": "euler",
    "cmd_delay_ms": 0.0, "torque_bw_hz": 0.0,
    "physics_hz": 1000.0,
    "props_json": "[]",
    # The hardware link (locomotion/hardware_link.py)
    "link_command_port": 20001, "link_state_host": "127.0.0.1", "link_state_port": 30010,
    "link_watchdog_ms": 100.0, "max_catch_up_s": 0.008,
}
