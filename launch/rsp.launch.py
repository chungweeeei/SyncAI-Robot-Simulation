"""robot_state_publisher for the sim session: the static TF tree of the URDF.

The workspace's bringup.launch.py does this and also starts the Livox driver,
which raises without a `[sensor.lidar] ip` in the INI — and the sim has no
lidar to drive. So the robot_state_publisher half is repeated here with the
same two settings that launch gives it: the robot_id namespace and the
`<robot_id>/` frame prefix (a namespace does not prefix frame ids). Passing the
URDF as a `-p robot_description:=...` command-line argument instead was tried
and rcl refuses to parse a 20 KB string with the URDF's punctuation in it.

`read_robot_id()` is the copy every launch file in the workspace carries (no
shared helper there either); the INI is config/system.ini relative to the cwd,
the workspace root, where every pane runs.
"""

import configparser

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.logging import get_logger

logger = get_logger("rsp.launch")
FALLBACK_ROBOT_ID = "default_robot"


def read_robot_id(config_path: str) -> str:
    config = configparser.ConfigParser()
    if not config.read(config_path):
        logger.warning(f"System config '{config_path}' not found; falling back to "
                       f"robot_id '{FALLBACK_ROBOT_ID}'")
        return FALLBACK_ROBOT_ID
    robot_id = config.get("system", "robot_id", fallback="").strip()
    if not robot_id:
        logger.warning(f"No [system] robot_id in '{config_path}'; falling back to "
                       f"'{FALLBACK_ROBOT_ID}'")
        return FALLBACK_ROBOT_ID
    return robot_id


def launch_setup(context, *args, **kwargs):
    config_path = LaunchConfiguration("system_config").perform(context)
    urdf_path = LaunchConfiguration("urdf").perform(context)
    robot_id = read_robot_id(config_path)
    with open(urdf_path) as f:
        robot_description = f.read()
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=robot_id,
            output="screen",
            parameters=[{
                "robot_description": robot_description,
                "frame_prefix": f"{robot_id}/",
                "use_sim_time": False,
            }],
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("system_config", default_value="config/system.ini"),
        DeclareLaunchArgument("urdf", default_value="src/syncai_bringup/description/G23.urdf"),
        OpaqueFunction(function=launch_setup),
    ])
