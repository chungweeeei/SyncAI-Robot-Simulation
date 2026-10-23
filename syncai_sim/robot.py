"""The G23 in MuJoCo: the workspace's URDF, rigid, with a mocap handle.

The URDF (`src/syncai_bringup/description/G23.urdf`) is the single description
of the robot — robot_state_publisher serves it on the real robot and in the sim
session — so the sim loads that file and nothing else: MuJoCo parses URDF
natively. Two compiler switches are injected into the text first, because they
cannot come from a URDF (`fusestatic="false"`, or the root link base_link is
fused into the world and cannot be given a free joint; `discardvisual="false"`,
or the STL meshes are dropped and only the collision primitives remain), with
`meshdir` pointing at the URDF's directory so its `../meshes/*.STL` resolve.

Then the model is saved back as MJCF and edited (ElementTree) into what the
sim needs:

* `base_link` gets a **free joint**; a **mocap body** `base_mocap` is welded to
  it (`<equality><weld>`), and `kinematics.py` moves the mocap. The weld is soft
  (`solref`) so a contact with a prop pushes back instead of being overpowered.
* The twelve leg joints are **removed** and each child body rotated to a fixed
  standing angle: there is no gait policy, the robot is a statue that is
  carried, so the legs are decoration. The angles are visual, chosen so the
  feet stand on the ground when the body is at the standing height.
* Leg and foot collision primitives become **non-colliding** (a carried statue
  must not catch its feet on a riser), hidden in group 3 behind the STL
  visuals (group 1). The trunk's two boxes keep collision, in the **prop**
  bitmask (contype/conaffinity 2) so the trunk meets props but never the
  world plates — the world is handled by `VoxelGrid.disc_blocked`.
* `lidar_top` and `lidar_top_imu` stay as bodies (fixed children of
  base_link); their world poses come straight from MjData, so the sensor
  extrinsics are the URDF's, read, not copied. `pointlio_body` is
  `lidar_top_imu` (CLAUDE.md), and the planner's `lidar_height` 0.481 is that
  frame's height over the ground when standing, which fixes the body height.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco
import numpy as np

# Visual standing angles (rad), by joint-name suffix. Found by forward
# kinematics: with these the feet sit ~0.37 m under the hips, which is where the
# ground is when lidar_top_imu stands 0.481 m over it (see RobotModel.h_stand).
DEFAULT_STANDING = {"HipX": 0.0, "HipY": -0.447, "Knee": 0.867}

# Collision bitmasks. World plates are 1/1 (scene.py); props are 2/3.
TRUNK_CONTYPE = 2
TRUNK_CONAFFINITY = 2


@dataclass
class RobotConfig:
    urdf_path: str
    base_link: str = "base_link"
    lidar_body: str = "lidar_top"
    imu_body: str = "lidar_top_imu"
    lidar_height: float = 0.481  # the planner's / controllers' value: imu frame over ground
    standing: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_STANDING))
    weld_solref: tuple[float, float] = (0.02, 1.0)
    mocap_name: str = "base_mocap"


@dataclass
class RobotModel:
    mjcf: str  # the edited MJCF text, ready for MjSpec.from_string
    cfg: RobotConfig
    base_to_lidar: np.ndarray  # 4x4, base_link -> lidar_top (URDF fixed joint)
    base_to_imu: np.ndarray  # 4x4, base_link -> lidar_top_imu
    h_stand: float  # base_link origin height over the ground when standing

    @property
    def lidar_in_imu(self) -> np.ndarray:
        """Translation of lidar_top in the lidar_top_imu frame (r_il is identity)."""
        return (np.linalg.inv(self.base_to_imu) @ self.base_to_lidar)[:3, 3]


def _axis_angle_quat(axis, angle) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), a[0] * s, a[1] * s, a[2] * s])


def _quat_mul(q, r) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(q, float), np.asarray(r, float))
    return out


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _frame_of(model: mujoco.MjModel, body: str) -> np.ndarray:
    """4x4 pose of a body in base_link, from the kinematic chain at qpos 0."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if bid < 0 or base < 0:
        raise ValueError(f"URDF has no body {body!r} / base_link")

    def T(i):
        M = np.eye(4)
        M[:3, :3] = data.xmat[i].reshape(3, 3)
        M[:3, 3] = data.xpos[i]
        return M

    return np.linalg.inv(T(base)) @ T(bid)


def load_robot(cfg: RobotConfig) -> RobotModel:
    with open(cfg.urdf_path) as f:
        urdf = f.read()
    meshdir = os.path.dirname(os.path.abspath(cfg.urdf_path))
    inject = (f'<mujoco><compiler fusestatic="false" discardvisual="false" '
              f'meshdir="{meshdir}" balanceinertia="true"/></mujoco>')
    urdf = re.sub(r"(<robot\b[^>]*>)", r"\1" + inject, urdf, count=1)
    model = mujoco.MjModel.from_xml_string(urdf)
    base_to_lidar = _frame_of(model, cfg.lidar_body)
    base_to_imu = _frame_of(model, cfg.imu_body)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "robot.xml")
        mujoco.mj_saveLastXML(path, model)
        root = ET.parse(path).getroot()

    worldbody = root.find("worldbody")
    base = next(b for b in worldbody.iter("body") if b.get("name") == cfg.base_link)
    ET.SubElement(base, "freejoint", name="base_free")

    # Legs: remove joints, bake the standing angle into the child body's quat.
    for body in list(root.iter("body")):
        joint = body.find("joint")
        if joint is None or body is base:
            continue
        name = joint.get("name", "")
        angle = next((a for suffix, a in cfg.standing.items() if suffix in name), 0.0)
        axis = [float(x) for x in joint.get("axis", "0 0 1").split()]
        q_joint = _axis_angle_quat(axis, angle)
        q_body = [float(x) for x in body.get("quat", "1 0 0 0").split()]
        body.set("quat", _fmt(_quat_mul(q_body, q_joint)))
        body.remove(joint)

    # Geoms: STL visuals stay in group 1; primitives lose collision except the
    # trunk's two boxes, which get the prop bitmask.
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            if geom.get("type") == "mesh":
                geom.set("group", "1")
                continue
            if body is base and geom.get("type") == "box":
                geom.set("contype", str(TRUNK_CONTYPE))
                geom.set("conaffinity", str(TRUNK_CONAFFINITY))
                geom.set("group", "3")
                geom.set("rgba", "0.2 0.5 0.9 0.35")
            else:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "3")

    # Mocap handle + weld.
    mocap = ET.SubElement(worldbody, "body", name=cfg.mocap_name, mocap="true")
    ET.SubElement(mocap, "geom", type="box", size="0.04 0.04 0.04", contype="0",
                  conaffinity="0", group="4", rgba="1 0.2 0.2 0.5")
    eq = root.find("equality")
    if eq is None:
        eq = ET.SubElement(root, "equality")
    ET.SubElement(eq, "weld", name="base_weld", body1=cfg.mocap_name, body2=cfg.base_link,
                  solref=_fmt(cfg.weld_solref))

    h_stand = cfg.lidar_height - float(base_to_imu[2, 3])
    return RobotModel(mjcf=ET.tostring(root, encoding="unicode"), cfg=cfg,
                      base_to_lidar=base_to_lidar, base_to_imu=base_to_imu, h_stand=h_stand)


def default_urdf(workspace: str) -> str:
    return os.path.join(workspace, "src", "syncai_bringup", "description", "G23.urdf")
