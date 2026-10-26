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

With `articulated` (the `policy` locomotion, legged.py) the model is the robot
instead of a statue: the twelve joints stay, with the motor model's armature
(and, `j60`, its friction and damping; the constants below); there is
no mocap or weld, the free base is carried by its legs; every leg and foot
primitive collides with the terrain patch (terrain_patch.py) and the trunk
with the patch and the props; a gyro and an accelerometer sit on
`lidar_top_imu`. The legs never collide with each other or the trunk (the
URDF's hip cylinders overlap its trunk boxes), and not with props (a leg in a
prop would need a contact model the props were never tuned for). What else
the walking body can be given — the vendor's trunk inertial, the vendor's
contact parameters, MuJoCo position actuators — is `RobotConfig`, each a
knob for the A/B against the robot (scripts/gait_bench.py) with the sim of
before as its default.
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

# Collision bitmasks. World plates are 1/1 (scene.py); props are 2/3; the
# terrain patch is 0/4 (articulated only).
TRUNK_CONTYPE = 2
TRUNK_CONAFFINITY = 2
TERRAIN_BIT = 4
# Rotor inertia reflected to each joint (kg m^2), motor_model `ideal`. Not in
# the URDF; the policies walk the same with 0.005 and 0.01 (flat harness,
# 2026-10-23), and without any the PD at kp 30 rings at a 1 ms step.
JOINT_ARMATURE = 0.01

# Motor model `j60` (RobotConfig.motor_model): the G23's J60 actuators as
# g2_deploy_rs's own MuJoCo simulation models them
# (interface/robot/simulation/mujoco_simulation_G23.py, its J60_* constants
# and j60_torque_limits, 2026-10), per [HipX, HipY, Knee]: a DC motor's
# torque-speed envelope — driving, at most peak * (1 - |dq| / wmax); braking,
# the full peak (legged.py applies it every step) — and, on the joint, the
# rotor's inertia through the gear (1.1e-4 kg m^2 * 6^2 / 6^2 / 9^2), dry
# friction and viscous damping fitted so that the loss at top speed is
# 0.81 / 0.81 / 0.92 Nm. `ideal` (the default) is the sim before 2026-10-26:
# JOINT_ARMATURE everywhere, no friction, the URDF's effort as a flat clamp.
J60_PEAK_TORQUE = (19.94, 19.94, 30.50)  # Nm
J60_MAX_SPEED = (24.18, 24.18, 15.49)  # rad/s
J60_ARMATURE = (1.1e-4 * 36, 1.1e-4 * 36, 1.1e-4 * 81)
J60_FRICTIONLOSS = (0.40, 0.40, 0.45)  # Nm
J60_DAMPING = ((0.81 - 0.40) / 24.6, (0.81 - 0.40) / 24.6, (0.92 - 0.45) / 15.5)  # Nm s/rad
JOINT_KINDS = ("HipX", "HipY", "Knee")  # the suffix each joint name carries

# Trunk inertial `vendor` (RobotConfig.trunk_inertial): the TORSO of
# g2_deploy_rs's G23 MJCF (robot/g23/mjcf/g23_mjcf.xml), 9.368 kg against the
# workspace URDF's 8.03 and a pitch inertia 2.5x the URDF's (0.1425 vs 0.058
# kg m^2) — the URDF's base_link looks like the bare torso without the lidar
# head and the computer. Which is the robot is to be weighed (findings.md 5:
# the sim's stair falls are pitch swings); until then `urdf` is the default.
VENDOR_TRUNK_INERTIAL = {
    "pos": "0.023640945 0.001529477 0.033866753", "mass": "9.368",
    "fullinertia": "0.058025245 0.142509325 0.150322311 0 -0.014586129 0",
}


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
    articulated: bool = False
    imu_site: str = "imu_site"
    # The walking body (articulated only; the constants above say what each is):
    motor_model: str = "ideal"  # ideal | j60
    trunk_inertial: str = "urdf"  # urdf | vendor
    # The legs' and feet's contact with the terrain boxes (scene.py gives the
    # boxes the same solref, so the pair's mean is this). The defaults are
    # MuJoCo's solref and a 1.0 sliding friction; g2_deploy_rs's MuJoCo sim
    # has solref (0.005, 1) and friction (1, 0.01, 0.01).
    foot_solref: tuple[float, float] = (0.02, 1.0)
    foot_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    # qfrc: the PD torque computed in legged.py every step and applied as a
    # generalized force (the sim before 2026-10-26). position: a MuJoCo
    # position actuator per joint, gains written by legged.py, meant for the
    # implicitfast integrator (scene.py) which integrates the -kd dq term
    # implicitly instead of relying on armature to keep it from ringing.
    actuator: str = "qfrc"  # qfrc | position
    # Visual STLs decimated (open3d quadric) into this directory, each to
    # `mesh_decimation` of its triangles, never under `mesh_min_triangles`.
    # The URDF's meshes are 274 k triangles (body 100 k, each shank 27 k):
    # 29 k look the same in the browser viewer (web_view.py sends them once
    # as GLB) and cost the llvmpipe window of before ~50 ms a frame less
    # (2026-10-24). Collision is primitives, so this is the picture only.
    # None = the URDF's files as they are.
    mesh_cache_dir: str | None = None
    mesh_decimation: float = 0.15
    mesh_min_triangles: int = 3000


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


def load_robot(cfg: RobotConfig, log=print) -> RobotModel:
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
    if cfg.mesh_cache_dir:
        _use_decimated_meshes(root, meshdir, cfg, log)

    worldbody = root.find("worldbody")
    base = next(b for b in worldbody.iter("body") if b.get("name") == cfg.base_link)
    ET.SubElement(base, "freejoint", name="base_free")
    if cfg.articulated:
        return _articulated(root, base, cfg, base_to_lidar, base_to_imu)

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


def joint_kind(name: str) -> int:
    """0 HipX, 1 HipY, 2 Knee, by the suffix the URDF's joint names carry."""
    for k, suffix in enumerate(JOINT_KINDS):
        if suffix in name:
            return k
    raise ValueError(f"joint {name!r} is none of {JOINT_KINDS}")


def _articulated(root: ET.Element, base: ET.Element, cfg: RobotConfig,
                 base_to_lidar: np.ndarray, base_to_imu: np.ndarray) -> RobotModel:
    from .policy import Q_DEFAULT

    if cfg.motor_model not in ("ideal", "j60"):
        raise ValueError(f"motor_model {cfg.motor_model!r}: ideal | j60")
    if cfg.trunk_inertial not in ("urdf", "vendor"):
        raise ValueError(f"trunk_inertial {cfg.trunk_inertial!r}: urdf | vendor")
    if cfg.actuator not in ("qfrc", "position"):
        raise ValueError(f"actuator {cfg.actuator!r}: qfrc | position")
    hinges = [j for j in root.iter("joint") if j.get("type", "hinge") == "hinge"]
    for joint in hinges:
        if cfg.motor_model == "j60":
            k = joint_kind(joint.get("name", ""))
            joint.set("armature", f"{J60_ARMATURE[k]:.6g}")
            joint.set("frictionloss", f"{J60_FRICTIONLOSS[k]:.6g}")
            joint.set("damping", f"{J60_DAMPING[k]:.6g}")
        else:
            joint.set("armature", str(JOINT_ARMATURE))
    if cfg.trunk_inertial == "vendor":
        inertial = base.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(base, "inertial")
        for attr in ("quat", "diaginertia", "fullinertia"):
            inertial.attrib.pop(attr, None)
        for attr, value in VENDOR_TRUNK_INERTIAL.items():
            inertial.set(attr, value)
    if cfg.actuator == "position":
        # Gains 0 until legged.py writes the state's kp / kd into gainprm and
        # biasprm; the force range is written every step too (the j60
        # envelope), so the compiled range only has to exist.
        act = root.find("actuator")
        if act is None:
            act = ET.SubElement(root, "actuator")
        for joint in hinges:
            name = joint.get("name", "")
            ET.SubElement(act, "position", name=f"act_{name}", joint=name, kp="0", kv="0",
                          forcerange="-100 100")
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            if geom.get("type") == "mesh":
                geom.set("group", "1")
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                continue
            geom.set("group", "3")
            if body is base and geom.get("type") == "box":
                geom.set("contype", str(TRUNK_CONTYPE | TERRAIN_BIT))
                geom.set("conaffinity", str(TRUNK_CONAFFINITY))
                geom.set("rgba", "0.2 0.5 0.9 0.35")
            else:
                geom.set("contype", str(TERRAIN_BIT))
                geom.set("conaffinity", "0")
                geom.set("friction", _fmt(cfg.foot_friction))
                geom.set("solref", _fmt(cfg.foot_solref))
    imu = next(b for b in root.iter("body") if b.get("name") == cfg.imu_body)
    ET.SubElement(imu, "site", name=cfg.imu_site, size="0.01")
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "gyro", name="imu_gyro", site=cfg.imu_site)
    ET.SubElement(sensor, "accelerometer", name="imu_acc", site=cfg.imu_site)
    mjcf = ET.tostring(root, encoding="unicode")

    # Standing height: the base over the ground with the legs at the policy's
    # default pose and the lowest foot sphere touching it.
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    for leg in ("FL", "FR", "HL", "HR"):
        for k, j in enumerate(("HipX", "HipY", "Knee")):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{j}_joint")
            data.qpos[model.jnt_qposadr[jid]] = Q_DEFAULT[k]
    mujoco.mj_forward(model, data)
    bottoms = [data.geom_xpos[g][2] - model.geom_size[g][0] for g in range(model.ngeom)
               if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
               and model.body(model.geom_bodyid[g]).name.endswith("_FOOT")]
    if not bottoms:
        raise ValueError("URDF has no *_FOOT sphere to stand on")
    h_stand = float(data.xpos[1][2] - min(bottoms))
    return RobotModel(mjcf=mjcf, cfg=cfg, base_to_lidar=base_to_lidar, base_to_imu=base_to_imu,
                      h_stand=h_stand)


def _use_decimated_meshes(root: ET.Element, meshdir: str, cfg: RobotConfig, log) -> None:
    """Point the MJCF's <mesh file=...> at decimated copies (see RobotConfig)."""
    try:
        import open3d as o3d
    except ImportError:
        log("[robot] open3d not installed: the URDF's meshes are drawn as they are")
        return
    asset = root.find("asset")
    if asset is None:
        return
    os.makedirs(cfg.mesh_cache_dir, exist_ok=True)
    for mesh in asset.findall("mesh"):
        src = mesh.get("file", "")
        if not src:
            continue
        src_path = src if os.path.isabs(src) else os.path.normpath(os.path.join(meshdir, src))
        if not os.path.isfile(src_path):
            continue
        st = os.stat(src_path)
        stem = os.path.splitext(os.path.basename(src_path))[0]
        out = os.path.join(cfg.mesh_cache_dir,
                           f"{stem}-{st.st_size}-{int(st.st_mtime)}-r{cfg.mesh_decimation:g}"
                           f"-m{cfg.mesh_min_triangles}.stl")
        if not os.path.isfile(out):
            m = o3d.io.read_triangle_mesh(src_path)
            n = len(m.triangles)
            target = max(cfg.mesh_min_triangles, int(n * cfg.mesh_decimation))
            if n > target:
                m = m.simplify_quadric_decimation(target)
            m.compute_triangle_normals()
            o3d.io.write_triangle_mesh(out, m)
            log(f"[robot] {os.path.basename(src_path)}: {n} -> {len(m.triangles)} triangles")
        mesh.set("file", out)


def default_urdf(workspace: str) -> str:
    return os.path.join(workspace, "src", "syncai_bringup", "description", "G23.urdf")
