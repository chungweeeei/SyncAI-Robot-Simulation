"""The G23 in MuJoCo: the workspace's URDF, its twelve joints driven by motors.

The URDF (`src/syncai_bringup/description/G23.urdf` in the workspace) is the
single description of the robot — the real robot's TF tree comes from it —
so the sim loads that file and nothing else: MuJoCo parses URDF natively. Two compiler switches are injected into the text first, because they
cannot come from a URDF (`fusestatic="false"`, or the root link base_link is
fused into the world and cannot be given a free joint; `discardvisual="false"`,
or the STL meshes are dropped and only the collision primitives remain), with
`meshdir` pointing at the URDF's directory so its `../meshes/*.STL` resolve.

Then the model is saved back as MJCF and edited (ElementTree) into what the
sim needs:

* `base_link` gets a **free joint**: the base is carried by its legs.
* The twelve leg joints stay, with the motor model's armature (and, `j60`,
  its friction and damping; the constants below). Their torque is applied by
  `actuator/motors.py` as a generalized force — the whole command a
  controller sends (a target velocity and a feed-forward torque included).
* Every leg and foot primitive collides with the terrain patch
  (terrain_patch.py), the trunk's two boxes with the patch and the props.
  The legs never collide with each other or the trunk (the URDF's hip
  cylinders overlap its trunk boxes), and not with props (a leg in a prop
  would need a contact model the props were never tuned for).
* Two IMUs: a gyro and an accelerometer on `lidar_top_imu` (the MID360's
  own) and on base_link (the body's, sensor/proprioception.py).
* `lidar_top` and `lidar_top_imu` stay as bodies (fixed children of
  base_link); their world poses come straight from MjData, so the sensor
  extrinsics are the URDF's, read, not copied.

What else the body can be given — the vendor's trunk inertial, the vendor's
contact parameters — is `RobotConfig`, each a knob with the sim of before as
its default (config/params/sim_params.yaml sets the vendor's).

Until 2026-11-13 there was also a rigid statue on a mocap weld, its legs
baked at a standing angle, carried by rules (the `kinematic` locomotion);
removed with the in-process policies when the sim became the controller's
hardware only.
"""

from __future__ import annotations

import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

# Collision bitmasks. World plates are 1/1 (scene.py); props are 2/3; the
# terrain patch is 0/4.
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
# the full peak (actuator/motors.py applies it every step) — and, on the joint, the
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

# The controller's joint order, matched by name to the URDF's joints: (FL, FR,
# HL, HR) x (HipX, HipY, Knee), which is its (LF, RF, LR, RR) x (hip, thigh,
# calf) with the same signs (on flat ground PPO walks the URDF as is and falls
# within a second with the hip signs flipped, 2026-10-23). Every layer above —
# the motors, the body's record, the hardware link — uses this order.
LEGS = ("FL", "FR", "HL", "HR")
JOINTS = ("HipX", "HipY", "Knee")
# The controller's default joint pose (its dof_pos_default): what a spawn puts
# the joints at, and the stand-in controller's goal (hardware_link.py).
Q_DEFAULT = np.array([0.0, -0.65, 1.3] * 4, dtype=np.float32)

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
    imu_site: str = "imu_site"  # the lidar's IMU, on imu_body
    base_imu_site: str = "base_imu"  # the body's IMU, on base_link
    # The body's physics (the constants above say what each is):
    motor_model: str = "ideal"  # ideal | j60
    trunk_inertial: str = "urdf"  # urdf | vendor
    # The legs' and feet's contact with the terrain boxes (scene.py gives the
    # boxes the same solref, so the pair's mean is this). The defaults are
    # MuJoCo's solref and a 1.0 sliding friction; g2_deploy_rs's MuJoCo sim
    # has solref (0.005, 1) and friction (1, 0.01, 0.01).
    foot_solref: tuple[float, float] = (0.02, 1.0)
    foot_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
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
    return _articulated(root, base, cfg, base_to_lidar, base_to_imu)


def joint_kind(name: str) -> int:
    """0 HipX, 1 HipY, 2 Knee, by the suffix the URDF's joint names carry."""
    for k, suffix in enumerate(JOINT_KINDS):
        if suffix in name:
            return k
    raise ValueError(f"joint {name!r} is none of {JOINT_KINDS}")


def _articulated(root: ET.Element, base: ET.Element, cfg: RobotConfig,
                 base_to_lidar: np.ndarray, base_to_imu: np.ndarray) -> RobotModel:
    if cfg.motor_model not in ("ideal", "j60"):
        raise ValueError(f"motor_model {cfg.motor_model!r}: ideal | j60")
    if cfg.trunk_inertial not in ("urdf", "vendor"):
        raise ValueError(f"trunk_inertial {cfg.trunk_inertial!r}: urdf | vendor")
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
    # Two IMUs, as on the robot: the lidar's own on lidar_top_imu (the MID360's
    # built-in one) and the body's on base_link (what the controller's hardware
    # layer reports: sensor/proprioception.py). The lidar's sits ~0.4 m over
    # the trunk, and its accelerometer carries that lever arm.
    imu = next(b for b in root.iter("body") if b.get("name") == cfg.imu_body)
    ET.SubElement(imu, "site", name=cfg.imu_site, size="0.01")
    ET.SubElement(base, "site", name=cfg.base_imu_site, size="0.01")
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "gyro", name="imu_gyro", site=cfg.imu_site)
    ET.SubElement(sensor, "accelerometer", name="imu_acc", site=cfg.imu_site)
    ET.SubElement(sensor, "gyro", name="base_gyro", site=cfg.base_imu_site)
    ET.SubElement(sensor, "accelerometer", name="base_acc", site=cfg.base_imu_site)
    mjcf = ET.tostring(root, encoding="unicode")

    # Standing height: the base over the ground with the legs at the
    # controller's default pose and the lowest foot sphere touching it.
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    for leg in LEGS:
        for k, j in enumerate(JOINTS):
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
