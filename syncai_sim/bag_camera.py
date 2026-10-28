"""The camera's side of a rosbag2 file, read by hand like livox_bag.py reads
the lidar's: the images, the camera's intrinsics and the static transforms
that place it on the robot. No ROS on the path, so voxel_colour.py's build
runs on the Mac as well as in the container.

Three message types, all flat enough for a small CDR cursor (XCDR1, little
endian: a 4-byte encapsulation header, then every primitive aligned to its
own size counted from the end of that header; a string is its uint32 length
including the NUL, then the bytes; a sequence its uint32 count, then the
elements):

    sensor_msgs/CompressedImage      std_msgs/Header header; string format; uint8[] data
    sensor_msgs/CameraInfo           header; uint32 height, width; string distortion_model;
                                     float64[] d; float64[9] k, r; float64[12] p;
                                     uint32 binning_x, binning_y; RegionOfInterest roi
    tf2_msgs/TFMessage               TransformStamped[]: header; string child_frame_id;
                                     Vector3 translation; Quaternion rotation (x y z w)

What the G23's bag says (rec_20261027_031456, 2026-10-28): the feeder
publishes 1280 x 720 JPEG at ~28 Hz, stamped ~20 ms before the bag received
it; `camera_info` once, latched, with the `equidistant` (fisheye) model and
four coefficients; `/tf_static` once, with lidar_top -> camera_optical_frame
(the URDF's camera_optical_joint) and lidar_top -> lidar_top_imu (Point-LIO's
body frame, t_il).
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass

import numpy as np

from .livox_bag import bag_files

IMAGE_TOPIC = "/robot01/image_raw/compressed"
CAMERA_INFO_TOPIC = "/robot01/camera_info"
TF_STATIC_TOPIC = "/tf_static"


class CdrReader:
    """A cursor over one CDR-serialised message."""

    def __init__(self, blob: bytes):
        self.b = bytes(blob)
        self.off = 4  # past the encapsulation header; alignment counts from here

    def _align(self, n: int) -> None:
        self.off += (-(self.off - 4)) % n

    def _unpack(self, fmt: str, size: int):
        self._align(size)
        v = struct.unpack_from("<" + fmt, self.b, self.off)[0]
        self.off += size
        return v

    def u8(self) -> int:
        return self._unpack("B", 1)

    def u32(self) -> int:
        return self._unpack("I", 4)

    def i32(self) -> int:
        return self._unpack("i", 4)

    def f64(self) -> float:
        return self._unpack("d", 8)

    def f64s(self, n: int) -> np.ndarray:
        self._align(8)
        v = np.frombuffer(self.b, dtype="<f8", count=n, offset=self.off)
        self.off += 8 * n
        return v.astype(np.float64)

    def string(self) -> str:
        n = self.u32()
        s = self.b[self.off:self.off + max(n - 1, 0)].decode()
        self.off += n
        return s

    def raw(self, n: int) -> bytes:
        v = self.b[self.off:self.off + n]
        self.off += n
        return v

    def header(self) -> tuple[float, str]:
        """std_msgs/Header -> (stamp in seconds, frame_id)."""
        sec, nsec = self.i32(), self.u32()
        return sec + nsec * 1e-9, self.string()


@dataclass
class ImageFrame:
    stamp_s: float  # header.stamp, the exposure's time
    frame_id: str
    format: str  # "jpeg" on the G23
    data: bytes  # the encoded image as published


@dataclass
class CameraModel:
    width: int
    height: int
    model: str  # distortion_model: "equidistant" (fisheye) or "plumb_bob"
    k: np.ndarray  # (3, 3)
    d: np.ndarray  # the coefficients, (4,) for equidistant
    frame_id: str


@dataclass
class StaticTransform:
    parent: str
    child: str
    t: np.ndarray  # (3,) the child's origin in the parent, m
    q: np.ndarray  # (4,) w x y z, the child's axes in the parent


def parse_compressed_image(blob: bytes) -> ImageFrame:
    r = CdrReader(blob)
    stamp, frame_id = r.header()
    fmt = r.string()
    n = r.u32()
    return ImageFrame(stamp_s=stamp, frame_id=frame_id, format=fmt, data=r.raw(n))


def parse_camera_info(blob: bytes) -> CameraModel:
    r = CdrReader(blob)
    _, frame_id = r.header()
    height, width = r.u32(), r.u32()
    model = r.string()
    d = r.f64s(r.u32())
    k = r.f64s(9).reshape(3, 3)
    return CameraModel(width=width, height=height, model=model, k=k, d=d, frame_id=frame_id)


def parse_tf_message(blob: bytes) -> list[StaticTransform]:
    r = CdrReader(blob)
    out = []
    for _ in range(r.u32()):
        _, parent = r.header()
        child = r.string()
        t = r.f64s(3)
        x, y, z, w = r.f64s(4)
        out.append(StaticTransform(parent=parent, child=child, t=t, q=np.array([w, x, y, z])))
    return out


def iter_messages(bag: str, topic: str):
    """Yield (bag timestamp ns, blob) of `topic` in time order over the bag's files."""
    for path in bag_files(bag):
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT id FROM topics WHERE name = ?", (topic,)).fetchone()
            if row is None:
                continue
            cur = con.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? "
                              "ORDER BY timestamp", (row[0],))
            yield from cur
        finally:
            con.close()


def iter_images(bag: str, topic: str = IMAGE_TOPIC, every: int = 1):
    """Yield every `every`-th ImageFrame of `topic`, in time order."""
    for k, (_, blob) in enumerate(iter_messages(bag, topic)):
        if k % every == 0:
            yield parse_compressed_image(blob)


def read_camera_info(bag: str, topic: str = CAMERA_INFO_TOPIC) -> CameraModel:
    for _, blob in iter_messages(bag, topic):
        return parse_camera_info(blob)
    raise ValueError(f"no {topic} in {bag}")


def read_static_transforms(bag: str, topic: str = TF_STATIC_TOPIC) -> list[StaticTransform]:
    """Every transform of the bag's /tf_static messages (one per publisher)."""
    out: list[StaticTransform] = []
    for _, blob in iter_messages(bag, topic):
        out.extend(parse_tf_message(blob))
    return out


def find_transform(transforms: list[StaticTransform], parent: str, child: str) -> StaticTransform:
    """The transform whose frames end with `parent` / `child` (the robot_id
    prefix, robot01/lidar_top, is the bag's; the URDF's names are what the
    caller knows)."""
    for tf in transforms:
        if tf.parent.split("/")[-1] == parent and tf.child.split("/")[-1] == child:
            return tf
    raise ValueError(f"no static transform {parent} -> {child} in the bag")


def rotation_matrix(q: np.ndarray) -> np.ndarray:
    """(3, 3) of a unit quaternion (w x y z): the rotation that takes a vector
    in the child's axes to the parent's."""
    w, x, y, z = (float(v) for v in q / np.linalg.norm(q))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def decode_image(frame: ImageFrame) -> np.ndarray:
    """(H, W, 3) uint8 RGB of a JPEG / PNG frame (Pillow)."""
    import io

    from PIL import Image

    with Image.open(io.BytesIO(frame.data)) as im:
        return np.asarray(im.convert("RGB"))
