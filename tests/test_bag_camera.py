"""bag_camera.py against hand-built CDR blobs and a two-table sqlite bag: the
image, camera_info and tf_static layouts, the lookups over the bag and the
quaternion's matrix."""

import sqlite3
import struct

import numpy as np
import pytest

from syncai_sim import bag_camera
from syncai_sim.bag_camera import (find_transform, iter_images, parse_camera_info,
                                   parse_compressed_image, parse_tf_message, read_camera_info,
                                   read_static_transforms, rotation_matrix)


class Cdr:
    """A writer that aligns like rmw_cyclonedds's XCDR1 serialiser."""

    def __init__(self):
        self.b = bytearray(b"\x00\x01\x00\x00")

    def align(self, n):
        while (len(self.b) - 4) % n:
            self.b += b"\0"

    def u32(self, v):
        self.align(4)
        self.b += struct.pack("<I", v)

    def i32(self, v):
        self.align(4)
        self.b += struct.pack("<i", v)

    def f64s(self, vs):
        self.align(8)
        self.b += struct.pack(f"<{len(vs)}d", *vs)

    def string(self, s):
        raw = s.encode() + b"\0"
        self.u32(len(raw))
        self.b += raw

    def header(self, sec, nsec, frame_id):
        self.i32(sec)
        self.u32(nsec)
        self.string(frame_id)

    def raw(self, data):
        self.b += data


def image_blob(stamp=(1793070930, 302000000), fmt="jpeg", data=b"\xff\xd8\xff\xd9"):
    c = Cdr()
    c.header(*stamp, "robot01/camera_optical_frame")
    c.string(fmt)
    c.u32(len(data))
    c.raw(data)
    return bytes(c.b)


K = [710.4, 0.0, 663.2, 0.0, 707.3, 371.4, 0.0, 0.0, 1.0]
D = [-0.0701, 0.0200, -0.0336, 0.0148]


def camera_info_blob():
    c = Cdr()
    c.header(1793070896, 0, "robot01/camera_optical_frame")
    c.u32(720)
    c.u32(1280)
    c.string("equidistant")
    c.u32(len(D))
    c.f64s(D)
    c.f64s(K)
    c.f64s([1, 0, 0, 0, 1, 0, 0, 0, 1])
    c.f64s(K[:3] + [0] + K[3:6] + [0] + K[6:] + [0])
    c.u32(0)
    c.u32(0)
    for _ in range(4):
        c.u32(0)
    c.raw(b"\0")  # roi.do_rectify
    return bytes(c.b)


TFS = [("robot01/lidar_top", "robot01/camera_optical_frame", (0.084, 0.0, -0.046),
        (-0.4328, 0.4365, -0.5635, 0.5519)),  # x y z w as the message carries it
       ("robot01/lidar_top", "robot01/lidar_top_imu", (0.011, 0.0233, -0.0441), (0, 0, 0, 1)),
       ("robot01/base_link", "robot01/lidar_top", (0.195, 0.0, 0.155), (0, 0.1247, 0, 0.9922))]


def tf_blob(tfs=TFS):
    c = Cdr()
    c.u32(len(tfs))
    for parent, child, t, q in tfs:
        c.header(0, 0, parent)
        c.string(child)
        c.f64s(t)
        c.f64s(q)
    return bytes(c.b)


def make_bag(path, messages):
    """A rosbag2 sqlite file with the two tables the readers use;
    `messages` is [(topic, timestamp_ns, blob)]."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, "
                "serialization_format TEXT, offered_qos_profiles TEXT)")
    con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, "
                "timestamp INTEGER, data BLOB)")
    ids = {}
    for topic, ts, blob in messages:
        if topic not in ids:
            ids[topic] = len(ids) + 1
            con.execute("INSERT INTO topics VALUES (?, ?, ?, ?, ?)", (ids[topic], topic, "", "cdr", ""))
        con.execute("INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
                    (ids[topic], ts, blob))
    con.commit()
    con.close()


def test_parse_compressed_image():
    f = parse_compressed_image(image_blob())
    assert f.stamp_s == pytest.approx(1793070930.302)
    assert f.frame_id == "robot01/camera_optical_frame" and f.format == "jpeg"
    assert f.data == b"\xff\xd8\xff\xd9"


def test_parse_camera_info():
    m = parse_camera_info(camera_info_blob())
    assert (m.width, m.height, m.model) == (1280, 720, "equidistant")
    assert np.allclose(m.k, np.array(K).reshape(3, 3)) and np.allclose(m.d, D)
    assert m.frame_id == "robot01/camera_optical_frame"


def test_parse_tf_message_and_find():
    tfs = parse_tf_message(tf_blob())
    assert [(t.parent, t.child) for t in tfs] == [(p, c) for p, c, _, _ in TFS]
    cam = find_transform(tfs, "lidar_top", "camera_optical_frame")
    assert np.allclose(cam.t, (0.084, 0.0, -0.046))
    assert np.allclose(cam.q, (0.5519, -0.4328, 0.4365, -0.5635))  # w x y z
    with pytest.raises(ValueError):
        find_transform(tfs, "lidar_top", "nowhere")


def test_readers_over_a_bag(tmp_path):
    bag = tmp_path / "bag"
    bag.mkdir()
    make_bag(str(bag / "bag_0.db3"), [
        (bag_camera.TF_STATIC_TOPIC, 10, tf_blob()),
        (bag_camera.CAMERA_INFO_TOPIC, 20, camera_info_blob()),
        (bag_camera.IMAGE_TOPIC, 100, image_blob((1, 0), data=b"a")),
        (bag_camera.IMAGE_TOPIC, 200, image_blob((2, 0), data=b"b")),
        (bag_camera.IMAGE_TOPIC, 300, image_blob((3, 0), data=b"c")),
    ])
    assert read_camera_info(str(bag)).width == 1280
    assert len(read_static_transforms(str(bag))) == 3
    assert [f.data for f in iter_images(str(bag))] == [b"a", b"b", b"c"]
    assert [f.stamp_s for f in iter_images(str(bag), every=2)] == [1.0, 3.0]
    with pytest.raises(ValueError):
        read_camera_info(str(bag), "/nothing")


def test_rotation_matrix_is_the_childs_axes_in_the_parent():
    s = np.sqrt(0.5)
    r = rotation_matrix(np.array([s, 0, 0, s]))  # 90 deg about z
    assert np.allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    # The optical convention (z forward, x right, y down) in a lidar frame
    # (x forward, y left, z up).
    r = rotation_matrix(np.array([0.5, -0.5, 0.5, -0.5]))
    assert np.allclose(r @ [0, 0, 1], [1, 0, 0]) and np.allclose(r @ [1, 0, 0], [0, -1, 0])
    assert np.allclose(r @ [0, 1, 0], [0, 0, -1])
