"""voxel_colour.py: the fisheye projection, the occlusion buffer, the pose
interpolation, the per-voxel median, the file, and a whole build from a
synthetic bag (a lidar over a floor, a camera looking at it, one green
frame) that checks the frame chain lidar -> body -> map -> camera."""

import io

import numpy as np
import pytest

from syncai_sim import bag_camera, voxel_colour
from syncai_sim.btree import TREE_MAX_VAL
from syncai_sim.voxel_colour import (Builder, Camera, PoseTrack, build_from_bag, colour_points,
                                     load, load_for, read_odom, save)
from syncai_sim.voxel_grid import VoxelGrid
from tests.test_bag_camera import Cdr, image_blob, make_bag, tf_blob
from tests.test_livox_bag import custom_msg_blob

ORG = TREE_MAX_VAL
RES = 0.1


def camera(width=640, height=480, d=(0.0, 0.0, 0.0, 0.0), model="equidistant"):
    return Camera(fx=300.0, fy=300.0, cx=320.0, cy=240.0, d=np.array(d), width=width,
                  height=height, model=model)


def test_project_equidistant_angles_and_bounds():
    cam = camera()
    th = 0.3
    p = np.array([[0.0, 0.0, 1.0], [np.sin(th), 0.0, np.cos(th)], [0.0, -np.sin(th), np.cos(th)],
                  [0.0, 0.0, -1.0], [5.0, 0.0, 1.0]])
    u, v, ok = cam.project(p)
    assert ok.tolist() == [True, True, True, False, False]  # behind; off the image
    assert (u[0], v[0]) == pytest.approx((320.0, 240.0))
    assert u[1] == pytest.approx(320.0 + 300.0 * th) and v[1] == pytest.approx(240.0)
    assert v[2] == pytest.approx(240.0 - 300.0 * th)  # y down: a point above the axis is higher
    # A coefficient bends the ray: k1 > 0 pushes it out.
    u2, _, _ = camera(d=(0.1, 0, 0, 0)).project(p[1:2])
    assert u2[0] > u[1]


def test_project_plumb_bob_matches_pinhole_without_distortion():
    cam = camera(model="plumb_bob", d=(0, 0, 0, 0, 0))
    u, v, ok = cam.project(np.array([[0.5, -0.25, 2.0]]))
    assert ok[0] and u[0] == pytest.approx(320.0 + 300.0 * 0.25) \
        and v[0] == pytest.approx(240.0 - 300.0 * 0.125)


def test_colour_points_takes_the_pixel_and_drops_what_is_behind():
    cam = camera()
    image = np.zeros((480, 640, 3), np.uint8)
    image[:, :320] = (200, 10, 10)
    image[:, 320:] = (10, 10, 200)
    near = np.array([[-0.2, 0.0, 2.0]])  # left half, 2 m
    same = near * 1.05  # the same ray, 5 % farther: the same surface
    behind = near * 3.0  # the same ray, 6 m: hidden by `near`
    right = np.array([[0.4, 0.0, 2.0]])
    rgb, keep = colour_points(np.concatenate([near, same, behind, right]), cam, image)
    assert keep.tolist() == [True, True, False, True]
    assert rgb[0].tolist() == [200, 10, 10] and rgb[3].tolist() == [10, 10, 200]


def test_pose_track_interpolates_and_turns_the_short_way():
    s = np.sqrt(0.5)
    track = PoseTrack(np.array([10.0, 11.0]), np.array([[0, 0, 0], [2, 0, 0]]),
                      np.array([[1, 0, 0, 0], [-s, 0, 0, -s]]))  # 90 deg yaw, flipped sign
    R, p = track.at(np.array([10.5, 9.0, 12.0]))
    assert np.allclose(p, [[1, 0, 0], [0, 0, 0], [2, 0, 0]])
    yaw = np.degrees(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    assert yaw[0] == pytest.approx(45.0, abs=0.5) and yaw[1] == pytest.approx(0.0, abs=1e-6)
    assert yaw[2] == pytest.approx(90.0, abs=1e-6)  # clamped to the ends


def test_read_odom(tmp_path):
    p = tmp_path / "odom.txt"
    p.write_text("# stamp x y z qw qx qy qz\n10.0 1 2 3 1 0 0 0\n11.0 2 2 3 1 0 0 0\n")
    track = read_odom(str(p))
    assert track.span == (10.0, 11.0) and np.allclose(track.p[1], [2, 2, 3])
    p.write_text("10.0 1 2 3\n")
    with pytest.raises(ValueError):
        read_odom(str(p))


def floor_grid():
    """A 2 x 2 m floor at z in [-0.1, 0), voxels (ORG..ORG+19, ORG..ORG+19, ORG-1)."""
    x, y = np.meshgrid(range(ORG, ORG + 20), range(ORG, ORG + 20), indexing="ij")
    vox = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, ORG - 1)))
    return VoxelGrid(vox, RES, z_clip_above_lowest_m=0)


def test_builder_median_lookup_and_file(tmp_path):
    g = floor_grid()
    b = Builder(g)
    p = np.array([[0.55, 0.55, -0.05]] * 3 + [[1.55, 1.55, -0.05], [9.0, 9.0, 0.0]])
    rgb = np.array([[250, 0, 0], [240, 0, 10], [0, 0, 250], [0, 200, 0], [1, 1, 1]])
    assert b.add(p, rgb) == 4  # the last is off the grid
    t = b.table()
    assert t.keys.shape[0] == 2 and t.count.tolist() == [3, 1]
    got = t.lookup(np.array([[ORG + 5, ORG + 5, ORG - 1], [ORG + 15, ORG + 15, ORG - 1],
                             [ORG + 1, ORG + 1, ORG - 1], [ORG + 500, 0, 0]]))
    assert got[0].tolist() == [240, 0, 10]  # per channel medians
    assert got[1].tolist() == [0, 200, 0]
    assert np.isnan(got[2]).all() and np.isnan(got[3]).all()
    path = str(tmp_path / "m.npz")
    save(path, t, {"bag": "test"})
    t2, meta = load(path)
    assert meta["bag"] == "test" and np.array_equal(t2.keys, t.keys) and np.array_equal(t2.rgb, t.rgb)
    assert t2.shape == t.shape and np.array_equal(t2.lo, t.lo)


def test_load_for_checks_the_grid(tmp_path):
    g = floor_grid()
    t = Builder(g).table()

    class Stats:
        map_name = "m"

    class World:
        stats = Stats()
        grid = g
    lines = []
    assert load_for(World(), log=lines.append, colour_dir=str(tmp_path)) is None
    assert "no " in lines[-1]
    save(str(tmp_path / "m.npz"), t, {})
    assert load_for(World(), log=lines.append, colour_dir=str(tmp_path)) is not None
    other = VoxelGrid(np.array([[ORG, ORG, ORG]]), RES, z_clip_above_lowest_m=0)
    World.grid = other
    assert load_for(World(), log=lines.append, colour_dir=str(tmp_path)) is None
    assert "another grid" in lines[-1]
    assert voxel_colour.file_key("m", str(tmp_path)) is not None
    assert voxel_colour.file_key("none", str(tmp_path)) is None


def green_jpeg(w=640, h=480, rgb=(10, 200, 30)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), rgb).save(buf, format="PNG")  # PNG: the exact colour back
    return buf.getvalue()


def test_build_from_bag_colours_the_floor_the_camera_saw(tmp_path):
    """The lidar 0.5 m over a floor at the map's origin, the body = the lidar
    (no offset), the camera optical frame on the lidar looking forward: the
    floor returns 1-4 m ahead project under the image's centre and take the
    frame's green; the voxels under them are green in the table."""
    pytest.importorskip("PIL")
    T = 1_000_000_000_000  # the scan's timebase, ns
    pts = []  # returns off the floor (z -0.55 under the lidar: the voxels at z -0.1..0)
    for i, x in enumerate(np.arange(1.0, 2.0, 0.05)):
        for y in (0.05, 0.45, 1.05):
            pts.append((i * 4947, x, y, -0.55, 20, 0, len(pts) % 4))
    scan = custom_msg_blob(pts, timebase=T, stamp=(T // 10 ** 9, T % 10 ** 9))
    tfs = [("robot01/lidar_top", "robot01/camera_optical_frame", (0.0, 0.0, 0.0),
            (-0.5, 0.5, -0.5, 0.5)),  # x y z w: optical (z forward) in the lidar's axes
           ("robot01/lidar_top", "robot01/lidar_top_imu", (0.0, 0.0, 0.0), (0, 0, 0, 1))]
    bag = tmp_path / "bag"
    bag.mkdir()
    cam_info = Cdr()  # the test camera, not the G23's
    cam_info.header(0, 0, "cam")
    cam_info.u32(480)
    cam_info.u32(640)
    cam_info.string("equidistant")
    cam_info.u32(4)
    cam_info.f64s([0, 0, 0, 0])
    cam_info.f64s([300, 0, 320, 0, 300, 240, 0, 0, 1])
    cam_info.f64s([1, 0, 0, 0, 1, 0, 0, 0, 1])
    cam_info.f64s([0] * 12)
    for _ in range(6):
        cam_info.u32(0)
    cam_info.raw(b"\0")
    make_bag(str(bag / "b_0.db3"), [
        (bag_camera.TF_STATIC_TOPIC, 1, tf_blob(tfs)),
        (bag_camera.CAMERA_INFO_TOPIC, 2, bytes(cam_info.b)),
        (bag_camera.IMAGE_TOPIC, T - 10 ** 8, image_blob((999, 900_000_000), "png", green_jpeg())),
        (bag_camera.IMAGE_TOPIC, T + 5 * 10 ** 7, image_blob((1000, 50_000_000), "png", green_jpeg())),
        ("/robot01/livox/lidar", T + 10 ** 8, scan),
    ])
    # The robot stands still at the origin, 0.5 m up, for the whole scan.
    track = PoseTrack(np.array([999.0, 1001.0]), np.array([[0, 0, 0.5], [0, 0, 0.5]]),
                      np.array([[1, 0, 0, 0], [1, 0, 0, 0]]))
    g = floor_grid()  # x, y in [0, 2): the returns at x 1-2 m land on it
    model = bag_camera.read_camera_info(str(bag))
    stfs = bag_camera.read_static_transforms(str(bag))
    table, meta = build_from_bag(
        str(bag), track, g, Camera.from_model(model),
        bag_camera.find_transform(stfs, "lidar_top", "camera_optical_frame"),
        bag_camera.find_transform(stfs, "lidar_top", "lidar_top_imu"), log=lambda s: None)
    assert meta["scans"] == 1 and meta["scans_coloured"] == 1
    got = table.lookup(np.array([[ORG + 15, ORG, ORG - 1], [ORG + 15, ORG + 4, ORG - 1],
                                 [ORG + 15, ORG + 10, ORG - 1], [ORG + 15, ORG + 15, ORG - 1]]))
    assert got[0].tolist() == [10, 200, 30] and got[1].tolist() == [10, 200, 30]
    assert got[2].tolist() == [10, 200, 30]  # y 1.05: 46 deg off the axis, still in the image
    assert np.isnan(got[3]).all()  # y = 1.5: no return there
    assert table.keys.shape[0] == 30  # x 1.0-1.95 at three y: 10 voxel columns x 3


def test_shade_photo_known_and_unknown():
    from syncai_sim import visual

    kind = np.array([visual.KIND_FLOOR, visual.KIND_WALL, visual.KIND_UNDER, visual.KIND_FLOOR])
    photo = np.array([[200, 100, 50], [200, 100, 50], [200, 100, 50], [np.nan] * 3])
    out = visual.shade_photo(kind, np.array([20.0, 20.0, 20.0, 120.0]), photo)
    assert out.dtype == np.uint8 and out[0].tolist() == [200, 100, 50]
    assert np.all(out[1] < out[0]) and np.all(out[2] < out[1])  # walls a little, undersides more
    expected = (visual.PHOTO_UNKNOWN * visual.intensity_factor(np.array([120.0]))[0]).clip(0, 1)
    assert out[3].tolist() == (expected * 255 + 0.5).astype(np.uint8).tolist()
    plain = visual.shade_photo(kind[:1], None, photo[3:4])
    assert plain[0].tolist() == (visual.PHOTO_UNKNOWN * visual.UNKNOWN_FACTOR * 255 + 0.5
                                 ).astype(np.uint8).tolist()
