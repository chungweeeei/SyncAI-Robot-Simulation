"""tsdf_mesh.py: the cube faces, the depth images a patch renders to, the
pose file, and (with open3d) a surface integrated from a few keyframes of a
synthetic room."""

import math

import numpy as np
import pytest

from syncai_sim import visual
from syncai_sim.btree import TREE_MAX_VAL
from syncai_sim.tsdf_mesh import (FACES, TsdfOptions, build, depth_faces, quat_wxyz_matrix,
                                  read_poses)
from syncai_sim.voxel_grid import VoxelGrid

ORG = TREE_MAX_VAL


def test_cube_faces_are_right_handed_cameras_looking_down_each_axis():
    assert FACES.shape == (6, 3, 3)
    for k, d in enumerate(([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])):
        R = FACES[k]
        assert np.allclose(np.linalg.det(R), 1.0)  # a rotation, not a reflection
        assert np.allclose(R @ np.array(d, float), [0, 0, 1])  # the axis is the optical axis
    # Forward camera: +y (left) is image-left, +z (up) is image-up.
    assert np.allclose(FACES[0] @ np.array([0, 1, 0.0]), [-1, 0, 0])
    assert np.allclose(FACES[0] @ np.array([0, 0, 1.0]), [0, -1, 0])


def test_depth_faces_put_each_return_on_one_face_at_its_depth():
    pts = np.array([[3.0, 0.0, 0.0], [-2.0, 0.1, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, -1.5],
                    [0.2, 0.0, 0.0], [50.0, 0.0, 0.0]])  # the last two: too near, too far
    faces = depth_faces(pts, np.array([10.0, 20.0, 30.0, 300.0, 1.0, 1.0]), 64, 0.3, 25.0)
    got = {k: (depth, colour) for k, depth, colour in faces}
    assert set(got) == {0, 1, 2, 5}
    for k, want_depth, want_grey in ((0, 3.0, 10), (1, 2.0, 20), (2, 4.0, 30), (5, 1.5, 255)):
        depth, colour = got[k]
        hit = np.argwhere(depth > 0)
        assert hit.shape[0] == 1 and np.isclose(depth[tuple(hit[0])], want_depth)
        assert colour[tuple(hit[0])].tolist() == [want_grey] * 3
    # Dead ahead lands on the face's centre pixel.
    assert tuple(np.argwhere(got[0][0] > 0)[0]) == (32, 32)
    # The nearer of two returns on one pixel wins.
    two = depth_faces(np.array([[3.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), np.array([1.0, 2.0]), 64,
                      0.3, 25.0)
    assert np.isclose(two[0][1].max(), 2.0) and two[0][2].max() == 2


def test_read_poses_wxyz(tmp_path):
    s2 = math.sqrt(0.5)
    (tmp_path / "poses.txt").write_text(
        f"0.pcd 1.0 2.0 3.0 1 0 0 0\n1.pcd 0 0 0 {s2} 0 0 {s2}\nbad line\n")
    poses = read_poses(str(tmp_path))
    assert [n for n, _ in poses] == ["0.pcd", "1.pcd"]
    assert np.allclose(poses[0][1][:3, 3], [1, 2, 3]) and np.allclose(poses[0][1][:3, :3], np.eye(3))
    assert np.allclose(poses[1][1][:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-9)  # a 90 deg yaw
    assert np.allclose(quat_wxyz_matrix([1, 0, 0, 0]), np.eye(3))


class Stats:
    cache_key = "test"
    options = {}


class Room:
    """A 4 x 4 m room, floor at z = 0, walls 2 m high, and a partition at
    x = 1.5 m, as a cleaned grid (on_site never keeps the grid's outermost
    layer, so the partition, not the outer wall, is what the test looks at)."""

    def __init__(self):
        xs = range(ORG - 20, ORG + 20)
        zs = range(ORG, ORG + 20)
        floor = np.array([[x, y, ORG - 1] for x in xs for y in xs])
        walls = np.array([[x, y, z] for x in xs for y in xs for z in zs
                          if x in (ORG - 20, ORG + 19, ORG + 15) or y in (ORG - 20, ORG + 19)])
        self.grid = VoxelGrid(np.concatenate([floor, walls]), 0.1, z_clip_above_lowest_m=0)
        self.storey_bounds = []
        self.stats = Stats()


def test_build_integrates_a_room_from_keyframes(tmp_path):
    pytest.importorskip("open3d")
    rng = np.random.default_rng(0)
    # Surfaces of the room, sampled; the keyframes see them from inside.
    n = 40_000
    floor = np.column_stack((rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), np.zeros(n)))
    wall = np.column_stack((np.full(n // 4, 1.55), rng.uniform(-2, 2, n // 4),
                            rng.uniform(0, 2, n // 4)))  # the partition
    surface = np.concatenate([floor, wall])
    intensity = np.where(surface[:, 2] > 0.01, 80.0, 20.0)  # the wall brighter than the floor
    patches = tmp_path / "patches"
    patches.mkdir()
    lines = []
    for k in range(6):
        origin = np.array([-1.0 + 0.4 * k, 0.0, 0.5])
        local = surface - origin  # identity rotation: the patch is the room seen from here
        head = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
                f"COUNT 1 1 1 1\nWIDTH {local.shape[0]}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
                f"POINTS {local.shape[0]}\nDATA binary\n")
        data = np.column_stack((local, intensity)).astype(np.float32)
        (patches / f"{k}.pcd").write_bytes(head.encode() + data.tobytes())
        lines.append(f"{k}.pcd {origin[0]} {origin[1]} {origin[2]} 1 0 0 0")
    (tmp_path / "poses.txt").write_text("\n".join(lines) + "\n")
    mesh = build(str(tmp_path), Room(), TsdfOptions(voxel=0.05, face_px=128, max_range=10.0),
                 log=lambda *a: None)
    assert mesh.faces.shape[0] > 1000
    floor_faces = mesh.kinds == visual.KIND_FLOOR
    assert floor_faces.sum() > 500 and (mesh.kinds == visual.KIND_WALL).sum() > 100
    zf = mesh.vertices[mesh.faces[floor_faces]][:, :, 2]
    assert abs(float(np.median(zf))) < 0.05  # the floor where it was
    assert np.all(mesh.storeys == 0)
