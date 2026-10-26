"""site_tiles.py and voxel_intensity.py on a small slab with a wall: the
intensity per voxel, the atlas packing, the painted texels, and tiles the
browser's loader (trimesh, here) reads back with their texture."""

import io

import numpy as np
import pytest

from syncai_sim import site_tiles, visual
from syncai_sim.btree import TREE_MAX_VAL
from syncai_sim.site_tiles import pack, surface_tiles, voxel_tiles
from syncai_sim.voxel_grid import VoxelGrid
from syncai_sim.voxel_intensity import build as build_intensity
from syncai_sim.world import exposed_face_rects, quad_corners

ORG = TREE_MAX_VAL
RES = 0.1


def _grid(x, y, z):
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)


def slab_and_wall():
    """A 2 x 1 m floor slab (z in [-0.1, 0)) with a 0.5 m wall along x = 1.0."""
    floor = _grid(range(ORG, ORG + 20), range(ORG, ORG + 10), [ORG - 1])
    wall = _grid([ORG + 10], range(ORG, ORG + 10), range(ORG, ORG + 5))
    return np.concatenate([floor, wall])


def test_rects_and_corners_of_one_voxel():
    rects = exposed_face_rects(np.array([[ORG, ORG, ORG]]), RES)
    assert rects.shape == (6, 7)
    assert set(map(tuple, rects[:, :2].tolist())) == set(visual.DIRECTIONS)
    assert np.all(rects[:, 4] == ORG) and np.all(rects[:, 5:] == 1)  # the voxel's own index
    corners, texels = quad_corners(rects, RES)
    assert corners.shape == (6, 4, 3) and texels.shape == (6, 4, 2)
    for r, c, t in zip(rects, corners, texels):
        axis, sign = r[0], r[1]
        assert np.allclose(c[:, axis], RES if sign > 0 else 0.0)
        assert sorted(map(tuple, t.tolist())) == [(0, 0), (0, 1), (1, 0), (1, 1)]
        # The corner at texel (0, 0) is the rectangle's lowest corner.
        assert np.allclose(c[np.flatnonzero((t == 0).all(axis=1))[0]][[a for a in range(3)
                                                                      if a != axis]], 0.0)


def test_voxel_intensity_median_and_neighbour_fill():
    vox = slab_and_wall()
    g = VoxelGrid(vox, RES, z_clip_above_lowest_m=0)
    # Points in the floor voxel at (0.5, 0.5): intensities 10, 20, 90 -> median 20;
    # one point in an empty voxel above the floor at (1.5, 0.5, 0.05): 70.
    pts = np.array([[0.55, 0.55, -0.05], [0.56, 0.55, -0.05], [0.57, 0.55, -0.05],
                    [1.55, 0.55, 0.05]])
    table = build_intensity(pts, np.array([10.0, 20.0, 90.0, 70.0]), g)
    assert table.lookup(np.array([[ORG + 5, ORG + 5, ORG - 1]]))[0] == pytest.approx(20.0)
    # The floor voxel under the lone point had none: it takes its neighbour's 70.
    assert table.lookup(np.array([[ORG + 15, ORG + 5, ORG - 1]]))[0] == pytest.approx(70.0)
    # A voxel with no point anywhere near stays unknown; off the grid too.
    assert np.isnan(table.lookup(np.array([[ORG + 10, ORG + 5, ORG + 4], [ORG + 500, 0, 0]]))).all()


def test_pack_rows_and_parts():
    w = np.array([10, 10, 10, 4])
    h = np.array([4, 4, 4, 8])
    x, y, a, heights = pack(w, h, width=16, max_height=16)
    assert heights == [16] and np.all(a == 0)
    assert (x[3], y[3]) == (0, 0)  # the tallest first
    assert sorted(zip(x[:3].tolist(), y[:3].tolist())) == [(0, 8), (0, 12), (4, 0)]
    _, _, a, heights = pack(w, h, width=16, max_height=8)  # too low: a second atlas
    assert len(heights) == 2 and set(a.tolist()) == {0, 1}


def test_voxel_tiles_paint_and_round_trip():
    trimesh = pytest.importorskip("trimesh")
    vox = slab_and_wall()
    rects = exposed_face_rects(vox, RES)
    storeys = np.zeros(rects.shape[0], np.int8)

    def lookup(idx):  # bright stripe on the floor at x in [0.5, 0.6): a "marking"
        return np.where(idx[:, 0] == ORG + 5, 120.0, 15.0)
    tiles = voxel_tiles(rects, storeys, RES, lookup)
    assert len(tiles) == 1 and tiles[0].storey == 0 and (tiles[0].ix, tiles[0].iy) == (0, 0)
    assert tiles[0].triangles == 2 * rects.shape[0]
    mesh = trimesh.load(io.BytesIO(tiles[0].glb), file_type="glb", force="mesh")
    assert mesh.faces.shape[0] == tiles[0].triangles
    uv = mesh.visual.uv
    assert uv.shape[0] == 4 * rects.shape[0] and uv.min() >= 0.0 and uv.max() <= 1.0
    image = np.asarray(mesh.visual.material.baseColorTexture)
    assert image.shape[1] >= 64 and image.shape[0] >= 8  # a power of two each way
    # The floor's top is a few rectangles (the wall's column splits it): their
    # texels hold the marking as a bright column and the plain floor around
    # it, in the ramp's floor colour, one texel per voxel face.
    top = np.flatnonzero((rects[:, 0] == 2) & (rects[:, 1] > 0))
    assert top.size >= 2 and rects[top, 5].sum() * 1 + 0 >= 19  # the whole slab minus the wall
    floor_rgb = visual.shade(np.array([visual.KIND_FLOOR]), np.array([0.0]), np.array([15.0]),
                             (0.0, 3.0))[0]
    mark_rgb = visual.shade(np.array([visual.KIND_FLOOR]), np.array([0.0]), np.array([120.0]),
                            (0.0, 3.0))[0]
    px = image.reshape(-1, image.shape[-1])[:, :3]
    assert (px == floor_rgb).all(axis=1).sum() >= 18 * 10  # 200 faces less the wall's and the marking's columns
    assert (px == mark_rgb).all(axis=1).sum() >= 10  # the marking's column


def test_surface_tiles_colour_vertices():
    pytest.importorskip("trimesh")

    class Mesh:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]],
                            np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 5], [0, 5, 4]], np.int32)
        kinds = np.array([visual.KIND_FLOOR, visual.KIND_FLOOR, visual.KIND_WALL, visual.KIND_WALL],
                         np.int8)
        storeys = np.array([0, 0, 0, 0], np.int8)
    tiles = surface_tiles(Mesh(), np.zeros(3), np.array([ORG, ORG, ORG]), RES, None)
    assert len(tiles) == 1 and tiles[0].triangles == 4
    assert tiles[0].name == "/site/s0/t0_0/p0"
    assert site_tiles.TILE_M == 10.0
