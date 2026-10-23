import numpy as np

from syncai_sim.btree import TREE_MAX_VAL
from syncai_sim.world import (
    clear_track_volume,
    clip_spikes,
    column_floor_filter,
    exposed_face_mesh,
    fill_track_floor,
)

ORG = TREE_MAX_VAL


def _grid(x, y, z):
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)


def test_single_voxel_mesh_is_a_cube():
    v, f = exposed_face_mesh(np.array([[ORG, ORG, ORG]]), 0.1)
    assert v.shape == (8, 3) and f.shape == (12, 3)
    assert np.allclose(v.min(axis=0), [0, 0, 0]) and np.allclose(v.max(axis=0), [0.1, 0.1, 0.1])
    # Every triangle's normal points away from the cube centre (outward winding).
    c = np.array([0.05, 0.05, 0.05])
    for tri in f:
        p0, p1, p2 = v[tri]
        n = np.cross(p1 - p0, p2 - p0)
        assert np.dot(n, (p0 + p1 + p2) / 3 - c) > 0


def test_slab_faces_merge():
    slab = _grid(range(ORG, ORG + 20), range(ORG, ORG + 10), [ORG])
    v, f = exposed_face_mesh(slab, 0.1)
    # Top and bottom merge to one rectangle each; the four sides to one each:
    # 6 rectangles -> 12 triangles, 8 shared vertices.
    assert f.shape == (12, 3)
    assert v.shape == (8, 3)


def test_column_floor_filter_trims_tall_columns():
    floor = _grid(range(ORG, ORG + 5), range(ORG, ORG + 5), [ORG])
    wall = _grid([ORG], range(ORG, ORG + 5), range(ORG + 1, ORG + 40))  # 4 m tall at x = 0
    kept = column_floor_filter(np.concatenate([floor, wall]), 0.1, 2.5)
    assert kept.shape[0] == floor.shape[0] + 5 * 25  # 2.5 m of wall per column
    assert (kept[:, 2] <= ORG + 25).all()
    assert column_floor_filter(wall, 0.1, 0.0).shape[0] == wall.shape[0]


def test_clip_spikes_removes_one_column_posts_only():
    floor = _grid(range(ORG, ORG + 8), range(ORG, ORG + 8), [ORG])
    spike = _grid([ORG + 3], [ORG + 3], range(ORG + 1, ORG + 4))  # 0.3 m post on one column
    pair = _grid([ORG + 6, ORG + 7], [ORG + 1], range(ORG + 1, ORG + 4))  # two columns: kept
    wall = _grid([ORG], range(ORG, ORG + 8), range(ORG + 1, ORG + 10))  # a wall: kept
    vox = np.concatenate([floor, spike, pair, wall])
    out = clip_spikes(vox, 0.1, 0.2)
    assert out.shape[0] == vox.shape[0] - spike.shape[0]
    assert not any((v[0] == ORG + 3 and v[1] == ORG + 3 and v[2] > ORG) for v in out.tolist())
    assert clip_spikes(vox, 0.1, 0.0).shape[0] == vox.shape[0]


def test_clear_track_volume_erases_fence_keeps_floor_and_risers():
    floor = _grid(range(ORG, ORG + 30), range(ORG, ORG + 10), [ORG - 1])  # top at z = 0
    fence = _grid([ORG + 15], range(ORG, ORG + 10), range(ORG, ORG + 4))  # 0.4 m thin wall across
    # A flight starting at x = 2.5 m: treads 0.3 m deep rising 0.2 m each, as
    # a mapped stair looks (a thin layer of top voxels per tread, plus risers).
    treads = []
    for k in range(4):
        x0 = ORG + 25 + 3 * k
        treads.append(_grid(range(x0, x0 + 3), range(ORG, ORG + 10), [ORG - 1 + 2 * (k + 1)]))
        treads.append(_grid([x0], range(ORG, ORG + 10), range(ORG - 1 + 2 * k, ORG + 1 + 2 * k)))
    stairs = np.concatenate(treads)
    vox = np.unique(np.concatenate([floor, fence, stairs]), axis=0)
    # The mapping robot walked along y = 0.5 m at IMU height ground + 0.481,
    # through the fence and up the whole flight.
    track = [[x, 0.5, 0.481] for x in np.arange(0.2, 2.5, 0.1)]
    track += [[2.5 + 0.3 * k + 0.15, 0.5, 0.481 + 0.2 * (k + 1)] for k in range(4)]
    out = clear_track_volume(vox, 0.1, np.array(track), 0.30, 0.20, 0.30, 0.481, window=3)
    # The fence has a gap where the track crossed it, from the floor margin
    # (0.2 m, so its lowest two layers stay as a kerb) up; the rest of it stands.
    fence_left = out[(out[:, 0] == ORG + 15) & (out[:, 2] >= ORG)]
    assert 0 < fence_left.shape[0] < fence.shape[0]
    gap = fence_left[np.abs(fence_left[:, 1] - (ORG + 5)) <= 2]
    assert gap.shape[0] > 0 and (gap[:, 2] <= ORG + 1).all()
    far = fence_left[np.abs(fence_left[:, 1] - (ORG + 5)) >= 4]
    assert (far[:, 2] == ORG + 3).any()
    # Not one voxel of the floor or of the flight is gone.
    stair_mask = vox[:, 0] >= ORG + 25
    assert (out[:, 0] >= ORG + 25).sum() == stair_mask.sum()
    assert (out[:, 2] == ORG - 1).sum() == (vox[:, 2] == ORG - 1).sum()


def test_fill_track_floor_closes_holes_under_the_track_only():
    floor = _grid(range(ORG, ORG + 20), range(ORG, ORG + 10), [ORG - 1])  # top z = 0
    hole = (floor[:, 0] >= ORG + 8) & (floor[:, 0] <= ORG + 12)  # a 0.5 m gap across the floor
    riser = _grid([ORG + 16], range(ORG, ORG + 10), [ORG, ORG + 1])  # a 0.2 m step at x = 1.6
    vox = np.concatenate([floor[~hole], riser])
    track = np.array([[x, 0.5, 0.481] for x in np.arange(0.2, 1.5, 0.1)])
    out = fill_track_floor(vox, 0.1, track, 0.30, 0.481)
    added = out[out.shape[0] - (out.shape[0] - vox.shape[0]):]
    assert added.shape[0] > 0 and (added[:, 2] == ORG - 1).all()  # floor voxels, top at z = 0
    # Filled within 0.3 m of the track line (y = 0.5), not across the whole gap.
    assert (np.abs(added[:, 1] - (ORG + 5)) <= 3).all()
    assert added[:, 0].min() >= ORG + 8 and added[:, 0].max() <= ORG + 12
    # The riser column is within the window: nothing added under it.
    assert not ((added[:, 0] == ORG + 16)).any()
    assert fill_track_floor(vox, 0.1, track, 0.0, 0.481).shape[0] == vox.shape[0]
