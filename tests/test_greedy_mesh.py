import numpy as np

from syncai_sim.greedy_mesh import boxes_volume, merge_voxels


def _grid(x, y, z):
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)


def test_solid_block_is_one_box():
    vox = _grid(range(3, 8), range(10, 12), range(0, 4))
    boxes = merge_voxels(vox)
    assert boxes.shape == (1, 6)
    assert boxes[0].tolist() == [3, 10, 0, 5, 2, 4]


def test_volume_is_conserved_and_boxes_disjoint():
    rng = np.random.default_rng(1)
    vox = np.unique(rng.integers(0, 12, size=(600, 3)), axis=0)
    boxes = merge_voxels(vox)
    assert boxes_volume(boxes) == vox.shape[0]
    # Re-expand and compare the sets: every voxel covered exactly once.
    covered = []
    for x0, y0, z0, nx, ny, nz in boxes:
        covered.append(_grid(range(x0, x0 + nx), range(y0, y0 + ny), range(z0, z0 + nz)))
    covered = np.concatenate(covered)
    assert covered.shape[0] == vox.shape[0]
    assert np.array_equal(np.unique(covered, axis=0), vox)


def test_floor_and_wall_merge_into_few_boxes():
    floor = _grid(range(0, 40), range(0, 30), [0])
    wall = _grid([0], range(0, 30), range(1, 25))
    boxes = merge_voxels(np.concatenate([floor, wall]))
    assert boxes.shape[0] == 2
    assert boxes_volume(boxes) == floor.shape[0] + wall.shape[0]


def test_duplicates_and_empty():
    vox = np.array([[1, 1, 1], [1, 1, 1], [2, 1, 1]])
    boxes = merge_voxels(vox)
    assert boxes.tolist() == [[1, 1, 1, 2, 1, 1]]
    assert merge_voxels(np.zeros((0, 3), dtype=int)).shape == (0, 6)
