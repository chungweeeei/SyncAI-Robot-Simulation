"""The parser against a tree written by hand in the .bt encoding.

Encoding reminder (btree.py): per inner node two bytes, child i -> bits (2i,
2i+1) of byte i // 4 (low bit first); 01 (hi only) = occupied leaf, 10 (lo only)
= free leaf, 11 = inner node, whose own two bytes follow in child order.
"""

import numpy as np
import pytest

from syncai_sim.btree import TREE_MAX_VAL, cubes_to_voxels, read_occupied_leaves


def _node(children: dict[int, str]) -> bytes:
    """children: child index -> 'occ' | 'free' | 'inner'."""
    b = [0, 0]
    for i, kind in children.items():
        lo, hi = {"free": (1, 0), "occ": (0, 1), "inner": (1, 1)}[kind]
        j = i % 4
        b[i // 4] |= (lo << (2 * j)) | (hi << (2 * j + 1))
    return bytes(b)


def _write(path, res, nodes: list[bytes]):
    body = b"".join(nodes)
    header = (
        b"# Octomap OcTree binary file\n# comment\nid OcTree\nsize %d\nres %s\ndata\n"
        % (len(nodes), str(res).encode())
    )
    path.write_bytes(header + body)


def test_depth_chain_to_single_voxel(tmp_path):
    # Root -> child 7 (+x +y +z) inner at every depth, until depth 16 where
    # child 0 is an occupied voxel and child 7 a free one.
    nodes = [_node({7: "inner"}) for _ in range(15)]  # depths 0..14 -> children at 1..15
    nodes.append(_node({0: "occ", 7: "free"}))  # the depth-15 node's children are voxels
    p = tmp_path / "t.bt"
    _write(p, 0.1, nodes)
    leaves = read_occupied_leaves(str(p))
    assert leaves.resolution == 0.1
    assert len(leaves) == 1
    assert leaves.free_leaves == 1
    x, y, z, edge = leaves.cubes[0].tolist()
    assert edge == 1
    # Always the +side child: the depth-15 node sits at key 32768 + 16383 ... the
    # exact value matters less than the invariant the planner relies on: the
    # top-right-most voxel of the + + + octant chain is the last index.
    # Depth-15 node's "minus" child 0 is the even voxel of its pair.
    assert x == y == z
    assert x % 2 == 0
    assert x == 65534  # one below the last voxel (65535), which is child 7


def test_coarse_leaf_expands_to_cube(tmp_path):
    # Root's child 0 is an occupied leaf at depth 1: half the tree on each axis.
    p = tmp_path / "c.bt"
    _write(p, 0.5, [_node({0: "occ"})])
    leaves = read_occupied_leaves(str(p))
    x, y, z, edge = leaves.cubes[0].tolist()
    assert edge == 1 << 15
    assert (x, y, z) == (0, 0, 0)  # the -x -y -z half: voxel indices 0 .. 32767
    lo, hi = leaves.bounds_m()
    assert np.allclose(lo, -TREE_MAX_VAL * 0.5)
    assert np.allclose(hi, 0.0)


def test_cubes_to_voxels_expands_edges():
    cubes = np.array([[10, 10, 10, 1], [20, 20, 20, 2]], dtype=np.int32)
    vox = cubes_to_voxels(cubes)
    assert vox.shape == (9, 3)
    assert [10, 10, 10] in vox.tolist()
    assert [21, 21, 21] in vox.tolist()
    assert [22, 20, 20] not in vox.tolist()


def test_rejects_other_files(tmp_path):
    p = tmp_path / "x.bt"
    p.write_bytes(b"not a tree")
    with pytest.raises(ValueError):
        read_occupied_leaves(str(p))
