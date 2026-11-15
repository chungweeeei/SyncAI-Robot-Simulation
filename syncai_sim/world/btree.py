"""Reader for OctoMap `.bt` files (binary occupancy trees), dependency-free.

`map/<name>/octomap.bt` is written by syncai_mapping's `clean_map --octomap 1`
with `OcTree::writeBinary()`. The sim needs the OCCUPIED leaves as integer voxel
cubes to greedy-mesh into MuJoCo boxes (`greedy_mesh.py`). The `octomap-python`
binding would do the same, but it has no aarch64 wheel and builds its own
octomap, so this parser exists instead: the format is two bytes per inner node
plus a text header, and nothing else.

Format (OcTreeBaseImpl::writeBinaryNode):

    # Octomap OcTree binary file
    # ...comments...
    id OcTree
    size <node count>
    res <resolution>
    data
    <binary: the root node, then depth-first its inner children>

Each inner node is 16 bits, 2 per child, child i in bits (2i, 2i+1) of byte 0
for i < 4 and of byte 1 for i >= 4 (low bit first):

    00  no child            10  free leaf        (bit 2i set)
    01  occupied leaf       11  inner node       (both set)

The subtrees of the inner children follow, in child order 0..7. Only inner
nodes are written; leaves are the two bits in their parent.

Keys follow OcTreeKey::computeChildKey: the root is (32768, 32768, 32768) at
depth 0, a child at depth d+1 adds +off on the axes whose bit is set in i (x=1,
y=2, z=4) and -off (or -1 when off is 0) on the others, off = 32768 >> (d+1).
A key at depth d < 16 names the cell's CENTRE voxel, so the cell's lowest voxel
index on each axis is `((k - 32768) >> diff << diff) + 32768`, diff = 16 - d,
and it spans `1 << diff` voxels. Voxel index i covers metres
[(i - 32768) * res, (i - 32767) * res).

Pure Python walks dp's 8.2 M inner nodes in about a minute; the result is
cached by `world.py`, so that is paid once per map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TREE_DEPTH = 16
TREE_MAX_VAL = 1 << (TREE_DEPTH - 1)  # 32768: the root key, and the index of x = 0


@dataclass
class OccupiedLeaves:
    """Occupied leaf cubes in voxel-index units, plus what the header said."""

    resolution: float
    node_count: int  # the header's `size`
    # (N, 4) int32: lowest voxel index on x, y, z, and the edge length in voxels
    # (1 for a full-depth leaf, 2 / 4 / ... for a pruned coarser one).
    cubes: np.ndarray
    free_leaves: int  # counted, not stored

    def __len__(self) -> int:
        return int(self.cubes.shape[0])

    def bounds_m(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds of the occupied space, metres, as (min, max)."""
        lo = (self.cubes[:, :3].min(axis=0) - TREE_MAX_VAL) * self.resolution
        hi_idx = self.cubes[:, :3] + self.cubes[:, 3:4]
        hi = (hi_idx.max(axis=0) - TREE_MAX_VAL) * self.resolution
        return lo, hi


def _parse_header(blob: bytes) -> tuple[float, int, int]:
    """Return (resolution, node_count, offset of the binary data)."""
    if not blob.startswith(b"# Octomap OcTree binary file"):
        raise ValueError("not an OctoMap .bt file (bad first line)")
    pos = 0
    res = None
    size = None
    while True:
        end = blob.find(b"\n", pos)
        if end < 0:
            raise ValueError("truncated .bt header")
        line = blob[pos:end].strip()
        pos = end + 1
        if line.startswith(b"#"):
            continue
        if line == b"data":
            break
        key, _, value = line.partition(b" ")
        if key == b"id":
            if value != b"OcTree":
                raise ValueError(f"unsupported tree id {value!r} (want OcTree)")
        elif key == b"size":
            size = int(value)
        elif key == b"res":
            res = float(value)
    if res is None or size is None:
        raise ValueError(".bt header lacks `res` or `size`")
    return res, size, pos


# For each byte value: the four children it describes, as (child j, lo bit, hi bit)
# with lo = bit 2j and hi = bit 2j+1 — looked up instead of shifted eight times
# per node, which is most of the inner loop.
_CHILD_BITS = [
    [((b >> (2 * j)) & 1, (b >> (2 * j + 1)) & 1) for j in range(4)] for b in range(256)
]


def read_occupied_leaves(path: str) -> OccupiedLeaves:
    """Walk a .bt file and return its occupied leaves (see module docstring)."""
    with open(path, "rb") as f:
        blob = f.read()
    res, size, pos = _parse_header(blob)
    data = memoryview(blob)[pos:]

    # Explicit stack instead of recursion: the tree is 16 deep but 8 M nodes
    # wide, and a Python frame per node would be slower still.
    stack = [(TREE_MAX_VAL, TREE_MAX_VAL, TREE_MAX_VAL, 0)]
    occ_x: list[int] = []
    occ_y: list[int] = []
    occ_z: list[int] = []
    occ_d: list[int] = []
    free = 0
    child_bits = _CHILD_BITS
    n = len(data)
    i = 0
    while stack:
        kx, ky, kz, depth = stack.pop()
        if i + 2 > n:
            raise ValueError(f"truncated .bt data at node {i // 2}")
        b0 = data[i]
        b1 = data[i + 1]
        i += 2
        cdepth = depth + 1
        off = TREE_MAX_VAL >> cdepth
        minus = -off if off else -1
        inner = []
        for byte, base in ((b0, 0), (b1, 4)):
            for j, (lo, hi) in enumerate(child_bits[byte]):
                if not (lo or hi):
                    continue
                c = base + j
                cx = kx + off if c & 1 else kx + minus
                cy = ky + off if c & 2 else ky + minus
                cz = kz + off if c & 4 else kz + minus
                if lo and hi:
                    inner.append((cx, cy, cz, cdepth))
                elif hi:
                    occ_x.append(cx)
                    occ_y.append(cy)
                    occ_z.append(cz)
                    occ_d.append(cdepth)
                else:
                    free += 1
        # Subtrees follow in child order 0..7; a stack pops the last push first.
        stack.extend(reversed(inner))
    if i != n:
        raise ValueError(f".bt data has {n - i} trailing bytes after the tree")

    keys = np.array([occ_x, occ_y, occ_z], dtype=np.int64).T  # (N, 3) centre keys
    depth = np.array(occ_d, dtype=np.int64)
    diff = TREE_DEPTH - depth
    lowest = (((keys - TREE_MAX_VAL) >> diff[:, None]) << diff[:, None]) + TREE_MAX_VAL
    edge = np.left_shift(1, diff)
    cubes = np.concatenate([lowest, edge[:, None]], axis=1).astype(np.int32)
    return OccupiedLeaves(resolution=res, node_count=size, cubes=cubes, free_leaves=free)


def cubes_to_voxels(cubes: np.ndarray) -> np.ndarray:
    """Expand leaf cubes to (M, 3) int32 full-depth voxel indices.

    Pruned leaves coarser than one voxel are rare in a lidar map (eight equal
    children only merge where a solid block is fully observed), so expanding
    them is cheap, and `greedy_mesh` wants one representation.
    """
    single = cubes[cubes[:, 3] == 1, :3]
    coarse = cubes[cubes[:, 3] > 1]
    if coarse.shape[0] == 0:
        return single.astype(np.int32, copy=False)
    parts = [single]
    for edge in np.unique(coarse[:, 3]):
        sel = coarse[coarse[:, 3] == edge, :3]
        r = np.arange(edge, dtype=np.int32)
        grid = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
        parts.append((sel[:, None, :] + grid[None, :, :]).reshape(-1, 3))
    return np.concatenate(parts, axis=0).astype(np.int32, copy=False)


def _morton(voxels: np.ndarray) -> np.ndarray:
    """Per voxel its path from the root: 16 octal digits, the root's child
    first (bit 0 x, 1 y, 2 z: the key's bit 15 - d at depth d)."""
    v = np.asarray(voxels, np.int64)
    m = np.zeros(v.shape[0], np.int64)
    for d in range(TREE_DEPTH):
        bit = TREE_DEPTH - 1 - d
        digit = ((v[:, 0] >> bit) & 1) | (((v[:, 1] >> bit) & 1) << 1) | (((v[:, 2] >> bit) & 1) << 2)
        m = (m << 3) | digit
    return m


def write_occupied(path: str, voxels: np.ndarray, resolution: float) -> int:
    """Write full-depth voxel indices ((N, 3), as `cubes_to_voxels` returns
    them) as an OcTree `.bt` that `read_occupied_leaves` reads back: every
    voxel an occupied leaf at depth 16, nothing free, nothing pruned. For
    maps made rather than measured (scripts/make_scenes.py). Returns the
    header's `size`, the tree's node count (inner nodes and leaves) as
    OcTree::size() counts it."""
    m = np.unique(_morton(voxels))
    out = bytearray()
    leaves = int(m.size)
    inner = 0
    # Depth first, children in order 0..7: the reader's order (module docstring).
    stack = [(0, 0, int(m.size))]
    while stack:
        depth, a, b = stack.pop()
        shift = 3 * (TREE_DEPTH - 1 - depth)
        digits = (m[a:b] >> shift) & 7
        starts = np.searchsorted(digits, np.arange(9))
        node = [0, 0]
        kids = []
        for c in range(8):
            if starts[c] == starts[c + 1]:
                continue
            code = 2 if depth == TREE_DEPTH - 1 else 3  # occupied leaf (hi bit) or inner node
            node[c // 4] |= code << (2 * (c % 4))
            if code == 3:
                kids.append((depth + 1, a + int(starts[c]), a + int(starts[c + 1])))
        out += bytes(node)
        inner += 1
        stack.extend(reversed(kids))
    size = inner + leaves
    header = (b"# Octomap OcTree binary file\n# written by syncai_sim.world.btree.write_occupied\n"
              b"id OcTree\nsize %d\nres %s\ndata\n" % (size, repr(float(resolution)).encode()))
    with open(path, "wb") as f:
        f.write(header + bytes(out))
    return size
