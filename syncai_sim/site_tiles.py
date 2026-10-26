"""The site for the browser, tile by tile: every voxel face painted with its
lidar intensity on the height ramp, as GLB files the viewer sends once.

The voxel mesh is rectangles (world.exposed_face_rects: coplanar faces
merged), and a rectangle's corners cannot carry a colour per voxel face — a
5 m floor slab is one rectangle. So each tile (TILE_M squares of the site,
one per storey) gets a texture: an atlas in which every rectangle has a
patch of n1 x n2 texels, one per voxel face, painted by visual.shade from
the face's height and its voxel's intensity (voxel_intensity.py), with a
one-texel gutter of its own edge around it so the GPU's linear filtering
never blends a neighbour in. The rectangle's four corners carry the patch's
texture coordinates (world.quad_corners says which corner is which), so the
geometry stays what it was: two triangles per rectangle. A tile is ~1600
rectangles and a 512 x 512 atlas on dp; the GLB's PNG is a few tens of KB.

`visual_mesh: pointcloud` (pcd_mesh.py's Poisson surface) and `tsdf`
(tsdf_mesh.py) have no voxel faces: their vertices are a few centimetres
apart, so there the intensity goes on the vertices.

Cached as cache/<map>/site_tiles-<visual_mesh>.npz (the GLB bytes of every
tile), keyed on the world, the intensities, the look (visual.SHADE_VERSION)
and, for the surface, the surface's own cache.
"""

from __future__ import annotations

import io
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np

from . import visual
from .btree import TREE_MAX_VAL
from .world import World, quad_corners

TILE_M = 10.0
FORMAT = 1
ATLAS_MAX = 2048  # texels a side; a tile that needs more gets more parts
GUTTER = 1


@dataclass
class Tile:
    storey: int
    ix: int
    iy: int
    part: int
    glb: bytes
    triangles: int

    @property
    def name(self) -> str:
        return f"/site/s{self.storey}/t{self.ix}_{self.iy}/p{self.part}"


def pack(w: np.ndarray, h: np.ndarray, width: int, max_height: int
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Shelf-pack rectangles of texel sizes (w, h) into atlases `width` wide
    and at most `max_height` tall, tallest first: (x, y, atlas index) per
    rectangle and each atlas's used height."""
    n = w.shape[0]
    x = np.zeros(n, np.int64)
    y = np.zeros(n, np.int64)
    a = np.zeros(n, np.int64)
    heights: list[int] = []
    cx = cy = row_h = 0
    atlas = 0
    for i in np.argsort(-h, kind="stable"):
        wi, hi = int(w[i]), int(h[i])
        if cx + wi > width:
            cx, cy, row_h = 0, cy + row_h, 0
        if cy + hi > max_height and (cx or cy):
            heights.append(cy + row_h)
            atlas += 1
            cx = cy = row_h = 0
        x[i], y[i], a[i] = cx, cy, atlas
        cx += wi
        row_h = max(row_h, hi)
    heights.append(cy + row_h)
    return x, y, a, heights


def _pow2(n: int) -> int:
    return 1 << max(0, math.ceil(math.log2(max(n, 1))))


def _paint(r: np.ndarray, x: np.ndarray, y: np.ndarray, width: int, height: int,
           resolution: float, lookup, z_range: tuple[float, float]) -> np.ndarray:
    """The atlas (height, width, 3) uint8 of rectangles `r` (rows of
    exposed_face_rects, int64) placed at texel (x, y) with their gutters."""
    axis, sign = r[:, 0], r[:, 1]
    n1, n2 = r[:, 5], r[:, 6]
    w, h = n1 + 2 * GUTTER, n2 + 2 * GUTTER
    area = w * h
    total = int(area.sum())
    rid = np.repeat(np.arange(r.shape[0]), area)
    offsets = np.repeat(np.cumsum(area) - area, area)
    t = np.arange(total) - offsets
    i_pad = t % w[rid]
    j_pad = t // w[rid]
    i = np.clip(i_pad - GUTTER, 0, n1[rid] - 1)
    j = np.clip(j_pad - GUTTER, 0, n2[rid] - 1)
    idx = np.empty((total, 3), np.int64)
    ax = axis[rid]
    for a in range(3):
        a1, a2 = [k for k in range(3) if k != a]
        sel = ax == a
        idx[sel, a1] = r[rid[sel], 2] + i[sel]
        idx[sel, a2] = r[rid[sel], 3] + j[sel]
        idx[sel, a] = r[rid[sel], 4]
    up = (ax == 2) & (sign[rid] > 0)
    down = (ax == 2) & (sign[rid] < 0)
    kind = np.where(up, visual.KIND_FLOOR, np.where(down, visual.KIND_UNDER, visual.KIND_WALL))
    z = np.where(ax == 2, (r[rid, 4] + (sign[rid] > 0) - TREE_MAX_VAL) * resolution,
                 (idx[:, 2] + 0.5 - TREE_MAX_VAL) * resolution)
    intensity = lookup(idx) if lookup is not None else None
    colour = visual.shade(kind, z, intensity, z_range)
    image = np.zeros((height, width, 3), np.uint8)
    image[y[rid] + j_pad, x[rid] + i_pad] = colour
    return image


def _glb(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray | None,
         image: np.ndarray | None, colours: np.ndarray | None) -> bytes:
    import trimesh
    from PIL import Image

    if image is not None:
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(image), metallicFactor=0.0, roughnessFactor=0.9)
        vis = trimesh.visual.TextureVisuals(uv=uv, material=material)
        mesh = trimesh.Trimesh(vertices, faces, visual=vis, process=False)
    else:
        mesh = trimesh.Trimesh(vertices, faces, vertex_colors=colours, process=False)
    with io.BytesIO() as buf:
        mesh.export(buf, file_type="glb")
        return buf.getvalue()


def voxel_tiles(rects: np.ndarray, storeys: np.ndarray, resolution: float, lookup,
                tile_m: float = TILE_M) -> list[Tile]:
    """The voxel mesh's rectangles as textured tiles. `lookup(idx (N, 3)) ->
    intensities` (voxel_intensity.VoxelIntensity.lookup), or None for the
    ramp alone."""
    if rects.shape[0] == 0:
        return []
    r = rects.astype(np.int64)
    axis, sign = r[:, 0], r[:, 1]
    centre = np.zeros((r.shape[0], 3))
    for a in range(3):
        a1, a2 = [k for k in range(3) if k != a]
        sel = axis == a
        centre[sel, a1] = r[sel, 2] + r[sel, 5] / 2.0
        centre[sel, a2] = r[sel, 3] + r[sel, 6] / 2.0
        centre[sel, a] = r[sel, 4] + 0.5
    centre = (centre - TREE_MAX_VAL) * resolution
    ix = np.floor(centre[:, 0] / tile_m).astype(np.int64)
    iy = np.floor(centre[:, 1] / tile_m).astype(np.int64)
    up = (axis == 2) & (sign > 0)
    z_range = visual.height_range((r[up, 4] + 1 - TREE_MAX_VAL) * resolution)
    groups, inverse = np.unique(np.column_stack((storeys.astype(np.int64), ix, iy)), axis=0,
                                return_inverse=True)
    inverse = inverse.reshape(-1)
    tiles: list[Tile] = []
    for g, (st, tx, ty) in enumerate(groups):
        sel = np.flatnonzero(inverse == g)
        rs = r[sel]
        w, h = rs[:, 5] + 2 * GUTTER, rs[:, 6] + 2 * GUTTER
        width = min(ATLAS_MAX, max(64, _pow2(math.ceil(math.sqrt(float((w * h).sum()) * 1.15)))))
        x, y, atlas, heights = pack(w, h, width, ATLAS_MAX)
        for part, used in enumerate(heights):
            part_sel = np.flatnonzero(atlas == part)
            rp = rs[part_sel]
            height = _pow2(used)
            image = _paint(rp, x[part_sel], y[part_sel], width, height, resolution, lookup, z_range)
            corners, texels = quad_corners(rp.astype(np.int32), resolution)
            px = x[part_sel][:, None] + GUTTER + texels[:, :, 0]
            py = y[part_sel][:, None] + GUTTER + texels[:, :, 1]
            # trimesh's uv has its origin at the image's bottom-left; the atlas
            # is painted from the top (row 0): v counts down from 1.
            uv = np.stack([px / width, 1.0 - py / height], axis=-1).reshape(-1, 2)
            n = rp.shape[0]
            base = np.arange(n) * 4
            faces = np.concatenate([np.column_stack((base, base + 1, base + 2)),
                                    np.column_stack((base, base + 2, base + 3))])
            glb = _glb(corners.reshape(-1, 3).astype(np.float32), faces, uv, image, None)
            tiles.append(Tile(int(st), int(tx), int(ty), part, glb, 2 * n))
    return tiles


def surface_tiles(mesh, origin_m: np.ndarray, lo: np.ndarray, resolution: float, lookup,
                  tile_m: float = TILE_M) -> list[Tile]:
    """pcd_mesh.PcdMesh (vertices, faces, kinds, storeys per face) as tiles
    with the intensity on the vertices: a Poisson surface's vertices are a
    few centimetres apart, which is the texture's resolution anyway."""
    v, f = mesh.vertices, mesh.faces
    if f.shape[0] == 0:
        return []
    centroid = v[f].mean(axis=1)
    ix = np.floor(centroid[:, 0] / tile_m).astype(np.int64)
    iy = np.floor(centroid[:, 1] / tile_m).astype(np.int64)
    kinds = mesh.kinds.astype(np.int64)
    vkind = np.full(v.shape[0], visual.KIND_WALL, np.int64)
    vkind[f[:, 0]] = kinds  # a vertex shared by kinds takes one of its faces'
    vkind[f[:, 1]] = kinds
    vkind[f[:, 2]] = kinds
    floor_v = np.unique(f[kinds == visual.KIND_FLOOR])
    z_range = visual.height_range(v[floor_v][:, 2] if floor_v.size else np.zeros(0))
    intensity = None
    if lookup is not None:
        idx = np.floor((v.astype(np.float64) - origin_m[None, :]) / resolution).astype(np.int64) \
            + lo[None, :]
        intensity = lookup(idx)
    colours = visual.shade(vkind, v[:, 2], intensity, z_range)
    groups, inverse = np.unique(np.column_stack((mesh.storeys.astype(np.int64), ix, iy)), axis=0,
                                return_inverse=True)
    inverse = inverse.reshape(-1)
    tiles: list[Tile] = []
    for g, (st, tx, ty) in enumerate(groups):
        fs = f[inverse == g]
        used, local = np.unique(fs, return_inverse=True)
        glb = _glb(v[used].astype(np.float32), local.reshape(-1, 3), None, None, colours[used])
        tiles.append(Tile(int(st), int(tx), int(ty), 0, glb, int(fs.shape[0])))
    return tiles


def _key(world: World, visual_mesh: str, intensity_key: str | None, surface_key: str | None) -> str:
    return json.dumps({"format": FORMAT, "shade": visual.SHADE_VERSION, "tile_m": TILE_M,
                       "world": [world.stats.cache_key, world.stats.options],
                       "visual_mesh": visual_mesh, "intensity": intensity_key,
                       "surface": surface_key}, sort_keys=True)


def load_or_build(world: World, map_dir: str, cache_dir: str, visual_mesh: str,
                  log=print) -> list[Tile]:
    """The site's tiles for the browser, from the cache or built (dp: ~10 s
    for the voxels; the surface adds pcd_mesh.py's own build the first time)."""
    from . import voxel_intensity

    if visual_mesh not in ("voxels", "pointcloud", "tsdf"):
        raise ValueError(f"visual_mesh {visual_mesh!r}: voxels | pointcloud | tsdf")
    pcd_path = os.path.join(map_dir, "map.pcd")
    intensity_key = None
    if os.path.isfile(pcd_path):
        st = os.stat(pcd_path)
        intensity_key = f"{st.st_size}-{int(st.st_mtime)}-{voxel_intensity.FORMAT}"
    surface_key = None
    surface = None
    if visual_mesh != "voxels":
        if visual_mesh == "pointcloud":
            from .pcd_mesh import load_or_build as load_surface
            cache_file = "pcd_mesh.npz"
        else:
            from .tsdf_mesh import load_or_build as load_surface
            cache_file = "tsdf_mesh.npz"
        surface = load_surface(map_dir, cache_dir, world, log=log)
        if surface is None:
            log("[tiles] no surface: the voxels are shown instead")
            visual_mesh = "voxels"
        else:
            st = os.stat(os.path.join(cache_dir, cache_file))
            surface_key = f"{st.st_size}-{int(st.st_mtime)}"
    key = _key(world, visual_mesh, intensity_key, surface_key)
    path = os.path.join(cache_dir, f"site_tiles-{visual_mesh}.npz")
    if os.path.isfile(path):
        z = np.load(path)
        if str(z["key"]) == key:
            meta = json.loads(str(z["meta"]))
            tiles = [Tile(m[0], m[1], m[2], m[3], z[f"t{k}"].tobytes(), m[4])
                     for k, m in enumerate(meta)]
            log(f"[tiles] cached: {len(tiles)} tiles, "
                f"{sum(t.triangles for t in tiles) / 1e6:.1f} M triangles")
            return tiles
    t0 = time.perf_counter()
    table = voxel_intensity.load_or_build(map_dir, cache_dir, world, log=log)
    lookup = table.lookup if table is not None else None
    if visual_mesh == "voxels":
        tiles = voxel_tiles(world.mesh_rects, world.mesh_rect_storeys, world.grid.res, lookup)
    else:
        tiles = surface_tiles(surface, world.grid.origin_m, world.grid.lo.astype(np.int64),
                              world.grid.res, lookup)
    log(f"[tiles] built {len(tiles)} tiles, {sum(t.triangles for t in tiles) / 1e6:.1f} M "
        f"triangles, {sum(len(t.glb) for t in tiles) / 1e6:.0f} MB of GLB in "
        f"{time.perf_counter() - t0:.1f} s")
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, key=key,
             meta=json.dumps([[t.storey, t.ix, t.iy, t.part, t.triangles] for t in tiles]),
             **{f"t{k}": np.frombuffer(t.glb, dtype=np.uint8) for k, t in enumerate(tiles)})
    os.replace(tmp, path)
    return tiles
