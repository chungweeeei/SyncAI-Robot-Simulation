"""web_view.py on a stand-in scene (a floor tile, a robot with a mesh, a prop,
a terrain pool), no browser: the colours, the box mesh, and a server that
takes the scene, a sync and a scan."""

import math
import os

import mujoco
import numpy as np
import pytest

from syncai_sim import visual
from syncai_sim.scene import SceneInfo
from syncai_sim.web_view import boxes_mesh, mesh_of, tile_colours

SCENE = """
<mujoco>
  <asset>
    <mesh name="tile" vertex="0 0 0  10 0 0  10 10 0  0 10 0" face="0 1 2  0 2 3" inertia="shell"/>
    <mesh name="trunk" vertex="-0.2 -0.1 0  0.2 -0.1 0  0.2 0.1 0  -0.2 0.1 0  0 0 0.1"
          face="0 1 4  1 2 4  2 3 4  3 0 4  0 2 1  0 3 2"/>
  </asset>
  <worldbody>
    <geom name="site_visual4_s0_t0_0" type="mesh" mesh="tile" contype="0" conaffinity="0" group="0"/>
    <body name="base_link" pos="1 1 0.5">
      <freejoint/>
      <geom type="box" size="0.2 0.1 0.05" group="3"/>
      <geom name="trunk_visual" type="mesh" mesh="trunk" group="1" contype="0" conaffinity="0"/>
      <body name="FL_HIP" pos="0.2 0.1 0">
        <joint name="FL_HipX_joint" axis="1 0 0"/>
        <geom type="sphere" size="0.03" group="3"/>
      </body>
    </body>
    <body name="prop_crate" pos="3 3 0.25">
      <freejoint/>
      <geom name="prop_crate_geom" type="box" size="0.25 0.25 0.25" group="2" rgba="0.9 0.6 0.2 1"/>
    </body>
    <body name="terrain0" mocap="true" pos="0 0 -1000"><geom name="terrain0_geom" type="box" size="0.049 0.049 0.049" group="3"/></body>
    <body name="terrain1" mocap="true" pos="1 0 -1000"><geom name="terrain1_geom" type="box" size="0.049 0.049 0.049" group="3"/></body>
  </worldbody>
</mujoco>
"""


class FakeState:
    x = 1.0
    y = 1.0
    yaw = 0.3
    z_ground = 0.0
    pitch = 0.0


class FakeCfg:
    visual_mesh = "voxels"
    map_dir = ""
    cache_dir = ""


class FakeSim:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(SCENE)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        m = self.model
        self.info = SceneInfo(
            base_body=m.body("base_link").id, mocap_body=-1, mocap_id=-1,
            lidar_body=m.body("base_link").id, imu_body=m.body("base_link").id,
            trunk_geoms=[], prop_bodies={"crate": m.body("prop_crate").id}, prop_mocap_ids={},
            site_geoms=[m.geom("site_visual4_s0_t0_0").id], site_storeys={0: [0]},
            site_tiles=np.array([[0, 5.0, 5.0, 7.07]]), storey_bounds=[],
            terrain_mocap_ids=np.array([0, 1]),
            terrain_geoms=np.array([m.geom("terrain0_geom").id, m.geom("terrain1_geom").id]),
            stair_geoms=[])
        self.cfg = FakeCfg()
        self.state = FakeState()
        self.last_scan = np.array([[1.0, 0.0, 0.0, 100.0], [0.0, 2.0, -0.4, 100.0]], np.float32)
        self.data.mocap_pos[0] = (1.0, 1.0, -0.05)  # one terrain box in use

    def status(self):
        return {"sim_time": 1.5, "last_event": "idle", "x": 1.0, "y": 1.0, "yaw": 0.3,
                "z_ground": 0.0, "pitch": 0.0, "motion_state": 1, "policy": 0, "falls": 0,
                "collisions": 0}

    def T_map_imu(self):
        T = np.eye(4)
        T[:3, 3] = (1.0, 1.0, 0.9)
        return T


def test_tile_colours_follow_the_ramp_by_kind():
    v = np.array([[0, 0, 0.0], [0, 0, 5.0], [0, 0, 10.0]])
    floor = tile_colours(v, visual.KIND_FLOOR, (0.0, 10.0))
    assert floor.dtype == np.uint8 and floor.shape == (3, 3)
    assert np.array_equal(floor[0], (np.array(visual.PALETTE[0][1]) * 255 + 0.5).astype(np.uint8))
    assert np.array_equal(floor[2], (np.array(visual.PALETTE[-1][1]) * 255 + 0.5).astype(np.uint8))
    wall = tile_colours(v, visual.KIND_WALL, (0.0, 10.0))
    assert np.all(wall[0] > floor[0])  # paler
    under = tile_colours(v, visual.KIND_UNDER, (0.0, 10.0))
    assert np.all(under == under[0])  # flat grey


def test_boxes_mesh_and_mesh_of():
    v, f = boxes_mesh(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), np.array([[1.0, 2.0, 3.0]] * 2))
    assert v.shape == (16, 3) and f.shape == (24, 3)
    assert np.allclose(v[:8].min(axis=0), [-1, -2, -3]) and np.allclose(v[8:].max(axis=0), [6, 2, 3])
    assert f[12:].min() == 8  # the second box indexes its own vertices
    sim = FakeSim()
    tv, tf = mesh_of(sim.model, sim.info.site_geoms[0])
    assert tv.shape == (4, 3) and tf.shape == (2, 3)


def test_web_view_serves_the_scene_without_a_browser():
    pytest.importorskip("viser")
    from syncai_sim.web_view import WebView
    sim = FakeSim()
    port = 18200 + os.getpid() % 500
    view = WebView(sim, teleop=None, port=port, log=lambda *a: None, site_in_background=False)
    try:
        assert view.rate_hz > 0 and view.is_running()
        assert len(view._site[0]) == 1  # the one tile, on storey 0
        assert [sim.model.body(b).name for b, _ in view._robot] == ["base_link", "FL_HIP"]
        assert len(view._props) == 1
        sim.data.qpos[0:3] = (2.0, 3.0, 0.6)
        view.sync(sim)
        base_handle = view._robot[0][1]
        assert np.allclose(base_handle.position, [2.0, 3.0, 0.6])
        view._terrain.value = True
        view._next_terrain = 0.0
        view.sync(sim)
        assert view._terrain_handle is not None
        view.show_scan(sim)
        assert view._shown == {0}
        assert "falls 0" in view._status.content
        assert math.isfinite(view._wanted().__len__())
    finally:
        view.close()
    assert not view.is_running()
