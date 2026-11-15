"""terrains/friction.py: a map's zones of floor with their own sliding
friction, and how they reach a contact (scene.py)."""

import mujoco
import numpy as np
import pytest

from syncai_sim.scene import scene as sc
from syncai_sim.terrains import friction as fr

ZONES = [("wet", (2.0, 5.0, -2.0, 2.0), 0.4), ("ice", (4.0, 6.0, 0.0, 1.0), 0.2)]


def write(tmp_path, text):
    path = tmp_path / "site.yaml"
    path.write_text(text)
    return str(path)


def test_a_map_without_a_file_has_no_zones(tmp_path):
    assert fr.read_zones(str(tmp_path / "nowhere.yaml")) == []
    assert fr.read_zones(write(tmp_path, "")) == []


def test_the_file_is_named_after_the_maps_folder():
    assert fr.zones_path("/maps/dp/", "/cfg") == "/cfg/dp.yaml"
    assert fr.zones_path("/home/sim/scenes/course").endswith("config/friction/course.yaml")


def test_zones_are_read_with_their_box_and_friction(tmp_path):
    path = write(tmp_path, "zones:\n  - name: wet\n    box: [2, 5, -2, 2]\n    friction: 0.4\n")
    assert fr.read_zones(path) == [("wet", (2.0, 5.0, -2.0, 2.0), 0.4)]


@pytest.mark.parametrize("zone", [
    "box: [5, 2, -2, 2]\n    friction: 0.4",  # x1 before x0
    "box: [2, 5, -2]\n    friction: 0.4",  # three numbers
    "box: [2, 5, -2, 2]\n    friction: 0",  # no friction at all
    "box: [2, 5, -2, 2]\n    friction: -0.1",
    "box: [2, 5, -2, 2]\n    friction: 3",
])
def test_a_zone_that_makes_no_sense_is_refused(tmp_path, zone):
    with pytest.raises(ValueError):
        fr.read_zones(write(tmp_path, f"zones:\n  - name: bad\n    {zone}\n"))


def test_the_floor_is_the_default_outside_every_zone_and_the_later_zone_where_two_overlap():
    xy = np.array([[0.0, 0.0],  # before the wet floor
                   [3.0, 0.5],  # wet
                   [4.5, 0.5],  # wet and ice: ice is listed later
                   [5.5, 0.5],  # ice alone
                   [4.5, -1.0],  # wet, beside the ice
                   [5.0, 0.0 - 1e-9],  # x1 is outside the wet floor
                   [2.0, -2.0]])  # x0, y0 are inside
    assert fr.sliding(ZONES, xy, 1.0).tolist() == [1.0, 0.4, 0.2, 0.2, 0.4, 1.0, 0.4]
    assert fr.sliding([], xy, 0.7).tolist() == [0.7] * 7


def _pool():
    """Two terrain boxes a metre apart and a foot on the first, as scene.py
    builds them."""
    spec = mujoco.MjSpec()
    for i in range(2):
        body = spec.worldbody.add_body(name=f"terrain{i}", mocap=True, pos=(float(i), 0, 0))
        body.add_geom(name=f"terrain{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05, 0.05, 0.05))
    foot = spec.worldbody.add_body(name="foot", pos=(0, 0, 0.07))
    foot.add_freejoint()
    foot.add_geom(name="foot_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.022, 0, 0),
                  friction=(1.0, 0.01, 0.01))
    model = spec.compile()
    geoms = np.array([model.geom(f"terrain{i}_geom").id for i in range(2)])
    info = sc.SceneInfo(base_body=0, lidar_body=0, imu_body=0, trunk_geoms=[], prop_bodies={},
                        prop_mocap_ids={}, terrain_geoms=geoms)
    return model, info, model.geom("foot_geom").id


def test_the_terrains_friction_becomes_the_contacts_not_the_larger_of_the_two():
    model, info, foot = _pool()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # As built: the foot's 1.0 wins over anything lower on the box.
    model.geom_friction[info.terrain_geoms, 0] = 0.2
    mujoco.mj_forward(model, data)
    assert data.ncon == 1 and data.contact[0].friction[0] == 1.0

    sc.let_terrain_set_friction(model, info, torsional=0.01, rolling=0.01)
    sc.set_terrain_friction(model, info, np.array([0.2, 0.6]))
    mujoco.mj_forward(model, data)
    assert data.contact[0].friction[0] == 0.2
    # Torsional and rolling stay the foot's.
    assert data.contact[0].friction[2] == 0.01 and data.contact[0].friction[3] == 0.01
    assert model.geom_friction[info.terrain_geoms, 0].tolist() == [0.2, 0.6]
    # The foot itself is untouched.
    assert model.geom_priority[foot] == 0 and model.geom_friction[foot, 0] == 1.0


def test_fewer_frictions_than_boxes_leaves_the_rest():
    model, info, _ = _pool()
    sc.set_terrain_friction(model, info, np.array([0.3]))
    assert model.geom_friction[info.terrain_geoms[0], 0] == 0.3
    assert model.geom_friction[info.terrain_geoms[1], 0] == 1.0  # MuJoCo's default
