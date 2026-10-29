# config/colour — what the camera saw, per voxel, per map

`<map>.npz` is the colour of each voxel of `map/<map>/octomap.bt` as the
camera of the mapping run saw it (`syncai_sim/voxel_colour.py`): the
viewer's `photo` look (the panel's Look switch, `visual_look`) paints the
site with it instead of the height ramp. Nothing the lidar, the feet or the
planner see reads these files; they are the picture only.

Built from the run's bag — its lidar scans, images, `camera_info` and
`/tf_static`, read without ROS — and Point-LIO's odometry replayed from the
same bag (`scripts/bag_odom.py` says how; `scripts/colour_map.py` builds the
table). The bag must be the mapping run of that map, so that the odometry's
frame is the map's. A voxel the camera never looked at (the robot's back
and sides, the ceilings) has no entry and is drawn grey; a table is for one
grid (the octomap's bounds) and the sim refuses one built for another.

| map | bag | scans | coverage | built |
|---|---|---|---|---|
| outdoor_1027 | workspace `record/rec_20261027_031456_replay` (robot01, 524 s, 14836 frames) | 5240 (5236 coloured), 17.4 M observations | 422 946 of 550 854 occupied voxels (77 %), 9.1 MB, 48 s | 2026-10-28 |

```bash
# in the sim container, from the workspace root
python3 /home/syncrobotic/sim/scripts/colour_map.py \
    --bag record/rec_20261027_031456_replay --odom record/rec_20261027_031456_replay/odom.txt \
    --map map/outdoor_1027 --preview /tmp/outdoor_1027.png
```

`np.load(<map>.npz)["meta"]` holds the build's numbers (scans coloured,
observations, occupied voxels coloured, the camera model). A few MB per map,
compressed; the bags themselves are not in this repository.
