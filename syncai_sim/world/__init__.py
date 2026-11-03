"""The site as the sim senses it: octomap.bt -> a voxel grid (`build.py`), the
truth for lidar rays, the ground under the robot and whether the trunk is
blocked (`voxel_grid.py`, numba). Nothing in here is drawn or stood on by
MuJoCo: the picture is `visuals/`, the feet's floor is `terrains/`.
`python3 -m syncai_sim.world map/<name>` rebuilds a map's cache."""
