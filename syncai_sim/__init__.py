"""SyncAI robot simulation: MuJoCo world from a site's octomap, a rigid G23 driven
kinematically, a MID360 by ray casting, and the ROS 2 face of Point-LIO, the
localizer and driver_manager — enough to run the workspace's 3D navigation stack
end to end without a robot. See README.md."""

import os

# The repository root (config/, cache/, policy/ hang off it). Modules sit two
# levels under it since the layered subpackages (2026-11-03); naming it once
# keeps the four modules that reach config/ or cache/ from counting dirnames.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
