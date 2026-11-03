"""SyncAI robot simulation: the G23 in MuJoCo on a real site's voxels, walking on
its own policies or on a controller process's joint commands, a MID360 cast
on the voxels, a browser viewer, and UDP as the only way out (the gait
controller's ASCII link, the hardware link). See README.md and
docs/architecture.md."""

import os

# The repository root (config/, cache/, policy/ hang off it). Modules sit two
# levels under it since the layered subpackages (2026-11-03); naming it once
# keeps the four modules that reach config/ or cache/ from counting dirnames.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
