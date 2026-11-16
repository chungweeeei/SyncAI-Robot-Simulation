"""The panel's teleport back to the spawn (web_view.py), queued from the
viewer's thread and applied on the main loop under the sim lock.

Until 2026-11-15 this was teleop.py and also held the panel's driving: the
vx / vy / wz sliders sent as AXES to the controller (driver/udp_driver.py).
They went when the controller became SyncAI-Robot-Control, which speaks a
binary protocol and is driven from its own `syncai_ctl teleop` in the
terminal; the sim's panel drives nothing. Before that (until 2026-10-26)
the driving was the GLFW window's keyboard.
"""

from __future__ import annotations


class Teleport:
    def __init__(self, spawn):
        self.spawn = spawn  # x, y, z, yaw: where `home` puts the robot
        self._pending = []  # closures applied under the sim lock

    def home(self) -> None:
        """From the panel (any thread)."""
        x, y, z, yaw = self.spawn
        self._pending.append(lambda sim: sim.teleport(x, y, yaw, z))

    def apply(self, sim) -> None:
        """On the main loop, under the sim lock."""
        for fn in self._pending:
            fn(sim)
        self._pending.clear()
