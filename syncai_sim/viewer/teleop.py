"""Teleop from the browser viewer's panel (web_view.py): a command held until
the next one or a stop, and the teleport home, each applied on the main loop
under the sim lock. The command goes to the controller process
(driver/udp_driver.py): the sim walks nowhere by itself. The controller's
motion keys and policy toggle were here until 2026-11-15 (udp_driver.py).

A command, once set, is sent every loop while it is non-zero and once more
as a stop when it goes to zero, then nothing: a teleop that set 0 every loop
would override the AXES arriving on the UDP link with the panel untouched,
so the panel and a program on the link (driver_manager under a navigation
goal) drive together, and the panel needs no switch: until 2026-11-15 a
--teleop flag turned its driving on, and no run was ever started without it.

Until 2026-10-26 this was the GLFW window's keyboard (hold-to-drive arrows,
press timers for a browser that reported no key releases); the window went
with the VNC display when the browser viewer replaced it.
"""

from __future__ import annotations


class Teleop:
    def __init__(self, spawn, log=print):
        self.spawn = spawn  # x, y, z, yaw: where `teleport_home` puts the robot
        self.log = log
        self._cmd: tuple[float, float, float] | None = None
        self._pending = []  # closures applied under the sim lock
        self._driving = False

    # -- from the panel (any thread) ----------------------------------------------
    def set_cmd(self, vx: float, vy: float, wz: float) -> None:
        self._cmd = (float(vx), float(vy), float(wz))

    def stop(self) -> None:
        self._cmd = (0.0, 0.0, 0.0)

    def teleport_home(self) -> None:
        x, y, z, yaw = self.spawn
        self._pending.append(lambda sim: sim.teleport(x, y, yaw, z))

    # -- on the main loop, under the sim lock --------------------------------------
    def apply(self, sim) -> None:
        for fn in self._pending:
            fn(sim)
        self._pending.clear()
        if self._cmd is None:
            return
        moving = any(abs(v) > 1e-9 for v in self._cmd)
        if moving or self._driving:
            sim.set_cmd(*self._cmd)
        self._driving = moving
