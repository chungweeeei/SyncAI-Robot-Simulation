"""Teleop from the browser viewer's panel (web_view.py): a command held until
the next one or a stop, and the controller's keys, each applied on the main
loop under the sim lock.

A command, once set, is sent every loop while it is non-zero and once more
as a stop when it goes to zero, then nothing: a teleop that set 0 every loop
would override cmd_vel / the driver's AXES with the panel untouched, and the
viewer and the nav stack can run together. `drive=False` (the session, no
--teleop) keeps only the teleport: the nav stack owns cmd_vel and
driver_manager the motion keys, but putting the robot back on the spawn
after a fall is wanted there most.

Until 2026-10-26 this was the GLFW window's keyboard (hold-to-drive arrows,
press timers for a browser that reported no key releases); the window went
with the VNC display when the browser viewer replaced it.
"""

from __future__ import annotations


class Teleop:
    def __init__(self, spawn, log=print, drive: bool = True):
        self.spawn = spawn  # x, y, z, yaw: where `teleport_home` puts the robot
        self.log = log
        self.drive = drive
        self._cmd: tuple[float, float, float] | None = None
        self._pending = []  # closures applied under the sim lock
        self._driving = False

    # -- from the panel (any thread) ----------------------------------------------
    def set_cmd(self, vx: float, wz: float) -> None:
        if self.drive:
            self._cmd = (float(vx), 0.0, float(wz))

    def stop(self) -> None:
        if self.drive:
            self._cmd = (0.0, 0.0, 0.0)

    def motion_key(self, key: str) -> None:
        """set_motion_key through the driver: "1" RL, "5" MPC (fake_driver.py)."""
        if self.drive:
            self._pending.append(
                lambda sim: self.log(f"[teleop] key {key} -> {sim.driver.set_motion_key(key)}"))

    def toggle_policy(self) -> None:
        if not self.drive:
            return

        def toggle(sim):
            new = 0 if sim.driver.policy == 1 else 1
            self.log(f"[teleop] policy -> {new} {sim.driver.set_policy_mode(new)}")
        self._pending.append(toggle)

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
