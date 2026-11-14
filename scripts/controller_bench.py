#!/usr/bin/env python3
"""The sim as a controller's hardware, headless: the bench for the hardware link.

    # the link end to end with no controller process (the stand-in answers every
    # state with a PD to the default pose): 0 falls, ~5000 states in 5 s
    python3 scripts/controller_bench.py --fake-controller --seconds 5

    # a controller process on the link (its --backend sim): stand, enter its MPC,
    # walk at 0.5 m/s, reading its MODE_STATE from the telemetry it sends
    python3 scripts/controller_bench.py \\
        --send "MODE Z" --send "MODE M" --gap 3 --axes 0.5 0 0 --seconds 15 --listen

The sim is built as `python3 -m syncai_sim` builds it (the defaults, any
--profile, -p),
spawned at the INI's pose or --pose, and stepped at wall-clock pace with no
viewer. Lines given with --send go to the controller's command port in order,
--gap seconds apart (MODE Z, then MODE M or MODE C; RESET_SAFETY after a fall);
--axes is sent at 10 Hz for the rest of the run, which keeps a controller's
AXES watchdog fed. --listen binds the controller's telemetry port (50010) and
reads MODE_STATE into the driver, so the report has the state transitions;
only one program can bind it, so not while driver_manager runs.

The report is falls, the pitch's mean / 5th / 95th percentile / extremes,
roll, speed, distance, plus the link's counters (states sent,
commands received, watchdog trips), the state transitions seen, and the last
command's gains and feed-forward, so what the controller sent can be read
next to what the body did. One JSON line per run in log/bench/<name>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from syncai_sim.config import (PARAM_DEFAULTS, apply_overrides, build_sim_config,  # noqa: E402
                               load_profile, read_ini)

STATE_NAMES = {v: k for k, v in {"STAND": 0, "LOCOMOTION": 1, "LIE_DOWN": 2, "DAMPING": 3,
                                 "ESTOP": 4, "MPC": 5, "UNKNOWN": 8}.items()}


class Telemetry:
    """The controller's ASCII telemetry on its port, read for MODE_STATE only."""

    def __init__(self, port: int, log=print):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.log = log
        self.lines = 0
        self.transitions: list[tuple[float, int, int]] = []  # (sim time, policy, state)

    def poll(self, driver, t: float) -> None:
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                return
            for line in data.decode("ascii", "replace").splitlines():
                parts = line.split()
                if "MODE_STATE" not in parts:
                    continue
                self.lines += 1
                i = parts.index("MODE_STATE")
                try:
                    policy, state = int(parts[i + 1]), int(parts[i + 2])
                except (IndexError, ValueError):
                    continue
                if (policy, state) != (driver.policy, driver.motion_state):
                    driver.policy, driver.motion_state = policy, state
                    self.transitions.append((round(t, 3), policy, state))
                    self.log(f"[bench] t={t:6.2f} controller: policy {policy}, "
                             f"state {state} ({STATE_NAMES.get(state, '?')})")

    def close(self) -> None:
        self.sock.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ini", default=None, help="the instance INI (default config/system.ini)")
    ap.add_argument("--map", default=None, help="map/<name> directory (default: the INI's)")
    ap.add_argument("--profile", action="append", default=[], metavar="PARAMS.yaml",
                    help="a params YAML over the defaults (repeatable, in order)")
    ap.add_argument("-p", dest="overrides", action="append", default=[], metavar="key:=value")
    ap.add_argument("--pose", type=float, nargs=4, metavar=("X", "Y", "Z", "YAW_DEG"),
                    help="where to spawn (default: the INI's initial_pose)")
    ap.add_argument("--seconds", type=float, default=10.0, help="wall time to run")
    ap.add_argument("--fake-controller", action="store_true",
                    help="no controller process: the stand-in answers every state with a PD to "
                         "the default pose (locomotion/hardware_link.py StandInController)")
    ap.add_argument("--send", action="append", default=[], metavar="LINE",
                    help='a command line for the controller ("MODE Z", "MODE M", "RESET_SAFETY"), '
                         'repeatable, sent in order --gap seconds apart')
    ap.add_argument("--gap", type=float, default=1.0, help="s between --send lines (default 1)")
    ap.add_argument("--axes", type=float, nargs=3, metavar=("VX", "VY", "WZ"),
                    help="AXES sent at 10 Hz after the last --send line")
    ap.add_argument("--listen", action="store_true",
                    help="read MODE_STATE from the controller's telemetry port "
                         "(--telemetry-port; not while driver_manager runs)")
    ap.add_argument("--telemetry-port", type=int, default=50010,
                    help="where the controller sends its telemetry (its telemetry_target)")
    ap.add_argument("--name", default="bench", help="label of this run")
    ap.add_argument("--out", default=None, help="jsonl (default log/bench/<name>.jsonl)")
    a = ap.parse_args(argv)

    params = dict(PARAM_DEFAULTS)
    params["points_per_scan"] = 100  # no scans here; a small model is a quick start
    for path in a.profile:
        params.update(load_profile(path))
    apply_overrides(params, a.overrides)
    if a.map:
        params["map_dir"] = a.map
    ini = read_ini(a.ini)
    cfg = build_sim_config(ini, params)

    from syncai_sim.envs.pacing import Pacer
    from syncai_sim.envs.simulator import Simulator
    sim = Simulator(cfg, log=print)
    if a.pose:
        pose = (a.pose[0], a.pose[1], a.pose[2], math.radians(a.pose[3]))
    elif params.get("use_ini_initial_pose", True) and ini.initial_pose is not None:
        x, y, z, yaw = ini.initial_pose
        pose = (x, y, z, yaw)
    else:
        s = params["spawn"]
        pose = (float(s[0]), float(s[1]), float(s[2]), float(s[3]))
    sim.spawn(pose[0], pose[1], pose[3], pose[2])

    ctrl = None
    if a.fake_controller and cfg.joint_link == "can":
        from syncai_sim.locomotion.can_link import StandInCanController
        ctrl = StandInCanController(cfg.can_interfaces, cfg.can_motor_ids)
        ctrl.start()
        print(f"[bench] stand-in controller on {', '.join(cfg.can_interfaces)}: the motors "
              f"enabled, a PD to the default pose at 1 kHz")
    elif a.fake_controller:
        from syncai_sim.locomotion.hardware_link import StandInController
        ctrl = StandInController(cfg.link_state_port, cfg.link_command_port, host=cfg.link_state_host)
        ctrl.start()
        print(f"[bench] stand-in controller on {cfg.link_state_host}:{cfg.link_state_port} -> "
              f"{cfg.link_command_port}: a PD to the default pose")
    telemetry = Telemetry(a.telemetry_port) if a.listen else None
    print(f"[bench] {a.name}: {a.seconds:g} s, sends {a.send}, axes {a.axes}, "
          f"commands to {cfg.controller_command_ip}:{cfg.controller_command_port}")

    dt = sim.model.opt.timestep
    sends = list(a.send)
    next_send = 0.0  # sim time
    axes_from = len(sends) * a.gap
    next_axes = axes_from
    base = sim.info.base_body
    vadr = sim.model.jnt_dofadr[sim.model.body_jntadr[base]]
    pitch, roll, fwd, driven = [], [], [], []
    falls0 = sim.falls
    z0, x0, y0 = sim.state.z_ground, sim.state.x, sim.state.y
    t_start = time.perf_counter()
    pacer = Pacer(dt, cfg.max_catch_up_s, t_start)  # envs/pacing.py
    try:
        while time.perf_counter() - t_start < a.seconds:
            now = time.perf_counter()
            steps = pacer.steps(now)
            for _ in range(steps):
                t = sim.sim_time
                if sends and t >= next_send:
                    line = sends.pop(0)
                    sim.driver.send(line)
                    print(f"[bench] t={t:6.2f} -> {line}")
                    next_send = t + a.gap
                if a.axes is not None and t >= next_axes:
                    sim.set_cmd(*a.axes)  # forwarded as AXES to the controller
                    next_axes = t + 0.1
                sim.step()
                if telemetry is not None:
                    telemetry.poll(sim.driver, sim.sim_time)
                r, p = sim.sensors.roll_pitch()
                pitch.append(math.degrees(p))
                roll.append(math.degrees(r))
                v = sim.data.qvel[vadr:vadr + 3]
                h = sim.state.yaw
                fwd.append(float(v[0] * math.cos(h) + v[1] * math.sin(h)))
                driven.append(a.axes is not None and sim.sim_time >= axes_from)
            sleep = pacer.sleep_s(time.perf_counter())
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        if ctrl is not None:
            ctrl.stop()
        if telemetry is not None:
            telemetry.close()
        sim.close()

    wall = time.perf_counter() - t_start
    pitch_a, roll_a, fwd_a, driven_a = (np.array(v) for v in (pitch, roll, fwd, driven))
    last = sim.motors.last
    st = sim.status()
    row = {
        "name": a.name, "duration": round(sim.sim_time, 3), "realtime": round(sim.sim_time / wall, 3),
        "falls": sim.falls - falls0,
        "pitch_mean": float(pitch_a.mean()), "pitch_p5": float(np.percentile(pitch_a, 5)),
        "pitch_p95": float(np.percentile(pitch_a, 95)),
        "pitch_min": float(pitch_a.min()), "pitch_max": float(pitch_a.max()),
        "roll_min": float(roll_a.min()), "roll_max": float(roll_a.max()),
        "speed_driven": float(fwd_a[driven_a].mean()) if driven_a.any() else None,
        "climbed": round(float(sim.state.z_ground - z0), 3),
        "distance": round(float(math.hypot(sim.state.x - x0, sim.state.y - y0)), 3),
        "link": st["link"], "can": st.get("can"), "imu": st.get("imu"), "bms": st.get("bms"),
        "transitions": telemetry.transitions if telemetry is not None else None,
        "telemetry_lines": telemetry.lines if telemetry is not None else None,
        "last_command": None if last is None else {
            "kp": [round(float(v), 2) for v in last.kp], "kd": [round(float(v), 2) for v in last.kd],
            "tau_ff_abs_max": round(float(np.abs(last.tau_ff).max()), 3),
            "dq_des_abs_max": round(float(np.abs(last.dq_des).max()), 3)},
        "motion_state": st["motion_state"], "policy": st["policy"],
        "sends": a.send, "axes": a.axes, "fake_controller": a.fake_controller,
        "start": [round(v, 3) for v in pose[:3]] + [round(math.degrees(pose[3]), 2)],
    }
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "log", "bench",
                                f"{a.name}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[bench] {sim.sim_time:.1f} s sim in {wall:.1f} s wall (x{row['realtime']}): "
          f"falls {row['falls']}, pitch mean {row['pitch_mean']:.1f} "
          f"[{row['pitch_min']:.0f}..{row['pitch_max']:.0f}] deg, "
          f"speed {row['speed_driven']}, link {row['link']}, can {row['can']}, "
          f"imu {row['imu']}, bms {row['bms']}, "
          f"state {STATE_NAMES.get(row['motion_state'], row['motion_state'])}")
    print(f"[bench] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
