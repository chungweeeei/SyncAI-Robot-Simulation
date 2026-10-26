#!/usr/bin/env python3
"""The walking body against a scripted command, without ROS: the A/B tool for
the body's physics knobs (motor_model, trunk_inertial, foot_solref,
foot_friction, actuator, integrator, cmd_delay_ms, torque_bw_hz, physics_hz;
config/params/sim_params.yaml).

    # dp's first fitted flight, HimLoco, 0.5 m/s for 12 s, ten runs with a
    # few cm of start jitter, the sim as it is:
    python3 scripts/gait_bench.py --flight 0 --policy 1 --cmd "0.5 0 0 12" --runs 10 --name base
    # the same with the vendor's motor model and trunk:
    python3 scripts/gait_bench.py --flight 0 --policy 1 --cmd "0.5 0 0 12" --runs 10 \\
        -p motor_model:=j60 -p trunk_inertial:=vendor --name j60-vendor

Each run spawns the robot (`--pose`, or `--before` metres before flight
`--flight`'s first riser, facing up it), lets it stand `--settle` seconds,
then drives every `--cmd "vx vy wz seconds"` segment in turn, sampling the
body every physics step; a fall (the controller's posture check) ends the
run. What it reports per run and on average is the list docs/findings.md
finding 5 asks the robot for, so the two can be read side by side: whether
it fell and when, the pitch (mean, 5th / 95th percentile, extremes), the
roll extremes, stalls (the command above 0.3 m/s while the body makes less
than 0.05 m/s for over 0.3 s), the speed along the command, the heading
drift and the height climbed. One JSON line per run goes to --out (default
log/gait/<name>.jsonl under this repository), the summary to stdout.

Runs are deterministic: the sim has no noise of its own in the gait, so
the spread comes from the start jitter (--jitter-xy, --jitter-yaw) — a
smaller spread than the navigation stack's timing gives, but the same
physics. The stack's own 1F <-> 2F batches (nav_test.py through the
session) are the final word on a knob; this is what to run first.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from syncai_sim.config import PARAM_DEFAULTS, build_sim_config, read_ini  # noqa: E402
from syncai_sim.fake_driver import LOCOMOTION  # noqa: E402

STALL_SPEED = 0.05  # m/s: below this the body is standing still
STALL_CMD = 0.3  # m/s: a command this large should move it
STALL_S = 0.3  # s: standing still this long under such a command is a stall


def parse_override(text: str) -> tuple[str, object]:
    """key:=value as `ros2 run ... -p` takes it: JSON where it parses, else a string."""
    if ":=" not in text:
        raise argparse.ArgumentTypeError(f"-p wants key:=value, got {text!r}")
    key, raw = text.split(":=", 1)
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def parse_cmd(text: str) -> tuple[float, float, float, float]:
    parts = text.split()
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"--cmd wants 'vx vy wz seconds', got {text!r}")
    return tuple(float(v) for v in parts)  # type: ignore[return-value]


def flight_pose(sim, k: int, before: float, descend: bool) -> tuple[float, float, float, float]:
    """(x, y, z, yaw): `before` metres before fitted flight k's first riser on
    its centre line, facing up the flight, on its lower floor — or, `descend`,
    that far past its last riser on the upper floor, facing the same way (the
    controllers walk stairs down backwards: a negative vx from there)."""
    if sim.surface is None or not sim.surface.flights:
        raise SystemExit("this map has no fitted flights (surface_heights.py needs map.pcd)")
    if not 0 <= k < len(sim.surface.flights):
        raise SystemExit(f"--flight {k}: the map has flights 0..{len(sim.surface.flights) - 1}")
    f = sim.surface.flights[k]
    run = (f.n - 1) * f.tread
    landing_lo = f.u0 - f.u_lo if math.isfinite(f.u_lo) else float("nan")
    landing_hi = f.u_hi - f.u1 if math.isfinite(f.u_hi) else float("nan")
    print(f"[gait] flight {k} ({f.name}): {f.n} risers of {f.riser:.3f} m on {f.tread:.3f} m "
          f"treads, {run:.2f} m long, floors {f.bottom:.2f} -> {f.top:.2f}; level floor "
          f"{landing_lo:.2f} m before and {landing_hi:.2f} m after it")
    if descend:
        u = f.u1 + f.tread + before
        if math.isfinite(f.u_hi) and u > f.u_hi:
            print(f"[gait] --before {before} puts the start past the upper floor's "
                  f"{landing_hi:.2f} m; it is on whatever is beyond")
    else:
        u = f.u0 - before
    along = f.sign * u
    x, y = (along, f.across) if f.axis == 0 else (f.across, along)
    yaw = math.atan2(f.sign if f.axis == 1 else 0.0, f.sign if f.axis == 0 else 0.0)
    return x, y, f.top if descend else f.bottom, yaw


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def one_run(sim, pose, policy: int, cmds, settle: float, rng, jitter_xy: float,
            jitter_yaw: float) -> dict:
    x, y, z, yaw = pose
    x += rng.uniform(-jitter_xy, jitter_xy)
    y += rng.uniform(-jitter_xy, jitter_xy)
    yaw += math.radians(rng.uniform(-jitter_yaw, jitter_yaw))
    sim.spawn(x, y, yaw, z)
    sim.driver.switch_delay = 0.0
    sim.driver.motion_state = LOCOMOTION
    sim.driver.set_policy_mode(policy)
    sim.driver.update()
    sim.driver.policy = policy
    falls0 = sim.falls
    dt = sim.model.opt.timestep
    base = sim.info.base_body
    vadr = sim.model.jnt_dofadr[sim.model.body_jntadr[base]]
    z0 = sim.state.z_ground
    yaw0 = sim.state.yaw
    pitch, roll, fwd, cmd_vx, turning, policy_ms = [], [], [], [], [], []
    t = 0.0
    t_fall = None
    still_since = None
    stalls = 0
    segments = [(0.0, 0.0, 0.0, settle)] + list(cmds)
    for vx, vy, wz, seconds in segments:
        for _ in range(int(round(seconds / dt))):
            sim.set_cmd(vx, vy, wz)
            sim.step()
            t += dt
            r, p = sim.legs.roll_pitch()
            pitch.append(math.degrees(p))
            roll.append(math.degrees(r))
            v = sim.data.qvel[vadr:vadr + 3]
            h = sim.state.yaw
            speed = float(v[0] * math.cos(h) + v[1] * math.sin(h))
            fwd.append(speed)
            cmd_vx.append(vx)
            turning.append(abs(wz) > 1e-6)
            policy_ms.append(sim.legs.policy_ms)
            if abs(vx) > STALL_CMD and abs(speed) < STALL_SPEED:
                if still_since is None:
                    still_since = t
                elif t - still_since >= STALL_S:
                    stalls += 1
                    still_since = None
            else:
                still_since = None
            if sim.falls > falls0:
                t_fall = t
                break
        if t_fall is not None:
            break
    pitch_a, roll_a, fwd_a = np.array(pitch), np.array(roll), np.array(fwd)
    driven = np.abs(np.array(cmd_vx)) > STALL_CMD
    straight = driven & ~np.array(turning)
    heading = np.array([math.degrees(wrap(sim.state.yaw - yaw0))])  # at the end
    return {
        "fell": t_fall is not None, "t_fall": t_fall, "duration": round(t, 3),
        "pitch_mean": float(pitch_a.mean()), "pitch_p5": float(np.percentile(pitch_a, 5)),
        "pitch_p95": float(np.percentile(pitch_a, 95)),
        "pitch_min": float(pitch_a.min()), "pitch_max": float(pitch_a.max()),
        "roll_min": float(roll_a.min()), "roll_max": float(roll_a.max()),
        "speed_driven": float(fwd_a[driven].mean()) if driven.any() else None,
        "stalls": stalls, "heading_drift": round(float(heading[0]), 2) if straight.any() else None,
        "climbed": round(float(sim.state.z_ground - z0), 3),
        "distance": round(float(math.hypot(sim.state.x - x, sim.state.y - y)), 3),
        "policy_ms": float(np.mean(policy_ms)),
        "start": [round(x, 3), round(y, 3), round(z, 3), round(math.degrees(yaw), 2)],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ini", default=None, help="the instance INI (default config/system.ini)")
    ap.add_argument("--map", default=None, help="map/<name> directory (default: the INI's)")
    ap.add_argument("--pose", type=float, nargs=4, metavar=("X", "Y", "Z", "YAW_DEG"),
                    help="where to start, instead of --flight")
    ap.add_argument("--flight", type=int, default=0, help="fitted flight to climb (default 0)")
    ap.add_argument("--before", type=float, default=1.0, help="m before its first riser")
    ap.add_argument("--descend", action="store_true",
                    help="start --before m past the last riser instead, on the upper floor, "
                         "facing up the flight: walk it down backwards with a negative vx")
    ap.add_argument("--policy", type=int, default=1, help="0 PPO, 1 HimLoco (default)")
    ap.add_argument("--cmd", type=parse_cmd, action="append", required=True,
                    help='"vx vy wz seconds", repeatable, driven in order')
    ap.add_argument("--settle", type=float, default=1.0, help="s standing (policy, zero command) first")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--jitter-xy", type=float, default=0.03, help="m, uniform, per run")
    ap.add_argument("--jitter-yaw", type=float, default=2.0, help="deg, uniform, per run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-p", dest="params", type=parse_override, action="append", default=[],
                    help="key:=value parameter override (sim_params.yaml names), repeatable")
    ap.add_argument("--name", default="base", help="label of this configuration")
    ap.add_argument("--out", default=None, help="jsonl (default log/gait/<name>.jsonl)")
    a = ap.parse_args(argv)

    params = dict(PARAM_DEFAULTS)
    params["points_per_scan"] = 100  # no scans here; a small model is a quick start
    for key, value in a.params:
        if key not in params:
            raise SystemExit(f"-p {key}: not a parameter (see config/params/sim_params.yaml)")
        params[key] = value
    if a.map:
        params["map_dir"] = a.map
    ini = read_ini(a.ini)
    cfg = build_sim_config(ini, params)
    if cfg.locomotion != "policy":
        raise SystemExit("gait_bench is for locomotion policy")

    from syncai_sim.simulator import Simulator
    sim = Simulator(cfg, log=lambda *x: None)
    if a.pose:
        pose = (a.pose[0], a.pose[1], a.pose[2], math.radians(a.pose[3]))
    else:
        pose = flight_pose(sim, a.flight, a.before, a.descend)
    knobs = {k: getattr(cfg, k) for k in ("motor_model", "trunk_inertial", "foot_solref",
                                           "foot_friction", "actuator", "integrator",
                                           "cmd_delay_ms", "torque_bw_hz", "physics_hz")}
    print(f"[gait] {a.name}: {knobs}")
    print(f"[gait] start ({pose[0]:.2f}, {pose[1]:.2f}, z {pose[2]:.2f}, yaw "
          f"{math.degrees(pose[3]):.0f} deg), policy {a.policy}, cmds {a.cmd}, {a.runs} runs")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "log", "gait",
                                f"{a.name}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    t0 = time.perf_counter()
    with open(out, "a") as f:
        for i in range(a.runs):
            r = one_run(sim, pose, a.policy, a.cmd, a.settle, rng, a.jitter_xy, a.jitter_yaw)
            r.update({"name": a.name, "run": i, "params": dict(a.params), "policy": a.policy,
                      "cmds": a.cmd, "when": time.strftime("%Y-%m-%d %H:%M:%S")})
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            fell = f"FELL at {r['t_fall']:.1f} s" if r["fell"] else "ok"
            print(f"[gait] run {i + 1}/{a.runs}: {fell}; pitch mean {r['pitch_mean']:.1f} "
                  f"p5/p95 {r['pitch_p5']:.1f}/{r['pitch_p95']:.1f} min/max {r['pitch_min']:.1f}/"
                  f"{r['pitch_max']:.1f} deg; roll {r['roll_min']:.1f}/{r['roll_max']:.1f}; "
                  f"stalls {r['stalls']}; speed "
                  f"{r['speed_driven'] if r['speed_driven'] is None else round(r['speed_driven'], 2)}; "
                  f"climbed {r['climbed']} m; heading {r['heading_drift']}")
    n = len(rows)
    if n == 0:
        return 0
    falls = sum(r["fell"] for r in rows)

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")
    print(f"[gait] {a.name}: {falls}/{n} fell; pitch mean {mean('pitch_mean'):.1f}, p5 "
          f"{mean('pitch_p5'):.1f}, p95 {mean('pitch_p95'):.1f}, extremes {mean('pitch_min'):.1f}/"
          f"{mean('pitch_max'):.1f} deg; roll {mean('roll_min'):.1f}/{mean('roll_max'):.1f}; "
          f"stalls {mean('stalls'):.1f}/run; speed {mean('speed_driven'):.2f} m/s; climbed "
          f"{mean('climbed'):.2f} m; heading {mean('heading_drift'):.1f} deg; policy "
          f"{mean('policy_ms'):.2f} ms; {time.perf_counter() - t0:.0f} s wall -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
