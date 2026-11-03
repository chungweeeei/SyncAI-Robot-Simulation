"""python3 -m syncai_sim: run the simulator at wall-clock pace.

    python3 -m syncai_sim [--teleop] [--headless] [--system-config INI] [--map DIR]
                          [--seconds S] [--locomotion policy|kinematic|udp]
                          [--visual-mesh voxels|pointcloud|tsdf] [--visual-look ramp|photo]
                          [--profile PARAMS.yaml ...] [-p key:=value ...]

Settings: config.PARAM_DEFAULTS, then config/params/sim_params.yaml (always,
it documents every key), then each --profile in order (e.g.
config/params/controller_profile.yaml), then -p, then the flags. The INI
(config/system.ini relative to the cwd, i.e. the workspace root, or
$SYNCAI_SYSTEM_INI) gives the map and the spawn pose.

The sim talks to the outside over UDP only:

    driver_link udp      the sim answers as the gait controller on the ASCII
                         link (driver/controller_link.py: AXES / MODE in on
                         50051, IMU / joints / MODE_STATE / BMS out to 50010)
    locomotion udp       the joints are a controller process's
                         (locomotion/hardware_link.py: the body's record out to
                         30010, joint commands in on 20001)

The browser viewer (viewer/web_view.py, the `web_view_port` parameter) is
served unless --headless; --teleop puts the driving controls on its panel.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# Before numpy is imported anywhere below: numpy's OpenBLAS starts a worker
# per CPU and the workers busy-wait after each call. The sim's BLAS work is
# small 4x4 / 3x3 algebra many times a second, single-threaded anyway, and the
# calls come faster than the workers' spin timeout, so three of them never
# slept: 350 % CPU for a headless sim, 47 % with one thread (2026-10-24).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

DEFAULT_PROFILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "config", "params", "sim_params.yaml")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python3 -m syncai_sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teleop", action="store_true", help="driving controls on the viewer's panel")
    ap.add_argument("--headless", action="store_true", help="no browser viewer")
    ap.add_argument("--system-config", default=None, help="the instance INI (default config/system.ini)")
    ap.add_argument("--map", default=None, help="map/<name> directory (default: the INI's [map])")
    ap.add_argument("--seconds", type=float, default=None, help="exit after this much wall time")
    ap.add_argument("--locomotion", choices=("policy", "kinematic", "udp"), default=None,
                    help="override the locomotion parameter (default: the profiles', policy)")
    ap.add_argument("--visual-mesh", choices=("voxels", "pointcloud", "tsdf"), default=None,
                    help="the site the viewer starts with (default: voxels); the panel switches")
    ap.add_argument("--visual-look", choices=("ramp", "photo"), default=None,
                    help="the site's look (default: ramp); photo needs config/colour/<map>.npz "
                         "(scripts/colour_map.py)")
    ap.add_argument("--profile", action="append", default=[], metavar="PARAMS.yaml",
                    help="a params YAML applied after sim_params.yaml (repeatable, in order); "
                         "config/params/controller_profile.yaml is the udp locomotion's")
    ap.add_argument("-p", action="append", default=[], metavar="key:=value", dest="overrides",
                    help="one parameter (applied after the profiles)")
    a = ap.parse_args(argv)

    from syncai_sim.config import read_ini
    ini = read_ini(a.system_config)
    print(f"[sim] INI {ini.path}: map {ini.map_dir or '(none)'}, initial_pose {ini.initial_pose}",
          flush=True)
    return run(ini, a)


def _viewer(a, sim, spawn, web_port: int, log):
    """(teleop, viewer factory): the browser viewer unless --headless; its
    panel drives the robot with --teleop, teleports in any case."""
    if a.headless:
        return None, None
    from syncai_sim.viewer.teleop import Teleop
    teleop = Teleop(spawn, log=log, drive=a.teleop)

    def factory():
        try:
            from syncai_sim.viewer.web_view import WebView
        except ImportError as err:
            # An image built before viser was in requirements.txt runs without
            # a picture rather than not at all.
            log(f"[web] no browser viewer: {err} (rebuild the image)")
            return None
        return WebView(sim, teleop, port=web_port, log=log)
    return teleop, factory


def settings(a) -> dict:
    """PARAM_DEFAULTS, sim_params.yaml, the profiles, -p, then the flags."""
    from syncai_sim.config import PARAM_DEFAULTS, apply_overrides, load_profile
    params = dict(PARAM_DEFAULTS)
    for path in [DEFAULT_PROFILE, *a.profile]:
        params.update(load_profile(path))
    apply_overrides(params, a.overrides)
    if a.map:
        params["map_dir"] = a.map
    if a.locomotion:
        params["locomotion"] = a.locomotion
    if a.visual_mesh:
        params["visual_mesh"] = a.visual_mesh
    if a.visual_look:
        params["visual_look"] = a.visual_look
    return params


def _spawn_pose(ini, params) -> tuple[float, float, float, float]:
    if params.get("use_ini_initial_pose", True) and ini.initial_pose is not None:
        return tuple(ini.initial_pose)
    s = params.get("spawn", [0.0, 0.0, 0.0, 0.0])
    return float(s[0]), float(s[1]), float(s[2]), float(s[3])


def run(ini, a) -> int:
    """The physics loop at wall-clock pace: the UDP links polled and ticked,
    the lidar scanned at its rate, the viewer synced at its own."""
    import time
    from syncai_sim.config import build_sim_config
    from syncai_sim.envs.simulator import Simulator
    params = settings(a)
    if params["driver_link"] not in ("fake", "udp"):
        raise SystemExit(f"driver_link {params['driver_link']!r}: fake | udp")
    if params["driver_link"] == "udp" and params["locomotion"] == "udp":
        # Both are the controller's command port: the sim cannot be the
        # controller and the controller's hardware at once.
        raise SystemExit("driver_link udp plays the gait controller; locomotion udp hands the "
                         "joints to one: pick one")
    cfg = build_sim_config(ini, params)
    sim = Simulator(cfg)
    spawn = _spawn_pose(ini, params)
    sim.spawn(*spawn[:2], spawn[3], spawn[2])
    link = None
    if params["driver_link"] == "udp":
        from syncai_sim.driver.controller_link import ControllerLink, LinkConfig
        link = ControllerLink(sim, sim.driver, LinkConfig(
            command_port=params["controller_command_port"],
            telemetry_port=params["controller_telemetry_port"],
            telemetry_hz=params["controller_telemetry_hz"],
            battery_soc=params["battery_soc"],
            battery_drain_per_hour=params["battery_drain_per_hour"]))
    teleop, factory = _viewer(a, sim, spawn, params["web_view_port"], print)
    viewer = factory() if factory else None
    dt = sim.model.opt.timestep
    t_start = time.perf_counter()
    next_t = t_start
    next_scan = 0.0
    next_view = 0.0
    next_print = 5.0
    scan_ms = 0.0
    n = 0
    # A late loop catches up in a burst of at most this many steps (the udp
    # profile keeps it to 8 ms: locomotion/hardware_link.py says why).
    max_catch_up = max(1, int(round(cfg.max_catch_up_s / dt)))
    try:
        while True:
            now = time.perf_counter()
            if a.seconds is not None and now - t_start > a.seconds:
                break
            if viewer is not None and not viewer.is_running():
                break
            steps = max(1, min(int((now - next_t) / dt) + 1, max_catch_up))
            if link is not None:
                link.poll()
            if teleop:
                teleop.apply(sim)
            for _ in range(steps):
                sim.step()
            if link is not None:
                link.tick(sim.sim_time)
            next_t += steps * dt
            if sim.sim_time >= next_scan:
                next_scan = sim.sim_time + 1.0 / cfg.lidar.scan_rate_hz
                t0 = time.perf_counter()
                xyzi = sim.scan()
                n = xyzi.shape[0]
                scan_ms = (time.perf_counter() - t0) * 1000
                if viewer is not None:
                    viewer.show_scan(sim)
            if viewer is not None and now >= next_view:
                next_view = now + 1 / viewer.rate_hz
                viewer.sync(sim)
            if sim.sim_time >= next_print:  # steps come in catch-up batches: never ==
                next_print += 5.0
                st = sim.status()
                hw = (f"  link cmds={st['link']['commands']} states={st['link']['states']} "
                      f"trips={st['link']['watchdog_trips']}" if "link" in st else "")
                rt = sim.sim_time / (now - t_start)
                print(f"[sim] t={sim.sim_time:6.1f}  x{rt:.2f}  scan {scan_ms:.1f} ms ({n} pts)  "
                      f"{st['last_event']}  xy=({st['x']:.2f},{st['y']:.2f}) z={st['z_ground']:.2f} "
                      f"pitch={math.degrees(st['pitch']):.1f}deg  coll={st['collisions']} "
                      f"falls={st['falls']}{hw}", flush=True)
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        if link is not None:
            link.close()
        sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
