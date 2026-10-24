"""python3 -m syncai_sim: run the simulator, as a ROS 2 node by default.

    python3 -m syncai_sim [--teleop] [--headless] [--show-scan] [--no-ros]
                          [--system-config INI] [--map DIR] [--seconds S]
                          [--locomotion policy|kinematic] [--visual-mesh pointcloud|voxels]
                          [--ros-args --params-file ... -p key:=value ...]

Everything after `--ros-args` goes to rclpy (the params file is
config/params/sim_params.yaml; see it for every setting). The INI
(config/system.ini relative to the cwd, i.e. the workspace root) gives the
robot_id, the map and the spawn pose, as it does for every launch file.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# Before numpy is imported anywhere below: numpy's OpenBLAS starts a worker
# per CPU and the workers busy-wait after each call. The node's only BLAS
# work is 4x4 TF algebra at 50 Hz (publish_pose), which is single-threaded
# anyway, and the calls come faster than the workers' spin timeout, so three
# of them never slept: the ROS node alone, headless, was 350 % CPU; with one
# thread it is 47 % (2026-10-24).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--ros-args" in argv:
        i = argv.index("--ros-args")
        own, ros_args = argv[:i], argv[i:]
    else:
        own, ros_args = argv, []
    ap = argparse.ArgumentParser(prog="python3 -m syncai_sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teleop", action="store_true", help="drive with the viewer's keyboard")
    ap.add_argument("--headless", action="store_true", help="no viewer window")
    ap.add_argument("--show-scan", action="store_true", help="draw a thinned lidar scan in the viewer")
    ap.add_argument("--no-ros", action="store_true", help="run without ROS (viewer / teleop only)")
    ap.add_argument("--system-config", default=None, help="the instance INI (default config/system.ini)")
    ap.add_argument("--map", default=None, help="map/<name> directory (default: the INI's [map])")
    ap.add_argument("--seconds", type=float, default=None, help="exit after this much wall time")
    ap.add_argument("--locomotion", choices=("policy", "kinematic"), default=None,
                    help="override the locomotion parameter (default: the params file's, policy)")
    ap.add_argument("--visual-mesh", choices=("pointcloud", "voxels"), default=None,
                    help="what the viewer draws of the site (default: the params file's, pointcloud)")
    a = ap.parse_args(own)

    from .config import read_ini
    ini = read_ini(a.system_config)
    print(f"[sim] INI {ini.path}: robot_id {ini.robot_id}, map {ini.map_dir or '(none)'}, "
          f"initial_pose {ini.initial_pose}", flush=True)

    if a.no_ros:
        return _run_without_ros(ini, a)

    import rclpy
    from .ros_node import ParamNode, SimNode, Spinner, build_sim_config, make_frames, run_loop
    from .simulator import Simulator

    rclpy.init(args=ros_args)
    pn = ParamNode(ini.robot_id)
    params = pn.values()
    pn.destroy_node()
    if a.map:
        params["map_dir"] = a.map
    if a.locomotion:
        params["locomotion"] = a.locomotion
    if a.visual_mesh:
        params["visual_mesh"] = a.visual_mesh
    cfg = build_sim_config(ini, params)
    log = print
    sim = Simulator(cfg, log=log)
    spawn = _spawn_pose(ini, params)
    sim.spawn(*spawn[:2], spawn[3], spawn[2])

    node = SimNode(ini, sim, params, make_frames(ini.robot_id))
    if params["publish_map_cloud"]:
        from .map_cloud import load_or_build
        clouds = load_or_build(cfg.map_dir, cfg.cache_dir, sim.world, log=log)
        if clouds:
            node.publish_map_clouds(clouds)
    spinner = Spinner(node)
    spinner.start()
    teleop = None
    viewer_factory = None
    if not a.headless:
        from .viewer import Teleop, Viewer
        if a.teleop:
            from .viewer import HELP
            teleop = Teleop(spawn, log=log)
            log(HELP)
        viewer_factory = lambda: Viewer(sim, teleop)  # noqa: E731
    try:
        run_loop(node, viewer_factory, teleop, a.show_scan, a.seconds, log)
    except KeyboardInterrupt:
        pass
    finally:
        # rclpy raises from a spinning executor on shutdown; a Ctrl-C is a
        # normal end here, so swallow what the teardown itself throws.
        for fn in (spinner.stop, node.destroy_node, rclpy.shutdown):
            try:
                fn()
            except Exception:
                pass
    return 0


def _spawn_pose(ini, params) -> tuple[float, float, float, float]:
    if params.get("use_ini_initial_pose", True) and ini.initial_pose is not None:
        return tuple(ini.initial_pose)
    s = params.get("spawn", [0.0, 0.0, 0.0, 0.0])
    return float(s[0]), float(s[1]), float(s[2]), float(s[3])


def _run_without_ros(ini, a) -> int:
    """Viewer + teleop + the same physics loop, no publications."""
    import time
    from .config import PARAM_DEFAULTS, build_sim_config
    from .simulator import Simulator
    params = dict(PARAM_DEFAULTS)
    if a.map:
        params["map_dir"] = a.map
    if a.locomotion:
        params["locomotion"] = a.locomotion
    if a.visual_mesh:
        params["visual_mesh"] = a.visual_mesh
    cfg = build_sim_config(ini, params)
    sim = Simulator(cfg)
    spawn = _spawn_pose(ini, params)
    sim.spawn(*spawn[:2], spawn[3], spawn[2])
    viewer = None
    teleop = None
    if not a.headless:
        from .viewer import HELP, Teleop, Viewer
        teleop = Teleop(spawn) if a.teleop else None
        if teleop:
            print(HELP)
        viewer = Viewer(sim, teleop)
    dt = sim.model.opt.timestep
    t_start = time.perf_counter()
    next_t = t_start
    next_scan = 0.0
    next_view = 0.0
    next_print = 5.0
    scan_ms = 0.0
    n = 0
    try:
        while True:
            now = time.perf_counter()
            if a.seconds is not None and now - t_start > a.seconds:
                break
            if viewer is not None and not viewer.is_running():
                break
            steps = max(1, min(int((now - next_t) / dt) + 1, int(round(0.1 / dt))))
            if teleop:
                teleop.apply(sim)
            for _ in range(steps):
                sim.step()
            next_t += steps * dt
            if sim.sim_time >= next_scan:
                next_scan = sim.sim_time + 1.0 / cfg.lidar.scan_rate_hz
                t0 = time.perf_counter()
                xyzi = sim.scan()
                n = xyzi.shape[0]
                scan_ms = (time.perf_counter() - t0) * 1000
                if viewer is not None and a.show_scan:
                    viewer.show_scan(sim)
            if viewer is not None and now >= next_view:
                next_view = now + 1 / viewer.rate_hz
                viewer.sync(sim)
            if sim.sim_time >= next_print:  # steps come in catch-up batches: never ==
                next_print += 5.0
                st = sim.status()
                print(f"[sim] t={sim.sim_time:6.1f}  scan {scan_ms:.1f} ms ({n} pts)  "
                      f"{st['last_event']}  xy=({st['x']:.2f},{st['y']:.2f}) z={st['z_ground']:.2f} "
                      f"pitch={math.degrees(st['pitch']):.1f}deg  coll={st['collisions']}", flush=True)
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
