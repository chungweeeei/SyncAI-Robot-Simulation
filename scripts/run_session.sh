#!/usr/bin/env bash
# Run a session spec (config/sessions/*.yaml) as background processes, no byobu.
#
#   scripts/run_session.sh [spec]          default config/sessions/start_sim.yaml
#   scripts/run_session.sh --stop          stop a session this script started
#
# Why not NodeManager: its sessions are the robot's operating modes; see the
# spec's header. This is the workspace's .github/scripts/smoke_test.sh pane loop
# (run_spec), kept as a tool: one `setsid bash -c "sleep N && exec cmd"` per pane
# so each launch and every node under it is one process group, stdout+stderr to
# log/stack/<robot_id>/<log>/current under the WORKSPACE (the robot writes its
# multilogs there too; `tail -f` works the same), and a stop that sends SIGINT to
# every group, waits, then SIGKILLs — what byobu's kill-session amounts to.
#
# Runs inside the sim container, from the workspace root (the launch files read
# the relative config/system.ini). The ROS environment must be sourced first:
#   source /opt/ros/humble/setup.bash && source /opt/ros2_rust_underlay/install/setup.bash \
#     && source install/setup.bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${ROBOT_WS_DIR:-/home/syncrobotic/robot_ws}"
PIDFILE="${WS}/log/stack/.sim_session.pids"

die() { echo "run_session: $*" >&2; exit 1; }

stop_session() {
    [ -f "${PIDFILE}" ] || die "no session pidfile at ${PIDFILE}"
    mapfile -t pids < "${PIDFILE}"
    echo "stopping ${#pids[@]} pane(s): SIGINT"
    for p in "${pids[@]}"; do kill -INT -- "-${p}" 2>/dev/null || true; done
    sleep 5
    for p in "${pids[@]}"; do kill -KILL -- "-${p}" 2>/dev/null || true; done
    rm -f "${PIDFILE}"
    ros2 daemon stop >/dev/null 2>&1 || true
    echo "stopped"
}

if [ "${1:-}" = "--stop" ]; then stop_session; exit 0; fi
spec="${1:-${SIM_DIR}/config/sessions/start_sim.yaml}"
[ -f "${spec}" ] || die "no such spec: ${spec}"
[ -d "${WS}/install" ] || die "${WS}/install missing: build the workspace first"
command -v ros2 >/dev/null || die "ros2 not on PATH: source the three setup.bash files first"
# The pid file lives in the mounted workspace, so it outlives the container: a
# `docker restart` kills every pane but leaves the file, and the next start
# refused "already running" with nothing running (2026-10-24). A file whose
# process groups are all gone is stale and is dropped.
if [ -f "${PIDFILE}" ]; then
    alive=0
    while read -r p; do [ -n "${p}" ] && kill -0 -- "-${p}" 2>/dev/null && alive=1; done < "${PIDFILE}"
    [ "${alive}" = 1 ] && die "a session is already running (${PIDFILE}); --stop it first"
    echo "dropping a stale ${PIDFILE} (none of its panes is alive)"
    rm -f "${PIDFILE}"
fi

cd "${WS}"
ROBOT_ID="$(python3 - <<'PY'
import configparser
c = configparser.ConfigParser(); c.read("config/system.ini")
print(c.get("system", "robot_id", fallback="default_robot").strip())
PY
)"
export ROBOT_ID
echo "workspace ${WS}, robot_id ${ROBOT_ID}, spec ${spec}"

# sleep <TAB> log <TAB> cmd, one line per pane (panes with enter: false are skipped).
panes="$(python3 - "${spec}" <<'PY'
import sys, yaml
spec = yaml.safe_load(open(sys.argv[1]))
for w in spec["windows"]:
    for p in w["panes"]:
        if p.get("enter", True) is False:
            continue
        log = p.get("log") or w["name"]
        print(f"{p.get('sleep', 0)}\t{log}\t{' '.join(p['cmd'].split())}")
PY
)" || die "cannot parse ${spec}"

mkdir -p "$(dirname "${PIDFILE}")"
: > "${PIDFILE}"
while IFS=$'\t' read -r sleep_s log cmd; do
    dir="${WS}/log/stack/${ROBOT_ID}/${log}"
    mkdir -p "${dir}"
    echo "  +${sleep_s}s  ${log}: ${cmd}"
    # Own session per pane, so the stop takes the launch and every node under it
    # down as one process group. The command is expanded by this bash (the
    # $(cat ...) of the URDF, ${ROBOT_ID}) exactly as a byobu pane would.
    setsid bash -c "sleep ${sleep_s} && exec ${cmd}" > "${dir}/current" 2>&1 &
    echo "$!" >> "${PIDFILE}"
done <<< "${panes}"
echo "started; logs under ${WS}/log/stack/${ROBOT_ID}/  (stop with: $0 --stop)"
