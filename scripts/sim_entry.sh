#!/usr/bin/env bash
# The sim container's command (docker-compose.yml): start the MuJoCo sim and
# its browser viewer (viser, http://localhost:8080 on the host) when the
# container comes up, then keep the container alive whatever the sim does.
#
#   SIM_AUTOSTART  1 (default): run `python3 -m syncai_sim $SIM_ARGS`. 0: nothing.
#   SIM_ARGS       the sim's arguments (default `--teleop`: the controller's
#                  hardware, the viewer's panel driving the controller). The
#                  robot's lidar on the network too: `--teleop -p lidar_link:=livox`
#
# Why the container outlives the sim: `docker compose exec sim bash` is how
# everything else here is run (the tests, the benches), and a sim that stops
# on an error should leave its log and the container to look at, not take
# both away. Its output is the container's: `docker compose logs -f sim`.
set -uo pipefail

if [ "${SIM_AUTOSTART:-1}" = "1" ]; then
    # shellcheck disable=SC2086  # SIM_ARGS is a word list on purpose
    echo "[entry] python3 -m syncai_sim ${SIM_ARGS:---teleop} (cwd $(pwd))"
    python3 -m syncai_sim ${SIM_ARGS:---teleop}
    echo "[entry] the sim exited with status $?; the container stays up (docker compose exec sim bash)"
else
    echo "[entry] SIM_AUTOSTART=0: nothing started (docker compose exec sim bash)"
fi
exec sleep infinity
