#!/usr/bin/env bash
# The sim container's command (docker-compose.yml): start the MuJoCo sim and
# its browser viewer (viser, http://localhost:8080 on the host) when the
# container comes up, then keep the container alive whatever the sim does.
#
# No ROS here: the sim runs `--no-ros` (MuJoCo, the lidar on the voxels, the
# viewer; the hardware link in the udp profile), so nothing is sourced and no
# DDS participant is created. The nav session, which does need ROS, is
# scripts/run_session.sh run by hand inside the container.
#
#   SIM_AUTOSTART  1 (default): run `python3 -m syncai_sim --no-ros $SIM_ARGS`.
#                  0: start nothing — for scripts/run_session.sh, whose sim pane
#                  would otherwise be a second sim on the same viewer port.
#   SIM_ARGS       extra arguments (default `--teleop`: the sim walking on its
#                  own policies, driven from the viewer's panel). The
#                  controller's hardware instead:
#                  `--teleop --profile /home/syncrobotic/sim/config/params/controller_profile.yaml`
#
# Why the container outlives the sim: `docker compose exec sim bash` is how
# everything else here is run (the tests, the benches, the session), and a
# sim that stops on an error should leave its log and the container to look
# at, not take both away. Its output is the container's: `docker compose logs -f sim`.
set -uo pipefail

if [ "${SIM_AUTOSTART:-1}" = "1" ]; then
    # shellcheck disable=SC2086  # SIM_ARGS is a word list on purpose
    echo "[entry] python3 -m syncai_sim --no-ros ${SIM_ARGS:---teleop} (cwd $(pwd))"
    python3 -m syncai_sim --no-ros ${SIM_ARGS:---teleop}
    echo "[entry] the sim exited with status $?; the container stays up (docker compose exec sim bash)"
else
    echo "[entry] SIM_AUTOSTART=0: nothing started (docker compose exec sim bash)"
fi
exec sleep infinity
