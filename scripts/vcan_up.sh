#!/usr/bin/env bash
# The CAN interfaces of `joint_link: can` (syncai_sim/locomotion/can_link.py):
# four virtual ones with the robot's names, on the host the sim and the
# controller share.
#
#   scripts/vcan_up.sh                 # can1 can2 can3 can4, up
#   scripts/vcan_up.sh down            # remove them
#   VCAN_IFS="vcan0 vcan1 vcan2 vcan3" scripts/vcan_up.sh
#
# Run on the host, not in the sim container (which has neither the privilege
# nor `ip`). On a Mac the host is Docker's VM (OrbStack: its kernel has vcan
# built in, checked 2026-11-13 on 7.2.12-orbstack; no vxcan, so the sim and
# the controller must both be on the host's network to see the same
# interfaces: SIM_NETWORK_MODE=host), reached through a privileged container.
# On Linux with `ip` and root it is done directly. The interfaces are gone
# after a restart of the VM / the machine: run this again.
#
# A vcan interface has no bitrate, arbitration or error frames: a frame
# arrives at once, always (can_link.py's can_delay_ms / can_drop_rate put a
# bus's faults back). `ip link set can2 down` takes one leg's bus away.
set -euo pipefail

IFS_LIST="${VCAN_IFS:-can1 can2 can3 can4}"
ACTION="${1:-up}"

if [ "$ACTION" = "down" ]; then
    CMDS="for i in $IFS_LIST; do ip link del \"\$i\" 2>/dev/null && echo \"\$i removed\" || echo \"\$i: not there\"; done"
elif [ "$ACTION" = "up" ]; then
    CMDS="for i in $IFS_LIST; do
        ip link show \"\$i\" >/dev/null 2>&1 || ip link add dev \"\$i\" type vcan
        ip link set up \"\$i\"
        ip -brief link show \"\$i\"
    done"
else
    echo "usage: $0 [up|down]" >&2
    exit 2
fi

if [ "$(uname -s)" = "Linux" ] && command -v ip >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
    modprobe vcan 2>/dev/null || true  # built in on some kernels
    sh -c "$CMDS"
else
    docker run --rm --privileged --net host alpine sh -c "apk add -q iproute2 >/dev/null 2>&1; $CMDS"
fi
