#!/usr/bin/env bash
# Start the container's display: Xvnc on :1 (TigerVNC, Mesa llvmpipe behind it)
# and noVNC on 6080 so a browser on the host shows it.
#
#   scripts/vnc.sh            start (idempotent)
#   scripts/vnc.sh --stop
#
# Then open http://localhost:6080/vnc.html?autoconnect=1&resize=remote
# (docker-compose.yml publishes 6080), or a VNC client on localhost:5901.
# resize=remote has noVNC ask Xvnc to resize the screen to the browser window
# (Xvnc accepts by default, AcceptSetDesktopSize), and the viewer keeps its
# window the size of the screen (viewer.py, _FitToDisplay), so the picture
# fills whatever the browser is. GEOMETRY is only the size until a client
# asks otherwise, and what a native VNC client gets. No password: the port is published to the
# host's loopback by Docker and the container holds nothing but a simulation.
#
# Why a VNC display and not X11 to XQuartz: XQuartz only offers indirect GLX
# (OpenGL 1.4), which MuJoCo's renderer cannot use; here GLFW gets a direct
# llvmpipe context on a local X server and the whole viewer UI works.
set -euo pipefail
DISPLAY_NUM="${SIM_DISPLAY_NUM:-1}"
GEOMETRY="${SIM_VNC_GEOMETRY:-1600x1000}"
LOG=/tmp/sim-vnc
mkdir -p "${LOG}"

if [ "${1:-}" = "--stop" ]; then
    pkill -f "websockify.*6080" 2>/dev/null || true
    pkill -f "Xtigervnc :${DISPLAY_NUM}" 2>/dev/null || pkill -f "Xvnc :${DISPLAY_NUM}" 2>/dev/null || true
    echo "vnc stopped"; exit 0
fi

if ! pgrep -f "X(tiger)?vnc :${DISPLAY_NUM}" >/dev/null; then
    # -SecurityTypes None: see header. -localhost no: websockify connects over
    # the container's loopback, but Docker's published 5901 arrives on eth0.
    Xvnc ":${DISPLAY_NUM}" -geometry "${GEOMETRY}" -depth 24 -SecurityTypes None \
        -localhost no -AlwaysShared -rfbport 5901 > "${LOG}/xvnc.log" 2>&1 &
    for _ in $(seq 1 50); do [ -e "/tmp/.X11-unix/X${DISPLAY_NUM}" ] && break; sleep 0.1; done
fi
if ! pgrep -f "websockify.*6080" >/dev/null; then
    websockify --web /usr/share/novnc 6080 localhost:5901 > "${LOG}/websockify.log" 2>&1 &
fi
sleep 0.5
echo "display :${DISPLAY_NUM} up; browser: http://localhost:6080/vnc.html?autoconnect=1&resize=remote  (logs in ${LOG})"
