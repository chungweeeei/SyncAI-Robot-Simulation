# =============================================================================
# The simulation container: the robot's own dev image plus MuJoCo.
#
# FROM the workspace's `dev` stage (built there as `syncai-robot-base`, see its
# docker-compose.build.yml) rather than a ros:humble image, because the nav
# stack this container runs is the workspace's build/ + install/ bind-mounted
# in, and a .so built against that image's exact ROS / GTSAM / xtensor set only
# loads in that image. Everything the sim adds on top is Python.
#
# The GUI: MuJoCo's viewer is a GLFW/OpenGL window, and the container has no
# GPU and no display, so it gets an X server of its own — TigerVNC's Xvnc —
# with Mesa's llvmpipe behind it (the base image already carries Mesa and
# libglfw3), and noVNC serves that display to a browser on port 6080. This is
# what works everywhere the same: forwarding X11 to XQuartz on a Mac was tried
# first and fails at GLX context creation (XQuartz offers only indirect GLX,
# OpenGL 1.4, below what mjr needs), and a Linux host would need its socket
# mounted. `scripts/vnc.sh` starts the pair; `python3 -m syncai_sim` renders on
# DISPLAY=:1. mesa-utils is only glxinfo, for when rendering misbehaves.
#
# numpy: the base image has ROS Humble's python3-numpy 1.21 and cv_bridge & co.
# are compiled against it, so mujoco is installed with numpy pinned < 2 — pip
# would otherwise pull a 2.x into /usr/local that shadows the apt one and breaks
# those imports. mujoco's manylinux_2_28 aarch64 wheel imports fine on 1.21.
# =============================================================================
FROM syncai-robot-base:latest

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglfw3 \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxi6 \
        x11-apps \
        mesa-utils \
        tigervnc-standalone-server \
        novnc \
        websockify \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/sim-requirements.txt
RUN pip3 install --no-cache-dir "numpy<2" -r /tmp/sim-requirements.txt \
    && python3 -c "import mujoco, numpy; print('mujoco', mujoco.__version__, 'numpy', numpy.__version__)" \
    && rm /tmp/sim-requirements.txt

USER syncrobotic

# The sim package is bind-mounted at /home/syncrobotic/sim (docker-compose.yml);
# on PYTHONPATH so `python3 -m syncai_sim` works from the workspace root, which
# is where every pane runs (the launch files resolve config/system.ini
# relative to the cwd).
ENV PYTHONPATH=/home/syncrobotic/sim
RUN echo 'export PYTHONPATH=/home/syncrobotic/sim:${PYTHONPATH}' >> /home/syncrobotic/.bashrc

WORKDIR /home/syncrobotic/robot_ws
CMD ["bash"]
