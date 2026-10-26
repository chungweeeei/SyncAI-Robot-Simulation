# =============================================================================
# The simulation container: the robot's own dev image plus MuJoCo.
#
# FROM the workspace's `dev` stage (built there as `syncai-robot-base`, see its
# docker-compose.build.yml) rather than a ros:humble image, because the nav
# stack this container runs is the workspace's build/ + install/ bind-mounted
# in, and a .so built against that image's exact ROS / GTSAM / xtensor set only
# loads in that image. Everything the sim adds on top is Python.
#
# No display, no GPU, no rendering in here: the picture is the browser
# viewer (syncai_sim/web_view.py, viser), a websocket the host's browser
# draws from with its own GPU. Until 2026-10-26 the container ran an X server
# of its own (TigerVNC's Xvnc with Mesa's llvmpipe, noVNC on 6080) for
# MuJoCo's GLFW window, which llvmpipe drew at ten frames a second; XQuartz
# had been tried before that and fails at GLX context creation (OpenGL 1.4).
# libgl1 stays: open3d's wheel (pcd_mesh.py) links libGL.so.1 to import.
#
# numpy: the base image has ROS Humble's python3-numpy and cv_bridge & co. are
# compiled against it, so mujoco is installed with numpy pinned < 2 — pip
# would otherwise pull a 2.x into /usr/local that shadows the apt one and breaks
# those imports. mujoco's manylinux_2_28 aarch64 wheel imports fine on 1.x.
# =============================================================================
FROM syncai-robot-base:latest

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/sim-requirements.txt
RUN pip3 install --no-cache-dir "numpy<2" -r /tmp/sim-requirements.txt \
    && python3 -c "import mujoco, numpy, viser; print('mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'viser', viser.__version__)" \
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
