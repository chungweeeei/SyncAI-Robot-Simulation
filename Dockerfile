# =============================================================================
# The simulation container: Ubuntu 22.04 + Python + MuJoCo. No ROS.
#
# The sim talks to the outside over UDP only (the gait controller's ASCII link,
# the hardware link to a controller process), so nothing here needs ROS: the
# image is the Python packages in requirements.txt on the distribution the
# robot runs (22.04, Python 3.10 — ruff.toml's py310). Until 2026-11-03 it was
# FROM the workspace's ROS image, for the nav session it ran; that is gone.
#
# No display, no GPU, no rendering in here: the picture is the browser viewer
# (syncai_sim/viewer/web_view.py, viser), a websocket the host's browser draws
# from with its own GPU. libgl1 and libgomp1 stay: open3d's wheel links
# libGL.so.1 to import, numba's parallel ray caster runs on OpenMP.
# =============================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/sim-requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/sim-requirements.txt \
    && python3 -c "import mujoco, numpy, numba, viser, yaml; print('mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'viser', viser.__version__)" \
    && rm /tmp/sim-requirements.txt

# docker-compose.yml runs the container as the host's uid/gid (so cache/ and
# log/ written into the mounts keep the host user's ownership) with HOME here:
# world-writable, as no passwd entry exists for that uid.
RUN mkdir -p /home/syncrobotic && chmod 1777 /home/syncrobotic

# The sim package is bind-mounted at /home/syncrobotic/sim (docker-compose.yml);
# on PYTHONPATH so `python3 -m syncai_sim` works from the workspace root, where
# the instance INI's relative paths resolve.
ENV PYTHONPATH=/home/syncrobotic/sim \
    HOME=/home/syncrobotic

WORKDIR /home/syncrobotic/robot_ws
CMD ["bash"]
