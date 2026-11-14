#!/usr/bin/env bash
# Build scripts/spin_sleep.c into run/spin_sleep.so: the LD_PRELOAD shim that
# makes a controller's sleeps end on time on OrbStack's VM (the .c says what
# was measured). Then, for docker-compose.controller.yml:
#
#   scripts/spin_sleep.sh
#   G2_SPIN_SLEEP=/run/syncai/spin_sleep.so docker compose -f ../g2-device-control/docker-compose.yml -f docker-compose.controller.yml up -d
#
# Compiled in a throwaway ubuntu:22.04 container, the controller image's base
# (glibc 2.35), for this machine's architecture; the sim's image has no
# compiler. The first run installs gcc there (a minute, needs the network).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p run
docker run --rm -v "$PWD/scripts/spin_sleep.c:/src/spin_sleep.c:ro" -v "$PWD/run:/out" ubuntu:22.04 sh -c '
    apt-get update -qq >/dev/null && apt-get install -y -qq gcc libc6-dev >/dev/null 2>&1
    gcc -O2 -Wall -shared -fPIC -o /out/spin_sleep.so /src/spin_sleep.c -ldl'
ls -l run/spin_sleep.so
