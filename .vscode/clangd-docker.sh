#!/usr/bin/env bash
exec docker exec -i viw-dev /usr/bin/clangd-10 \
  --compile-commands-dir=/root/catkin_ws/build \
  --path-mappings=/home/nio/code/VIW-Fusion=/root/catkin_ws/src/VIW-Fusion \
  "$@"
