#!/bin/bash
# Bring up the KV260 DPU overlay and launch the xilinx/smartcam:2022.1 container.
# Usage:
#   ./bringup.sh          # B3136 (kv260-smartcam)
#   ./bringup.sh -b4096   # B4096 (kv260-benchmark-b4096)

set -e

APP="kv260-smartcam"
if [ "$1" = "-b4096" ] || [ "$1" = "--b4096" ]; then
  APP="kv260-benchmark-b4096"
fi

sudo xmutil unloadapp || true
sudo xmutil loadapp "$APP"
sudo xmutil listapps

XCLBIN="/lib/firmware/xilinx/${APP}/${APP}.xclbin"

docker run --rm -it --privileged \
  -v /dev:/dev \
  -v /lib/firmware/xilinx:/lib/firmware/xilinx \
  -v "$PWD":/workspace \
  -w /workspace \
  -e XLNX_VART_FIRMWARE="$XCLBIN" \
  xilinx/smartcam:2022.1 bash
