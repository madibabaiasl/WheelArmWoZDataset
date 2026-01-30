#!/usr/bin/env bash
set -uo pipefail

PORTS=(10005 11005 5000)

echo "Stopping ZMQ/Gunicorn ports…"
for p in "${PORTS[@]}"; do
  sudo lsof -t -i tcp:$p | xargs -r kill -15
done
sleep 1
for p in "${PORTS[@]}"; do
  sudo lsof -t -i tcp:$p | xargs -r kill -9 || true
done

echo "Killing common RealSense holders…"
pgrep -fa 'realsense|realsense2_camera|realsense-viewer|ffmpeg|python.*realsense|ros2' \
  | awk '{print $1}' | xargs -r sudo kill -9

echo "Releasing /dev/video* and USB handles…"
sudo fuser -kv /dev/video* 2>/dev/null || true
sudo lsof /dev/bus/usb -nP | grep -i realsense | awk '{print $2}' | xargs -r sudo kill -9

echo "Verifying ports…"
for p in "${PORTS[@]}"; do
  sudo lsof -i tcp:$p || echo "✓ port $p free"
done

echo "Done."

