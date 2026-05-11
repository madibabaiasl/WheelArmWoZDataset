#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(dirname "$(realpath "$0")")"

mkdir -p "$PROJECT_DIR/logs"
nohup python3 "$PROJECT_DIR/robot_camera.py" > "$PROJECT_DIR/logs/camera_log.txt" &
cd "$PROJECT_DIR/server"
nohup gunicorn -w 12 -b 0.0.0.0:5000 -k gevent --timeout 0 --worker-connections 2 'monitor:app' > "$PROJECT_DIR/logs/cam_server.txt" &
