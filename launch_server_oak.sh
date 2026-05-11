#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(dirname "$(realpath "$0")")"

mkdir -p "$PROJECT_DIR/oak_logs"
nohup python3 "$PROJECT_DIR/wide_range_robot_camera.py" > "$PROJECT_DIR/oak_logs/camera_log.txt" &
