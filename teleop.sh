#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(dirname "$(realpath "$0")")"
DDS_XML="$PROJECT_DIR/cyclonedds_ros2.xml"

gnome-terminal --title="Teleoperation" -- bash -c "
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openteach
source /opt/ros/humble/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
unset CYCLONEDDS_URI
export CYCLONEDDS_URI=file://$DDS_XML

cd $PROJECT_DIR
python teleop.py robot=kinova_gen3
exec bash
"
