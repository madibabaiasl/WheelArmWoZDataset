#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(dirname "$(realpath "$0")")"
DDS_XML="$PROJECT_DIR/cyclonedds_ros2.xml"

gnome-terminal --title="Data Collection" -- bash -c "
source ~/anaconda3/bin/activate openteach
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file://$DDS_XML
cd $PROJECT_DIR
python data_collection_GUI.py robot=kinova_gen3 demo_num=1
exec bash
"
