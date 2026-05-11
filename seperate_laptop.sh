#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(dirname "$(realpath "$0")")"
DDS_XML="$PROJECT_DIR/cyclonedds_ros2.xml"

gnome-terminal --title="Luxonis" -- bash -c "
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openteach
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file://$DDS_XML
cd $PROJECT_DIR
ros2 launch depthai_ros_driver camera.launch.py
exec bash
"

sleep 2

gnome-terminal --title="ros_cam_to_zmq" -- bash -c "
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openteach
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file://$DDS_XML
cd $PROJECT_DIR
python3 ros_cam_to_zmq.py
exec bash
"

gnome-terminal --title="ee_pose_publisher" -- bash -c "
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openteach
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file://$DDS_XML
cd $PROJECT_DIR
python3 ee_pose_publisher.py
exec bash
"

sleep 3

gnome-terminal --title="Display" -- bash -c "
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openteach
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file://$DDS_XML
cd $PROJECT_DIR
bash launch_server_oak.sh
exec bash
"

sleep 3

gnome-terminal --title="Collection Router" -- bash -c "
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openteach
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file://$DDS_XML
cd $PROJECT_DIR
python3 collect_signal_router.py
exec bash
"