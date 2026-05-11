#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(dirname "$(realpath "$0")")"
DDS_XML="$PROJECT_DIR/cyclonedds_ros2.xml"

gnome-terminal --title="Kinova" -- bash -c "
source ~/anaconda3/bin/activate openteach
source /opt/ros/humble/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file://$DDS_XML
ros2 launch kinova_gen3_6dof_robotiq_2f_85_moveit_config robot.launch.py robot_ip:=192.168.1.10
exec bash
"

sleep 3

gnome-terminal --title="Kinova Vision" -- bash -c "
source ~/anaconda3/bin/activate openteach
source /opt/ros/humble/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file://$DDS_XML
ros2 launch kinova_vision kinova_vision.launch.py
exec bash
"

sleep 2

gnome-terminal --title="Whill" -- bash -c "
source ~/anaconda3/bin/activate openteach
source /opt/ros/humble/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file://$DDS_XML
ros2 launch whill_bringup whill_launch.py
exec bash
"

sleep 2

gnome-terminal --title="Server" -- bash -c "
source ~/anaconda3/bin/activate openteach
source /opt/ros/humble/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file://$DDS_XML
cd $PROJECT_DIR
python3 server_control.py
"
