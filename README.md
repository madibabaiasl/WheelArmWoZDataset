# A Multimodal Data Collection Framework for Dialogue-Driven Assistive Robotics to Clarify Ambiguities: A Wizard-of-Oz Pilot Study
Authors: Guangping Liu, Nicholas Hawkins, Tipu Sultan, Flavio Esposito, Madi Babaiasl

Page: [project](https://madibabaiasl.github.io/WheelArmWoZDataset/) | Paper: [arxiv](https://arxiv.org/abs/2601.16870) | Code: [github](https://github.com/madibabaiasl/WheelArmWoZDataset) | Dataset: will be published soon

Welcome to the WheelArm Multimodel Dataset for the wheelchair and wheelchair-mounted robotic arm! In this research, we developed a real-time teleoperation and data collection framework using Wizard-of-Oz tailored for the Kinova Gen3 robotic arm and Whill Model CR2 wheelchair. Our work is developed on [OpenTeach](https://github.com/aadhithya14/Open-Teach) by customizing the Unity application, expanding the manipulation to navigation, and simplifying the hand detection to controller tracking. We propose a multimodal data collection framework that employs a dialogue-based interaction protocol and a two-room Wizard-of-Oz (WoZ) setup to simulate robot autonomy while eliciting natural user behavior. The framework records five synchronized modalities: RGB-D video, conversational audio, inertial measurement unit (IMU) signals, end-effector Cartesian pose, and whole-body joint states across five assistive tasks. Using this framework, we collected a pilot dataset of 53 trials from five participants and validated its quality through motion smoothness analysis and user feedback. The results show that the framework effectively captures diverse ambiguity types and supports natural dialogue-driven interaction, demonstrating its suitability for scaling to a larger dataset for learning, benchmarking, and evaluation of ambiguity-aware assistive control.

![Overview](assests/overview.png)

## Getting Started!
### Hardware 
Two laptops: Precision 5570 and 7780
Kinova Gen3 robotic Arm
Whill Model CR2 wheelchair
Luxion OAK-D W canmera
A battery box for Kinova movable power(please email guangping.liu@slu.edu if you need more info)

### Environment Setup
Download the repo to your local directory.
```
git clone https://github.com/madibabaiasl/WheelArmWoZDataset.git
```
Build your conda environment.
```
conda env create -f environment.yml
pip install -e .
```
### ROS2 Packages Setup
Kinova Gen3 6-DOF Arm: [Kinova ROS2 Control Humble](https://github.com/Kinovarobotics/ros2_kortex) <br>
Whill Model CR2: [WHILL MODEL](https://github.com/whill-labs/ros2_whill) <br>
Luxion OAK-D W: [vision](https://docs.luxonis.com/software-v3/depthai/ros/) <br>

### Program Setup
We set up the program running on two laptops, laptop A (5570;10.0.0.1) and laptop B(7780;10.0.0.2). Please build the environment on both laptops.
Laptop A runs the teleoperation and data collection framework, while laptop B executes the wheelchair and robotic arm.

### Launch
Step 1: On laptop B:
```
# launch wheelchair
export CYCLONEDDS_URI=file://$HOME/cyclonedds_ros2.xml
cd /whill/ros/package/path
source install/setup.bash
sudo chmod a+rw /dev/ttyUSB0
ros2 launch whill_bringup whill_launch.py

# launch the robotic arm
source /opt/ros/humble/setup.bash
cd /kinova/path/
source install/setup.bash
ros2 launch kinova_gen3_6dof_robotiq_2f_85_moveit_config robot.launch.py   robot_ip:=192.168.1.10

# launch the kinova vision
cd /kinova/vision/path/
source install/setup.bash
ros2 launch kinova_vision kinova_vision.launch.py

# launch the execution server

```
Step 2: On laptop A
```
# launch the camera
ros2 launch depthai_ros_driver camera.launch.py

# launch the in

## Dataset

## License

## Citation
