# A Multimodal Data Collection Framework for Dialogue-Driven Assistive Robotics to Clarify Ambiguities: A Wizard-of-Oz Pilot Study
**Authors**: Guangping Liu, Nicholas Hawkins, Tipu Sultan, Flavio Esposito, Madi Babaiasl

**Page**: [project](https://madibabaiasl.github.io/WheelArmWoZDataset/) | **Paper**: [arxiv](https://arxiv.org/abs/2601.16870) | **Code**: [github](https://github.com/madibabaiasl/WheelArmWoZDataset) | **Dataset**: will be published soon
![Overview](assests/overview.png)

## News
**[05/11/2026]** We have upgraded our program on a ***Jetson AGX Orin*** and a ***laptop***. Please follow the instructions in the branch of WheelArm_Tele_v2. <br>
**[04/20/2026]** Our paper has been accepted by ***IEEE International Conference on Biomedical Robotics and Biomechatronics (BioRob 2026)***. <br>
**[01/01/2026]** Our main branch is for WheelArm_Tele implementation for ***two laptops***. <br>

## Getting Started!
### Hardware List
- Two laptops: Precision 5570 and 7780 <br>
- Kinova Gen3 robotic Arm <br>
- Whill Model CR2 wheelchair <br>
- Luxion OAK-D W camera <br>
- A battery box for Kinova movable power <br>
- Two microphones <br>
- Three headphones<br>
### Hardware Setup
WheelArm Setup:
<table>
  <tr>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/82d1e152-cc04-465b-929f-1ba6ca96dd8e" width="100%"/>
    </td>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/b38ea148-c5ce-4e60-ab2d-5bc174d8e427" width="100%"/>
    </td>
  </tr>
</table>
(Please email guangping.liu@slu.edu if you need more info about the portable battery box for Kinova Gen3 charging.) <br>

Hardware connected to laptop A: <br>
- Luxion OAK-D W camera <br>
- A collar microphone <br>

Hardware connected to laptop B: <br>
- Kinova Gen3 robotic Arm <br>
- Whill Model CR2 wheelchair <br>

The laptop A and B are connected through an Ethernet cable.
<img width="809" height="568" alt="laptopAB" src="https://github.com/user-attachments/assets/84281f16-ea0b-440b-a566-856d4f8d30f3" style="width: 50%; display: inline-block; margin-right: 8%;"/>

### Wizard-of-Oz
<img width="1158" height="561" alt="woz" src="https://github.com/user-attachments/assets/1640ec2e-5c0c-4ad4-b696-3801b15a721a" />

Besides remote teleoperation, another important aspect of Wizard-of-Oz is the use of a real-time voice changer model. We use [w-okada](https://github.com/w-okada/voice-changer/tree/master) to convert the teleoperator's voice to *robot voice*. Additionally, the teleoperator, Researcher A, and the participants should be in the same Zoom room with three devices. In our setting, the participants were provided with earphones to hear the teleoperator's changed voice and talk. The teleoperator and Research A used two workstations to join the Zoom.

### Environment Setup
Download the repo to your local directory.
```
git clone https://github.com/madibabaiasl/WheelArmWoZDataset.git
```
Build your conda environment.
```
conda env create -f environment.yml
pip install -e .
conda activate openteach
sudo apt install libportaudio2 libsndfile1 ros-humble-tf-transformations
pip install transforms3d
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
export CYCLONEDDS_URI=file://$PROJ_DIR/cyclonedds_ros2.xml
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
conda activate openteach
cd $PROJ_DIR
source ~/workspace/whill_ws/install/setup.bash
python3 server_control.py

```
Step 2: On laptop A
```
# launch the OAK-D
ros2 launch depthai_ros_driver camera.launch.py

# launch the monitors in VR headsets
conda activate openteach
cd /proj/path
./stop_pid.sh
bash launch_server_oak.sh

# calculate the end effector cartesian pose
source /opt/ros/humble/setup.bash
cd ~/workspace/ros2_kortex_ws
source install/setup.bash
cd /kinova/path/src/ros2_kortex/kortex_bringup/kortex_bringup
python3 ee_pose_publisher.py

# send kinova vision to VR headset
conda activate openteach
cd /proj/path/
python3 ros_cam_to_zmq.py
```
Step 3: Install WheelArm.apk to Meta Quest VR Headset (Make sure your Meta Quest is in developer mode before this step)
```
cd /proj/path/VR
adb devices
```
Select *Allow* in the pop-up of the USB connection request in the VR Headset.
```
adb install WheelArm.apk
```
Launch the software WheelArm after installing. Change the host IP to your network address to ensure the VR headset, laptop A, and laptop B are on the same network. Input the host IP after clicking the Steam button in the WheelArm software.

Step 4: After launching the WheelArm successfully, launch teleoperation and data collection
```
# start teleoperation
conda activate openteach
cd /proj/path/
python teleop.py robot=kinova_gen3

# start data collection
conda activate openteach
cd /proj/path/
source /whill/ros/package/path/install/setup.bash
python data_collection_GUI.py robot=kinova_gen3 demo_num=1
```
## Dataset
Pilot dataset is available: 

## License
This repository is released under the License in this repo.

## Citation
@article{liu2026multimodal,<br>
  title={A Multimodal Data Collection Framework for Dialogue-Driven Assistive Robotics to Clarify Ambiguities: A Wizard-of-Oz Pilot Study},<br>
  author={Liu, Guangping and Hawkins, Nicholas and Madden, Billy and Sultan, Tipu and Esposito, Flavio and Babaiasl, Madi},<br>
  journal={arXiv preprint arXiv:2601.16870},<br>
  year={2026}<br>
}
