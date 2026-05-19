# A Multimodal Data Collection Framework for Dialogue-Driven Assistive Robotics to Clarify Ambiguities: A Wizard-of-Oz Pilot Study
**Authors**: Guangping Liu, Nicholas Hawkins, Tipu Sultan, Flavio Esposito, Madi Babaiasl

**Page**: [project](https://madibabaiasl.github.io/WheelArmWoZDataset/) | **Paper**: [arxiv](https://arxiv.org/abs/2601.16870) | **Code**: [github](https://github.com/madibabaiasl/WheelArmWoZDataset) | **Dataset**: [hugging face](https://huggingface.co/datasets/Cordelia/WheelArm_WoZ_Pilot_Dataset)
![Overview](assests/overview.png)

## News
**[05/11/2026]** We have upgraded our program on a ***Jetson AGX Orin*** and a ***laptop***. Please follow the instructions in the branch of WheelArm_Tele_v2. <br>
**[04/20/2026]** Our paper has been accepted by ***IEEE International Conference on Biomedical Robotics and Biomechatronics (BioRob 2026)***. <br>
**[01/01/2026]** Our main branch is for WheelArm_Tele implementation for ***two laptops***. <br>

## Getting Started!
We upgraded our hardware architecture to a Jetson-laptop pair to better manage high computational loads. Distributing the teleoperation and data collection tasks across separate units allowed for a more efficient framework, resulting in higher control frequencies and minimized latency in the vision pipeline.

### Hardware List
- Edge Device: Nvidia Jetson AGX Orin <br>
- Laptops: Precision 7780 <br>
- Kinova Gen3 robotic Arm <br>
- Whill Model CR2 wheelchair <br>
- Luxion OAK-D W camera <br>
- A battery box for Kinova movable power <br>
- Two microphones <br>
- Three headphones<br>
### Hardware Setup
<img width="2080" height="964" alt="jetsonLaptop" src="https://github.com/user-attachments/assets/dc56aa99-4e22-41c9-be09-08d33dbcdc15" />

Hardware connected to laptop A: <br>
- Luxion OAK-D W camera <br>

Hardware connected to Jetson: <br>
- A collar microphone <br>
- Kinova Gen3 robotic Arm <br>
- Whill Model CR2 wheelchair <br>

The laptop A and Jetson are connected through an Ethernet cable.

### Environment Setup
Download the repo to your local directory.
```
git clone https://github.com/madibabaiasl/WheelArmWoZDataset.git
```
Build your conda environment on both devices.
```
conda env create -f environment.yml
pip install -e .
conda activate openteach
sudo apt install libportaudio2 libsndfile1 ros-humble-tf-transformations
pip install transforms3d
```
### Network
Kinova Gen3 IP Address: 192.168.1.10 <br>
Laptop A: 10.0.0.1 <br>
Jetson: 10.0.0.2 <br>
Jetson and Laptop are synchronized using cyclone DDS. Please find the Ethernet interface on both devices using:
```
ip address
```
Then fill your interface to cyclonedds_ros2.xml

### ROS2 Packages Setup
Jetson:<br>
- Kinova Gen3 6-DOF Arm: [Kinova ROS2 Control Humble](https://github.com/Kinovarobotics/ros2_kortex) <br>
- Whill Model CR2: [WHILL MODEL](https://github.com/whill-labs/ros2_whill) <br>

Laptop:<br>
- Luxion OAK-D W: [vision](https://docs.luxonis.com/software-v3/depthai/ros/) <br>

### Program Setup
In this version, we route VR commands from the laptop to Jetson, so the data collection pipeline and teleoperation pipeline can be separated into two devices. With the upgraded framework, the Jetson handles robot execution and real-time data collection, while the laptop computes the end-effector twist and processes and uploads camera streams.

### Launch
Step 1: On Jetson:
```
./seperate_jetson.sh
```
Step 2: On laptop A
```
./seperate_laptop.sh
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

Step 4: After launching the WheelArm successfully, launch data collection on Jetson
```
./collection.sh
```
Step 5: Launch teleoperation on the laptop
```
./teleop.sh
```

## License
This repository is released under the License in this repo.

## Citation
@article{liu2026multimodal,<br>
  title={A Multimodal Data Collection Framework for Dialogue-Driven Assistive Robotics to Clarify Ambiguities: A Wizard-of-Oz Pilot Study},<br>
  author={Liu, Guangping and Hawkins, Nicholas and Madden, Billy and Sultan, Tipu and Esposito, Flavio and Babaiasl, Madi},<br>
  journal={arXiv preprint arXiv:2601.16870},<br>
  year={2026}<br>
}
