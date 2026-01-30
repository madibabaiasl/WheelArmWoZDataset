import numpy as np
import time
import threading
from copy import deepcopy as copy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu

from .kinova_proxy import KinovaGen3Controller
from whill_bringup.whill_controller import WhillController
from whill_msgs.msg import ModelCr2State

KINOVA_JOINT_STATE_TOPIC = '/joint_states'
KINOVA_CARTESIAN_STATE_TOPIC = '/tf/end_effector2base'
KINOVA_HOME_VALUES = [0.0, 0.261 , -2.27, 0.0, 0.96, 1.5708]

class DexArmControl(Node):
    def __init__(self, record_type=None, robot_type='kinova_gen3'):
        try:
            rclpy.init()  # ROS2 initialization (no disable_signals option)
        except Exception:
            pass  # Can occur if rclpy is already initialized
        
        super().__init__('dex_arm')

        self.record_type = record_type
        self.robot_type = robot_type
        
        if robot_type == 'kinova_gen3':
            self._init_kinova_arm_control()

        self.get_logger().info(f'DexArmControl initialized with robot_type: {robot_type}')

    # Controller initializers
    def _init_kinova_arm_control(self):
        self.kinova = KinovaGen3Controller()
        self.whill = WhillController()
        

        self.kinova_joint_state = None
        self.create_subscription(
            JointState, 
            KINOVA_JOINT_STATE_TOPIC, 
            self._callback_kinova_joint_state, 
            1
        )

        self.kinova_cartesian_state = None
        self.create_subscription(
            PoseStamped,
            KINOVA_CARTESIAN_STATE_TOPIC,
            self._callback_kinova_cartesian_state,
            1
        )

        self.wheelchair_joy_commands = None
        self.create_subscription(
            Joy,
            '/whill/controller/joy',
            self._callback_whill_joy,
            1
        )

        self.wheelchair_states = None
        self.create_subscription(
            ModelCr2State,
            '/whill/states/model_cr2',
            self._callback_whill_states,
            1
        )

        self.imu_data = None
        self.create_subscription(
            Imu,
            '/oak/imu/data',
            self._callback_imu,
            10
        )

        print("Entering Kinova Arm Control and Whill Control")
    
    # ── spin both nodes in the background ────────────────────────
        exec_ = rclpy.executors.MultiThreadedExecutor()
        exec_.add_node(self)
        # exec_.add_node(self.kinova)
        threading.Thread(target=exec_.spin, daemon=True).start()

        self.get_logger().info("DexArmControl ready")

    # Rostopic callback functions
    def _callback_imu(self, imu_data):
        self.imu_data = imu_data 

    def _callback_whill_states(self, whill_states):
        self.wheelchair_states = whill_states

    def _callback_kinova_joint_state(self, joint_state):
        self.kinova_joint_state = joint_state
        # print(f"[Kinova Control(roslink)]: joint states {joint_state}")

    def _callback_kinova_cartesian_state(self, cartesian_state):
        self.kinova_cartesian_state = cartesian_state
    
    def _callback_whill_joy(self, joy_msg):
        self.wheelchair_joy_commands = joy_msg
    
    def get_imu(self):
        if self.imu_data is None:
            return None
        
        raw_imu = copy(self.imu_data)

        imu = dict(
            orientation = np.array([
                raw_imu.orientation.x, raw_imu.orientation.y, raw_imu.orientation.z, raw_imu.orientation.w
            ], dtype = np.float32),
            orientation_covariance = np.array(raw_imu.orientation_covariance, dtype = np.float32),
            angular_velocity = np.array([
                raw_imu.angular_velocity.x, raw_imu.angular_velocity.y, raw_imu.angular_velocity.z
            ], dtype = np.float32),
            angular_velocity_covariance = np.array(raw_imu.angular_velocity_covariance, dtype = np.float32),
            linear_acceleration = np.array([
                raw_imu.linear_acceleration.x, raw_imu.linear_acceleration.y, raw_imu.linear_acceleration.z
            ], dtype = np.float32),
            linear_acceleration_covariance = np.array(raw_imu.linear_acceleration_covariance, dtype = np.float32),
            timestamp = raw_imu.header.stamp.sec + (raw_imu.header.stamp.nanosec * 1e-9)
        )
        return imu
    
    def get_whill_states(self):
        if self.wheelchair_states is None:
            return None
        
        raw_states = copy(self.wheelchair_states)

        states = dict(
            right_motor_angle = raw_states.right_motor_angle,
            left_motor_angle = raw_states.left_motor_angle,
            right_motor_speed = raw_states.right_motor_speed,
            left_motor_speed = raw_states.left_motor_speed
        )
        return states
    
    def get_whill_joy(self):
        if self.wheelchair_joy_commands is None:
            return None
        
        raw_joy_conmmands = copy(self.wheelchair_joy_commands)

        joy_commands = dict(
            axes = np.array(raw_joy_conmmands.axes[:], dtype = np.float32),
            buttons = np.array(raw_joy_conmmands.buttons[:], dtype = np.int32),
            timestamp = raw_joy_conmmands.header.stamp.sec + (raw_joy_conmmands.header.stamp.nanosec * 1e-9)
        )
        return joy_commands
    
    def get_arm_cartesian_state(self):
        # print(f"[KinovaGEn3(roslink)]: get_arm_cartesian {self.kinova_cartesian_state}")
        if self.kinova_cartesian_state is None:
            return None

        raw_cartesian_state = copy(self.kinova_cartesian_state)

        cartesian_state = dict(
            position = np.array([
                raw_cartesian_state.pose.position.x, raw_cartesian_state.pose.position.y, raw_cartesian_state.pose.position.z
            ], dtype = np.float32),
            orientation = np.array([
                raw_cartesian_state.pose.orientation.x, raw_cartesian_state.pose.orientation.y, raw_cartesian_state.pose.orientation.z, raw_cartesian_state.pose.orientation.w
            ], dtype = np.float32),
            timestamp = raw_cartesian_state.header.stamp.sec + (raw_cartesian_state.header.stamp.nanosec * 1e-9)
        )
        return cartesian_state
    
    def get_arm_joint_state(self):
        if self.kinova_joint_state is None:
            return None

        raw_joint_state = copy(self.kinova_joint_state)
        # print(f"raw_joint_state: {raw_joint_state}")

        joint_state = dict(
            # name = list(raw_joint_state.name[:]),
            position = np.array(raw_joint_state.position[:], dtype = np.float32),
            velocity = np.array(raw_joint_state.velocity[:], dtype = np.float32),
            effort = np.array(raw_joint_state.effort[:], dtype = np.float32),
            timestamp = raw_joint_state.header.stamp.sec + (raw_joint_state.header.stamp.nanosec * 1e-9)
        )
        return joint_state
    
    def get_arm_position(self):
        if self.kinova_joint_state is None:
            return None
        
        return np.array(self.kinova_joint_state.position, dtype = np.float32)

    def get_arm_velocity(self):
        if self.kinova_joint_state is None:
            return None
        
        return np.array(self.kinova_joint_state.velocity, dtype = np.float32)

    def get_arm_torque(self):
        if self.kinova_joint_state is None:
            return None

        return np.array(self.kinova_joint_state.effort, dtype = np.float32)

    def get_arm_cartesian_coords(self):
        if self.kinova_cartesian_state is None:
            return None

        cartesian_state  =[
            self.kinova_cartesian_state.pose.position.x,
            self.kinova_cartesian_state.pose.position.y,
            self.kinova_cartesian_state.pose.position.z,
            self.kinova_cartesian_state.pose.orientation.x,
            self.kinova_cartesian_state.pose.orientation.y,
            self.kinova_cartesian_state.pose.orientation.z,
            self.kinova_cartesian_state.pose.orientation.w
        ]
        return np.array(cartesian_state)


    # Movement functions
    def move_arm(self, kinova_angles):
        self.kinova.joint_movement(kinova_angles, False) # needed to be written->fixed in controller.py
    
    def move_arm_cartesian(self, kinova_cartesian_values):
        self.kinova.cartesian_movement(kinova_cartesian_values) # needed to be written->fixed in controller.py, defult orientation is in quaternion

    def move_arm_cartesian_velocity(self, cartesian_velocity_values, duration):
        self.kinova.publish_cartesian_velocity(cartesian_velocity_values, duration) # needed to be written -> finsed in controller.py

    def home_arm(self):
        self.kinova.joint_movement(KINOVA_HOME_VALUES) 

    def reset_arm(self):
        self.home_arm()
    
    def move_whillchair(self, x: float, y: float):
        """
        Move the wheelchair using the WhillController.
        x: Angular movement (turning)
        y: Linear movement (forward/backward)
        """
        self.whill.move_wheelchair(x, y)

    # Reset the Robot
    def reset_robot(self):
        pass

    # Full robot commands
    def move_robot(self, kinova_angles):
        self.kinova.joint_movement(kinova_angles, False)
        # self.allegro.hand_pose(allegro_angles)
    
    def move_gripper(self, gripper_value):
        self.kinova.gripper_movement(gripper_value)

    def home_robot(self):
        self.home_arm()
        # self.home_hand()
    
    def shutdown(self):
        self.destroy_node()
        self.kinova.destroy_node()
        self.whill.destroy_node()
