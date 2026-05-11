import numpy as np
import threading
from copy import deepcopy as copy

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState, Joy, Imu
from geometry_msgs.msg import PoseStamped
from whill_msgs.msg import ModelCr2State

from .kinova_proxy import KinovaGen3Controller


KINOVA_JOINT_STATE_TOPIC = '/joint_states'
KINOVA_CARTESIAN_STATE_TOPIC = '/tf/end_effector2base'
WHILL_JOY_TOPIC = '/whill/controller/joy'
WHILL_STATE_TOPIC = '/whill/states/model_cr2'
IMU_TOPIC = '/oak/imu/data'

KINOVA_HOME_VALUES = [0.0, 0.261, -2.27, 0.0, 0.96, 1.5708]


class DexArmControl(Node):
    def __init__(self, record_type=None, robot_type='kinova_gen3', start_whill=True):
        try:
            rclpy.init()
        except Exception:
            pass

        super().__init__('dex_arm')

        self.record_type = record_type
        self.robot_type = robot_type
        self.start_whill = start_whill

        self._executor = None
        self._executor_thread = None

        if robot_type == 'kinova_gen3':
            self._init_kinova_arm_control()

        self.get_logger().info(f'DexArmControl initialized with robot_type: {robot_type}')


    def _init_kinova_arm_control(self):
        # ---- robot helper ----
        self.kinova = KinovaGen3Controller()

        # ---- latest received states ----
        self.kinova_joint_state = None
        self.kinova_cartesian_state = None
        self.wheelchair_joy_commands = None
        self.wheelchair_states = None
        self.imu_data = None

        # ---- WHILL command state (merged from WhillController) ----
        self.latest_x = 0.0
        self.latest_y = 0.0

        # ---- subscriptions ----
        self.create_subscription(
            JointState,
            KINOVA_JOINT_STATE_TOPIC,
            self._callback_kinova_joint_state,
            1
        )

        self.create_subscription(
            PoseStamped,
            KINOVA_CARTESIAN_STATE_TOPIC,
            self._callback_kinova_cartesian_state,
            1
        )

        self.create_subscription(
            Joy,
            WHILL_JOY_TOPIC,
            self._callback_whill_joy,
            1
        )

        self.create_subscription(
            ModelCr2State,
            WHILL_STATE_TOPIC,
            self._callback_whill_states,
            1
        )

        self.create_subscription(
            Imu,
            IMU_TOPIC,
            self._callback_imu,
            10
        )

        self.whill_pub = None
        self.whill_timer = None
        if self.start_whill:
            # ---- WHILL publisher (merged from WhillController) ----
            self.whill_pub = self.create_publisher(Joy, WHILL_JOY_TOPIC, 10)

            # 20 Hz publish loop
            self.whill_timer = self.create_timer(0.05, self._publish_whill_loop)

            self.get_logger().info("Entering Kinova Arm Control and merged WHILL control")
        else:
            self.get_logger().info("Entering Kinova Arm Control WITHOUT WHILL control")

        # ---- spin only this node ----
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self)
        self._executor_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()

        self.get_logger().info("DexArmControl ready")

    # -------------------------
    # Callbacks
    # -------------------------
    def _callback_imu(self, imu_data):
        self.imu_data = imu_data

    def _callback_whill_states(self, whill_states):
        self.wheelchair_states = whill_states

    def _callback_kinova_joint_state(self, joint_state):
        self.kinova_joint_state = joint_state

    def _callback_kinova_cartesian_state(self, cartesian_state):
        self.kinova_cartesian_state = cartesian_state

    def _callback_whill_joy(self, joy_msg):
        self.wheelchair_joy_commands = joy_msg

    # -------------------------
    # WHILL merged publisher loop
    # -------------------------
    def _publish_whill_loop(self):
        msg = Joy()
        msg.axes = [self.latest_x, self.latest_y]
        msg.buttons = []
        self.whill_pub.publish(msg)
        # Uncomment for debug only; 20 Hz printing is noisy.
        # self.get_logger().info(f"[WHILL] publishing axes={[self.latest_x, self.latest_y]}")

    def move_whillchair(self, x: float, y: float):
        """
        x: angular command
        y: linear command
        """
        self.latest_x = max(-0.1, min(0.1, float(x)))
        self.latest_y = max(-0.1, min(0.1, float(y)))
        # self.get_logger().info(f"[WHILL] updated command x={self.latest_x}, y={self.latest_y}")

    # -------------------------
    # Getters
    # -------------------------
    def get_imu(self):
        if self.imu_data is None:
            return None

        raw_imu = copy(self.imu_data)

        imu = dict(
            orientation=np.array([
                raw_imu.orientation.x,
                raw_imu.orientation.y,
                raw_imu.orientation.z,
                raw_imu.orientation.w
            ], dtype=np.float32),
            orientation_covariance=np.array(raw_imu.orientation_covariance, dtype=np.float32),
            angular_velocity=np.array([
                raw_imu.angular_velocity.x,
                raw_imu.angular_velocity.y,
                raw_imu.angular_velocity.z
            ], dtype=np.float32),
            angular_velocity_covariance=np.array(raw_imu.angular_velocity_covariance, dtype=np.float32),
            linear_acceleration=np.array([
                raw_imu.linear_acceleration.x,
                raw_imu.linear_acceleration.y,
                raw_imu.linear_acceleration.z
            ], dtype=np.float32),
            linear_acceleration_covariance=np.array(raw_imu.linear_acceleration_covariance, dtype=np.float32),
            timestamp=raw_imu.header.stamp.sec + (raw_imu.header.stamp.nanosec * 1e-9)
        )
        return imu

    def get_whill_states(self):
        if self.wheelchair_states is None:
            return None

        raw_states = copy(self.wheelchair_states)

        states = dict(
            right_motor_angle=raw_states.right_motor_angle,
            left_motor_angle=raw_states.left_motor_angle,
            right_motor_speed=raw_states.right_motor_speed,
            left_motor_speed=raw_states.left_motor_speed
        )
        return states

    def get_whill_joy(self):
        if self.wheelchair_joy_commands is None:
            return None

        raw_joy_commands = copy(self.wheelchair_joy_commands)

        joy_commands = dict(
            axes=np.array(raw_joy_commands.axes[:], dtype=np.float32),
            buttons=np.array(raw_joy_commands.buttons[:], dtype=np.int32),
            timestamp=raw_joy_commands.header.stamp.sec + (raw_joy_commands.header.stamp.nanosec * 1e-9)
        )
        return joy_commands

    def get_arm_cartesian_state(self):
        if self.kinova_cartesian_state is None:
            return None

        raw_cartesian_state = copy(self.kinova_cartesian_state)

        cartesian_state = dict(
            position=np.array([
                raw_cartesian_state.pose.position.x,
                raw_cartesian_state.pose.position.y,
                raw_cartesian_state.pose.position.z
            ], dtype=np.float32),
            orientation=np.array([
                raw_cartesian_state.pose.orientation.x,
                raw_cartesian_state.pose.orientation.y,
                raw_cartesian_state.pose.orientation.z,
                raw_cartesian_state.pose.orientation.w
            ], dtype=np.float32),
            timestamp=raw_cartesian_state.header.stamp.sec + (raw_cartesian_state.header.stamp.nanosec * 1e-9)
        )
        return cartesian_state

    def get_arm_joint_state(self):
        if self.kinova_joint_state is None:
            return None

        raw_joint_state = copy(self.kinova_joint_state)

        joint_state = dict(
            position=np.array(raw_joint_state.position[:], dtype=np.float32),
            velocity=np.array(raw_joint_state.velocity[:], dtype=np.float32),
            effort=np.array(raw_joint_state.effort[:], dtype=np.float32),
            timestamp=raw_joint_state.header.stamp.sec + (raw_joint_state.header.stamp.nanosec * 1e-9)
        )
        return joint_state

    def get_arm_position(self):
        if self.kinova_joint_state is None:
            return None
        return np.array(self.kinova_joint_state.position, dtype=np.float32)

    def get_arm_velocity(self):
        if self.kinova_joint_state is None:
            return None
        return np.array(self.kinova_joint_state.velocity, dtype=np.float32)

    def get_arm_torque(self):
        if self.kinova_joint_state is None:
            return None
        return np.array(self.kinova_joint_state.effort, dtype=np.float32)

    def get_arm_cartesian_coords(self):
        if self.kinova_cartesian_state is None:
            return None

        cartesian_state = [
            self.kinova_cartesian_state.pose.position.x,
            self.kinova_cartesian_state.pose.position.y,
            self.kinova_cartesian_state.pose.position.z,
            self.kinova_cartesian_state.pose.orientation.x,
            self.kinova_cartesian_state.pose.orientation.y,
            self.kinova_cartesian_state.pose.orientation.z,
            self.kinova_cartesian_state.pose.orientation.w
        ]
        return np.array(cartesian_state, dtype=np.float32)

    # -------------------------
    # Arm / robot movement
    # -------------------------
    def move_arm(self, kinova_angles):
        self.kinova.joint_movement(kinova_angles, False)

    def move_arm_cartesian(self, kinova_cartesian_values):
        self.kinova.cartesian_movement(kinova_cartesian_values)

    def move_arm_cartesian_velocity(self, cartesian_velocity_values, duration):
        self.kinova.publish_cartesian_velocity(cartesian_velocity_values, duration)

    def home_arm(self):
        self.kinova.joint_movement(KINOVA_HOME_VALUES)

    def reset_arm(self):
        self.home_arm()

    def reset_robot(self):
        pass

    def move_robot(self, kinova_angles):
        self.kinova.joint_movement(kinova_angles, False)

    def move_gripper(self, gripper_value):
        self.kinova.gripper_movement(gripper_value)

    def home_robot(self):
        self.home_arm()

    # -------------------------
    # Cleanup
    # -------------------------
    def shutdown(self):
        self.get_logger().info("Shutting down DexArmControl")

        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:
                pass

        if self._executor_thread is not None and self._executor_thread.is_alive():
            self._executor_thread.join(timeout=1.0)

        # Only do this if kinova is actually a node-like object with destroy_node()
        try:
            self.kinova.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
