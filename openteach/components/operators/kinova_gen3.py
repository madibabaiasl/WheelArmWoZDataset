import numpy as np
import matplotlib.pyplot as plt
import zmq
import time
import rclpy
from pathlib import Path
import csv
from time import perf_counter

from copy import deepcopy as copy
from asyncio import threads
from openteach.constants import *
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
from openteach.utils.vectorops import *
from openteach.utils.files import *

from openteach.robot.robot import RobotWrapper
from openteach.robot.kinova_gen3 import KinovaGen3
from scipy.spatial.transform import Rotation, Slerp
from .operator import Operator
from scipy.spatial.transform import Rotation as R
from numpy.linalg import pinv
from scipy.linalg import logm

from scipy.spatial.transform import Rotation, Slerp

class Filter:
    def __init__(self, state, comp_ratio=0.6):
        """
        state: 7D pose [x, y, z, qx, qy, qz, qw]
        comp_ratio: how much to trust the previous state (0..1).
                    higher = smoother but more laggy.
        """
        state = np.asarray(state, dtype=float)

        self.pos_state = state[:3].copy()      # (3,)
        self.ori_state = state[3:7].copy()     # (4,) quaternion [x,y,z,w]
        self.comp_ratio = comp_ratio

    def __call__(self, next_state):
        next_state = np.asarray(next_state, dtype=float)
        alpha = self.comp_ratio

        # --- 1) Low-pass filter position (exponential moving average) ---
        self.pos_state = alpha * self.pos_state + (1.0 - alpha) * next_state[:3]

        # --- 2) Slerp for orientation (geodesic interpolation on SO(3)) ---
        rot_old = Rotation.from_quat(self.ori_state)       # [x,y,z,w]
        rot_new = Rotation.from_quat(next_state[3:7])

        # Define a Slerp object between two keyframes t=0 -> old, t=1 -> new
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([rot_old, rot_new]))

        # Move by (1 - alpha) fraction toward new orientation
        beta = 0.2
        rot_filt = slerp([1.0 - beta])[0]
        self.ori_state = rot_filt.as_quat()

        # Return filtered 7D pose
        return np.concatenate([self.pos_state, self.ori_state])


#Template arm operator class
class KinovaGen3Operator(Operator):
    def __init__(
        self,
        host, 
        transformed_keypoints_port,
        moving_average_limit,
        use_filter=False,
        arm_resolution_port = None, 
        teleoperation_reset_port = None,
        gripper_index_port = None,
        gripper_grip_port = None,
        wheelchair_port = None,
        tele_gui_port = None,
        home_port = None):

        self.notify_component_start('kinova gen3 operator')
        # Transformed Arm Keypoint Subscriber
        self._transformed_arm_keypoint_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=transformed_keypoints_port,
            topic='transformed_hand_frame'
        )

        self._gripper_index_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=gripper_index_port,
            topic='gripper_index'
        )
        self._gripper_grip_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=gripper_grip_port,
            topic='gripper_grip'
        )
        self._wheelchair_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=wheelchair_port,
            topic='joystick'
        )
        # GUI
        self._tele_gui_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=tele_gui_port,
            topic='tele_gui'
        )
        self._home_gui_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=home_port,
            topic='home_button'
        )
        
        # Define Robot object
        self._robot = KinovaGen3()

        # self.arm_teleop_state = ARM_TELEOP_STOP # We will start as the cont

        self._arm_resolution_subscriber = ZMQKeypointSubscriber(
            host = host,
            port = arm_resolution_port,
            topic = 'button'
        )

        self._arm_teleop_state_subscriber = ZMQKeypointSubscriber(
            host = host, 
            port = teleoperation_reset_port,
            topic = 'pause'
        )
        
        # Get the initial pose of the robot
        time.sleep(1)
        robot_coords = self._get_initial_pose()# !!! upadate to get_cartesian_pose
        self.robot_init_H =  self.cartesian_to_homo(robot_coords)
        self.is_first_frame = True

        # Use the filter
        self.use_filter = use_filter
        if use_filter:
            robot_init_cart = self._homo2cart(self.robot_init_H)
            self.comp_filter = Filter(robot_init_cart, comp_ratio=0.8)
        
        # Getting the bounds to perform linear transformation
        bounds_file = get_path_in_package(
            'components/operators/configs/kinovaGen3.yaml')
        bounds_data = get_yaml_data(bounds_file)
        self.velocity_threshold = bounds_data['velocity_threshold']

        # Frequency timer
        self._timer = FrequencyTimer(VR_FREQ)

        # Moving average queues
        self.moving_Average_queue = []
        self.moving_average_limit = moving_average_limit
            
        # Class variables
        # self.resolution_scale =0.7
        # self.arm_teleop_state = ARM_TELEOP_STOP
        # self.prev_time = 0.0

        # --- Gripper control config/state ---
        self.gripper_min = 0.0
        self.gripper_max = 0.8
        self._gripper_pos = 0.0

        #-----Gui state -----
        self.tele_gui_state = 0         # current state (0 or 1)
        self.last_tele_gui_state = 0    # for detecting 0→1 transitions
        self.home_state = 0            # current state (0 or 1)

    @property
    def timer(self):
        return self._timer

    @property
    def robot(self):
        return self._robot

    @property
    def transformed_hand_keypoint_subscriber(self):
        return self._transformed_hand_keypoint_subscriber

    @property
    def transformed_arm_keypoint_subscriber(self):
        return self._transformed_arm_keypoint_subscriber
    
    @property
    def gripper_index_subscriber(self):
        return self._gripper_index_subscriber
    
    @property
    def gripper_grip_subscriber(self):
        return self._gripper_grip_subscriber
    
    @property
    def wheelchair_subscriber(self):
        return self._wheelchair_subscriber
    
    def _read_tele_gui(self):
        """Return latest tele_gui value as int (0/1). Blocking = False."""
        try:
            data = self._tele_gui_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        if data is None:
            return None
        arr = np.asanyarray(data).reshape(-1)
        if arr.size == 0:
            return None
        try:
            return int(float(arr[0]))
        except Exception:
            return None
    
    def _read_home_gui(self):
        """Return latest home button value as int (0/1). Blocking = False."""
        try:
            data = self._home_gui_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        if data is None:
            return None
        arr = np.asanyarray(data).reshape(-1)
        if arr.size == 0:
            return None
        try:
            return int(float(arr[0]))
        except Exception:
            return None

    def get_joystick_commands(self):
        """Read joystick commands from the wheelchair subscriber."""
        try:
            data = self.wheelchair_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None, None
        if data is None:
            print(f"[Operation]: joystick is {data}")
            return 0.0, 0.0
        arr = np.asanyarray(data).reshape(-1)
        return float(-arr[0]), float(arr[1])  # x, y
    
    def send_joystick_commands(self, x: float, y: float):
        """Send joystick commands to the wheelchair."""
        try:
            self.robot.move_whillchair(x, y)
        except Exception as e:
            print(f"Error sending wheelchair commands: {e}")
    
    def wheelchair_control(self):
        """Control the wheelchair based on joystick input."""
        x, y = self.get_joystick_commands()
        if x is not None and y is not None:
            x = 0.4*x
            self.send_joystick_commands(x, y)
        else:
            print("Operators: No joystick commands received.")

    def _get_gripper_index_position(self):
        try:
            data = self.gripper_index_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
            # print(f"Operator: Gripper index position data: {data}")
        except zmq.Again:
            return None
        if data is None:
            return None
        arr = np.asanyarray(data).reshape(-1)
        val = float(arr[0])
        # print(f"Operator: Gripper index position value: {val}")
        return val


    def _get_gripper_grip_position(self):
        try:
            data = self.gripper_grip_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
            # print(f"Operator: Gripper grip position data: {data}")
        except zmq.Again:
            return None
        if data is None:
            return None
        arr = np.asanyarray(data).reshape(-1)
        val = float(arr[0])
        # print(f"Operator: Gripper grip position value: {val}")
        return val

        
    def _read_trigger_nonblocking(self):
        index_trigger = self._get_gripper_index_position()
        grip_trigger = self._get_gripper_grip_position()
        # print(f"Operator: Gripper index trigger: {index_trigger}, grip trigger: {grip_trigger}")
        index_value = float(np.clip(index_trigger, 0.0, 1.0)) if index_trigger is not None else 0.0
        grip_value = float(np.clip(grip_trigger, 0.0, 1.0)) if grip_trigger is not None else 0.0
        return index_value, grip_value
    
    def _step_gripper(self, dt: float):
        """
        Read triggers and regard them as velocity trigger.
        """
        close, op = self._read_trigger_nonblocking()
        if close != 0.0:
            v_gripper = 0.4
        elif op != 0.0:
            v_gripper = -0.4
        else:
            v_gripper = 0.0
        
        net = v_gripper*dt
        if net!= 0.0:
            self._gripper_pos = float(np.clip(self._gripper_pos + net,
                                            self.gripper_min, self.gripper_max))
            # print(f"Operator: Gripper published pose is {self._gripper_pos}")
            self._send_gripper_position(self._gripper_pos)
            print(f"[Operator] gripper command sent {self._gripper_pos}")

    def _send_gripper_position(self, pos):
        """Send gripper position to the robot."""
        try:
            # print(f"Operator: Sending gripper position {pos} to the robot.")
            self.robot.move_gripper(pos)
            # print(f"Operator: Gripper position sent to the robot: {pos}")
        except Exception as e:
            print(f"Error sending gripper position to the robot: {e}")

    def _get_initial_pose(self, timeout=10.0):
        start = time.time()
        coords = None
        while coords is None:
            coords = self.robot.get_cartesian_position()
            # print(f"Operator: Robot Initial Pose is {coords} in quaternion")
            if time.time() - start > timeout:
                raise RuntimeError("Timed out waiting for first cartesian pose")
            time.sleep(0.05)                       # let background executor work
        return coords 
    
    def cartesian_to_homo(self,pose_aa: np.ndarray) -> np.ndarray:
        """Converts a robot pose in axis-angle format to an affine matrix.
        Args:
            pose_aa (list): [x, y, z, ax, ay, az, w] where (x, y, z) is the position and (ax, ay, az, w) is the quaternion.
            x, y, z are in mm and ax, ay, az, w are in quanternion. --I think all assumes about the orientation are wrong. It should be in quaternion.
        Returns:
            np.ndarray: 4x4 affine matrix [[R, t],[0, 1]]
        """        
        rotation = R.from_quat(pose_aa[3:]).as_matrix()
        translation = np.array(pose_aa[:3])
        return np.block([[rotation, translation[:, np.newaxis]],
                        [0, 0, 0, 1]])
    
    #Function to differentiate between real and simulated robot
    def return_real(self):
        return True

    def _get_hand_frame(self):
        """Block/retry until a valid 4x4 hand pose is received.
        Never returns None due to missing data; only returns a 4x4 or exits if headset_loop_flag is cleared externally.
        """
        while True:
            try:
                data = self.transformed_arm_keypoint_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
                # print(f"[Operators] transformed_arm_keypoint_subscriber data: {data}")
            except zmq.Again:
                data = None  # queue empty

            if data is not None:
                try:
                    H = np.asanyarray(data).reshape(4, 4)
                    R = H[:3, :3]  # 3x3 rotation part
                    if not (np.allclose(R @ R.T, np.eye(3), atol=1e-5) and np.isclose(np.linalg.det(R), 1.0, atol=1e-3)):
                        print("[SUB] WARN: received non-orthonormal rotation; waiting for next frame…")
                        time.sleep(0.01)
                        continue
                    return H  # success
                except ValueError as e:
                    # Malformed payload—keep trying instead of returning None
                    print("[SUB] WARN: reshape failed:", e, "— waiting for next frame…")
                    time.sleep(0.01)
                    continue
            else:
                # No data this round—keep polling
                print("[SUB] waiting for controller frame…")
                time.sleep(0.01)
                continue

    # Function to get the resolution scale mode
    # def _get_resolution_scale_mode(self):
    #     res_scale = ARM_HIGH_RESOLUTION
    #     return res_scale
    
    # Get the teleop state (Pause or Continue)
    # def _get_arm_teleop_state(self):
    #     #!!! I hard code the teleop state to be continue, this is for testing purpose only
    #     reset_stat = ARM_TELEOP_CONT
    #     return reset_stat

    # Function to get the arm teleop state from the hand keypoints
    # def _get_arm_teleop_state_from_hand_keypoints(self):
    #     pause_state ,pause_status,pause_right =self.get_pause_state_from_hand_keypoints()
    #     pause_status =np.asanyarray(pause_status).reshape(1)[0] 

    #     return pause_state,pause_status,pause_right

    # get the translation vector
    def _get_translation_vector(self,commanded_robot_position, current_robot_position):
        return commanded_robot_position - current_robot_position
    
    # Get the rotation angular displacement
    def _get_rotation_angles(self, robot_target_orientation,current_robot_rotation_values):
        # Calculating the angular displacement between the target hand frame and the current robot frame# 
        target_rotation_state = Rotation.from_quat(robot_target_orientation)
        robot_rotation_state = Rotation.from_quat(current_robot_rotation_values)

        # Calculating the angular displacement between the target hand frame and the current robot frame
        angular_displacement = Rotation.from_matrix(
            np.matmul(robot_rotation_state.inv().as_matrix(),target_rotation_state.as_matrix())
        ).as_rotvec() # in eular angles

        return angular_displacement 
    
    # Get the displacement vector
    def _get_displacement_vector(self, commanded_robot_position, current_robot_position):
        commanded_robot_pose = np.zeros(6)
        # Transformation from translation
        commanded_robot_pose[:3] = self._get_translation_vector(
            commanded_robot_position=commanded_robot_position[:3],
            current_robot_position = current_robot_position[:3]
        ) * KINOVA_VELOCITY_SCALING_FACTOR

        # Transformation from rotation
        commanded_robot_pose[3:] = self._get_rotation_angles(
            robot_target_orientation=commanded_robot_position[3:],
            current_robot_rotation_values = current_robot_position[3:]
        ) 
        return commanded_robot_pose

    # Function to turn a frame to a homogeneous matrix
    def _turn_frame_to_homo_mat(self, frame):
        t = frame[0]
        R = frame[1:]

        homo_mat = np.zeros((4, 4))
        homo_mat[:3, :3] = np.transpose(R)
        homo_mat[:3, 3] = t
        homo_mat[3, 3] = 1

        return homo_mat

    # Function to turn homogenous matrix to cartesian vector
    def _homo2cart(self, homo_mat):
        t = homo_mat[:3, 3]
        R = Rotation.from_matrix(
            homo_mat[:3, :3]).as_quat()

        cart = np.concatenate(
            [t, R], axis=0
        )

        return cart
    

    # Get the scaled resolution cartesian pose
    # def _get_scaled_cart_pose(self, moving_robot_homo_mat):
    #     # Get the cart pose without the scaling
    #     unscaled_cart_pose = self._homo2cart(moving_robot_homo_mat)

       
    #     robot_coords = self.robot.get_cartesian_position()
    #     current_homo_mat =  copy(self.cartesian_to_homo(robot_coords))
    #     current_cart_pose = self._homo2cart(current_homo_mat)

    #     # Get the difference in translation between these two cart poses
    #     diff_in_translation = unscaled_cart_pose[:3] - current_cart_pose[:3]
    #     scaled_diff_in_translation = diff_in_translation * self.resolution_scale
        
    #     scaled_cart_pose = np.zeros(7)
    #     scaled_cart_pose[3:] = unscaled_cart_pose[3:] # Get the rotation directly
    #     scaled_cart_pose[:3] = current_cart_pose[:3] + scaled_diff_in_translation # Get the scaled translation only

    #     return scaled_cart_pose

    # Reset the teleoperation
    def _reset_teleop(self):
        robot_coords = self.robot.get_cartesian_position()
        self.robot_init_H =  self.cartesian_to_homo(robot_coords)
        # print(f"Operator: the robot_init_H is {self.robot_init_H}")

        if self.use_filter:
            robot_init_cart = self._homo2cart(self.robot_init_H)
            self.comp_filter = Filter(robot_init_cart, comp_ratio=0.8)


        first_hand_frame = self._get_hand_frame()
        while first_hand_frame is None:
            first_hand_frame = self._get_hand_frame()
        # self.hand_init_H = self._turn_frame_to_homo_mat(first_hand_frame)
        self.hand_init_H = first_hand_frame
        self.hand_init_t = copy(self.hand_init_H[:3, 3])
        self.is_first_frame = False
        # self.prev_time = perf_counter()
        return first_hand_frame
    
    def twist_vector_from_twist_matrix(self, twist_matrix):
        """
        Compute the original 6D twist vector from a 4x4 twist matrix.

        Parameters:
        - twist_matrix: A 4x4 matrix representing the matrix form of the twist 

        Returns:
        - twist_vector: The 6D twist vector [w, v] corresponding to the input
                        twist matrix.
        """
        assert twist_matrix.shape == (4, 4), "Input matrix must be 4x4"

        w = np.array([twist_matrix[2, 1], twist_matrix[0, 2], twist_matrix[1, 0]])
        v = twist_matrix[:3, 3]

        return np.concatenate((v, w))
    
    def _get_twist(self, T_target, T_current):
        Tbd = np.linalg.inv(T_current) @ T_target
        matrix_Vb = logm(Tbd)
        Vb = self.twist_vector_from_twist_matrix(matrix_Vb).reshape((6, 1))
        Vb = Vb.astype(float).flatten() 
        return Vb


    def _apply_retargeted_angles(self, moving_hand_frame=None, log=False):
        # See if there is a reset in the teleop
        print("Applying retargeted angles...")

        current_robot_position = self.robot.get_cartesian_position()
        current_homo_mat = self.cartesian_to_homo(current_robot_position)

        # arm_teleoperation_scale_mode = self._get_resolution_scale_mode()

        # if arm_teleoperation_scale_mode == ARM_HIGH_RESOLUTION:
        #     self.resolution_scale = 1 # !!! change it for test temporally
        # elif arm_teleoperation_scale_mode == ARM_LOW_RESOLUTION:
        #     self.resolution_scale = 0.6
        
        # Find the moving hand frame
        self.hand_moving_H = moving_hand_frame
        # print(f"Operators: The transformation matrix of the hand pose in oculus is {self.hand_moving_H}")

        # Transformation code
        H_HI_HH = copy(self.hand_init_H) # Homo matrix that takes P_HI to P_HH - Point in Inital Hand Frame to Point in Home Hand Frame (Thome_init_o)
        H_HT_HH = copy(self.hand_moving_H) # Homo matrix that takes P_HT to P_HH (Thome_home_current_o)
        H_RI_RH = copy(self.robot_init_H) # Homo matrix that takes P_RI to P_RH !!!this should be initial the end effector relative to the base (Tb_ee_gen3)

        # Find the relative transformation in human hand space.
        H_HT_HI = np.linalg.pinv(H_HI_HH) @ H_HT_HH # Homo matrix that takes P_HT to P_HI (Tinit_hand_o)

        # Transformation matrix
        H_R_V= [[-1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]]
        
        # Find the relative transform and apply it to robot initial position
        H_R_R= (H_R_V@H_HT_HI@np.linalg.inv(H_R_V))[:3,:3] 
        H_R_T= (H_R_V@H_HT_HI@np.linalg.inv(H_R_V))[:3,3]
        H_F_H=np.block([[H_R_R,H_R_T.reshape(3,1)],[np.array([0,0,0]),1]]) # T_base_handpose
        H_RT_RH = H_RI_RH  @ H_F_H # Homo matrix that takes P_RT to P_RH (T)
        self.robot_moving_H = copy(H_RT_RH)


        final_pose_h = self.robot_moving_H
        final_pose_cart = self._homo2cart(final_pose_h)
        if self.use_filter:
            final_pose_cart = self.comp_filter(final_pose_cart)
        
        final_pose = self.cartesian_to_homo(final_pose_cart)

        calculated_twist = self._get_twist(final_pose, current_homo_mat) 

        calculated_twist = moving_average(
            calculated_twist,
            self.moving_Average_queue,
            self.moving_average_limit
        )

        # add parameters to the angular twist
        twist = []
        twist[0:3] = calculated_twist[0:3]
        twist[3:6] = 4*calculated_twist[3:6]
        for axis in range(len(twist[:3])):
            if abs(twist[axis]) < 0.01:
                twist[axis] = 0.0
        
        
        self.robot.move_velocity(twist, 1 / VR_FREQ) # twist control
        self._step_gripper(1 / VR_FREQ)
        self.wheelchair_control()


    # NOTE: This is for debugging should remove this when needed
    def stream(self):
        self.notify_component_start('{} control'.format(self.robot.name))
        print("Start controlling the robot hand using the Oculus Headset.\n")

        # Assume that the initial position is considered initial after 3 seconds of the start
        while True:
            try:
                # rob = self.robot.get_joint_position() 
                if self.robot.get_joint_position() is not None:
                    self.timer.start_loop()

                    val = self._read_tele_gui()
                    if val is not None:
                        self.tele_gui_state = val

                    home_button = self._read_home_gui()
                    # if home_button is not None:
                    #     self.home_state = home_button

                    if self.tele_gui_state == 1:
                        # Transition 0 → 1? reset mapping
                        if self.last_tele_gui_state == 0:
                            moving_hand_frame = self._reset_teleop()
                            self._apply_retargeted_angles(moving_hand_frame, log=False)

                        # normal teleop step
                        moving_hand_frame = self._get_hand_frame()
                        self._apply_retargeted_angles(moving_hand_frame, log=False)
                    else:
                        # tele_gui_state == 0 → teleop is stopped
                        try:
                            home_pressed = False

                            # Interpret home button ONLY when teleop is already stopped
                            if home_button is not None:
                                try:
                                    home_val = int(home_button)
                                except Exception:
                                    home_val = 0

                                # Rising edge detection: 0 → 1 while stopped
                                if home_val == 1 :
                                    home_pressed = True

                            if home_pressed:
                                print("Homing the robot...")
                                self.robot.home()
                                time.sleep(5)
                                # optional: after homing, keep home_state as 1 until GUI sends 0
                            else:
                                # Normal stop: just send zero twist
                                self.robot.move_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1 / VR_FREQ)
                                print("[Operator]: Stop teleoperation")
                        
                        except Exception as e:
                            print(f"[Operator] zero-twist stop error: {e}")

                    self.last_tele_gui_state = self.tele_gui_state
                    self.timer.end_loop()
                else:
                    print(f"[Operator]: cannot get joint position")
            except KeyboardInterrupt as e:
                print(f"[Operator]: Error is {e}")
                break

        self.transformed_arm_keypoint_subscriber.stop()
        print('Stopping the teleoperator!')
       
        


