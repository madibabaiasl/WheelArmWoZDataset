#!/usr/bin/env python3
from __future__ import annotations
import math
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.duration import Duration
from rclpy.clock import Clock, ClockType

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, PoseStamped
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory, GripperCommand
from builtin_interfaces.msg import Duration as MsgDuration
from moveit_msgs.srv import GetPositionIK
import threading, time
from rclpy.action.client import ClientGoalHandle

class KinovaGen3Controller(Node):

    JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]  # 6‑DOF arm
    IK_SERVICE = "/compute_ik"                         
    MOVE_GROUP = "manipulator"                                 
    POSE_FRAME = "base_link"                           

    def __init__(self, node_name: str = "gen3_node") -> None:
        super().__init__(node_name)

        # Internal state
        self.current_joint_state: JointState | None = None
        self._twist_ok: bool | None = None 
        self._joint_ok: bool | None = None

        # Subscribers and Publishers
        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            qos_profile_sensor_data,
        )
        self.twist_pub = self.create_publisher(
            Twist, "/twist_controller/commands", 1
        )

        # clients
        self.joint_traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )
        self._cm_list   = self.create_client(
                              ListControllers,
                              "/controller_manager/list_controllers")
        self._cm_switch = self.create_client(
                              SwitchController,
                              "/controller_manager/switch_controller")
        self.ik_srv = self.create_client(GetPositionIK, self.IK_SERVICE)
        self.gripper_client = ActionClient(
            self, GripperCommand, "/robotiq_gripper_controller/gripper_cmd"
        )

        # Parameters
        self.gripper_min = 0.0     # fully open
        self.gripper_max = 0.8     # fully close 
        self._gripper_last_send_ts = 0.0
        self._gripper_min_interval = 0.10   


        # Wait for endpoints
        self._wait_for_servers()

        self._gripper_lock = threading.Lock()
        self._gripper_active_goal: ClientGoalHandle | None = None

    def _joint_state_cb(self, msg: JointState) -> None:
        self.current_joint_state = msg

    def _wait_for_servers(self) -> None:
        if not self.joint_traj_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Joint trajectory action server not available")
        if not self.ik_srv.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("/compute_ik service not available — Cartesian pose moves disabled")
        if not self._cm_list.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("controller_manager/list_controllers service not available")
        if not self._cm_switch.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("controller_manager/switch_controller service not available")
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Gripper action server not available yet")

    # Joint Control
    def joint_movement(self, radians: List[float]) -> None:
        """Move the arm to specified joint angles (in radians)."""
        if not self._ensure_active_controller("joint_trajectory_controller"):
            self.get_logger().error("Joint trajectory controller not active")
            return

        if len(radians) != 6:
            raise ValueError("Expected 6 joint values for Gen 3 arm")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.JOINT_NAMES
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        
        duration = 7.0  # seconds
        pt = JointTrajectoryPoint()
        pt.positions = list(radians)
        pt.time_from_start = MsgDuration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        goal.trajectory.points.append(pt)

        self.get_logger().info("Sending joint trajectory goal…")
        future = self.joint_traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory rejected")
            return

        # wait for EXECUTION result
        res_future = goal_handle.get_result_async()
        self.get_logger().info("Waiting for trajectory execution to finish…")
        rclpy.spin_until_future_complete(self, res_future)     # blocks until done
        result = res_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(f"Trajectory failed ({result.error_code})")
            return
    
    # Cartesian Control
    def cartesian_movement(self, cartesian_pose: List[float] | None = None, ) -> None:
        if cartesian_pose is None:
            raise ValueError("No Cartesian pose provided for movement")
        if len(cartesian_pose) != 7:
            raise ValueError("Expected 7 values for Cartesian pose (x, y, z, qx, qy, qz, qw)")

        if not self.ik_srv.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("/compute_ik service unavailable — skipping pose move")
            return

        # Build IK request
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.MOVE_GROUP
        req.ik_request.pose_stamped = PoseStamped()
        req.ik_request.pose_stamped.header.frame_id = self.POSE_FRAME
        req.ik_request.pose_stamped.pose.position.x = cartesian_pose[0]
        req.ik_request.pose_stamped.pose.position.y = cartesian_pose[1]
        req.ik_request.pose_stamped.pose.position.z = cartesian_pose[2]
        req.ik_request.pose_stamped.pose.orientation.x = cartesian_pose[3]
        req.ik_request.pose_stamped.pose.orientation.y = cartesian_pose[4]
        req.ik_request.pose_stamped.pose.orientation.z = cartesian_pose[5]
        req.ik_request.pose_stamped.pose.orientation.w = cartesian_pose[6]

        self.get_logger().info("Calling /compute_ik …")
        future = self.ik_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        sol = future.result()
        if sol is None or sol.error_code.val != sol.error_code.SUCCESS:
            self.get_logger().error("IK failed — aborting pose move")
            return

        # Execute as a trajectory
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = sol.solution.joint_state.name
        pt = JointTrajectoryPoint()
        pt.positions = list(sol.solution.joint_state.position)
        goal.trajectory.points = [pt]

        self.get_logger().info("Executing IK result via JTC…")
        fut = self.joint_traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        if fut.result() is None:
            self.get_logger().error("Failed to send JTC goal")

    # Twist Control
    def publish_cartesian_velocity(self, twist_vec: List[float], duration: float = 1.0) -> None:
        if not self._ensure_active_controller("twist_controller"):
            self.get_logger().error("Twist controller not active")
            return

        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = twist_vec[:3]
        twist.angular.x, twist.angular.y, twist.angular.z = twist_vec[3:]

        hz = 100.0  # Hz for streaming
        period = duration / hz
        deadline = time.time() + duration
        while time.time() < deadline:
            self.twist_pub.publish(twist)
            time.sleep(period)

    # Gripper Control
    def gripper_movement(self, position_m: float, wait=True):
        with self._gripper_lock:
            # Debounce rapid repeats
            now = time.time()
            if now - self._gripper_last_send_ts < self._gripper_min_interval:
                time.sleep(self._gripper_min_interval - (now - self._gripper_last_send_ts))

            if not self.gripper_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("Gripper action server unavailable")
                return None

            # Clamp & build goal
            pos = float(max(self.gripper_min, min(self.gripper_max, position_m)))
            goal = GripperCommand.Goal()
            goal.command.position = pos

            self.get_logger().info(f"Gripper → pos={pos:.3f}")
            send_fut = self.gripper_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_fut)
            goal_handle = send_fut.result()
            if not goal_handle or not goal_handle.accepted:
                self.get_logger().error("Gripper goal rejected")
                return None

            self._gripper_active_goal = goal_handle
            self._gripper_last_send_ts = time.time()

            if not wait:
                return goal_handle

            res_fut = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, res_fut)
            result = res_fut.result().result if res_fut.result() else None
            if result is not None:
                self.get_logger().info(
                    f"Gripper result: reached={result.reached_goal}, "
                    f"stalled={result.stalled}, pos={result.position:.3f}, effort={result.effort:.1f}"
                )
            
            time.sleep(0.05)
            return result

    # Helper Functions
    def _call_list_controllers(self, timeout_sec: float = 5.0):
        req  = ListControllers.Request()
        fut  = self._cm_list.call_async(req)

        deadline = self.get_clock().now() + Duration(seconds=timeout_sec)

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)   # <-- pumps callbacks
            if fut.done():
                return fut.result()                  # success
            if self.get_clock().now() > deadline:
                break

        self.get_logger().error("controller_manager list timeout")
        return None
    
    def _ensure_active_controller(self, desired: str) -> bool:
        OTHER = {
            "joint_trajectory_controller": "twist_controller",
            "twist_controller":            "joint_trajectory_controller",
        }.get(desired)

        if OTHER is None:
            self.get_logger().error(f"Unknown controller “{desired}”")
            return False

        #  Ask controller_manager for current states
        resp = self._call_list_controllers(timeout_sec=5.0)
        if resp is None:                                  
            return False
        states = {c.name: c.state for c in resp.controller}
        desired_active = states.get(desired) == "active"
        other_active   = states.get(OTHER)   == "active"

        # Fast detection if the desired already active
        if desired_active:
            return True            # leave everything else untouched

        # Build a minimal switch request 
        req = SwitchController.Request()
        req.activate_controllers   = [desired]             # start the one we need
        if other_active:                                   # stop ONLY its twin
            req.deactivate_controllers.append(OTHER)
        req.strictness    = SwitchController.Request.STRICT
        req.activate_asap = True
        req.timeout       = MsgDuration(sec=2, nanosec=0)     

        # Call the service and wait
        self._cm_switch.call_async(req)          
        deadline = self.get_clock().now() + Duration(seconds=20)
        while self.get_clock().now() < deadline:
            resp = self._call_list_controllers(timeout_sec=2.0)
            if resp:
                states = {c.name: c.state for c in resp.controller}
                if states.get(desired) == "active" and states.get(OTHER) != "active":
                    self.get_logger().info(f"{desired} ACTIVE (switch confirmed)")
                    return True
        self.get_logger().error("Controller never reached desired state")
        return False