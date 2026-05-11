#!/usr/bin/env python3
"""
Publish a PoseStamped for the Kinova Gen 3 end‑effector based on TF.

• Looks up the transform from 'robotiq_85_base_link' (tool frame)
  to 'base_link' (robot base).
• Publishes /tf/end_effector2base as geometry_msgs/PoseStamped.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf_transformations import quaternion_from_matrix
import numpy as np


class EePosePublisher(Node):
    def __init__(self):
        super().__init__("ee_pose_pub")

        # Parameters (can be overridden via launch)
        self.declare_parameter("parent_frame", "base_link")
        self.declare_parameter("child_frame",  "robotiq_85_base_link")
        self.declare_parameter("publish_topic", "/tf/end_effector2base")
        self.declare_parameter("rate_hz", 50.0)

        parent = self.get_parameter("parent_frame").get_parameter_value().string_value
        child  = self.get_parameter("child_frame").get_parameter_value().string_value
        topic  = self.get_parameter("publish_topic").get_parameter_value().string_value
        rate   = self.get_parameter("rate_hz").get_parameter_value().double_value

        self.parent_frame = parent
        self.child_frame  = child

        # TF listener
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher
        self.pose_pub = self.create_publisher(PoseStamped, topic, 10)

        # Timer to publish at fixed rate
        self.create_timer(1.0 / rate, self.timer_callback)

        self.get_logger().info(
            f"Publishing PoseStamped ({child} → {parent}) on {topic} at {rate:.0f} Hz"
        )

    # ------------------------------------------------------------------
    def timer_callback(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.child_frame,
                rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            # Transform unavailable this cycle; just skip
            return

        pose = PoseStamped()
        pose.header.stamp    = t.header.stamp
        pose.header.frame_id = self.parent_frame
        pose.pose.position.x = t.transform.translation.x
        pose.pose.position.y = t.transform.translation.y
        pose.pose.position.z = t.transform.translation.z
        pose.pose.orientation = t.transform.rotation

        self.pose_pub.publish(pose)


def main():
    rclpy.init()
    try:
        rclpy.spin(EePosePublisher())
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
