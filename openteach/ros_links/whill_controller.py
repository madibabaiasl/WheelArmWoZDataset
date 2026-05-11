#!/usr/bin/env python3
"""
Simplest WHILL joystick publisher for ROS 2.
Publishes sensor_msgs/Joy to /whill/controller/joy.
Axes[0] = angular (turn), Axes[1] = linear (forward/back).
Range is typically -1.0 to +1.0.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

class WhillController(Node):
    def __init__(self):
        super().__init__('whill_controller')
        self.publisher_ = self.create_publisher(Joy, '/whill/controller/joy', 10)
        print(f"[Whill Controller] Publisher created")
        self.latest_x = 0.0
        self.latest_y = 0.0
        # 20 Hz timer
        self.timer = self.create_timer(0.05, self.publish_loop)

    def move_wheelchair(self, x: float, y: float):
        # only update
        self.latest_x = max(-0.05, min(0.05, x))
        self.latest_y = max(-0.05, min(0.05, y))

    def publish_loop(self):
        msg = Joy()
        print(f"[Whill Controller] x and y is {[self.latest_x, self.latest_y]}")
        msg.axes = [self.latest_x, self.latest_y]
        self.publisher_.publish(msg)
    

