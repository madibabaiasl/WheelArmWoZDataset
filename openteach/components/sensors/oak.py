# openteach/components/sensors/oak.py

import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, CameraInfo

from openteach.components import Component
from openteach.utils.images import rotate_image, rescale_image
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter
from openteach.constants import CAM_FPS, VIZ_PORT_OFFSET, DEPTH_PORT_OFFSET


class OakRosBridge(Node):
    """
    ROS -> ZMQ bridge for OAK RGB + Depth streams.

    Subscribes to:
      - /oak/rgb/image_raw/compressed        (sensor_msgs/CompressedImage)
      - /oak/stereo/image_raw/compressedDepth (sensor_msgs/CompressedImage)
      - /oak/stereo/camera_info              (sensor_msgs/CameraInfo)

    Publishes via ZMQ:
      - RGB frames on <port>
      - (optionally) Oculus viz stream on <port + VIZ_PORT_OFFSET>
      - Depth frames on <port + DEPTH_PORT_OFFSET>
      - Depth intrinsics once available
    """

    def __init__(
        self,
        node_name,
        cam_configs,
        stream_configs,
        stream_oculus,
        rgb_publisher,
        depth_publisher,
        rgb_viz_publisher=None,
    ):
        super().__init__(node_name)

        self.cam_configs = cam_configs
        self._stream_configs = stream_configs
        self._stream_oculus = stream_oculus
        self.rgb_publisher = rgb_publisher
        self.rgb_viz_publisher = rgb_viz_publisher
        self.depth_publisher = depth_publisher

        self.timer = FrequencyTimer(CAM_FPS)

        # Intrinsics (3x3) like in Realsense
        self.intrinsics_matrix = None
        self._intrinsics_published = False

        # Subscribe to the compressed RGB topic from depthai-ros
        self.create_subscription(
            CompressedImage,
            "/oak/rgb/image_raw/compressed",
            self._rgb_cb,
            10,
        )

        # Subscribe to compressed depth
        self.create_subscription(
            CompressedImage,
            "/oak/stereo/image_raw/compressedDepth",
            self._depth_cb,
            10,
        )

        # Subscribe to camera info (stereo/depth) for intrinsics
        self.create_subscription(
            CameraInfo,
            "/oak/stereo/camera_info",
            self._camera_info_cb,
            1,
        )

        self.get_logger().info(
            f"OAK bridge subscribing to:\n"
            f"  - /oak/rgb/image_raw/compressed\n"
            f"  - /oak/stereo/image_raw/compressedDepth\n"
            f"  - /oak/stereo/camera_info\n"
            f"and bridging to ZMQ {stream_configs['host']}:{stream_configs['port']}"
        )

    # ---------- Callbacks ----------

    def _rgb_cb(self, msg: CompressedImage):
        """Callback for incoming compressed RGB frames."""
        self.timer.start_loop()

        # Decode JPEG/PNG from ROS CompressedImage
        np_arr = np.frombuffer(msg.data, np.uint8)
        rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR uint8

        if rgb is None:
            self.get_logger().warning("Failed to decode OAK compressed RGB image")
            self.timer.end_loop()
            return

        # Apply rotation from your config (same as Realsense path)
        rgb = rotate_image(rgb, self.cam_configs.rotation_angle)

        # Use ROS timestamp if valid, otherwise wall time (ms)
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            ts_ms = msg.header.stamp.sec * 1000.0 + msg.header.stamp.nanosec / 1e6 # million seconds
        else:
            ts_ms = time.time() * 1000.0

        # Main RGB stream (for Open-Teach)
        self.rgb_publisher.pub_rgb_image(rgb, ts_ms)

        # Optional Oculus viz stream (upscaled)
        if self._stream_oculus and self.rgb_viz_publisher is not None:
            self.rgb_viz_publisher.send_image(rescale_image(rgb, 2))

        self.timer.end_loop()

    def _depth_cb(self, msg: CompressedImage):
        """Callback for incoming compressed depth frames."""
        # Decode PNG depth from compressedDepth
        depth_header_size = 12

        # Optional: sanity check – see if the first 12 bytes contain 'PNG'
        # If they do, header_size should be 0 instead.
        if b'PNG' in msg.data[:12]:
            depth_header_size = 0

        raw_png = msg.data[depth_header_size:]

        np_arr = np.frombuffer(raw_png, np.uint8)
        depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # print(f"[OAK Bridge]: depth is {msg.data}")
        # depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)  # keep 16U / 32F

        if depth is None:
            self.get_logger().warning("Failed to decode OAK compressed depth image")
            return

        # Rotate depth to match RGB orientation
        depth = rotate_image(depth, self.cam_configs.rotation_angle)

        # Timestamp: use ROS header when available
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            ts_ms = msg.header.stamp.sec * 1000.0 + msg.header.stamp.nanosec / 1e6
        else:
            ts_ms = time.time() * 1000.0

        # Publish depth frame
        self.depth_publisher.pub_depth_image(depth, ts_ms)

        # Publish intrinsics once we have them (mirrors Realsense behavior)
        if self.intrinsics_matrix is not None and not self._intrinsics_published:
            self.depth_publisher.pub_intrinsics(self.intrinsics_matrix)
            self._intrinsics_published = True

    def _camera_info_cb(self, msg: CameraInfo):
        """Build intrinsics matrix from CameraInfo (once)."""
        if self.intrinsics_matrix is not None:
            return

        # msg.K is row-major 3x3 (fx, 0, cx, 0, fy, cy, 0, 0, 1)
        K = msg.k  # or msg.K depending on ROS2 version; using 'k' here
        if len(K) != 9:
            self.get_logger().warning("CameraInfo has invalid K size; skipping intrinsics.")
            return

        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        self.intrinsics_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.get_logger().info(f"OAK intrinsics matrix set to:\n{self.intrinsics_matrix}")


class OakCamera(Component):
    """
    ROS-backed replacement for RealsenseCamera that publishes:
      - RGB frames on <port>
      - (optionally) Oculus viz stream on <port + VIZ_PORT_OFFSET>
      - Depth frames on <port + DEPTH_PORT_OFFSET>
      - Depth intrinsics (from /oak/stereo/camera_info)
    """

    def __init__(self, stream_configs, mxid, cam_id, cam_configs, stream_oculus=False):
        # mxid is unused now but kept in the signature for compatibility
        np.set_printoptions(suppress=True)
        self.cam_id = cam_id
        self.cam_configs = cam_configs
        self._mxid = mxid
        self._stream_configs = stream_configs
        self._stream_oculus = stream_oculus

        # ZMQ publishers (same interface as RealsenseCamera)
        self.rgb_publisher = ZMQCameraPublisher(
            host=stream_configs["host"],
            port=stream_configs["port"],
        )

        if self._stream_oculus:
            self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
                host=stream_configs["host"],
                port=stream_configs["port"] + VIZ_PORT_OFFSET,
            )
        else:
            self.rgb_viz_publisher = None

        self.depth_publisher = ZMQCameraPublisher(
            host=stream_configs["host"],
            port=stream_configs["port"] + DEPTH_PORT_OFFSET,
        )

        print()

    def stream(self):
        """Start ROS <-> ZMQ bridge instead of DepthAI pipeline."""
        self.notify_component_start("oak")
        print(
            f"Starting OAK ROS bridge on "
            f"{self._stream_configs['host']}:{self._stream_configs['port']}..."
        )
        if self._stream_oculus:
            print(
                f"Starting OAK oculus stream on port: "
                f"{self._stream_configs['port'] + VIZ_PORT_OFFSET}"
            )

        rclpy.init(args=None)

        bridge_node = OakRosBridge(
            node_name=f"oak_ros_bridge_{self.cam_id}",
            cam_configs=self.cam_configs,
            stream_configs=self._stream_configs,
            stream_oculus=self._stream_oculus,
            rgb_publisher=self.rgb_publisher,
            depth_publisher=self.depth_publisher,
            rgb_viz_publisher=self.rgb_viz_publisher,
        )

        try:
            rclpy.spin(bridge_node)
        finally:
            print(f"Shutting down OAK bridge for camera {self.cam_id}.")
            bridge_node.destroy_node()
            rclpy.shutdown()

            # Stop ZMQ publishers like Realsense does
            self.rgb_publisher.stop()
            if self._stream_oculus and self.rgb_viz_publisher is not None:
                self.rgb_viz_publisher.stop()
            self.depth_publisher.stop()
