#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage

import argparse, time, json
import zmq
import numpy as np
from cv_bridge import CvBridge
import cv2

from openteach.constants import *

# Port Configuration
cam_port_offset = 10005
info_port_offset = 20000
ros_arm_cam_number =  2
RGB_PORT         = cam_port_offset + ros_arm_cam_number                # e.g., 10007
RGB_JPEG_PORT    = cam_port_offset + ros_arm_cam_number + 1
JPEG_QUALITY     = 80
DEPTH_PORT       = cam_port_offset + ros_arm_cam_number + DEPTH_PORT_OFFSET  # e.g., 11007
RGB_INFO_PORT    = info_port_offset + ros_arm_cam_number               # e.g., 20002
DEPTH_INFO_PORT  = info_port_offset + ros_arm_cam_number + DEPTH_PORT_OFFSET 


class RosToZmq(Node):
    def __init__(self, host, rgb_port, depth_port, rgb_info_port, depth_info_port,
                 rgb_topic, depth_topic, rgb_info_topic, depth_info_topic, rgb_jpeg_port=None, jpeg_quality=80, republish_info_hz=1.0):
        super().__init__('ros_cam_to_zmq')

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        self.bridge = CvBridge() if CvBridge else None

        # Image PUBs (raw)
        ctx = zmq.Context.instance()
        self.sock_rgb = ctx.socket(zmq.PUB);   self.sock_rgb.setsockopt(zmq.SNDHWM, 10);   self.sock_rgb.bind(f"tcp://{host}:{rgb_port}")
        print(f"RGB ZMQ PUB bound to tcp://{host}:{rgb_port}")
        self.sock_d   = ctx.socket(zmq.PUB);   self.sock_d.setsockopt(zmq.SNDHWM, 10);     self.sock_d.bind(f"tcp://{host}:{depth_port}")
        print(f"Depth ZMQ PUB bound to tcp://{host}:{depth_port}")

        # Image Pubs (encoded)
        self.sock_rgb_jpeg = None
        self.jpeg_quality = int(jpeg_quality)
        self.sock_rgb_jpeg = ctx.socket(zmq.PUB)
        self.sock_rgb_jpeg.setsockopt(zmq.SNDHWM, 10)
        self.sock_rgb_jpeg.bind(f"tcp://{host}:{int(rgb_jpeg_port)}")
        print(f"RGB JPG PUB bound to tcp://{host}:{rgb_jpeg_port} (quality={self.jpeg_quality})")

        # CameraInfo PUBs
        self.sock_rgb_info = ctx.socket(zmq.PUB);  self.sock_rgb_info.setsockopt(zmq.SNDHWM, 1);  self.sock_rgb_info.bind(f"tcp://{host}:{rgb_info_port}")
        print(f"RGB CameraInfo ZMQ PUB bound to tcp://{host}:{rgb_info_port}")
        self.sock_d_info   = ctx.socket(zmq.PUB);  self.sock_d_info.setsockopt(zmq.SNDHWM, 1);    self.sock_d_info.bind(f"tcp://{host}:{depth_info_port}")
        print(f"Depth CameraInfo ZMQ PUB bound to tcp://{host}:{depth_info_port}")

        # Subscriptions
        self.sub_rgb  = self.create_subscription(CompressedImage,      rgb_topic,        self.cb_rgb_compressed,   qos)
        self.sub_d    = self.create_subscription(Image,      depth_topic,      self.cb_depth, qos)
        self.sub_ci_r = self.create_subscription(CameraInfo, rgb_info_topic,   self.cb_rgb_info, qos)
        self.sub_ci_d = self.create_subscription(CameraInfo, depth_info_topic, self.cb_depth_info, qos)

        # store latest info & rebroadcast periodically so late subscribers catch it
        self._rgb_info_json   = None
        self._depth_info_json = None
        self._last_info_pub   = 0.0
        self._republish_interval = 1.0 / max(republish_info_hz, 0.1)

        self.create_timer( self._republish_interval, self._tick_republish )

    @staticmethod
    def _ts_pair(msg_header):
        return msg_header.stamp.sec * 1_000_000_000 + msg_header.stamp.nanosec, time.monotonic_ns() # nanoseconds

    def _tick_republish(self):
        if self._rgb_info_json:
            self.sock_rgb_info.send(self._rgb_info_json)
        if self._depth_info_json:
            self.sock_d_info.send(self._depth_info_json)

    def _pack_cam_info(self, msg: CameraInfo):
        ts_ros, ts_host = self._ts_pair(msg.header)
        d = dict(
            width=msg.width, height=msg.height,
            distortion_model=msg.distortion_model,
            D=list(msg.d), K=list(msg.k), R=list(msg.r), P=list(msg.p),
            binning_x=msg.binning_x, binning_y=msg.binning_y,
            roi=dict(x_offset=msg.roi.x_offset, y_offset=msg.roi.y_offset,
                     height=msg.roi.height, width=msg.roi.width, do_rectify=msg.roi.do_rectify),
            header=dict(frame_id=msg.header.frame_id, stamp_ns=ts_ros),
            host_stamp_ns=ts_host
        )
        return json.dumps(d).encode('utf-8')

    def cb_rgb_info(self, msg: CameraInfo):
        self._rgb_info_json = self._pack_cam_info(msg)
        self.sock_rgb_info.send(self._rgb_info_json)

    def cb_depth_info(self, msg: CameraInfo):

        self._depth_info_json = self._pack_cam_info(msg)
        self.sock_d_info.send(self._depth_info_json)

    def cb_rgb(self, msg: Image):
        print(f"RGB image received {msg}")
        ts_ros, ts_host = self._ts_pair(msg.header)
        enc = msg.encoding
         
        if self.bridge is not None and enc not in ('bgr8', 'rgb8'):
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            enc = 'bgr8'
            payload = memoryview(img.data)
            h, w, step = img.shape[0], img.shape[1], img.shape[1]*3
        else:
            payload = memoryview(msg.data)
            h, w, step = msg.height, msg.width, msg.step

        header = json.dumps(dict(enc=enc, h=h, w=w, step=step, ts_ros=ts_ros, ts_host=ts_host)).encode('ascii')
        self.sock_rgb.send_multipart([b'RGB', header, payload], copy=False)

        if self.sock_rgb_jpeg is not None:
            if self.bridge is None and enc in ('bgr8','rgb8'):
                arr = np.frombuffer(payload, dtype=np.uint8),reshape(h, step//3, 3)
                if enc == 'rgb8':
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                try:
                    arr
                except NameError:
                    np.frombuffer(payload, dtype=np.uint8).reshape(h, step//3, 3)[:, :w, :]
                    if enc == 'rgb8':
                        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                print(f"[RosToZmq] ok is {ok}")
                jpg = buf.tobytes()
                header_jpeg = json.dumps(dict(enc='jpeg', h=h, w=w, ts_ros=ts_ros, ts_host=ts_host)).encode('ascii')
                self.sock_rgb_jpeg.send_multipart([b'RGB_JPEG', header_jpeg, jpg], copy=False)
        else:
            print(f"[RosToZmq]: the jpeg socket is {self.sock_rgb_jpeg}")

    def cb_rgb_compressed(self, msg):
        import re, json
        import numpy as np
        import cv2

        try:
            ts_ros, ts_host = self._ts_pair(msg.header)
        except Exception as e:
            self.get_logger().warn(f"timestamp extraction failed: {e}")
            return

        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)  
            if img is None:
                raise RuntimeError(f"cv2.imdecode returned None (format='{msg.format}')")

            # normalize to 3-channel BGR
            if img.ndim == 2:  # grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:  # BGRA -> BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            h, w = img.shape[:2]
            step = w * 3

            header_raw = json.dumps(
                dict(enc="bgr8", h=h, w=w, step=step, ts_ros=ts_ros, ts_host=ts_host)
            ).encode("ascii")

            # send RAW (uncompressed BGR) frame
            if getattr(self, "sock_rgb", None) is not None:
                self.sock_rgb.send_multipart([b"RGB", header_raw, memoryview(img.data)], copy=False)
        except Exception as e:
            self.get_logger().warn(f"Compressed decode->BGR failed: {e}")
            return

        # JPEG socket
        sock_jpeg = getattr(self, "sock_rgb_jpeg", None)
        if sock_jpeg is None:
            return

        fmt = (msg.format or "").lower()
        is_jpeg = bool(re.search(r"\b(jpe?g)\b", fmt))

        try:
            if is_jpeg:
                # Pass through original JPEG bytes
                header_jpeg = json.dumps(
                    dict(enc="jpeg", h=h, w=w, ts_ros=ts_ros, ts_host=ts_host)
                ).encode("ascii")
                sock_jpeg.send_multipart([b"RGB_JPEG", header_jpeg, memoryview(msg.data)], copy=False)
            else:
                # Re-encode (handles PNG or other formats)
                quality = int(getattr(self, "jpeg_quality", 85))
                ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if not ok:
                    raise RuntimeError("cv2.imencode('.jpg', ...) failed")
                header_jpeg = json.dumps(
                    dict(enc="jpeg", h=h, w=w, ts_ros=ts_ros, ts_host=ts_host)
                ).encode("ascii")
                sock_jpeg.send_multipart([b"RGB_JPEG", header_jpeg, enc.tobytes()], copy=False)
        except Exception as e:
            self.get_logger().warn(f"JPEG publish failed: {e}")

    def cb_depth(self, msg: Image):
        # print(f"Depth image received {msg}")
        ts_ros, ts_host = self._ts_pair(msg.header)
        header = json.dumps(dict(enc=msg.encoding, h=msg.height, w=msg.width, step=msg.step,
                                 ts_ros=ts_ros, ts_host=ts_host)).encode('ascii')
        self.sock_d.send_multipart([b'Depth', header, memoryview(msg.data)], copy=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--rgb-port', type=int, default=RGB_PORT)
    ap.add_argument('--depth-port', type=int, default=DEPTH_PORT)
    ap.add_argument('--rgb-info-port', type=int, default=RGB_INFO_PORT)
    ap.add_argument('--depth-info-port', type=int, default=DEPTH_INFO_PORT)
    ap.add_argument('--rgb-jpeg-port', type=int, default=RGB_JPEG_PORT)
    ap.add_argument('--jpeg-quality', type=int, default=JPEG_QUALITY)
    ap.add_argument('--rgb-topic', default='/camera/color/image_raw/compressed')
    ap.add_argument('--depth-topic', default='/camera/depth/image_raw')
    ap.add_argument('--rgb-info-topic', default='/camera/color/camera_info')
    ap.add_argument('--depth-info-topic', default='/camera/depth/camera_info')
    ap.add_argument('--republish-info-hz', type=float, default=1.0, help='rebroadcast camera_info at this rate')
    args = ap.parse_args()

    rclpy.init()
    node = RosToZmq(args.host, args.rgb_port, args.depth_port, args.rgb_info_port, args.depth_info_port,
                    args.rgb_topic, args.depth_topic, args.rgb_info_topic, args.depth_info_topic,
                    args.rgb_jpeg_port, args.jpeg_quality,
                    republish_info_hz=args.republish_info_hz)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
