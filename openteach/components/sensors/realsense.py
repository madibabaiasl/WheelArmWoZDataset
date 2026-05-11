import numpy as np
import pyrealsense2 as rs
import time
from openteach.components import Component
from openteach.utils.images import rotate_image, rescale_image
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter
from openteach.constants import *

class RealsenseCamera(Component):
    def __init__(self, stream_configs, cam_serial_num, cam_id, cam_configs, stream_oculus = False):
        # Disabling scientific notations
        np.set_printoptions(suppress=True)
        self.cam_id = cam_id
        self.cam_configs = cam_configs
        self._cam_serial_num = cam_serial_num
        self._stream_configs = stream_configs
        self._stream_oculus = stream_oculus

        # Different publishers to avoid overload
        self.rgb_publisher = ZMQCameraPublisher(
            host = stream_configs['host'],
            port = stream_configs['port']
        )
        
        if self._stream_oculus:
            self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
                host = stream_configs['host'],
                port = stream_configs['port'] + VIZ_PORT_OFFSET
            )

        self.depth_publisher = ZMQCameraPublisher(
            host = stream_configs['host'],
            port = stream_configs['port'] + DEPTH_PORT_OFFSET
        )

        self.timer = FrequencyTimer(CAM_FPS)

        # Starting the realsense pipeline
        self._start_realsense(self._cam_serial_num)

    def _start_realsense(self, cam_serial_num):
        config = rs.config()
        self.pipeline = rs.pipeline()
        config.enable_device(cam_serial_num)
        
        time.sleep(0.1 * (self.cam_id or 0))
        _set_depth_preset_before_start(cam_serial_num, name_hint="high accuracy")


        # Enabling camera streams
        config.enable_stream(
            rs.stream.color, 
            self.cam_configs.width, 
            self.cam_configs.height, 
            rs.format.bgr8, 
            self.cam_configs.fps
        )
        config.enable_stream(
            rs.stream.depth, 
            self.cam_configs.width, 
            self.cam_configs.height, 
            rs.format.z16, 
            self.cam_configs.fps
        )

        # Starting the pipeline
        cfg = self.pipeline.start(config)
        device = cfg.get_device()

        # Setting the depth mode to high accuracy mode
        depth_sensor = device.first_depth_sensor()
        print(f"Serial num is {cam_serial_num}")
        # depth_sensor.set_option(rs.option.visual_preset, self.cam_configs.processing_preset)
        self.realsense = self.pipeline

        # Obtaining the color intrinsics matrix for aligning the color and depth images
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        self.intrinsics_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy], 
            [0, 0, 1]
        ])

        # Align function - aligns other frames with the color frame
        self.align = rs.align(rs.stream.color)

    def get_rgb_depth_images(self):
        frames = None

        while frames is None:
            # Obtaining and aligning the frames
            frames = self.realsense.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Getting the images from the frames
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image, frames.get_timestamp()

    def stream(self):
        # Starting the realsense stream
        self.notify_component_start('realsense')
        print(f"Started the Realsense pipeline for camera: {self._cam_serial_num}!")
        print("Starting stream on {}:{}...\n".format(self._stream_configs['host'], self._stream_configs['port']))
        
        if self._stream_oculus:
            print('Starting oculus stream on port: {}\n'.format(self._stream_configs['port'] + VIZ_PORT_OFFSET))

        while True:
            #try:
                self.timer.start_loop()
                color_image, depth_image, timestamp = self.get_rgb_depth_images()

                color_image = rotate_image(color_image, self.cam_configs.rotation_angle)
                depth_image = rotate_image(depth_image, self.cam_configs.rotation_angle)

                # Publishing the rgb images
                self.rgb_publisher.pub_rgb_image(color_image, timestamp)
                # TODO - move the oculus publisher to a separate process - this cycle works at 40 FPS
                if self._stream_oculus:
                    self.rgb_viz_publisher.send_image(rescale_image(color_image, 2)) # 640 * 360->1280 * 720

                # Publishing the depth images
                self.depth_publisher.pub_depth_image(depth_image, timestamp)
                self.depth_publisher.pub_intrinsics(self.intrinsics_matrix) # Publishing inrinsics along with the depth publisher

                self.timer.end_loop()
            # except KeyboardInterrupt:
            #     break
        
        print('Shutting down realsense pipeline for camera {}.'.format(self.cam_id))
        self.rgb_publisher.stop()
        if self._stream_oculus:
            self.rgb_viz_publisher.stop()
        self.depth_publisher.stop()
        self.pipeline.stop()


#----------this is to fix the issue cause by sending visual preset after starting the pipeline----------
import pyrealsense2 as rs
import time

def _set_depth_preset_before_start(serial: str, name_hint: str = "high accuracy"):
    """
    Set depth visual_preset by human-readable name *before* starting the pipeline.
    Falls back to Default if name not found.
    """
    ctx = rs.context()
    dev = None
    for d in ctx.query_devices():
        if d.get_info(rs.camera_info.serial_number) == serial:
            dev = d
            break
    if not dev:
        print(f"[preset] Device with serial {serial} not found; leaving Default.")
        return

    depth_sensor = dev.first_depth_sensor()
    if not depth_sensor.supports(rs.option.visual_preset):
        print("[preset] visual_preset not supported; leaving Default.")
        return

    rng = depth_sensor.get_option_range(rs.option.visual_preset)

    # Try to resolve the integer value by name (if firmware exposes descriptions)
    target_val = None
    for v in range(int(rng.min), int(rng.max) + 1):
        try:
            desc = depth_sensor.get_option_value_description(rs.option.visual_preset, float(v)).lower()
            if name_hint.lower() in desc:  # e.g., "high accuracy"
                target_val = float(v)
                break
        except Exception:
            pass

    if target_val is None:
        # Common fallback: many D415 firmwares use 1 for High Accuracy
        # but we clamp to the valid range just in case.
        target_val = max(rng.min, min(1.0, rng.max))

    try:
        depth_sensor.set_option(rs.option.visual_preset, target_val)
        print(f"[preset] Set visual_preset to {target_val} for {serial}")
        # tiny settle delay is harmless
        time.sleep(0.05)
    except Exception as e:
        print(f"[preset] Failed to set visual_preset before start: {e}; leaving Default.")
