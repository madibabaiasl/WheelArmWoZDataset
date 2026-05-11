# import numpy as np
# import pyrealsense2 as rs
# import time
# from openteach.components import Component
# from openteach.utils.images import rotate_image, rescale_image
# from openteach.utils.timer import FrequencyTimer
# from openteach.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter
# from openteach.constants import *

# class RealsenseCamera(Component):
#     def __init__(self, stream_configs, cam_serial_num, cam_id, cam_configs, stream_oculus = False):
#         # Disabling scientific notations
#         np.set_printoptions(suppress=True)
#         self.cam_id = cam_id
#         self.cam_configs = cam_configs
#         self._cam_serial_num = cam_serial_num
#         self._stream_configs = stream_configs
#         self._stream_oculus = stream_oculus

#         # Different publishers to avoid overload
#         self.rgb_publisher = ZMQCameraPublisher(
#             host = stream_configs['host'],
#             port = stream_configs['port']
#         )
        
#         # if self._stream_oculus:
#         #     self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
#         #         host = stream_configs['host'],
#         #         port = stream_configs['port'] + VIZ_PORT_OFFSET
#         #     )

#         self.depth_publisher = ZMQCameraPublisher(
#             host = stream_configs['host'],
#             port = stream_configs['port'] + DEPTH_PORT_OFFSET
#         )

#         self.timer = FrequencyTimer(CAM_FPS)

#         # Starting the realsense pipeline
#         self._start_realsense(self._cam_serial_num)

#         #----GPU ENCODING FOR HEADSET CAM----#
#         self._use_gpu_headset = True
#         # self._headset_ip    = getattr(self.cam_configs, "headset_ip",   "10.178.46.240")
#         # self._headset_port  = getattr(self.cam_configs, "headset_port", 5004)
#         self._headset_br    = getattr(self.cam_configs, "bitrate_kbps", 4000)
#         self._nvenc = None

#         if self._stream_oculus and self._use_gpu_headset:
#             # Start NVENC pipeline that accepts raw frames via stdin
#             nvenc_host = self._stream_configs['host']                      # reuse existing host
#             nvenc_port = self._stream_configs['port'] + VIZ_PORT_OFFSET
#             self._nvenc = _NvencRtpSenderPipe(
#                 host=nvenc_host,
#                 port=snvenc_port,
#                 w=self.cam_configs.width,
#                 h=self.cam_configs.height,
#                 fps=self.cam_configs.fps,
#                 bitrate_kbps=self._headset_br
#             )
#         else:
#             if self._stream_oculus:
#                 self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
#                     host=self._stream_configs['host'],
#                     port=self._stream_configs['port'] + VIZ_PORT_OFFSET
#                 )


#     def _start_realsense(self, cam_serial_num):
#         config = rs.config()
#         self.pipeline = rs.pipeline()
#         config.enable_device(cam_serial_num)

#         # Enabling camera streams
#         config.enable_stream(
#             rs.stream.color, 
#             self.cam_configs.width, 
#             self.cam_configs.height, 
#             rs.format.bgr8, 
#             self.cam_configs.fps
#         )
#         config.enable_stream(
#             rs.stream.depth, 
#             self.cam_configs.width, 
#             self.cam_configs.height, 
#             rs.format.z16, 
#             self.cam_configs.fps
#         )

#         # Starting the pipeline
#         cfg = self.pipeline.start(config)
#         device = cfg.get_device()

#         # Setting the depth mode to high accuracy mode
#         depth_sensor = device.first_depth_sensor()
#         time.sleep(0.5)                       # let firmware finish startup

#         for _ in range(3):                    # retry up to 3 times
#             try:
#                 depth_sensor.set_option(
#                     rs.option.visual_preset,
#                     self.cam_configs.processing_preset
#                 )
#                 break                         # success → exit loop
#             except RuntimeError as e:
#                 # only retry if the sensor reports "busy"
#                 if "busy" in str(e).lower():
#                     time.sleep(0.5)
#                     continue
#                 raise             
#         # depth_sensor.set_option(rs.option.visual_preset, self.cam_configs.processing_preset)
#         self.realsense = self.pipeline

#         # Obtaining the color intrinsics matrix for aligning the color and depth images
#         profile = self.pipeline.get_active_profile()
#         color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
#         intrinsics = color_profile.get_intrinsics()
#         self.intrinsics_matrix = np.array([
#             [intrinsics.fx, 0, intrinsics.ppx],
#             [0, intrinsics.fy, intrinsics.ppy], 
#             [0, 0, 1]
#         ])

#         # Align function - aligns other frames with the color frame
#         self.align = rs.align(rs.stream.color)

#     def get_rgb_depth_images(self):
#         frames = None

#         while frames is None:
#             # Obtaining and aligning the frames
#             frames = self.realsense.wait_for_frames()
#             aligned_frames = self.align.process(frames)

#             depth_frame = aligned_frames.get_depth_frame()
#             color_frame = aligned_frames.get_color_frame()

#             # Getting the images from the frames
#             depth_image = np.asanyarray(depth_frame.get_data())
#             color_image = np.asanyarray(color_frame.get_data())

#         return color_image, depth_image, frames.get_timestamp()

#     def stream(self):
#         # Starting the realsense stream
#         self.notify_component_start('realsense')
#         print(f"Started the Realsense pipeline for camera: {self._cam_serial_num}!")
#         print("Starting stream on {}:{}...\n".format(self._stream_configs['host'], self._stream_configs['port']))
        
#         if self._stream_oculus:
#             print('Starting oculus stream on port: {}\n'.format(self._stream_configs['port'] + VIZ_PORT_OFFSET))

#         while True:
#             #try:
#                 self.timer.start_loop()
#                 color_image, depth_image, timestamp = self.get_rgb_depth_images()

#                 color_image = rotate_image(color_image, self.cam_configs.rotation_angle)
#                 depth_image = rotate_image(depth_image, self.cam_configs.rotation_angle)

#                 # Publishing the rgb images
#                 self.rgb_publisher.pub_rgb_image(color_image, timestamp)

#                 # Headset path: GPU or CPU
#                 if self._stream_oculus:
#                     if self._use_gpu_headset and self._nvenc:
#                         # push the original (or downscaled) BGR frame to NVENC
#                         self._nvenc.send(color_image)
#                     else:
#                         self.rgb_viz_publisher.send_image(rescale_image(color_image, 2))  # old CPU path


#                 # Publishing the depth images
#                 self.depth_publisher.pub_depth_image(depth_image, timestamp)
#                 self.depth_publisher.pub_intrinsics(self.intrinsics_matrix) # Publishing inrinsics along with the depth publisher

#                 self.timer.end_loop()
#             # except KeyboardInterrupt:
#             #     break
        
#         print('Shutting down realsense pipeline for camera {}.'.format(self.cam_id))
#         self.rgb_publisher.stop()
#         if self._stream_oculus and not self._use_gpu_headset and hasattr(self, "rgb_viz_publisher"):
#             self.rgb_viz_publisher.stop()
#         self.depth_publisher.stop()
#         self.pipeline.stop()
#         if self._nvenc:
#             self._nvenc.stop()


# #----GPU ENCODING FOR HEADSET CAM----#
# import subprocess, shlex, os, signal, sys
# import numpy as np

# class _NvencRtpSenderPipe:
#     """
#     Feed raw BGR frames from Python into GStreamer via stdin (fd=0) →
#     NVENC (H.264) → RTP/UDP. Avoids opening /dev/videoX a second time.
#     """
#     def __init__(self, host, port, w, h, fps, bitrate_kbps):
#         self.w, self.h = w, h
#         self.frame_bytes = w * h * 3  # BGR
#         pipeline = (
#             "gst-launch-1.0 "
#             f"fdsrc fd=0 blocksize={self.frame_bytes} ! "
#             f"videoparse format=BGR width={w} height={h} framerate={fps}/1 ! "
#             "videoconvert ! video/x-raw,format=NV12 ! "
#             f"nvh264enc preset=low-latency-hp rc-mode=cbr bitrate={bitrate_kbps} gop-size=60 zerolatency=true ! "
#             "h264parse ! rtph264pay pt=96 config-interval=1 ! "
#             f"udpsink host={host} port={port} sync=false"
#         )
#         self.p = subprocess.Popen(
#             shlex.split(pipeline),
#             stdin=subprocess.PIPE,
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.STDOUT,
#             bufsize=0,
#         )

#     def send(self, frame_bgr: np.ndarray):
#         # Expect contiguous uint8 BGR (H x W x 3)
#         if frame_bgr.shape[0] != self.h or frame_bgr.shape[1] != self.w or frame_bgr.shape[2] != 3:
#             raise ValueError(f"Unexpected frame shape {frame_bgr.shape}, expected ({self.h},{self.w},3)")
#         self.p.stdin.write(frame_bgr.tobytes())

#     def stop(self):
#         try:
#             if self.p and self.p.poll() is None:
#                 os.kill(self.p.pid, signal.SIGINT)
#                 self.p.wait(timeout=2)
#         except Exception:
#             try:
#                 self.p.kill()
#             except Exception:
#                 pass

