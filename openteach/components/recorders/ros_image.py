import os
import time
import cv2
import h5py
import numpy as np
import json, zmq
from .recorder import Recorder
from openteach.utils.network import ZMQCameraSubscriber
from openteach.constants import CAM_FPS, IMAGE_RECORD_RESOLUTION, DEPTH_RECORD_FPS  # reuse your constants
from openteach.utils.files import store_pickle_data
from openteach.utils.timer import FrequencyTimer

CAM3_FPS = 30
# --- camera_info helpers (ZMQ SUB, non-blocking) ---
def _make_info_sub(host: str, port: int | None):
    """Create a non-blocking SUB socket for camera_info JSON."""
    if not port:
        return None
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.RCVHWM, 1)
    s.setsockopt_string(zmq.SUBSCRIBE, '')
    s.connect(f"tcp://{host}:{port}")
    s.RCVTIMEO = 0  # non-blocking
    print(f"CameraInfo ZMQ SUB connected to tcp://{host}:{port}")
    return s

def _pump_info_nonblock(sock, info_state: dict):
    """
    Drain any pending camera_info JSON and keep newest in:
      info_state['first'], info_state['last']
    """
    if sock is None:
        return
    while True:
        try:
            payload = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        try:
            ci = json.loads(payload.decode('utf-8'))
        except Exception:
            # If decoding fails, keep a small hex preview to aid debugging
            ci = {'_raw_hex': payload[:64].hex(), '_note': 'decode_failed'}
        if 'first' not in info_state:
            info_state['first'] = ci
        info_state['last'] = ci
       
class RosRGBImageRecorder(Recorder):
    """
    Records ROS->ZMQ RGB frames using the same wire format as ros_cam_test.py:
      SUBSCRIBE ""
      recv_multipart() = [b"RGB", b'{"h":..,"w":..,"step":..,"enc":"bgr8","ts":...}', <raw bytes>]
    """
    def __init__(self, host, rgb_port, storage_path, filename, rgb_info_port=None):
        self.notify_component_start(f'ROS RGB stream: {rgb_port}')
        host = '127.0.0.1'
        self._host, self._rgb_port = host, rgb_port

        # --- raw SUB like ros_cam_test.py ---
        self._ctx = zmq.Context.instance()
        self._sub_rgb = self._ctx.socket(zmq.SUB)
        self._sub_rgb.setsockopt_string(zmq.SUBSCRIBE, "")   # accept all topics
        self._sub_rgb.setsockopt(zmq.RCVHWM, 10)
        self._rgb_endpoint = f"tcp://{host}:{rgb_port}"
        self._sub_rgb.connect(self._rgb_endpoint)

        self._rgb_poller = zmq.Poller()
        self._rgb_poller.register(self._sub_rgb, zmq.POLLIN)
        print(f"[ros_image] SUB RGB connect → {self._rgb_endpoint} (SUBSCRIBE '')")

        # File outputs
        os.makedirs(storage_path, exist_ok=True)
        self._recorder_file_name = os.path.join(storage_path, filename + ".avi")
        self._metadata_filename = os.path.join(storage_path, filename + ".metadata")

        # Use your configured resolution / fps; we will resize to this if needed
        self.recorder = cv2.VideoWriter(
            self._recorder_file_name,
            cv2.VideoWriter_fourcc(*"XVID"),
            CAM3_FPS,
            IMAGE_RECORD_RESOLUTION
        )
        print("VideoWriter opened:", self.recorder.isOpened(), "→", self._recorder_file_name)

        # CameraInfo side-channel (non-blocking)
        self._rgb_info_sock = _make_info_sub(host, rgb_info_port)
        self._info_state = {}  # {'first': {...}, 'last': {...}}

        # self.timer = FrequencyTimer(CAM_FPS)

        # state
        self.timestamps = []
        self.num_image_frames = 0
        self.record_start_time = None
        self.record_end_time = None

    def _decode_rgb(self, parts):
        """
        parts: [topic, header_json_bytes, payload_bytes] → returns (BGR uint8 HxWx3, header dict)
        """
        header = json.loads(parts[1].decode("ascii"))
        h, w, step = int(header["h"]), int(header["w"]), int(header["step"])
        enc = header.get("enc", "bgr8").lower()

        # reconstruct from bytes honoring 'step' (bytes/row; may include padding)
        buf = parts[2]
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, step)[:, : w * 3].reshape(h, w, 3)

        if enc == "rgb8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif enc == "bgr8":
            pass
        elif enc == "mono8":
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        # else: assume already BGR-like
        return arr, header

    def stream(self):
        print(f"Starting to record ROS RGB frames from {self._rgb_endpoint}")
        self.record_start_time = time.time()

        # --- First-frame handshake with timeout: fail loud if stream is dead/misconfigured ---
        deadline = time.time() + 3.0
        first_frame = None
        while time.time() < deadline:
            _pump_info_nonblock(self._rgb_info_sock, self._info_state)
            events = dict(self._rgb_poller.poll(timeout=100))
            if self._sub_rgb in events:
                parts = self._sub_rgb.recv_multipart()
                if len(parts) != 3:
                    continue
                # If multiple topics share the socket, ignore non-RGB topics
                if parts[0] not in (b"RGB", b"RosRGB", b"ros_rgb", b"color", b"cam_rgb"):
                    continue
                frame_bgr, header = self._decode_rgb(parts)
                # size to writer's expected resolution
                W, H = IMAGE_RECORD_RESOLUTION
                if (frame_bgr.shape[1], frame_bgr.shape[0]) != (W, H):
                    frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
                self.recorder.write(frame_bgr)
                ts = header.get("ts_ros") or header.get("ts") or time.time()
                self.timestamps.append(float(ts))
                self.num_image_frames = 1
                first_frame = True
                break

        if not first_frame:
            raise TimeoutError(f"No ROS RGB frames on {self._rgb_endpoint} within 3 s")

        try:
            while True:
                _pump_info_nonblock(self._rgb_info_sock, self._info_state)
                events = dict(self._rgb_poller.poll(timeout=100))
                if self._sub_rgb in events:
                    parts = self._sub_rgb.recv_multipart()
                    if len(parts) != 3:
                        continue
                    if parts[0] not in (b"RGB", b"RosRGB", b"ros_rgb", b"color", b"cam_rgb"):
                        continue
                    frame_bgr, header = self._decode_rgb(parts)
                    W, H = IMAGE_RECORD_RESOLUTION
                    if (frame_bgr.shape[1], frame_bgr.shape[0]) != (W, H):
                        frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
                    self.recorder.write(frame_bgr)
                    ts = header.get("ts_ros") or header.get("ts") or time.time()
                    self.timestamps.append(float(ts))
                    self.num_image_frames += 1
        except KeyboardInterrupt:
            self.record_end_time = time.time()

        # Teardown & metadata
        try:
            self._sub_rgb.close(0)
        except Exception:
            pass
        self.recorder.release()

        if hasattr(self, "_display_statistics"):
            self._display_statistics(self.num_image_frames)
        if hasattr(self, "_add_metadata"):
            self._add_metadata(self.num_image_frames)
        if hasattr(self, "metadata"):
            self.metadata["timestamps"] = self.timestamps
            self.metadata["recorder_ip_address"] = self._host
            self.metadata["recorder_image_stream_port"] = self._rgb_port
            if "first" in self._info_state:
                self.metadata["camera_info_first"] = self._info_state["first"]
            if "last" in self._info_state:
                self.metadata["camera_info_last"] = self._info_state["last"]

        print(f"Recorded ROS RGB {self.num_image_frames} frames.")
        store_pickle_data(self._metadata_filename, self.metadata)
        print(f"Stored the metadata in {self._metadata_filename}.")





# class RosDepthImageRecorder(Recorder):
#     """
#     Records depth frames arriving from a ROS->ZMQ bridge.
#     Behaves like DepthImageRecorder but **does not** throttle with FrequencyTimer.
#     """
#     def __init__(self, host, depth_port, storage_path, filename, depth_info_port=None):
#         self.notify_component_start(f'ROS Depth stream: {depth_port}')
#         self._host, self._depth_port = host, depth_port

#         self.image_subscriber = ZMQCameraSubscriber(
#             host=host,
#             port=depth_port,
#             topic_type=None
#         )
#         print(f"Subscribed to Depth image stream at tcp://{host}:{depth_port}")

#         self._filename = filename
#         self._recorder_file_name = os.path.join(storage_path, filename + '.h5')

#         self.depth_frames = []
#         self.timestamps = []

#         # NEW: camera_info subscriber + state
#         self._depth_info_sock = _make_info_sub(host, depth_info_port)
#         self._info_state = {}  # {'first': {...}, 'last': {...}}

#     def stream(self):
#         # sanity check
#         first = self.image_subscriber.recv_depth_image()
#         if first is None:
#             raise ValueError('Depth image stream is not active.')
#         else:
#             depth0, ts0 = first
#             self.depth_frames.append(depth0)
#             self.timestamps.append(ts0)

#         print(f'Starting to record ROS depth frames from port: {self._depth_port}')
#         self.num_image_frames = 1
#         self.record_start_time = time.time()

#         try:
#             while True:
#                 depth_data, timestamp = self.image_subscriber.recv_depth_image()  # write-on-arrival
#                 print(f"[RosDepthImageRecorder]: timestamp {timestamp}")
#                 print(f"[RosDepthImageRecorder]: depth shape {depth_data.shape}")
#                 _pump_info_nonblock(self._depth_info_sock, self._info_state)

#                 self.depth_frames.append(depth_data)
#                 self.timestamps.append(timestamp)
#                 self.num_image_frames += 1
#         except KeyboardInterrupt:
#             self.record_end_time = time.time()

#         # Close socket
#         self.image_subscriber.stop()

#         # Stats + metadata
#         self._display_statistics(self.num_image_frames)
#         self._add_metadata(self.num_image_frames)
#         print(f"Recorded ROS Depth {self.num_image_frames} frames.")
#         self.metadata['recorder_ip_address'] = self._host
#         self.metadata['recorder_image_stream_port'] = self._depth_port

#         # Write compressed HDF5 (+ attrs)
#         print('Compressing depth data...')
#         with h5py.File(self._recorder_file_name, "w") as f:
#             stacked = np.array(self.depth_frames, dtype=np.uint16)
#             f.create_dataset("depth_images", data=stacked, compression="gzip", compression_opts=6)
#             f.create_dataset("timestamps", data=np.array(self.timestamps, np.float64),
#                              compression="gzip", compression_opts=6)

#             # carry Recorder metadata alongside as datasets (kept to match your existing behavior)
#             # NOTE: if you prefer HDF5 attrs for *all* metadata, move these into f.attrs[...] instead.
#             try:
#                 f.update(self.metadata)  # as in your current file
#             except Exception:
#                 # Fallback: store a JSON metadata blob as an attribute
#                 f.attrs['recorder_metadata_json'] = json.dumps(self.metadata)

#             # NEW: camera_info as JSON attrs
#             if 'first' in self._info_state:
#                 f.attrs['camera_info_json_first'] = json.dumps(self._info_state['first'])
#             if 'last' in self._info_state:
#                 f.attrs['camera_info_json_last'] = json.dumps(self._info_state['last'])

#         print(f'Saved compressed depth data in {self._recorder_file_name}')
class RosDepthImageRecorder(Recorder):
    """
    Records ROS->ZMQ Depth frames with multipart format:
      SUBSCRIBE ""
      recv_multipart() = [b"Depth", b'{"h":..,"w":..,"step":..,"enc":"16UC1|32FC1","ts":...}', <raw bytes>]
    Saves frames to self.depth_frames (list of arrays); rest of your save/metadata code can stay as-is.
    """
    def __init__(self, host, depth_port, storage_path, filename, depth_info_port=None):
        self.notify_component_start(f'ROS Depth stream: {depth_port}')
        host = '127.0.0.1'
        self._host, self._depth_port = host, depth_port

        # --- raw SUB like ros_cam_test.py ---
        self._ctx = zmq.Context.instance()
        self._sub_depth = self._ctx.socket(zmq.SUB)
        self._sub_depth.setsockopt_string(zmq.SUBSCRIBE, "")   # accept all topics
        self._sub_depth.setsockopt(zmq.RCVHWM, 10)
        self._depth_endpoint = f"tcp://{host}:{depth_port}"
        self._sub_depth.connect(self._depth_endpoint)

        self._depth_poller = zmq.Poller()
        self._depth_poller.register(self._sub_depth, zmq.POLLIN)
        print(f"[ros_image] SUB Depth connect → {self._depth_endpoint} (SUBSCRIBE '')")

        # CameraInfo side-channel (non-blocking)
        self._depth_info_sock = _make_info_sub(host, depth_info_port)
        self._info_state = {}

        # state
        self.depth_frames = []
        self.timestamps = []
        self.num_image_frames = 0
        self.record_start_time = None
        self.record_end_time = None

        # File outputs (your existing writer/h5 handling likely occurs later in stream teardown)
        # os.makedirs(storage_path, exist_ok=True)
        # self._metadata_filename = os.path.join(storage_path, filename + ".metadata")
        # self._h5_filename = os.path.join(storage_path, filename + ".h5")
        
        # File outputs for depth
        os.makedirs(storage_path, exist_ok=True)
        self._recorder_file_name = os.path.join(storage_path, f"{filename}.h5")   # <-- needed by _display_statistics
        self._metadata_filename  = os.path.join(storage_path, f"{filename}.metadata")

    def _decode_depth(self, parts):
        header = json.loads(parts[1].decode("ascii"))
        h, w, step = int(header["h"]), int(header["w"]), int(header["step"])
        enc = header.get("enc", "16UC1")
        if enc in ("16UC1", "mono16"):
            depth = np.frombuffer(parts[2], dtype=np.uint16).reshape(h, step // 2)[:, :w]
        elif enc == "32FC1":
            depth = np.frombuffer(parts[2], dtype=np.float32).reshape(h, step // 4)[:, :w]
        else:
            # fallback assume 16-bit
            depth = np.frombuffer(parts[2], dtype=np.uint16).reshape(h, step // 2)[:, :w]
        return depth, header

    def stream(self):
        print(f"Starting to record ROS depth frames from {self._depth_endpoint}")
        self.record_start_time = time.time()

        # First-frame handshake with timeout
        deadline = time.time() + 3.0
        first_frame = None
        while time.time() < deadline:
            _pump_info_nonblock(self._depth_info_sock, self._info_state)
            events = dict(self._depth_poller.poll(timeout=100))
            if self._sub_depth in events:
                parts = self._sub_depth.recv_multipart()
                if len(parts) != 3:
                    continue
                if parts[0] not in (b"Depth", b"RosDepth", b"ros_depth", b"depth", b"cam_depth"):
                    continue
                arr, header = self._decode_depth(parts)
                ts = header.get("ts_ros") or header.get("ts") or time.time()
                self.depth_frames.append(arr)
                self.timestamps.append(float(ts))
                self.num_image_frames = 1
                first_frame = True
                break

        if not first_frame:
            raise TimeoutError(f"No ROS Depth frames on {self._depth_endpoint} within 3 s")

        try:
            while True:
                _pump_info_nonblock(self._depth_info_sock, self._info_state)
                events = dict(self._depth_poller.poll(timeout=100))
                if self._sub_depth in events:
                    parts = self._sub_depth.recv_multipart()
                    if len(parts) != 3:
                        continue
                    if parts[0] not in (b"Depth", b"RosDepth", b"ros_depth", b"depth", b"cam_depth"):
                        continue
                    arr, header = self._decode_depth(parts)
                    self.depth_frames.append(arr)
                    ts = header.get("ts_ros") or header.get("ts") or time.time()
                    self.timestamps.append(float(ts))
                    self.num_image_frames += 1
        except KeyboardInterrupt:
            self.record_end_time = time.time()

        # Teardown; leave your existing HDF5/compression code intact after this
        try:
            self._sub_depth.close(0)
        except Exception:
            pass
        
        # Show stats (uses _recorder_file_name) then write H5
        if hasattr(self, "_display_statistics"):
            self._display_statistics(self.num_image_frames)

        # Write .h5 if we actually captured frames
        if self.num_image_frames > 0:
            print("Compressing depth data...")
            import numpy as np
            depth_np = np.stack(self.depth_frames, axis=0)  # [N, H, W]
            with h5py.File(self._recorder_file_name, "w") as hf:
                hf.create_dataset("depth_image", data=depth_np, compression="gzip", compression_opts=4)
                hf.create_dataset("timestamps", data=np.asarray(self.timestamps, dtype=np.float64))
            print(f"Saved compressed depth data in {self._recorder_file_name}.")
        else:
            print("No ROS depth frames captured; skipping .h5 write.")

        # Write metadata as you already do
        if hasattr(self, "_add_metadata"):
            self._add_metadata(self.num_image_frames)
        if hasattr(self, "metadata"):
            self.metadata["timestamps"] = self.timestamps
            self.metadata["recorder_ip_address"] = self._host
            self.metadata["recorder_image_stream_port"] = self._depth_port
            if "first" in self._info_state: self.metadata["camera_info_first"] = self._info_state["first"]
            if "last" in self._info_state:  self.metadata["camera_info_last"]  = self._info_state["last"]
        store_pickle_data(self._metadata_filename, self.metadata)
        print(f"Stored the metadata in {self._metadata_filename}.")


        
