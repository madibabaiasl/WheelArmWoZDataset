import os
import cv2
import time
import h5py
import queue
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import CompressedImage

from .recorder import Recorder
from openteach.constants import (
    VR_FREQ,
    CAM_FPS,
    DEPTH_RECORD_FPS,
    IMAGE_RECORD_RESOLUTION,
    CAM_FPS_SIM,
    IMAGE_RECORD_RESOLUTION_SIM
)
from openteach.utils.files import store_pickle_data
from openteach.utils.timer import FrequencyTimer


def ros_time_to_float(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class _Ros2CompressedImageSubscriber:
    """
    Drop-in replacement for ZMQCameraSubscriber-style usage.

    Exposes:
        recv_rgb_image()
        recv_depth_image()
        stop()
    """
    def __init__(self, rgb_topic=None, depth_topic=None, queue_size=200):
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic

        self._rgb_queue = queue.Queue(maxsize=queue_size)
        self._depth_queue = queue.Queue(maxsize=queue_size)

        self._running = True

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = Node('image_recorder_subscriber_node')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self._rgb_sub = None
        self._depth_sub = None

        if self.rgb_topic is not None:
            self._rgb_sub = self.node.create_subscription(
                CompressedImage,
                self.rgb_topic,
                self._rgb_callback,
                10
            )

        if self.depth_topic is not None:
            self._depth_sub = self.node.create_subscription(
                CompressedImage,
                self.depth_topic,
                self._depth_callback,
                10
            )

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spin(self):
        while self._running and rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception as exc:
                print(f'ROS2 subscriber spin warning: {exc}')

    def _push_queue(self, q, item):
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _rgb_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                return

            timestamp = ros_time_to_float(msg.header.stamp)
            self._push_queue(self._rgb_queue, (image, timestamp))
        except Exception as exc:
            print(f'RGB callback decode warning: {exc}')

    def _depth_callback(self, msg):
        try:
            depth_image = self._decode_depth(msg)
            timestamp = ros_time_to_float(msg.header.stamp)
            self._push_queue(self._depth_queue, (depth_image, timestamp))
        except Exception as exc:
            print(f'Depth callback decode warning: {exc}')

    def _decode_depth(self, msg):
        """
        Handles both:
        - plain compressed png depth
        - compressedDepth payloads containing a PNG body
        """
        fmt = (msg.format or '').lower()

        if 'compresseddepth' not in fmt:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError('Failed to decode plain compressed depth image.')
            return depth

        raw = msg.data
        png_signature = b'\x89PNG\r\n\x1a\n'
        idx = raw.find(png_signature)
        if idx < 0:
            raise ValueError('Could not find PNG header inside compressedDepth payload.')

        png_bytes = raw[idx:]
        np_arr = np.frombuffer(png_bytes, dtype=np.uint8)
        depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError('Failed to decode PNG from compressedDepth payload.')

        return depth

    def recv_rgb_image(self, timeout=None):
        return self._rgb_queue.get(timeout=timeout)

    def recv_depth_image(self, timeout=None):
        return self._depth_queue.get(timeout=timeout)

    def stop(self):
        self._running = False

        try:
            if self.executor is not None:
                self.executor.shutdown()
        except Exception as exc:
            print(f'Warning: executor shutdown failed: {exc}')

        try:
            if self.node is not None:
                self.node.destroy_node()
        except Exception as exc:
            print(f'Warning: node destroy failed: {exc}')


# To record realsense streams
class RGBImageRecorder(Recorder):
    def __init__(
        self,
        host,
        image_stream_port,
        storage_path,
        filename,
        sim=False
    ):
        # Keep same signature for compatibility, though host/port are unused now
        self.notify_component_start('RGB stream: {}'.format(image_stream_port))

        self._host, self._image_stream_port = host, image_stream_port
        self._ros_topic = '/oak/rgb/image_raw/compressed'

        self.image_subscriber = _Ros2CompressedImageSubscriber(
            rgb_topic=self._ros_topic
        )

        self.sim = sim

        # Timer
        if self.sim == True:
            self.timer = FrequencyTimer(CAM_FPS_SIM)
        else:
            self.timer = FrequencyTimer(CAM_FPS)

        # Storage path for file
        self._filename = filename
        self._recorder_file_name = os.path.join(storage_path, filename + '.avi')
        self._metadata_filename = os.path.join(storage_path, filename + '.metadata')

        # Initializing the recorder
        if self.sim == True:
            self.recorder = cv2.VideoWriter(
                self._recorder_file_name,
                cv2.VideoWriter_fourcc(*'XVID'),
                CAM_FPS_SIM,
                IMAGE_RECORD_RESOLUTION_SIM
            )
        else:
            self.recorder = cv2.VideoWriter(
                self._recorder_file_name,
                cv2.VideoWriter_fourcc(*'XVID'),
                CAM_FPS,
                IMAGE_RECORD_RESOLUTION
            )

        self.timestamps = []

    def stream(self):
        print('Starting to record RGB frames from ROS2 topic: {}'.format(self._ros_topic))

        self.num_image_frames = 0
        self.record_start_time = time.time()

        while True:
            try:
                self.timer.start_loop()
                image, timestamp = self.image_subscriber.recv_rgb_image(timeout=1.0)

                # Resize only if input size differs from writer size
                target_resolution = IMAGE_RECORD_RESOLUTION_SIM if self.sim else IMAGE_RECORD_RESOLUTION
                if image.shape[1] != target_resolution[0] or image.shape[0] != target_resolution[1]:
                    image = cv2.resize(image, target_resolution)

                self.recorder.write(image)
                self.timestamps.append(timestamp)
                self.num_image_frames += 1
                self.timer.end_loop()

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                self.record_end_time = time.time()
                break

        self.image_subscriber.stop()

        # Displaying statistics
        self._display_statistics(self.num_image_frames)

        # Saving the metadata
        self._add_metadata(self.num_image_frames)
        self.metadata['timestamps'] = self.timestamps
        self.metadata['recorder_ip_address'] = self._host
        self.metadata['recorder_image_stream_port'] = self._image_stream_port
        self.metadata['recorder_ros_topic'] = self._ros_topic

        # Storing the data
        print('Storing the final version of the video...')
        self.recorder.release()
        store_pickle_data(self._metadata_filename, self.metadata)
        print('Stored the video in {}.'.format(self._recorder_file_name))
        print('Stored the metadata in {}.'.format(self._metadata_filename))


class DepthImageRecorder(Recorder):
    def __init__(
        self,
        host,
        image_stream_port,
        storage_path,
        filename
    ):
        # Keep same signature for compatibility, though host/port are unused now
        self.notify_component_start('Depth stream: {}'.format(image_stream_port))

        self._host, self._image_stream_port = host, image_stream_port
        self._ros_topic = '/oak/stereo/image_raw/compressed'

        self.image_subscriber = _Ros2CompressedImageSubscriber(
            depth_topic=self._ros_topic
        )

        # Timer
        self.timer = FrequencyTimer(DEPTH_RECORD_FPS)

        # Storage path for file
        self._filename = filename
        self._recorder_file_name = os.path.join(storage_path, filename + '.h5')
        self._metadata_filename = os.path.join(storage_path, filename + '.metadata')

        # Initializing the depth data containers
        self.depth_frames = []
        self.timestamps = []
        self.metadata = {}

    def _save_depth_file(self):
        if self.num_image_frames <= 0 or len(self.depth_frames) == 0:
            print('No depth frames captured; skipping .h5 write.')
            return

        print('Compressing depth data...')
        with h5py.File(self._recorder_file_name, 'w') as file:
            stacked_frames = np.asarray(self.depth_frames)

            file.create_dataset(
                'depth_images',
                data=stacked_frames,
                compression='gzip',
                compression_opts=6
            )

            timestamps = np.asarray(self.timestamps, dtype=np.float64)
            file.create_dataset(
                'timestamps',
                data=timestamps,
                compression='gzip',
                compression_opts=6
            )

            for key, value in self.metadata.items():
                try:
                    file.attrs[key] = value
                except Exception:
                    pass

        print('Saved compressed depth data in {}.'.format(self._recorder_file_name))

    def stream(self):
        print('Starting to record depth frames from ROS2 topic: {}'.format(self._ros_topic))

        self.num_image_frames = 0
        self.record_start_time = time.time()
        self.record_end_time = None

        try:
            while True:
                self.timer.start_loop()
                depth_data, timestamp = self.image_subscriber.recv_depth_image(timeout=1.0)
                self.depth_frames.append(depth_data)
                self.timestamps.append(timestamp)
                self.num_image_frames += 1
                self.timer.end_loop()

        except queue.Empty:
            pass
        except KeyboardInterrupt:
            print('Depth recorder interrupted. Saving buffered frames before exit...')
        finally:
            self.record_end_time = time.time()

            try:
                self.image_subscriber.stop()
            except Exception as exc:
                print('Warning: failed to stop depth subscriber cleanly: {}'.format(exc))

            try:
                self._display_statistics(self.num_image_frames)
            except Exception as exc:
                print('Warning: failed to display depth statistics: {}'.format(exc))

            try:
                self._add_metadata(self.num_image_frames)
            except Exception as exc:
                print('Warning: failed to build depth metadata: {}'.format(exc))
                if not hasattr(self, 'metadata') or self.metadata is None:
                    self.metadata = {}

            self.metadata['timestamps'] = self.timestamps
            self.metadata['recorder_ip_address'] = self._host
            self.metadata['recorder_image_stream_port'] = self._image_stream_port
            self.metadata['recorder_ros_topic'] = self._ros_topic

            try:
                store_pickle_data(self._metadata_filename, self.metadata)
                print('Stored the metadata in {}.'.format(self._metadata_filename))
            except Exception as exc:
                print('Warning: failed to store depth metadata: {}'.format(exc))

            self._save_depth_file()


class FishEyeImageRecorder(Recorder):
    def __init__(
        self,
        host,
        image_stream_port,
        storage_path,
        filename
    ):
        self.notify_component_start('RGB stream: {}'.format(image_stream_port))

        self._host, self._image_stream_port = host, image_stream_port
        self._ros_topic = '/oak/rgb/image_raw/compressed'

        self.image_subscriber = _Ros2CompressedImageSubscriber(
            rgb_topic=self._ros_topic
        )

        # Timer
        self.timer = FrequencyTimer(CAM_FPS)

        # Storage path for file
        self._filename = filename
        self._recorder_file_name = os.path.join(storage_path, filename + '.avi')
        self._metadata_filename = os.path.join(storage_path, filename + '.metadata')
        self._pickle_filename = os.path.join(storage_path, filename + '.pkl')

        # Initializing the recorder
        self.recorder = cv2.VideoWriter(
            self._recorder_file_name,
            cv2.VideoWriter_fourcc(*'XVID'),
            CAM_FPS,
            IMAGE_RECORD_RESOLUTION
        )
        self.timestamps = []
        self.frames = []

    def stream(self):
        print('Starting to record RGB frames from ROS2 topic: {}'.format(self._ros_topic))

        self.num_image_frames = 0
        self.record_start_time = time.time()

        while True:
            try:
                self.timer.start_loop()
                image, timestamp = self.image_subscriber.recv_rgb_image(timeout=1.0)

                if image.shape[1] != IMAGE_RECORD_RESOLUTION[0] or image.shape[0] != IMAGE_RECORD_RESOLUTION[1]:
                    image = cv2.resize(image, IMAGE_RECORD_RESOLUTION)

                self.recorder.write(image)
                self.timestamps.append(timestamp)
                self.frames.append(np.array(image))

                self.num_image_frames += 1
                self.timer.end_loop()

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                self.record_end_time = time.time()
                break

        self.image_subscriber.stop()

        # Displaying statistics
        self._display_statistics(self.num_image_frames)

        # Saving the metadata
        self._add_metadata(self.num_image_frames)
        self.metadata['timestamps'] = self.timestamps
        self.metadata['recorder_ip_address'] = self._host
        self.metadata['recorder_image_stream_port'] = self._image_stream_port
        self.metadata['recorder_ros_topic'] = self._ros_topic

        # Storing the data
        print('Storing the final version of the video...')
        self.recorder.release()
        store_pickle_data(self._metadata_filename, self.metadata)
        print('Stored the video in {}.'.format(self._recorder_file_name))
        print('Stored the metadata in {}.'.format(self._metadata_filename))
