import os
import cv2
import time
import h5py
import numpy as np
from .recorder import Recorder
from openteach.constants import VR_FREQ,CAM_FPS,DEPTH_RECORD_FPS,IMAGE_RECORD_RESOLUTION,CAM_FPS_SIM,IMAGE_RECORD_RESOLUTION_SIM
from openteach.utils.files import store_pickle_data
from openteach.utils.network import ZMQCameraSubscriber
from openteach.utils.timer import FrequencyTimer

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
        self.notify_component_start('RGB stream: {}'.format(image_stream_port))
        
        # Subscribing to the image stream port
        self._host, self._image_stream_port = host, image_stream_port
        self.image_subscriber = ZMQCameraSubscriber(
            host = host,
            port = image_stream_port,
            topic_type = 'RGB'
        )
        self.sim = sim
        # Timer
        if self.sim==True:
            self.timer = FrequencyTimer(CAM_FPS_SIM)
        else:
            self.timer = FrequencyTimer(CAM_FPS)

        # Storage path for file
        self._filename = filename
        self._recorder_file_name = os.path.join(storage_path, filename + '.avi')
        self._metadata_filename = os.path.join(storage_path, filename + '.metadata')

        # Initializing the recorder
        if self.sim==True:
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
        print('Starting to record RGB frames from port: {}'.format(self._image_stream_port))

        self.num_image_frames = 0
        self.record_start_time = time.time()

        while True:
            try:
                self.timer.start_loop()
                image, timestamp = self.image_subscriber.recv_rgb_image()
                self.recorder.write(image)
                self.timestamps.append(timestamp)
                self.num_image_frames += 1
                self.timer.end_loop()
            except KeyboardInterrupt:
                    self.record_end_time = time.time()
                    break
            
        # Closing the socket
        self.image_subscriber.stop()

        # Displaying statistics
        self._display_statistics(self.num_image_frames)
        
        # Saving the metadata
        self._add_metadata(self.num_image_frames)
        self.metadata['timestamps'] = self.timestamps
        self.metadata['recorder_ip_address'] = self._host
        self.metadata['recorder_image_stream_port'] = self._image_stream_port

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
        self.notify_component_start('Depth stream: {}'.format(image_stream_port))
        
        # Subscribing to the image stream port
        self._host, self._image_stream_port = host, image_stream_port
        self.image_subscriber = ZMQCameraSubscriber(
            host=host,
            port=image_stream_port,
            topic_type='Depth'
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
            stacked_frames = np.asarray(self.depth_frames, dtype=np.uint16)
            file.create_dataset('depth_images', data=stacked_frames, compression='gzip', compression_opts=6)

            timestamps = np.asarray(self.timestamps, dtype=np.float64)
            file.create_dataset('timestamps', data=timestamps, compression='gzip', compression_opts=6)

            for key, value in self.metadata.items():
                try:
                    file.attrs[key] = value
                except Exception:
                    pass

        print('Saved compressed depth data in {}.'.format(self._recorder_file_name))

    def stream(self):
        print('Starting to record depth frames from port: {}'.format(self._image_stream_port))

        self.num_image_frames = 0
        self.record_start_time = time.time()
        self.record_end_time = None

        try:
            while True:
                self.timer.start_loop()
                depth_data, timestamp = self.image_subscriber.recv_depth_image()
                self.depth_frames.append(depth_data)
                self.timestamps.append(timestamp)
                self.num_image_frames += 1
                self.timer.end_loop()
        except KeyboardInterrupt:
            print('Depth recorder interrupted. Saving buffered frames before exit...')
        finally:
            self.record_end_time = time.time()

            # Closing the socket
            try:
                self.image_subscriber.stop()
            except Exception as exc:
                print('Warning: failed to stop depth subscriber cleanly: {}'.format(exc))

            # Displaying statistics
            try:
                self._display_statistics(self.num_image_frames)
            except Exception as exc:
                print('Warning: failed to display depth statistics: {}'.format(exc))

            # Saving the metadata
            try:
                self._add_metadata(self.num_image_frames)
            except Exception as exc:
                print('Warning: failed to build depth metadata: {}'.format(exc))
                if not hasattr(self, 'metadata') or self.metadata is None:
                    self.metadata = {}

            self.metadata['timestamps'] = self.timestamps
            self.metadata['recorder_ip_address'] = self._host
            self.metadata['recorder_image_stream_port'] = self._image_stream_port

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
        
        # Subscribing to the image stream port
        print("Image Stream Port", image_stream_port)
        self._host, self._image_stream_port = host, image_stream_port
        self.image_subscriber = ZMQCameraSubscriber(
            host = host,
            port = image_stream_port,
            topic_type = 'RGB'
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
        print('Starting to record RGB frames from port: {}'.format(self._image_stream_port))

        self.num_image_frames = 0
        self.record_start_time = time.time()

        while True:
            try:
                self.timer.start_loop()
                image, timestamp = self.image_subscriber.recv_rgb_image()
                self.recorder.write(image)
                self.timestamps.append(timestamp)

                self.frames.append(np.array(image))
    
                self.num_image_frames += 1
                self.timer.end_loop()
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

        # Storing the data
        print('Storing the final version of the video...')
        self.recorder.release()
        store_pickle_data(self._metadata_filename, self.metadata)
        print('Stored the video in {}.'.format(self._recorder_file_name))
        print('Stored the metadata in {}.'.format(self._metadata_filename))