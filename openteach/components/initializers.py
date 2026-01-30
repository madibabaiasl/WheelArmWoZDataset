import os
import hydra
from abc import ABC
from .recorders.audio import HeadsetOpusRecorder, LaptopMicWavRecorder
from .recorders.image import RGBImageRecorder, DepthImageRecorder
from .recorders.ros_image import RosRGBImageRecorder, RosDepthImageRecorder
from .recorders.robot_state import RobotInformationRecord
from .recorders.sim_state import SimInformationRecord
from .recorders.sensors import XelaSensorRecorder
from .sensors import *
from .sensors.oak import OakCamera 
from multiprocessing import Process
from openteach.constants import *
import signal

class ProcessInstantiator(ABC):
    def __init__(self, configs):
        self.configs = configs
        self.processes = []

    def _start_component(self,configs):
        raise NotImplementedError('Function not implemented!')

    def get_processes(self):
        return self.processes


class RealsenseCameras(ProcessInstantiator):
    def __init__(self, configs):
        super().__init__(configs)
        self._init_camera_processes()

    def _start_component(self, cam_idx):
        component = RealsenseCamera(
            stream_configs = dict(
                host = self.configs.host_address,
                port = self.configs.cam_port_offset + cam_idx
            ),
            cam_serial_num = self.configs.robot_cam_serial_numbers[cam_idx],
            cam_id = cam_idx + 1,
            cam_configs = self.configs.cam_configs,
            stream_oculus = True if self.configs.oculus_cam == cam_idx else False
        )
        component.stream()

    def _init_camera_processes(self):
        for cam_idx in range(len(self.configs.robot_cam_serial_numbers)):
            self.processes.append(Process(
                target = self._start_component,
                args = (cam_idx, )
            ))

class OakCameras(ProcessInstantiator):
    def __init__(self, configs):
        super().__init__(configs)
        self._init_camera_processes()

    def _start_component(self, cam_idx):
        component = OakCamera(
            stream_configs=dict(
                host=self.configs.host_address,
                port=self.configs.cam_port_offset + cam_idx # the port that oculus publish to
            ),
            mxid=self.configs.oak_mxids[cam_idx],      # NEW list in your YAML (see below)
            cam_id=cam_idx + 1,
            cam_configs=self.configs.cam_configs,
            stream_oculus=True if self.configs.oculus_cam == cam_idx else False
        )
        component.stream()

    def _init_camera_processes(self):
        for cam_idx in range(len(self.configs.oak_mxids)):
            self.processes.append(Process(target=self._start_component, args=(cam_idx,)))


class TeleOperator(ProcessInstantiator):
    def __init__(self, configs):
        super().__init__(configs)
      
        # For Simulation environment start the environment as well
        if configs.sim_env:
            self._init_sim_environment()
        # Start the Hand Detector
        self._init_detector()
        # Start the keypoint transform
        self._init_keypoint_transform()
        self._init_visualizers()


        if configs.operate: 
            self._init_operator()
        
    #Function to start the components
    def _start_component(self, configs):    
        component = hydra.utils.instantiate(configs)
        component.stream()

    #Function to start the detector component
    def _init_detector(self):
        self.processes.append(Process(
            target = self._start_component,
            args = (self.configs.robot.detector, )
        ))

    #Function to start the sim environment
    def _init_sim_environment(self):
         for env_config in self.configs.robot.environment:
            self.processes.append(Process(
                target = self._start_component,
                args = (env_config, )
            ))

    #Function to start the keypoint transform
    def _init_keypoint_transform(self):
        for transform_config in self.configs.robot.transforms:
            self.processes.append(Process(
                target = self._start_component,
                args = (transform_config, )
            ))

    #Function to start the visualizers
    def _init_visualizers(self):
       
        for visualizer_config in self.configs.robot.visualizers:
            self.processes.append(Process(
                target = self._start_component,
                args = (visualizer_config, )
            ))
        # XELA visualizer
        if self.configs.run_xela:
            for visualizer_config in self.configs.xela_visualizers:
                self.processes.append(Process(cam_number, None))


    #Function to start the operator
    def _init_operator(self):
        for operator_config in self.configs.robot.operators:
            
            self.processes.append(Process(
                target = self._start_component,
                args = (operator_config, )
            ))
  
# Data Collector Class
class Collector(ProcessInstantiator):
    def __init__(self, configs, demo_num, session_name: str | None = None):
        super().__init__(configs)
        self.demo_num = demo_num
        folder = session_name.strip() if session_name else f"demonstration_{self.demo_num}"
        self._storage_path = os.path.join(self.configs.storage_path, folder)
       
        self._create_storage_dir()
        self._init_camera_recorders()

        if self.configs.sim_env is True:
            self._init_sim_recorders()
        else:
            print("Initialising robot recorders")
            self._init_robot_recorders()
        
        if self.configs.is_xela is True:
            self._init_sensor_recorders()
        
        if hasattr(self.configs, "audio"):
            if getattr(self.configs.audio, "record_headset", False):
                self.processes.append(Process(target=self._start_headset_audio, args=()))
            if getattr(self.configs.audio, "record_laptop_mic", False):
                self.processes.append(Process(target=self._start_laptop_mic, args=()))
        

    def _create_storage_dir(self):
        if os.path.exists(self._storage_path):
            return 
        else:
            os.makedirs(self._storage_path)

    #Function to start the components
    def _start_component(self, component):
        component.stream()
    
    def _start_ros_arm_rgb(self, cam_number):
        component = RosRGBImageRecorder(
            host=self.configs.host_address,
            rgb_port=self.configs.cam_port_offset + cam_number,
            storage_path=self._storage_path,
            filename=f'cam_{cam_number}_rgb_video',
            rgb_info_port=self.configs.info_port_offset + cam_number  # NEW
        )
        component.stream()

    def _start_ros_arm_depth(self, cam_number):
        component = RosDepthImageRecorder(
            host=self.configs.host_address,
            depth_port=self.configs.cam_port_offset + cam_number + DEPTH_PORT_OFFSET,
            storage_path=self._storage_path,
            filename=f'cam_{cam_number}_depth',
            depth_info_port=self.configs.info_port_offset + cam_number + DEPTH_PORT_OFFSET # NEW
        )
        component.stream()

    # Record the rgb components
    def _start_rgb_component(self, cam_idx=0):
        if self.configs.sim_env is False:
            print("RGB function")
            component = RGBImageRecorder(
                host = self.configs.host_address,
                image_stream_port = self.configs.cam_port_offset + cam_idx,
                storage_path = self._storage_path,
                filename = 'cam_{}_rgb_video'.format(cam_idx)
            )
        else:
            print("Reaching correct function")
            component = RGBImageRecorder(
            host = self.configs.host_address,
            image_stream_port = self.configs.sim_image_port+ cam_idx,
            storage_path = self._storage_path,
            filename = 'cam_{}_rgb_video'.format(cam_idx),
            sim = True
        )
        component.stream()

    # Record the depth components
    def _start_depth_component(self, cam_idx):
        if self.configs.sim_env is not True:
            component = DepthImageRecorder(
                host = self.configs.host_address,
                image_stream_port = self.configs.cam_port_offset + cam_idx + DEPTH_PORT_OFFSET,
                storage_path = self._storage_path,
                filename = 'cam_{}_depth'.format(cam_idx)
            )
        else:
            component = DepthImageRecorder(
                host = self.configs.host_address,
                image_stream_port = self.configs.sim_image_port + cam_idx + DEPTH_PORT_OFFSET,
                storage_path = self._storage_path,
                filename = 'cam_{}_depth'.format(cam_idx)
            )
        component.stream()
    
    # Audio Recording
    def _start_headset_audio(self):
        # Keep PortAudio's default OUTPUT pointed at your laptop speakers (Pulse default sink)
        import sounddevice as sd
        sd.default.device = (None, "default")  

        component = headset = HeadsetOpusRecorder(
            port=50010,
            storage_path=self._storage_path,
            filename="headset_audio",
            play=True,
            output_device="pipe",  # sentinel: skip PortAudio
            sink_name="alsa_output.pci-0000_00_1f.3.analog-stereo",  # from `pactl info`
            force_pipe=True,  # hard-stop PortAudio path
        )
        component.stream()
    
    def _start_laptop_mic(self):
        component = LaptopMicWavRecorder(
            storage_path=self._storage_path,
            filename="laptop_mic",
            samplerate=48000,          # your USB source advertises 48 kHz mono
            channels=1,                # it's mono (".mono-fallback")
            device="alsa_input.usb-Jieli_Technology_USB_Composite_Device_433130353331342E-00.mono-fallback",                 
            segment_seconds=getattr(self.configs.audio, "laptop_segment_seconds", None),
        )
        component.stream()

    def _init_camera_recorders(self):
        # keep your existing cameras
        if self.configs.sim_env is not True:
            for cam_idx in range(len(self.configs.robot_cam_serial_numbers)):
                self.processes.append(Process(target=self._start_rgb_component, args=(cam_idx,)))
                self.processes.append(Process(target=self._start_depth_component, args=(cam_idx,)))                
        else:
            for cam_idx in range(self.configs.num_cams):
                self.processes.append(Process(target=self._start_rgb_component, args=(cam_idx,)))
                self.processes.append(Process(target=self._start_depth_component, args=(cam_idx,)))

        ros_arm_cam_number = getattr(self.configs, "ros_arm_cam_number", None)
        if ros_arm_cam_number is not None:
            self.processes.append(Process(target=self._start_ros_arm_rgb, args=(ros_arm_cam_number,)))
            self.processes.append(Process(target=self._start_ros_arm_depth, args=(ros_arm_cam_number,)))

    #Function to start the sim recorders
    def _init_sim_recorders(self):
        port_configs = self.configs.robot.port_configs
        for key in self.configs.robot.recorded_data[0]:
            self.processes.append(Process(
                        target = self._start_sim_component,
                        args = (port_configs[0],key)))

    #Function to start the xela sensor recorders
    def _start_xela_component(self,
        controller_config
    ):
        component = XelaSensorRecorder(
            controller_configs=controller_config,
            storage_path=self._storage_path
        )
        component.stream()

    #Function to start the sensor recorders
    def _init_sensor_recorders(self):
        """
        For the XELA sensors or any other sensors
        """
        for controller_config in self.configs.robot.xela_controllers:
            self.processes.append(Process(
                target = self._start_xela_component,
                args = (controller_config, )
            ))

    #Function to start the robot recorders
    def _start_robot_component(
        self, 
        robot_configs, 
        recorder_function_key):
        component = RobotInformationRecord(
            robot_configs = robot_configs,
            recorder_function_key = recorder_function_key,
            storage_path = self._storage_path
        )

        component.stream()

    #Function to start the sim recorders
    def _start_sim_component(self,port_configs, recorder_function_key):
        component = SimInformationRecord(
                   port_configs = port_configs,
                   recorder_function_key= recorder_function_key,
                   storage_path=self._storage_path
        )
        component.stream()

    #Function to start the robot recorders
    def _init_robot_recorders(self):
        # Instantiating the robot classes
        for idx, robot_controller_configs in enumerate(self.configs.robot.controllers):
            for key in self.configs.robot.recorded_data[idx]:
                self.processes.append(Process(
                    target = self._start_robot_component,
                    args = (robot_controller_configs, key, )
                ))
    
    def stop(self):
        """Send SIGINT to child recorders so they flush/close cleanly."""
        for p in self.processes:
            if p.is_alive() and p.pid:
                try:
                    os.kill(p.pid, signal.SIGINT)
                except Exception:
                    pass
        for p in self.processes:
            try:
                p.join(timeout=5.0)
            except Exception:
                pass


    

   
