import zmq
from openteach.constants import VR_FREQ,  ARM_LOW_RESOLUTION, ARM_HIGH_RESOLUTION ,ARM_TELEOP_STOP,ARM_TELEOP_CONT
from openteach.components import Component
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import create_pull_socket, ZMQKeypointPublisher, ZMQButtonFeedbackSubscriber

            
class OculusVRHandDetector(Component):
    def __init__(self, 
        host, 
        oculus_port, 
        gripper_receiver_index_port,
        gripper_receiver_grip_port, 
        joystick_receiver_port, 
        keypoint_pub_port, 
        gripper_index_port, 
        gripper_grip_port, 
        wheelchair_port, 
        button_port,
        button_publish_port,
        teleop_reset_port, 
        teleop_reset_publish_port, 
        tele_port, 
        tele_publish_port,
        home_port,
        home_pub_port):
        self.notify_component_start('vr detector') 
        # Initializing the network socket for getting the raw right hand keypoints
        self.raw_keypoint_socket = create_pull_socket(host, oculus_port)
        print(f"[OculusVRHandDetector: INFO] Raw keypoint socket created on {host}:{oculus_port}")
        self.raw_joystick_socket = create_pull_socket(host, joystick_receiver_port)
        print(f"[OculusVRHandDetector: INFO] Raw joystick socket created on {host}:{joystick_receiver_port}")
        self.button_keypoint_socket = create_pull_socket(host, button_port)
        print(f"[OculusVRHandDetector: INFO] Button keypoint socket created on {host}:{button_port}")
        self.teleop_reset_socket = create_pull_socket(host, teleop_reset_port)
        print(f"[OculusVRHandDetector: INFO] Teleop reset socket created on {host}:{teleop_reset_port}")
        self.raw_gripper_index_socket = create_pull_socket(host, gripper_receiver_index_port)
        print(f"[OculusVRHandDetector: INFO] Gripper index socket created on {host}:{gripper_receiver_index_port}")
        self.raw_gripper_grip_socket = create_pull_socket(host, gripper_receiver_grip_port)
        print(f"[OculusVRHandDetector: INFO] Gripper grip socket created on {host}:{gripper_receiver_grip_port}")
        self.tele_gui_socket = create_pull_socket(host, tele_port)
        print(f"[OculusVRHandDetector: INFO] Teleoperation GUI socket created on {host}:{tele_port}")
        self.raw_home_socket = create_pull_socket(host, home_port)
        print(f"[OculusVRHandDetector: INFO] Home button socket created on {host}:{home_port}")
        
        # context = zmq.Context()
        # self.raw_gripper_socket = context.socket(zmq.PULL)
        # self.raw_gripper_socket.bind("tcp://*:{}".format(gripper_receiver_port))
        # print(f"[OculusVRHandDetector: INFO] Raw gripper socket created on {host}:{gripper_receiver_port}")


        # ZMQ Keypoint publisher
        self.hand_keypoint_publisher = ZMQKeypointPublisher(
            host = host,
            port = keypoint_pub_port
        )

        self.gripper_index_publisher = ZMQKeypointPublisher(
            host = host,
            port = gripper_index_port
        )

        self.gripper_grip_publisher = ZMQKeypointPublisher(
            host = host,
            port = gripper_grip_port
        )

        self.wheelchair_publisher = ZMQKeypointPublisher(
            host = host,
            port = wheelchair_port
        )

        # Socket For Resolution Button
        self.button_socket_publisher = ZMQKeypointPublisher(
            host =host,
            port =button_publish_port
        ) 
        # Socket For Teleop Reset
        self.pause_info_publisher = ZMQKeypointPublisher(
            host = host,
            port = teleop_reset_publish_port
        )

        #Socket For GUI
        self.tele_gui_publisher = ZMQKeypointPublisher(
            host = host,
            port = tele_publish_port
        )

        # Socket For Home Button
        self.home_button_publisher = ZMQKeypointPublisher(
            host = host,
            port = home_pub_port
        )

        self.timer = FrequencyTimer(VR_FREQ)

    # Function to process a token received from the VR
    def _process_data_token(self, data_token):
        return data_token.decode().strip()

    # Function to Extract the Keypoints from the String Token sent by the VR
    
    def _extract_data_from_token(self, token):
        """Parse a single incoming token from Unity into a structured dict.

        Supports:
          - controllerL: "controllerL:px,py,pz|qx,qy,qz,qw"
          - gripper:     "gripper_left_index:VAL:" or "gripper_left_grip:VAL:"
          - joystick:    "joystick:x,y"
        """
        data = self._process_data_token(token)
        # print(f"[OculusVRHandDetector: INFO] Received data token: {data}")
        information = {}

        try:
            if data.startswith("controllerL:"):
                _, content = data.split(":", 1)
                pos_str, rot_str = content.strip(":").split("|")
                px, py, pz = map(float, pos_str.split(","))
                qx, qy, qz, qw = map(float, rot_str.split(","))
                information["type"] = "controllerL"
                # Pack as [px,py,pz,qx,qy,qz,qw]
                information["keypoints"] = [px, py, pz, qx, qy, qz, qw]
            
            if data.startswith("gripper_left_index:"):
                _, content = data.split(":", 1)
                val_str = content.strip(":")
                val = float(val_str) if val_str else 0.0
                # print(f"[OculusVRHandDetector: INFO] Gripper index value: {val}")
                information["type"] = "gripper_index"
                information["gripper_index"] = val

            if data.startswith("gripper_left_grip:"):
                _, content = data.split(":", 1)
                val_str = content.strip(":")
                val = float(val_str) if val_str else 0.0
                # print(f"[OculusVRHandDetector: INFO] Gripper grip value: {val}")
                information["type"] = "gripper_grip"
                information["gripper_grip"] = val

            if data.startswith("joystick:"):
                _, content = data.split(":", 1)
                x_str, y_str = content.strip(":").split(",")
                information["type"] = "joystick"
                information["joystick"] = [float(x_str), float(y_str)]

        except Exception as e:
            print(f"[ERROR] Failed to parse token: {e}")
            information["type"] = "error"
            information["raw"] = data

        return information

    def _publish_parsed_data(self, information):
        """
        Publishes parsed controllerL, gripper, or joystick data
        using self.hand_keypoint_publisher.pub_keypoints().
        """
        if information["type"] == "controllerL":
            # Publish controller pose
            self.hand_keypoint_publisher.pub_keypoints(
                keypoint_array=information["keypoints"], 
                topic_name='right'
            )

        elif information["type"] == "gripper_index":
            # Wrap gripper value in a list so it's a numeric array
            self.gripper_index_publisher.pub_keypoints(
                keypoint_array=[information["gripper_index"]], 
                topic_name='gripper_index'
            )
        
        elif information["type"] == "gripper_grip":
            # Wrap gripper value in a list so it's a numeric array
            self.gripper_grip_publisher.pub_keypoints(
                keypoint_array=[information["gripper_grip"]], 
                topic_name='gripper_grip'
            )

        elif information["type"] == "joystick":
            # Publish joystick XY
            self.wheelchair_publisher.pub_keypoints(
                keypoint_array=information["joystick"], 
                topic_name='joystick'
            )

        else:
            print(f"[WARN] Unsupported type for publishing: {information['type']}")


    # Function to Publish the Resolution Button Feedback
    def _publish_button_data(self,button_feedback):
        self.button_socket_publisher.pub_keypoints(
            keypoint_array = button_feedback, 
            topic_name = 'button'
        )

    # Function to Publish the Teleop Reset Status
    def _publish_pause_data(self,pause_status):
        self.pause_info_publisher.pub_keypoints(
            keypoint_array = pause_status, 
            topic_name = 'pause'
        )
    
    def _publish_tele_gui_data(self,gui_info):
        self.tele_gui_publisher.pub_keypoints(
            keypoint_array = gui_info, 
            topic_name = 'tele_gui'
        )
    
    def _publish_home_button_data(self,home_info):
        self.home_button_publisher.pub_keypoints(
            keypoint_array = home_info, 
            topic_name = 'home_button'
        )

    def stream(self):
        poller = zmq.Poller()
        # Register sockets for non-blocking polling
        poller.register(self.raw_keypoint_socket, zmq.POLLIN)
        poller.register(self.raw_gripper_index_socket, zmq.POLLIN)
        poller.register(self.raw_gripper_grip_socket, zmq.POLLIN)
        poller.register(self.raw_joystick_socket, zmq.POLLIN)
        poller.register(self.button_keypoint_socket, zmq.POLLIN)
        poller.register(self.teleop_reset_socket, zmq.POLLIN)
        poller.register(self.tele_gui_socket, zmq.POLLIN)
        poller.register(self.raw_home_socket, zmq.POLLIN)

        while True:
            try:
                self.timer.start_loop()
                events = dict(poller.poll(timeout=20))  # ~20ms tick

                # Controller pose
                if self.raw_keypoint_socket in events:
                    token = self.raw_keypoint_socket.recv()
                    # print(f"[OculusVRHandDetector: INFO] Received controller L: {token}")
                    info = self._extract_data_from_token(token)
                    if info.get("type") == "controllerL":
                        # last["controllerL"] = info
                        self._publish_parsed_data(info)

                if self.raw_gripper_index_socket in events:
                    token = self.raw_gripper_index_socket.recv()
                    # print(f"[OculusVRHandDetector: INFO] Received raw gripper: {token}")
                    info = self._extract_data_from_token(token)
                    # print(f"[OculusVRHandDetector: INFO] Extracted gripper data: {info}")
                    if info.get("type") == "gripper_index":
                        # last["gripper"] = info
                        self._publish_parsed_data(info)
                
                if self.raw_gripper_grip_socket in events:
                    token = self.raw_gripper_grip_socket.recv()
                    # print(f"[OculusVRHandDetector: INFO] Received raw gripper: {token}")
                    info = self._extract_data_from_token(token)
                    # print(f"[OculusVRHandDetector: INFO] Extracted gripper data: {info}")
                    if (info.get("type") == "gripper_grip"):
                        self._publish_parsed_data(info)

                # Joystick
                if self.raw_joystick_socket in events:
                    token = self.raw_joystick_socket.recv()
                    # print(f"[OculusVRHandDetector: INFO] Received raw joystick: {token}")
                    info = self._extract_data_from_token(token)
                    if info.get("type") == "joystick":
                        # last["joystick"] = info
                        self._publish_parsed_data(info)

                # Resolution button feedback
                if self.button_keypoint_socket in events:
                    button_feedback = self.button_keypoint_socket.recv()
                    # Map text bytes b'Low'/b'High' -> numeric constants
                    # print(f"[OculusVRHandDetector: INFO] Received button feedback: {button_feedback}")
                    if button_feedback == b'Low':
                        button_feedback_num = ARM_LOW_RESOLUTION
                    else:
                        button_feedback_num = ARM_HIGH_RESOLUTION
                    self._publish_button_data([float(button_feedback_num)])

                if self.tele_gui_socket in events:
                    # print(f"[OculusVRHandDEtector: INFO] tele socket")
                    gui_info = self.tele_gui_socket.recv()
                    # print(f"[OculusVRHandDetector: INFO] Received teleop GUI info: {gui_info}")
                    gui_val_str = gui_info.decode().strip()  # "1" or "0"
                    # print(f"[OculusVRHandDetector: INFO] Teleop GUI value: {gui_val_str}")
                    if gui_val_str == "1":
                        gui_val = 1.0
                    elif gui_val_str == "0":
                        gui_val = 0.0
                    self._publish_tele_gui_data([gui_val])
                
                if self.raw_home_socket in events:
                    home_info = self.raw_home_socket.recv()
                    # print(f"[OculusVRHandDetector: INFO] Received home button info: {home_info}")
                    home_val_str = home_info.decode().strip()
                    if home_val_str == "1":
                        home_val = 1.0
                    self._publish_home_button_data([home_val])

                self.timer.end_loop()
            except KeyboardInterrupt:
                print("[OculusVRHandDetector: INFO] Stopped by user.")
                break
            except Exception as e:
                print(f"[OculusVRHandDetector: ERROR] stream loop error: {e}")
                # keep looping
                self.timer.end_loop()

if __name__ == "__main__":
    detector = OculusVRHandDetector(
        host="10.178.46.172",
        oculus_port=8087,
        gripper_receiver_index_port=8110,
        gripper_receiver_grip_port=8113,
        joystick_receiver_port=8111,
        keypoint_pub_port=8088,
        gripper_index_port=8115,
        gripper_grip_port=8112,
        wheelchair_port=8122,
        button_port=8095,
        button_publish_port=8093,
        teleop_reset_port=8100,
        teleop_reset_publish_port=8102
    )
    detector.stream()

