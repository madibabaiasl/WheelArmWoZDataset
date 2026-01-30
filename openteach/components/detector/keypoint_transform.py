import numpy as np
from copy import deepcopy as copy
from openteach.components import Component
from openteach.constants import *
from openteach.utils.vectorops import *
from openteach.utils.network import ZMQKeypointPublisher, ZMQKeypointSubscriber,ZMQButtonFeedbackSubscriber
from openteach.utils.timer import FrequencyTimer

class TransformHandPositionCoords(Component):
    def __init__(self, host, keypoint_port, transformation_port, moving_average_limit = 5):
        self.notify_component_start('keypoint position transform')
        
        # Initializing the subscriber for right hand keypoints
        self.original_keypoint_subscriber = ZMQKeypointSubscriber(host, keypoint_port, 'right')
        print(f"[TransformHandPositionCoords] Subscribed to original keypoints on {host}:{keypoint_port}")
        # Initializing the publisher for transformed right hand keypoints
        self.transformed_keypoint_publisher = ZMQKeypointPublisher(host, transformation_port)
        # Timer
        self.timer = FrequencyTimer(VR_FREQ)
        # Moving average queue
        self.moving_average_limit = moving_average_limit
        # Create a queue for moving average
        self.coord_moving_average_queue, self.frame_moving_average_queue = [], []

    def _quat_to_R(self, qx, qy, qz, qw):
        """
        Unity-style quaternion (x, y, z, w) -> 3x3 rotation matrix.
        Ensures normalization before conversion.
        """
        q = np.array([qx, qy, qz, qw], dtype=np.float64)
        n = np.linalg.norm(q)
        if n == 0.0:
            # identity if bad input
            return np.eye(3, dtype=np.float64)
        q /= n
        x, y, z, w = q

        # standard formula
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        R = np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
            [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
            [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
        ], dtype=np.float64)
        return R

    def _pose_to_T(self, px, py, pz, qx, qy, qz, qw):
        R = self._quat_to_R(qx, qy, qz, qw)
        t = np.array([[px], [py], [pz]], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = t.ravel()
        return T

    def _recv_pose(self):
        """
        Returns (px,py,pz,qx,qy,qz,qw) as floats.
        Assumes ZMQKeypointSubscriber.recv_keypoints() -> list/array of 7 floats.
        """
        arr = self.original_keypoint_subscriber.recv_keypoints()
        # Defensive checks
        if arr is None or len(arr) < 7:
            raise ValueError(f"Bad controller pose payload: {arr}")
        px, py, pz, qx, qy, qz, qw = map(float, arr[:7])
        return px, py, pz, qx, qy, qz, qw
    
    # Function to find hand coordinates with respect to the wrist
    def _translate_coords(self, hand_coords):
        return copy(hand_coords) - hand_coords[0]

    # Create a coordinate frame for the hand
    def _get_coord_frame(self, index_knuckle_coord, pinky_knuckle_coord):
        palm_normal = normalize_vector(np.cross(index_knuckle_coord, pinky_knuckle_coord))   # Current Z
        palm_direction = normalize_vector(index_knuckle_coord + pinky_knuckle_coord)         # Current Y
        cross_product = normalize_vector(np.cross(palm_direction, palm_normal))              # Current X
        return [cross_product, palm_direction, palm_normal]

    # Create a coordinate frame for the arm 
    def _get_hand_dir_frame(self, origin_coord, index_knuckle_coord, pinky_knuckle_coord):

        palm_normal = normalize_vector(np.cross(index_knuckle_coord, pinky_knuckle_coord))   # Unity space - Y
        palm_direction = normalize_vector(index_knuckle_coord + pinky_knuckle_coord)         # Unity space - Z
        cross_product = normalize_vector(index_knuckle_coord - pinky_knuckle_coord)              # Unity space - X
        
        return [origin_coord, cross_product, palm_normal, palm_direction]

    def transform_keypoints(self, hand_coords):
        translated_coords = self._translate_coords(hand_coords)
        original_coord_frame = self._get_coord_frame(
            translated_coords[self.knuckle_points[0]], 
            translated_coords[self.knuckle_points[1]]
        )

        # Finding the rotation matrix and rotating the coordinates
        rotation_matrix = np.linalg.solve(original_coord_frame, np.eye(3)).T
        transformed_hand_coords = (rotation_matrix @ translated_coords.T).T
        
        hand_dir_frame = self._get_hand_dir_frame(
            hand_coords[0],
            translated_coords[self.knuckle_points[0]], 
            translated_coords[self.knuckle_points[1]]
        )

        return transformed_hand_coords, hand_dir_frame

    def stream(self):
        while True:
            try:
                self.timer.start_loop()

                px, py, pz, qx, qy, qz, qw = self._recv_pose()
                T = self._pose_to_T(px, py, pz, qx, qy, qz, qw)

                self.transformed_keypoint_publisher.pub_keypoints(T.reshape(-1).tolist(), 'transformed_hand_frame')

                self.timer.end_loop()
            except:
                break
        
        self.original_keypoint_subscriber.stop()
        self.transformed_keypoint_publisher.stop()

        print('Stopping the keypoint position transform process.')
