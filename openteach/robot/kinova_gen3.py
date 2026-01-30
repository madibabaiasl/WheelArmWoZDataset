from openteach.ros_links.kinova_gen3_control import DexArmControl
from .robot import RobotWrapper

class KinovaGen3(RobotWrapper):
    def __init__(self,record_type=None):
        self._controller = DexArmControl(record_type=record_type, robot_type='kinova_gen3')
        self._data_frequency = 60

    @property
    def name(self):
        return 'kinova_gen3'

    @property
    def recorder_functions(self):
        return{
            'joint_states': self.get_joint_state,
            'cartesian_states': self.get_cartesian_state,
            'wheelchair_joy_commands': self.get_wheelchair_commands,
            'wheelchair_states': self.get_wheelchair_states,
            'imu': self.get_imu
        }

    @property
    def data_frequency(self):
        return self._data_frequency
    
    def get_imu(self):
        return self._controller.get_imu()
    
    def get_cartesian_state(self): # this includes timestamps
        return self._controller.get_arm_cartesian_state()

    def get_cartesian_position(self): # !!!this is cartesion pose instead of positition, update after testing!!! 
        return self._controller.get_arm_cartesian_coords()

    def get_joint_state(self):
        return self._controller.get_arm_joint_state()

    def get_joint_position(self):
        return self._controller.get_arm_position()

    def get_joint_velocity(self):
        return self._controller.get_arm_velocity()

    def get_joint_torque(self):
        return self._controller.get_arm_torque()
    
    def get_gripper_state(self):
        return self._controller.get_gripper_state()
    
    def get_wheelchair_commands(self):
        return self._controller.get_whill_joy()
    
    def get_wheelchair_states(self):
        return self._controller.get_whill_states()

    # Movement functions
    def home(self):
        return self._controller.home_arm()

    def move(self, input_angles):
        self._controller.move_arm(input_angles)

    def move_coords(self, input_coords):
        self._controller.move_arm_cartesian(cartesian_coords)
    
    def move_velocity(self, input_velocity_values, duration):
        self._controller.move_arm_cartesian_velocity(input_velocity_values, duration) 
    
    def move_gripper(self, input_gripper_value):
        self._controller.move_gripper(input_gripper_value)
    
    def move_whillchair(self, x: float, y: float):
        self._controller.move_whillchair(x, y)
    # @abstractmethod
    # def reset(self):
    #     pass

    # @abstractmethod
    # def arm_control(self):
    #     pass

    # @abstractmethod
    # def set_gripper_state(self):
    #     pass
