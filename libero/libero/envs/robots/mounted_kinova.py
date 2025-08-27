import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


# CHANGE 1: Rename the class
class MountedKinova(ManipulatorModel):
    """
    Kinova Gen3 is a sensitive single-arm robot designed by Kinova.
    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        # CHANGE 2: Use the Kinova3 XML file
        super().__init__(xml_path_completion("robots/kinova3/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount" # This is correct for Sawyer

    @property
    def default_gripper(self):
        # CHANGE 3: Use the Robotiq gripper, which is default for Kinova3
        return "Robotiq85Gripper"

    @property
    def default_controller_config(self):
        # CHANGE 4: Use the Kinova3 controller config
        return "default_kinova3"

    @property
    def init_qpos(self):
        return np.array(
            [0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4]
        )
    
    
    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
