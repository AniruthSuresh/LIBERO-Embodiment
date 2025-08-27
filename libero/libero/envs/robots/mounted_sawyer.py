import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion

# CHANGE 1: Rename the class
class MountedSawyer(ManipulatorModel):
    """
    Sawyer is a sensitive single-arm robot designed by Rethink Robotics.
    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/sawyer/robot.xml"), idn=idn)


    @property
    def default_mount(self):
        return "RethinkMount" # This is correct for Sawyer

    @property
    def default_gripper(self):
        # CHANGE 3: Use the Sawyer gripper
        return "RethinkGripper"

    @property
    def default_controller_config(self):
        # CHANGE 4: Use the Sawyer controller config
        return "default_sawyer"

    @property
    def init_qpos(self):
        # CHANGE 5: Use Sawyer's default joint positions
        return np.array([-0.395, -0.995, 0.005, 1.44, -0.005, 0.45, -0.005])

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
