"""
Dont't use this script. It is just for reference & still under development.
"""

import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R


# --- 1. Task Setup (Unchanged) ---
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10"
task_suite = benchmark_dict[task_suite_name]()

task_id = 2
task = task_suite.get_task(task_id)
task_name = task.name
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] Task: '{task_name}'")


demo_file_path = "../libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"

HDF5_KEY = "data/demo_0/obs/ee_states" 

print(f"[info] Loading 6D (XYZ, RPY) pose from key '{HDF5_KEY}'...")
with h5py.File(demo_file_path, "r") as f:

    pose_6d_traj = f[HDF5_KEY][()]
    eef_pos_traj = pose_6d_traj[:, :3] # First 3 values are position
    eef_rpy_traj = pose_6d_traj[:, 3:] # Rest 3 are orientation (RPY)
    
    gripper_actions = f["data/demo_0/actions"][:, -1]
    initial_state = f["data/demo_0/states"][0]

num_steps = len(eef_pos_traj)
print(f"[info] Loaded {num_steps} target poses for the trajectory.")


print("[info] Setting up environment with JOINT_POSITION controller for direct IK control...")


env_args = {
    "bddl_file_name": task_bddl_file,
    "robots": ["Panda"],
    "controller": "IK_POSE", 
    "camera_heights": 256,
    "camera_widths": 256,
}
env = OffScreenRenderEnv(**env_args)

obs = env.reset()
env.sim.set_state_from_flattened(initial_state)
env.sim.forward()


env.env._update_observables(force=True)
obs = env.env._get_observations()


output_dir = "output_scene_ik_retarget_sawyer"
os.makedirs(output_dir, exist_ok=True)
print(f"[info] Output frames will be saved in '{output_dir}/'")


print("[info] Moving robot by calculating joint angles with Inverse Kinematics...")

last_valid_joints = obs['robot0_joint_pos']
for t in tqdm(range(num_steps), desc="Sending EE Poses"):
    target_pos = eef_pos_traj[t]
    target_rpy = eef_rpy_traj[t]


    target_rot_mat = T.euler2mat(target_rpy)
    r = R.from_matrix(target_rot_mat)

    target_axis_angle = r.as_rotvec()

    action = np.concatenate([target_pos, target_axis_angle, [gripper_actions[t]]])
    
    obs, reward, done, info = env.step(action)

    img = obs["agentview_image"][::-1, :, ::-1]
    cv2.imwrite(os.path.join(output_dir, f"frame_{t:04d}.png"), img)

    if done:
        break

env.close()
print(f"\n[info] Successfully saved frames to '{output_dir}/'.")