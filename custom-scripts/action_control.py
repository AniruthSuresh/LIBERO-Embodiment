import os
import cv2
import h5py
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite as suite

import robosuite.macros as macros
from robosuite.controllers import load_controller_config


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10"
task_suite = benchmark_dict[task_suite_name]()

task_id = 2
task = task_suite.get_task(task_id)
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

demo_file_path = "../libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"

with h5py.File(demo_file_path, "r") as f:
    ee_states = f["data/demo_0/actions"][()]   
    print(f"[info] Loaded {len(ee_states)} end-effector states.")

env_args = {
    "bddl_file_name": task_bddl_file,
    "robots": ["Panda"],
    "camera_heights": 256,
    "camera_widths": 256,
}


env = OffScreenRenderEnv(**env_args)
obs = env.reset()

output_dir = "output_scene_ee_replay_sawyer"
os.makedirs(output_dir, exist_ok=True)

print("[info] Moving robot using EE poses from demo...")
for t, ee in enumerate(ee_states):
    pos = ee[:3]
    quat = ee[3:]   # [qw,qx,qy,qz]
    

    action = np.concatenate([pos, quat])
    
    obs, reward, done, info = env.step(action)

    img = obs["agentview_image"][::-1, :, ::-1]
    cv2.imwrite(os.path.join(output_dir, f"frame_{t:04d}.png"), img)

    if done:
        break

env.close()
print(f"[info] Saved {len(ee_states)} frames to {output_dir}/")
