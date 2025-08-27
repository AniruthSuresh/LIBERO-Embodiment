import os
import cv2
import h5py
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# --- 1. Basic Task Setup ---
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10"
task_suite = benchmark_dict[task_suite_name]()

task_id = 2
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] Retrieving task {task_id}: '{task_name}'")
print(f"[info] Language instruction: '{task_description}'")

print(task)

relative_demo_path = task_suite.get_task_demonstration(task_id)

datasets_base_path = get_libero_path("datasets")
demo_file_path = "../libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
print(f"[info] Found demonstration file: {os.path.basename(demo_file_path)}")

print("[info] Loading actions and initial state from demonstration file...")
with h5py.File(demo_file_path, "r") as f:
    actions = f["data/demo_0/actions"][()]
    initial_state = f["data/demo_0/states"][0]
num_actions = len(actions)
print(f"[info] Loaded a full sequence of {num_actions} actions.")

print(task_bddl_file)

env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 256, # Using a decent resolution for saved images
    "camera_widths": 256 , 
    
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()

env.set_init_state(initial_state)


output_dir = f"output_scene_{task_name}"
os.makedirs(output_dir, exist_ok=True)
print(f"[info] Full scene images will be saved in the '{output_dir}/' directory.")

print(f"[info] Starting execution of {num_actions} steps and saving all images...")
for step, action in enumerate(actions):
    obs, reward, done, info = env.step(action)
    
    image_from_step = obs["agentview_image"]
    
    filename = os.path.join(output_dir, f"step_{step:03d}.png") 
    
    cv2.imwrite(filename, image_from_step[::-1, :, ::-1])
    
    print(f"  -> Saved frame {step+1}/{num_actions}", end='\r')


print(f"\n[info] Full scene finished. Saved {num_actions} images to '{output_dir}/'.")
env.close()