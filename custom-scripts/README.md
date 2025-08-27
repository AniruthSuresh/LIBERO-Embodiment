# LIBERO Task Rendering Instructions

This repository demonstrates how to render robot actions from the LIBERO dataset using different robot models.

---

## 1. Download Task Data

1. Visit the LIBERO dataset page:  
   [https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/main/libero_10](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/main/libero_10)

2. Download the HDF5 file for the desired task. For example:  
```bash
KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
```

Save it inside the `libero_10` dataset folder.

> Note: This file corresponds to `task_2` in the code. If you want to change the task, download the corresponding HDF5 file and update the `task_no` accordingly.

---

## 2. Render Actions with Robots

### `render.py`
- Loads the Franka Panda robot.
- Executes the actions stored in the HDF5 file.
- Saves images of the robot performing the task.

### `render_any_robot.py`
- Supports multiple robots: Franka Panda, Sawyer, or Kinova.
- Moves the selected robot to its default position.
- Saves an initial pose image.
- Useful to test if robot swapping works correctly.
