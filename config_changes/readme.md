# Extra Patches and Configurations

This folder contains modified files essential for GPD (Grasp Pose Detection) to work with this project. Move each file to the directory indicated below, replacing the originals.

## 📂 File Destinations

* **`detect_grasps.cpp`** ↳ `~/gpd/src/detect_grasps.cpp`

* **`ros_eigen_params.cfg`** ↳ `~/gpd/cfg/ros_eigen_params.cfg`
    
    * In this file you need to change the path to your own. 

* **`lenet_15_channels.json`** ↳ `~/gpd/models/lenet/15chanels/params/lenet_15_channels.json`

---

## ⚙️ Mandatory Compilation

After moving the `.cpp` file, you **must recompile GPD** for the changes to take effect:

```bash
cd ~/gpd/build
make