# Franka Panda MuJoCo Simulation for Grasp Detection

This repository contains a modular **MuJoCo** simulation setup for the **Franka Emika Panda** robot, integrated with **ROS 2 Humble**.

It was developed by **Pedro Crisp**, **Enzo Kozonoe**, and **Murilo Gebra** within the scope of the **Chair of Cyber-Physical Systems at Montanuniversität Leoben**. It serves as a testbed for simulating robotic manipulation, generating synthetic RGB-D datasets, and testing computer vision algorithms for grasp detection.

## 🎯 Key Features

* **Real-time Physics:** High-fidelity simulation using MuJoCo.
* **ROS 2 & MoveIt Integration:** The simulation is synchronized with MoveIt 2 for motion planning and obstacle avoidance.
* **Computer Vision Pipeline:** Simulates an RGB-D camera attached to the end-effector and integrates **YOLOv8** for real-time object segmentation.
* **Synthetic Data Generation:** Tools to automatically generate labeled datasets for training vision models.

## 1. System Prerequisites
Before you begin, ensure you have the following installed:
**Operating System:** Ubuntu 22.04 (Recommended for ROS 2 Humble)
**ROS 2 Humble:** [Follow the official installation guide](https://docs.ros.org/en/humble/Installation.html)
**GPD:** GPD installation and its dependencies for ubuntu, if you are not using ubuntu you should [follow the offical installation guide](https://github.com/atenpas/gpd?tab=readme-ov-file#install):
* [Install Eigen (version 3.4.0)](https://libeigen.gitlab.io/)

* Install pcl
```bash
    sudo apt install libpcl-dev\
```

**MoveIt 2:** Motion planning framework.
    ```bash
    sudo apt install ros-humble-moveit
    ```
**Panda Configuration Package:**
    ```bash
    sudo apt install ros-humble-moveit-resources-panda-moveit-config
    ```
**Gnome Terminal:** Required for the launch script.
    ```bash
    sudo apt install gnome-terminal
    ```

## 2. Project Installation

### 1. System Dependencies (APT)
Install ROS 2 packages, MoveIt, and hardware controllers.

```bash
sudo apt update
sudo apt install -y \
    ros-humble-desktop \
    ros-humble-moveit \
    ros-humble-moveit-resources-panda-moveit-config \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-message-filters \
    ros-humble-tf2-ros \
    ros-humble-tf-transformations \
    python3-numpy \
    python3-venv
```

### 2. Install Python dependencies:

**⚠️ Important Note for Virtual Environments (venv):**
This project relies on `rclpy` (ROS 2 Python client), which is a system package and **cannot** be installed via pip. If you are using a virtual environment, you must create it with the `--system-site-packages` flag to allow access to ROS libraries:
    
    # Create venv with system packages access
    ```bash
    python3 -m venv venv --system-site-packages
    ```

    If not using a venv, simply run: `pip install -r requirements.txt`.

## 3. How to Run

The project includes an automated script that starts all necessary nodes.

1.  **Grant execution permission:**
    ```bash
    chmod +x run_all.sh
    ```

2.  **Setup Environment & Run:**
    You **must** load the ROS 2 environment (and your venv, if active) before running the script to avoid "Command not found" or import errors.
    
    ```bash
    # 1. Load ROS 2 Humble
    source /opt/ros/humble/setup.bash
    
    # 2. Activate your venv (if you created one)
    # source venv/bin/activate
    
    # 3. Run the system
    ./run_all.sh
    ```

This will open several terminal tabs executing the Simulation, MoveIt, YOLO Vision, and Control nodes.

## Common Troubleshooting
* **`ModuleNotFoundError: No module named 'rclpy'`:** You are likely in a standard `venv` that cannot see system packages. Recreate your venv using the `--system-site-packages` flag (see Section 2).
* **`ros2: command not found`:** You forgot to run `source /opt/ros/humble/setup.bash` before starting.