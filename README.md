# **Franka Emika Panda MuJoCo RGB-D Grasp Detection**

## 📝 **Project Description**

Developed by Pedro Crisp, Enzo Kozonoe, and Murilo Gebra at the Chair of Cyber-Physical Systems, Montanuniversität Leoben, this project implements a modular MuJoCo simulation framework integrated with ROS 2 for robotic manipulation and grasp detection.

### **Problem Statement**
The project was initiated to address three specific technical requirements for a robotic perception pipeline:

1. **Simulation Setup**: Model the Franka Emika Panda robot with a wrist-mounted RGB-D camera in a tabletop scenario using a high-fidelity simulator.

2. **Object Segmentation**: Integrate a baseline object segmentation approach, specifically YOLO, to segment color images pixel-wise.

3. **Grasp Synthesis**: Integrate a grasp pose detection method (such as GPD or Dex-Net) and combine it with the pixel-wise segmentation to derive 3D grasp poses for specific object segments.

### **Proposed Solution**
To fulfill these requirements, we engineered a ROS 2 node architecture that processes the data in stages:

- **Data Generation**: We utilized MuJoCo to simulate the robot and sensors, implementing a custom wrapper to linearize OpenGL depth buffers into metric depth maps and generating synchronized point clouds.

- **Semantic Segmentation**: We integrated YOLOv8 to process the RGB stream in real-time, outputting binary masks that isolate objects of interest from the background.

- **Targeted Grasping**: We implemented a filtering mechanism that maps 2D YOLO masks to 3D point cloud indices. These isolated clusters are then fed into GPD (Grasp Pose Detection) to sample and rank valid 6-DOF grasp candidates.

- **Motion Planning**: We employed MoveIt 2 to calculate collision-free trajectories for the selected grasp poses, executing the kinematic solutions on the simulated robot via a ROS 2 control interface.

## 🎯 **Key Features**

* **Real-time Physics**: High-fidelity simulation using MuJoCo for accurate contact dynamics and gravity compensation.

* **Computer Vision Pipeline**: Simulates an RGB-D camera attached to the end-effector and integrates YOLOv8 for real-time object segmentation.

* **Synthetic Data Generation**: Tools to automatically generate labeled datasets for training vision models.

* **ROS 2 & MoveIt Integration**: The simulation is synchronized with MoveIt 2 for motion planning and obstacle avoidance.

## 📸 **Layout & Visuals**
(Placeholder)

## ⚙️ **System Prerequisites**
Before you begin, ensure you have the following installed and configured:

* **Operating System**: Ubuntu 22.04 LTS (Recommended for ROS 2 Humble).

* **ROS 2 Humble:** Follow the [official installation guide](https://docs.ros.org/en/humble/Installation.html).

### Required Packages
Run the following commands to install the necessary system dependencies:

1. **MoveIt 2:** Motion planning framework.

    ```bash
    sudo apt install ros-humble-moveit
    ```
    
2. **Panda Configuration Package:** Resources for the Franka Emika Panda robot.

    ```bash
    sudo apt install ros-humble-moveit-resources-panda-moveit-config
    ```
    
3. **Gnome Terminal:** Required for the multi-tab launch script.
   
    ```bash
    sudo apt install gnome-terminal
    ```

4. **GPD Dependencies:**
- **Note**: If you are not using Ubuntu 22.04, please refer to the [Official GPD Installation Guide](https://github.com/atenpas/gpd?tab=readme-ov-file#install) to build dependencies from source.
- **PCL (Point Cloud Library)**:

```bash
    sudo apt install libpcl-dev\
```
- **Eigen (3.4.0)**: Install [Eigen (version 3.4.0)](https://libeigen.gitlab.io/). Strictly required. On Ubuntu 22.04, the default package is sufficient:

```bash
    sudo apt install libeigen3-dev
```




## 🚀  **Project Installation**

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

## ▶️ **How to Run**

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

## 🆕 **How to Add and Train New Objects**

If you want the robot to manipulate a new object (e.g., a banana, a specific tool, or a box), you need to add it to the simulation and retrain the neural network (YOLO) to recognize it. Follow the workflow below:

### **Add the Object to Simulation (MuJoco)**

1. Prepare the Mesh:
    * Obtain the 3D file of the object(formats `.stl` or `.obj`).
    * Place the file in the `model/objects/` folder.

2.  Edit the Scene XML (`model/scene.xml`):
    * Register the mesh in the `<asset>` section.
    * Add the object body to the `<worldbody>`.  

```xml
<asset>
    <mesh name="my_new_object_mesh" file="assets/my_object.stl" scale="0.001 0.001 0.001"/>
<asset>

<worldbody>
    <body name="my_new_object" pos="0.5 0.0 0.05">
        <freejoint/>
        <geom type="mesh" mesh="my_new_object_mesh" mass="0.1" rgba="1 1 0 1"/>
    </body>
</worldbody>
```

3. Verify: Run the following command in bash and ensure the object appears on the table.
```bash
python3 nodes/simulation_node.py
```

### **Data Collection & Labeling**
To teach the robot to "see" the new object, we need to generate a labeled dataset.

1. Automated Data Collection:
    * Run the simulation in one terminal.
    * In another terminal, run the collector script:
    ```bash
    python3 training/auto_collector.py
    ```
    * The robot will move the camera to various viewpoints and save images to `dataset_v2/images`.
    * Tip: The script will pause periodically. When it does, switch to the MuJoCo window and manually move/rotate the objects with your mouse to create variety, then press ENTER in the terminal to continue.
  
2. Annotation (Roboflow)
    * You need an account on Roboflow. If you don't have it, you can create it [here](https://app.roboflow.com/login).
    * Create a new project and select Instance Segmentation as the type.
    * Upload the images generated in `dataset_v2/images`.
    * **Annotate**:
        * Open an image.
        * Select the Smart Polygon tool (Magic Wand/Lightning icon).
        * Click on your new object to automatically generate the mask.
        * Assign a class name (e.g., banana).
    * Go to the "Generate" step.
    * Click "Export Dataset", select YOLOv8, and download the `.zip` file.
  
### **Training & Integration**

1. Prepare Workspace:
    * Extract the downloaded `.zip` into the project folder (e.g., `training/dataset`).
    * Ensure the `data.yaml` file inside correctly points to the images.

2. Train the Model:
    * Open `training/train_yolo.py` and update the `data` argument to point to the new `data.yaml`.
    * Run the training script:
   ```bash
   python3 training/train_yolo.py
   ```
   * Wait for the process to finish. The new weights will be saved at `runs/segment/YOUR_RUN_NAME/weights/best.pt`.
  
3. Update the Robot Code:
    * Copy the generated `best.pt` to the `training/` folder.
    * Update Segmentation Node (`nodes/object_segmentation_node.py`):
   ```python
   self.model = YOLO("training/best.pt") # Ensure it points to the new file
   ```
   * Update Commander Node (`nodes/commander_node.py`): Add the new object to the menu so you can select it:
   ```python
   print("4. New Object")
   # ... inside the loop ...
       elif choice == '4':
           node.set_target("new_object") # Must match the class name used in Roboflow
           # ...
   ```
