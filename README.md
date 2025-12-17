# Franka Panda MuJoCo Simulation for Grasp Detection

This repository contains a modular **MuJoCo** simulation setup for the **Franka Emika Panda** robot, integrated with **ROS 2 Humble**.

It was developed by **Pedro Crisp**, **Enzo Kozonoe**, and **Murilo Gebra** within the scope of the **Chair of Cyber-Physical Systems at Montanuniversität Leoben**. It serves as a testbed for simulating robotic manipulation, generating synthetic RGB-D datasets, and testing computer vision algorithms for grasp detection.

## 🎯 Key Features

* **Real-time Physics:** High-fidelity simulation using MuJoCo.
* **ROS 2 & MoveIt Integration:** The simulation is synchronized with MoveIt 2 for motion planning and obstacle avoidance.
* **Computer Vision Pipeline:** Simulates an RGB-D camera attached to the end-effector and integrates **YOLOv8** for real-time object segmentation.
* **Synthetic Data Generation:** Tools to automatically generate labeled datasets for training vision models.
