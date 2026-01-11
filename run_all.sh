#!/bin/bash

# 1. Launch MoveIt
gnome-terminal --tab --title="MoveIt" -- bash -c "ros2 launch moveit_resources_panda_moveit_config demo.launch.py; exec bash"

sleep 5

# 2. Launch simulation (MuJoCo)
gnome-terminal --tab --title="MuJoCo Sim" -- bash -c "python3 nodes/simulation_node.py; exec bash"

# 3. Launch segmentation (YOLO)
gnome-terminal --tab --title="YOLO Vision" -- bash -c "python3 nodes/object_segmentation_node.py; exec bash"

# 4. Launch grasp logic
# gnome-terminal --tab --title="Grasp Logic" -- bash -c "python3 grasp_detection_node.py; exec bash"

# 5. Launch commander
gnome-terminal --tab --title="Commander" -- bash -c "python3 nodes/commander_node.py; exec bash"

echo "System started!"
