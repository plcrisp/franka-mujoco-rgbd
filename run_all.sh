#!/bin/bash

# Launch MoveIt demo (Panda robot + planning scene)
gnome-terminal --tab --title="MoveIt" -- bash -c \
"ros2 launch moveit_resources_panda_moveit_config demo.launch.py; exec bash"

# Allow MoveIt to fully initialize
sleep 5

# Launch MuJoCo simulation node
gnome-terminal --tab --title="MuJoCo Sim" -- bash -c \
"python3 nodes/simulation_node.py; exec bash"

# Launch object segmentation (YOLO-based vision)
gnome-terminal --tab --title="YOLO Vision" -- bash -c \
"python3 nodes/object_segmentation_node.py; exec bash"

# Launch grasp detection pipeline (point cloud + GPD)
gnome-terminal --tab --title="Grasp Logic" -- bash -c \
"python3 nodes/grasp_detection_node.py; exec bash"

# Optional vision diagnostic node
# gnome-terminal --tab --title="Diagnostic" -- bash -c \
# "python3 nodes/diagnostic_node.py; exec bash"

# Launch MoveIt commander (user control + grasp execution)
gnome-terminal --tab --title="Commander" -- bash -c \
"python3 nodes/commander_node.py; exec bash"

# Publish static transform between robot hand and camera frame
gnome-terminal --tab --title="TF Camera" -- bash -c \
"ros2 run tf2_ros static_transform_publisher \
 --x 0.060 --y -0.018 --z 0.160 \
 --roll 0 --pitch 0 --yaw 0.785 \
 --frame-id panda_link7 \
 --child-frame-id camera_optical_frame; exec bash"

# Final status message
echo "System started!"