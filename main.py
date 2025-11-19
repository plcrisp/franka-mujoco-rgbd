import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import mujoco
import mujoco.viewer
import numpy as np 
import cv2

import warnings
warnings.filterwarnings("ignore")

XML_PATH = "model/scene.xml"
CAMERA_NAME = "end_effector_camera"

try:
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
except Exception as e:
    print(f"Error loading the model: {e}")
    print(f"Check if the file '{XML_PATH}' exists and if the internal references are correct.")
    exit()

ready_state = np.array([0., -1/4 * np.pi, 0., -3/4 * np.pi, 0., 1/2 * np.pi, 1/4 * np.pi])
m.qpos0[:7] = -ready_state 
mujoco.mj_resetData(m, d)

print("Franka Panda loaded. Use the mouse to rotate the model.")

cam_renderer = mujoco.Renderer(m, 720, 1280)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()
        cam_renderer.update_scene(d, camera=CAMERA_NAME)
        img = cam_renderer.render()
        cv2.imshow("End Effector Camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
