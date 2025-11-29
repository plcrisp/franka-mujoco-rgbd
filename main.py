import os
import time
import mujoco
import mujoco.viewer
import numpy as np
import cv2
import warnings

warnings.filterwarnings("ignore")

XML_PATH = "model/scene.xml"
CAMERA_NAME = "end_effector_camera"

m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

print("Loaded model with RGB-D camera.")

renderer = mujoco.Renderer(m, 720, 1280)

os.makedirs("photos", exist_ok=True)
last_photo = time.time()

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()

        renderer.update_scene(d, camera=CAMERA_NAME)
        colour = renderer.render()

        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()

        rgb_vis = cv2.cvtColor(colour, cv2.COLOR_RGB2BGR)
        cv2.imshow("End Effector Camera (RGB)", rgb_vis)
        cv2.waitKey(1)

        if time.time() - last_photo >= 10:
            t = int(time.time())

            cv2.imwrite(f"photos/{t}_rgb.png", rgb_vis)

            depth_vis = (depth * 255).astype(np.uint8)
            cv2.imwrite(f"photos/{t}_depth.png", depth_vis)

            print(f"Saved rgb_{t}.png and depth_{t}.png")
            last_photo = time.time()
