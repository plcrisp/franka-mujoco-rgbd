import os
import time
import mujoco
import mujoco.viewer
import numpy as np
import cv2

XML_PATH = "model/scene.xml"
CAMERA_NAME = "end_effector_camera"

m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

print("RGB-D + PointCloud capture enabled.")

renderer = mujoco.Renderer(m, 720, 1280)

os.makedirs("photos", exist_ok=True)
last_photo = time.time()

cam_id = m.camera(CAMERA_NAME).id
fovy = m.camera(CAMERA_NAME).fovy 

H, W = 720, 1280
f = 0.5 * H / np.tan(0.5 * fovy * np.pi/180)    
cx, cy = W/2, H/2                              

print(f"Camera intrinsics: f={f}, cx={cx}, cy={cy}")



with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()

        renderer.update_scene(d, camera=CAMERA_NAME)
        rgb = renderer.render()

        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()

        rgb_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("End Effector Camera (RGB)", rgb_vis)
        cv2.waitKey(1)

        if time.time() - last_photo >= 10:
            t = int(time.time())

            # SAVE RGB
            cv2.imwrite(f"photos/{t}_rgb.png", rgb_vis)

            # SAVE DEPTH VISUALIZATION
            depth_vis = (depth / depth.max() * 255).astype(np.uint8)
            cv2.imwrite(f"photos/{t}_depth.png", depth_vis)

            # SAVE DEPTH REAL (float32)
            np.save(f"photos/{t}_depth.npy", depth)

            print(f"Saved RGB, Depth PNG and Depth NPY at time {t}")

            last_photo = time.time()
