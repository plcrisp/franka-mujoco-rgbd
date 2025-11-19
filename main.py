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

print("Loaded model. Use Q/A W/S E/D R/F T/G Y/H U/J to move joints.")

renderer = mujoco.Renderer(m, 720, 1280)

joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
actuator_names = ["actuator1","actuator2","actuator3","actuator4","actuator5","actuator6","actuator7"]

actuator_ids = [m.actuator(name).id for name in actuator_names]

targets = np.zeros(7)
step = 0.05

keymap = {
    ord('q'):(0,+step), ord('a'):(0,-step),
    ord('w'):(1,+step), ord('s'):(1,-step),
    ord('e'):(2,+step), ord('d'):(2,-step),
    ord('r'):(3,+step), ord('f'):(3,-step),
    ord('t'):(4,+step), ord('g'):(4,-step),
    ord('y'):(5,+step), ord('h'):(5,-step),
    ord('u'):(6,+step), ord('j'):(6,-step),
}

last_photo = time.time()

os.makedirs("photos", exist_ok=True)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)

        key = cv2.waitKey(1) & 0xFF
        if key in keymap:
            idx, delta = keymap[key]
            targets[idx] += delta
            d.ctrl[actuator_ids[idx]] = targets[idx]

        renderer.update_scene(d, camera=CAMERA_NAME)
        img = renderer.render()
        cv2.imshow("End Effector Camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if time.time() - last_photo >= 10:
            filename = f"photos/frame_{int(time.time())}.png"
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Saved {filename}")
            last_photo = time.time()

        viewer.sync()
