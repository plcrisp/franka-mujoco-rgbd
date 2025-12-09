import os
import time
import mujoco
import mujoco.viewer
import numpy as np
import cv2

XML_PATH = "model/scene.xml"
CAMERA_NAME = "end_effector_camera"

CLASS_MAPPING = {
    0: {"name": "hammer", "radius": 0.15},    # Approximate size of the object (meters)
    1: {"name": "mug", "radius": 0.08},
    2: {"name": "bottle", "radius": 0.12}
}


def get_pixel_coord(cam_pos, cam_mat, obj_pos, f, cx, cy):
    # Transform object position to camera coordinate frame
    res = obj_pos - cam_pos
    cam_coord = cam_mat.T @ res

    x_cv = cam_coord[0]
    y_cv = -cam_coord[1]
    z_cv = -cam_coord[2]

    if z_cv <= 0.01: # 0.01 near clipping plane
        return None

    # Projection Pinhole to 2D pixel coordinates
    u = f * (x_cv / z_cv) + cx
    v = f * (y_cv / z_cv) + cy
    
    # Return u, v and depth (to calculate box size)
    return u, v, z_cv


m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

# print("RGB-D + PointCloud capture enabled.")

renderer = mujoco.Renderer(m, 720, 1280)

os.makedirs("photos", exist_ok=True)
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)
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

        # renderer.enable_depth_rendering()
        # depth = renderer.render()
        # renderer.disable_depth_rendering()
        
        rgb_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("End Effector Camera (RGB)", rgb_vis)
        cv2.waitKey(1)

        if time.time() - last_photo >= 0.2:
            t = int(time.time())

            # Camera position and orientation at this timestep
            cam_pos = d.cam_xpos[cam_id]
            cam_mat = d.cam_xmat[cam_id].reshape(3, 3)

            labels = []

            for cls_id, props in CLASS_MAPPING.items():
                try:
                    obj_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, props["name"])
                    if obj_body_id == -1:
                        continue
                    obj_pos = d.xpos[obj_body_id]

                    res = get_pixel_coord(cam_pos, cam_mat, obj_pos, f, cx, cy)

                    if res:
                        u, v, depth = float(res[0]), float(res[1]), float(res[2])

                        if 0 <= u < W and 0 <= v < H:
                            f_val = float(f)
                            rad_val = float(props["radius"])
                            
                            box_size_px = int(f_val * 2 * rad_val / depth)

                            # Normalize coordinates for YOLO format
                            x_center = float(u / W)
                            y_center = float(v / H)
                            width = float(box_size_px*2 / W)
                            height = float(box_size_px*2 / H)

                            labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                            
                            # Draw bounding box on RGB visualization
                            b_half = int(box_size_px / 2)
                            p1 = (int(u - b_half), int(v - b_half))
                            p2 = (int(u + b_half), int(v + b_half))
                            cv2.rectangle(rgb_vis, p1, p2, (0, 255, 0), 2)
                                
                except Exception as e:
                    print(f"Error processing object {props['name']}: {e}")

            if labels:
                cv2.imwrite(f"dataset/images/{t}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                with open(f"dataset/labels/{t}.txt", "w") as f_txt:
                    f_txt.write("\n".join(labels))
                # print(f"Saved dataset/{t} with {len(labels)} objects")

            cv2.imshow("Preview", rgb_vis) # Mostra com as boxes desenhadas
            last_photo = time.time()

            # # SAVE DEPTH VISUALIZATION
            # depth_vis = (depth / depth.max() * 255).astype(np.uint8)
            # cv2.imwrite(f"photos/{t}_depth.png", depth_vis)

            # # SAVE DEPTH REAL (float32)
            # np.save(f"photos/{t}_depth.npy", depth)

            # print(f"Saved RGB, Depth PNG and Depth NPY at time {t}")

            # last_photo = time.time()
