import os
import time
import mujoco
import mujoco.viewer
import numpy as np
import cv2

XML_PATH = "model/scene.xml"
CAMERA_NAME = "end_effector_camera"

CLASS_MAPPING = {
    0: {"name": "hammer", "radius": 0.15},    
    1: {"name": "mug", "radius": 0.08},
    2: {"name": "bottle", "radius": 0.12}
}

def get_pixel_coord(cam_pos, cam_mat, obj_pos, f, cx, cy):
    res = obj_pos - cam_pos
    cam_coord = cam_mat.T @ res

    x_cv = cam_coord[0]
    y_cv = -cam_coord[1]
    z_cv = -cam_coord[2]

    if z_cv <= 0.01: 
        return None

    u = f * (x_cv / z_cv) + cx
    v = f * (y_cv / z_cv) + cy
    
    return u, v, z_cv

m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

renderer = mujoco.Renderer(m, 720, 1280)

cam_id = m.camera(CAMERA_NAME).id
fovy = m.camera(CAMERA_NAME).fovy 

H, W = 720, 1280
f = 0.5 * H / np.tan(0.5 * fovy * np.pi/180)    
cx, cy = W/2, H/2                              


with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()

        renderer.update_scene(d, camera=CAMERA_NAME)
        rgb = renderer.render()
        
        rgb_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Calculate camera and object positions to draw the box (Visualization)
        cam_pos = d.cam_xpos[cam_id]
        cam_mat = d.cam_xmat[cam_id].reshape(3, 3)

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
                        
                        # Draw bounding box on visualization only (Preview)
                        b_half = int(box_size_px / 2)
                        p1 = (int(u - b_half), int(v - b_half))
                        p2 = (int(u + b_half), int(v + b_half))
                        cv2.rectangle(rgb_vis, p1, p2, (0, 255, 0), 2)
                        
                        # Add text
                        cv2.putText(rgb_vis, props["name"], (p1[0], p1[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
            except Exception as e:
                pass

        # Preview
        cv2.imshow("Preview", rgb_vis) 
        cv2.waitKey(1)