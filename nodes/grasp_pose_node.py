import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import mujoco
import mujoco.viewer
import numpy as np

# --- NEW IMPORTS FOR TF ---
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R

import logging
logging.getLogger('glfw').setLevel(logging.ERROR)

# Constants
XML_PATH = "../model/scene.xml"
CAM_NAME = "end_effector_camera"
WIDTH, HEIGHT = 640, 480 # Ensure this matches your YOLO/Grasp Node expectations

class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')

        # 1. Load Model & Start Passive Viewer
        self.m = mujoco.MjModel.from_xml_path(XML_PATH)
        self.d = mujoco.MjData(self.m)
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        
        # 2. Setup Camera Renderer
        self.renderer = mujoco.Renderer(self.m, HEIGHT, WIDTH)
        
        # 3. Setup ROS Publishers
        self.pub_rgb = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.bridge = CvBridge()

        # --- TF BROADCASTER ---
        self.tf_broadcaster = TransformBroadcaster(self)

        # 4. Setup Camera Intrinsics
        cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
        fovy = self.m.cam_fovy[cam_id]
        f = 0.5 * HEIGHT / np.tan(0.5 * fovy * np.pi/180)
        cx, cy = WIDTH / 2, HEIGHT / 2
        
        self.cam_info = CameraInfo()
        self.cam_info.width, self.cam_info.height = WIDTH, HEIGHT
        self.cam_info.k = [f, 0., cx, 0., f, cy, 0., 0., 1.]
        self.cam_info.p = [f, 0., cx, 0., 0., f, cy, 0., 0., 0., 1., 0.]

        # 5. Timer (30 Hz)
        self.create_timer(1.0/30.0, self.loop)
        self.get_logger().info("MuJoCo Node Started.")

    def loop(self):
        if not self.viewer.is_running():
            rclpy.shutdown()
            return

        # Physics & Viewer Sync
        mujoco.mj_step(self.m, self.d)
        self.viewer.sync()

        # Render
        self.renderer.update_scene(self.d, camera=CAM_NAME)
        rgb = self.renderer.render()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()

        # Publish Images
        timestamp = self.get_clock().now().to_msg()
        
        msg_rgb = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        msg_rgb.header.stamp = timestamp
        msg_rgb.header.frame_id = 'camera_optical_frame'
        self.pub_rgb.publish(msg_rgb)

        msg_depth = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        msg_depth.header.stamp = timestamp
        msg_depth.header.frame_id = 'camera_optical_frame'
        self.pub_depth.publish(msg_depth)

        self.cam_info.header.stamp = timestamp
        self.cam_info.header.frame_id = 'camera_optical_frame'
        self.pub_info.publish(self.cam_info)

        # --- PUBLISH TF (World -> Camera) ---
        self.publish_camera_tf(timestamp)

    def publish_camera_tf(self, timestamp):
        # 1. Get Camera Position and Rotation from MuJoCo
        cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
        pos = self.d.cam_xpos[cam_id]
        mat = self.d.cam_xmat[cam_id].reshape(3, 3)

        # 2. Coordinate Conversion (MuJoCo -> ROS Optical)
        # MuJoCo Camera: -Z is look direction, +Y is up
        # ROS Optical: +Z is look direction, +Y is down
        # We need to rotate 180 deg around X
        conversion_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        
        # Apply conversion
        final_rot_matrix = mat @ conversion_matrix
        quat = R.from_matrix(final_rot_matrix).as_quat() # x, y, z, w

        # 3. Create Transform Message
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = "world" # Or "panda_link0" if you prefer
        t.child_frame_id = "camera_optical_frame"

        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def destroy_node(self):
        self.viewer.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MujocoNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()