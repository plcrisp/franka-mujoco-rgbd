#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import subprocess
import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as R

# --- CONFIG ---
GPD_PATH = "/home/pedro/gpd/build/detect_grasps"
CFG_PATH = "/home/pedro/gpd/cfg/ros_eigen_params.cfg"
TEMP_PCD_PATH = "/tmp/temp_grasp.pcd"

class GraspDetector(Node):
    def __init__(self):
        super().__init__('grasp_detector_node')
        self.bridge = CvBridge()
        
        # Subscribers (Sync RGB, Depth, Mask)
        self.sub_rgb = message_filters.Subscriber(self, Image, '/camera/rgb/image_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.sub_mask = message_filters.Subscriber(self, Image, '/perception/mask')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_mask], 10, 0.1)
        self.ts.registerCallback(self.callback)

        # TF and Publishers
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.grasp_pub = self.create_publisher(PoseStamped, '/grasp_pose', 10)
        self.pcd_pub = self.create_publisher(PointCloud2, '/grasp/debug_cloud', 10)
        
        self.camera_info = None
        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)
        
        self.get_logger().info("GPD Node ready.")

    def info_callback(self, msg):
        self.camera_info = msg

    def callback(self, rgb_msg, depth_msg, mask_msg):
        if self.camera_info is None: return

        # 1. Convert Images
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        mask = self.bridge.imgmsg_to_cv2(mask_msg, "mono8")

        # 2. Filter Valid Pixels
        valid_indices = np.where(mask > 128)
        if len(valid_indices[0]) < 50: return 

        # 3. Generate Point Cloud (Camera Frame)
        # Correction factor to align cloud with simulation scale
        correction_factor = 0.86

        fx = self.camera_info.k[0] * correction_factor
        fy = self.camera_info.k[4] * correction_factor
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        u = valid_indices[1]
        v = valid_indices[0]
        z = depth[v, u]
        
        # Simple depth filter
        valid_z = (z > 0.1) & (z < 1.5)
        u = u[valid_z]; v = v[valid_z]; z = z[valid_z]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points = np.stack((x, y, z), axis=-1)

        # Publish raw cloud for debugging (RViz handles TF)
        self.publish_debug_cloud(points, rgb_msg.header.frame_id)

        # Save cloud for GPD (Camera Frame)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(TEMP_PCD_PATH, pcd, write_ascii=True)

        # 4. Run GPD Binary
        cmd = [GPD_PATH, CFG_PATH, TEMP_PCD_PATH]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            best_grasp = self.parse_gpd_output(result.stdout)
            
            if best_grasp:
                # Transform from Camera Frame -> Robot Frame
                self.transform_and_publish(best_grasp, rgb_msg.header.frame_id)
                
        except Exception as e:
            self.get_logger().error(f"GPD Execution Error: {e}")

    def parse_gpd_output(self, text):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if "Grasp 0 Score" in line:
                pos_line = lines[i+1]
                rot_line = lines[i+2]
                if "DATA_POS:" in pos_line:
                    # Parse Position and Rotation
                    pos = [float(x) for x in pos_line.split("DATA_POS:")[1].split()]
                    rot_floats = [float(x) for x in rot_line.split("DATA_ROT:")[1].split()]
                    rot_matrix = np.array(rot_floats).reshape(3,3)
                    quat = R.from_matrix(rot_matrix).as_quat()
                    return {'pos': pos, 'quat': quat}
        return None

    def transform_and_publish(self, grasp, source_frame):
        # Create Pose in Camera Frame
        pose_cam = PoseStamped()
        pose_cam.header.frame_id = source_frame
        pose_cam.header.stamp = self.get_clock().now().to_msg()
        pose_cam.pose.position.x, pose_cam.pose.position.y, pose_cam.pose.position.z = grasp['pos']
        pose_cam.pose.orientation.x = grasp['quat'][0]
        pose_cam.pose.orientation.y = grasp['quat'][1]
        pose_cam.pose.orientation.z = grasp['quat'][2]
        pose_cam.pose.orientation.w = grasp['quat'][3]

        try:
            # Transform to World Frame (panda_link0)
            transform = self.tf_buffer.lookup_transform(
                "panda_link0", source_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0)
            )
            pose_world = tf2_geometry_msgs.do_transform_pose(pose_cam.pose, transform)
            
            msg_out = PoseStamped()
            msg_out.header.frame_id = "panda_link0"
            msg_out.header.stamp = self.get_clock().now().to_msg()
            msg_out.pose = pose_world
            
            self.get_logger().info("Grasp published (transformed to panda_link0).")
            self.grasp_pub.publish(msg_out)

        except Exception as e:
            self.get_logger().warn(f"TF Error: {e}")

    def publish_debug_cloud(self, points, frame_id):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()
        
        self.pcd_pub.publish(msg)

def main():
    rclpy.init()
    node = GraspDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()