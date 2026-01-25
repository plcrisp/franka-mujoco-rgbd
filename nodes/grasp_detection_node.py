#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
import subprocess
import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as R
import time
from std_msgs.msg import Empty
from pathlib import Path

# --- CONFIG ---
HOME = Path.home()

GPD_PATH = HOME / "gpd" / "build" / "detect_grasps"
CFG_PATH = HOME / "gpd" / "cfg" / "ros_eigen_params.cfg"
TEMP_PCD_PATH = Path("/tmp/temp_grasp.pcd")

# Performance
VOXEL_SIZE = 0.005  # 5 mm

# Top-down filter
# Closer to 1.0 means more vertical
MIN_VERTICAL_ALIGNMENT = 1


class GraspDetector(Node):
    def __init__(self):
        super().__init__('grasp_detector_node')
        self.bridge = CvBridge()

        # Image subscribers (synchronized)
        self.sub_rgb = message_filters.Subscriber(self, Image, '/camera/rgb/image_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.sub_mask = message_filters.Subscriber(self, Image, '/perception/mask')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_mask], 10, 0.1
        )
        self.ts.registerCallback(self.callback)

        # TF and publishers
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publish only the best grasp
        self.grasp_pub = self.create_publisher(PoseStamped, '/grasp_pose', 10)
        self.pcd_pub = self.create_publisher(PointCloud2, '/grasp/debug_cloud', 10)

        self.camera_info = None
        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)

        self.create_subscription(Empty, '/grasp/reset', self.reset_callback, 10)

        # Start paused to save CPU
        self.paused = True
        self.get_logger().info("GPD node started (paused, waiting for reset)")

    def info_callback(self, msg):
        # Store camera intrinsics
        self.camera_info = msg

    def reset_callback(self, msg):
        # Allow new grasp detection
        self.get_logger().info("Reset received, enabling grasp detection")
        self.paused = False

    def callback(self, rgb_msg, depth_msg, mask_msg):
        if self.paused:
            return

        if self.camera_info is None:
            return

        # 1. Image processing
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        mask = self.bridge.imgmsg_to_cv2(mask_msg, "mono8")

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        valid_indices = np.where(mask > 128)
        if len(valid_indices[0]) < 50:
            return

        # 2. Point cloud generation
        fx = self.camera_info.k[0] * 0.72
        fy = self.camera_info.k[4] * 1.05
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        u = valid_indices[1]
        v = valid_indices[0]
        z = depth[v, u]

        valid_z = (z > 0.1) & (z < 1.5)
        u = u[valid_z]
        v = v[valid_z]
        z = z[valid_z]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack((x, y, z), axis=-1)

        # 3. Downsampling and filtering
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        pcd, _ = pcd.remove_radius_outlier(nb_points=4, radius=0.02)

        if len(pcd.points) < 10:
            return

        # Save and publish debug cloud
        final_points = np.asarray(pcd.points)
        self.publish_debug_cloud(final_points, rgb_msg.header.frame_id)
        o3d.io.write_point_cloud(TEMP_PCD_PATH, pcd, write_ascii=True)

        # 4. Run GPD
        cmd = [GPD_PATH, CFG_PATH, TEMP_PCD_PATH]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            # 5. Select best top-down grasp
            best_pose_world = self.find_best_top_down_grasp(
                result.stdout, rgb_msg.header.frame_id
            )

            if best_pose_world:
                self.grasp_pub.publish(best_pose_world)
                self.paused = True
                self.get_logger().info("Top-down grasp found and published")
            else:
                self.get_logger().warn("No valid top-down grasp found")

        except Exception as e:
            self.get_logger().error(f"GPD execution error: {e}")

    def find_best_top_down_grasp(self, text, source_frame):
        lines = text.split('\n')

        # Use time zero to avoid TF extrapolation issues
        time_zero = rclpy.time.Time().to_msg()

        if not self.tf_buffer.can_transform(
            "panda_link0", source_frame, rclpy.time.Time()
        ):
            return None

        # Z axis must point downwards
        MIN_Z_SCORE = -0.8

        for i, line in enumerate(lines):
            if "Grasp" in line and "Score" in line:
                try:
                    pos_line = lines[i + 1]
                    rot_line = lines[i + 2]

                    if "DATA_POS:" not in pos_line:
                        continue

                    pos = [float(x) for x in pos_line.split("DATA_POS:")[1].split()]
                    rot_part = rot_line.split("DATA_ROT:")[1] if "DATA_ROT:" in rot_line else rot_line
                    rot_floats = [float(x) for x in rot_part.split()]
                    rot_matrix = np.array(rot_floats).reshape(3, 3)
                    quat = R.from_matrix(rot_matrix).as_quat()

                    # Pose in camera frame
                    p_cam = PoseStamped()
                    p_cam.header.frame_id = source_frame
                    p_cam.header.stamp = time_zero
                    p_cam.pose.position.x = float(pos[0])
                    p_cam.pose.position.y = float(pos[1])
                    p_cam.pose.position.z = float(pos[2])
                    p_cam.pose.orientation.x = float(quat[0])
                    p_cam.pose.orientation.y = float(quat[1])
                    p_cam.pose.orientation.z = float(quat[2])
                    p_cam.pose.orientation.w = float(quat[3])

                    # Transform to world frame
                    p_world = self.tf_buffer.transform(
                        p_cam, "panda_link0",
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )

                    # Check world Z axis
                    q_world = [
                        p_world.pose.orientation.x,
                        p_world.pose.orientation.y,
                        p_world.pose.orientation.z,
                        p_world.pose.orientation.w
                    ]
                    mat_world = R.from_quat(q_world).as_matrix()
                    axis_z_world = mat_world[:, 2]
                    vertical_score_z = axis_z_world[2]

                    if vertical_score_z < MIN_Z_SCORE:
                        self.get_logger().info(
                            f"Top-down grasp selected (Z score: {vertical_score_z:.2f})"
                        )
                        return p_world

                except Exception:
                    continue

        return None

    def is_top_down(self, pose):
        # Check if gripper points down
        q = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ]
        rot_matrix = R.from_quat(q).as_matrix()

        approach = rot_matrix[:, 0]

        return approach[2] < -MIN_VERTICAL_ALIGNMENT

    def publish_debug_cloud(self, points, frame_id):
        # Publish point cloud for visualization
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
