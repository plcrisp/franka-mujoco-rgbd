#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R
import time


class VisionDiagnostic(Node):
    def __init__(self):
        super().__init__('vision_diagnostic')

        self.bridge = CvBridge()
        self.last_print_time = 0.0  # Print throttle

        # Vision subscribers
        self.sub_rgb = message_filters.Subscriber(
            self, Image, '/camera/rgb/image_raw'
        )
        self.sub_depth = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw'
        )
        self.sub_mask = message_filters.Subscriber(
            self, Image, '/perception/mask'
        )

        # Synchronization
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_mask],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.callback)

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cam_info = None
        self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_cb, 10
        )

        print("\n=== Vision diagnostic running (prints every 3s) ===\n")

    def camera_info_cb(self, msg):
        self.cam_info = msg

    def callback(self, rgb, depth, mask):
        if self.cam_info is None:
            return

        # Throttle prints
        now = time.time()
        if now - self.last_print_time < 3.0:
            return
        self.last_print_time = now

        # Convert images
        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth, "32FC1")
            mask_img = self.bridge.imgmsg_to_cv2(mask, "mono8")
        except Exception:
            return

        # Mask filtering
        v, u = np.where(mask_img > 100)
        if len(v) < 50:
            print("[Vision] Bottle not detected")
            return

        # Depth filtering
        z_vals = depth_img[v, u]
        z_vals = z_vals[(z_vals > 0.1) & (z_vals < 2.0)]
        if len(z_vals) == 0:
            print("[Vision] Invalid depth values")
            return

        # Camera frame computation
        fx, fy = self.cam_info.k[0], self.cam_info.k[4]
        cx, cy = self.cam_info.k[2], self.cam_info.k[5]

        z = np.mean(z_vals)
        u_m = np.mean(u)
        v_m = np.mean(v)

        x = (u_m - cx) * z / fx
        y = (v_m - cy) * z / fy

        point_cam = np.array([x, y, z])

        # Transform to robot frame
        try:
            tf = self.tf_buffer.lookup_transform(
                "panda_link0",
                "camera_optical_frame",
                rclpy.time.Time()
            )

            rot = R.from_quat([
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w
            ]).as_matrix()

            trans = np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ])

            point_robot = rot @ point_cam + trans

            reference = np.array([0.30, -0.20, 0.15])
            error = point_robot - reference

            print(
                f"[Vision]\n"
                f"  Camera : {point_cam.round(3)}\n"
                f"  Robot  : {point_robot.round(3)}\n"
                f"  Ref    : {reference}\n"
                f"  Error  : {error.round(3)}\n"
            )

        except Exception as e:
            print(f"[TF] Transform failed: {e}")


def main():
    rclpy.init()
    node = VisionDiagnostic()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
