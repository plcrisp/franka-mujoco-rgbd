import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO


class ObjectSegmentationNode(Node):
    def __init__(self):
        super().__init__('object_segmentation_node')

        # Load trained model
        self.model_path = "training/best.pt"
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"Model loaded: {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise SystemExit

        # Camera subscriber
        self.subscription = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.listener_callback,
            10
        )

        # Binary mask publisher
        self.mask_pub = self.create_publisher(
            Image,
            '/perception/mask',
            10
        )

        self.bridge = CvBridge()
        self.get_logger().info("Object segmentation node started")

    def listener_callback(self, msg):
        # ROS Image -> OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

        # Run YOLO segmentation
        results = self.model(
            cv_image,
            verbose=False,
            conf=0.75,
            retina_masks=True
        )
        result = results[0]

        final_mask = None

        # Mask processing
        if result.masks is None:
            self.get_logger().warn(
                "Objects detected but no masks returned"
            )
            self.publish_empty_mask(cv_image.shape, msg.header)
        else:
            try:
                masks_data = result.masks.data.cpu().numpy()
                combined_mask = np.any(masks_data > 0.5, axis=0)
                binary_mask = combined_mask.astype(np.uint8) * 255

                h, w = cv_image.shape[:2]
                final_mask = cv2.resize(
                    binary_mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

                mask_msg = self.bridge.cv2_to_imgmsg(
                    final_mask,
                    encoding="mono8"
                )
                mask_msg.header = msg.header
                self.mask_pub.publish(mask_msg)

            except Exception as e:
                self.get_logger().error(f"Mask processing error: {e}")
                self.publish_empty_mask(cv_image.shape, msg.header)

        # Visualization
        annotated_frame = result.plot()
        cv2.imshow("YOLO Segmentation", annotated_frame)

        if final_mask is not None:
            cv2.imshow("Binary Mask", final_mask)
        else:
            h, w = cv_image.shape[:2]
            cv2.imshow(
                "Binary Mask",
                np.zeros((h, w), dtype=np.uint8)
            )

        cv2.waitKey(1)

    def publish_empty_mask(self, shape, header):
        h, w = shape[:2]
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        mask_msg = self.bridge.cv2_to_imgmsg(
            empty_mask,
            encoding="mono8"
        )
        mask_msg.header = header
        self.mask_pub.publish(mask_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectSegmentationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
