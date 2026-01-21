import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO


class ObjectSegmentationNode(Node):
    def __init__(self):
        super().__init__('object_segmentation_node')

        # Load trained YOLO model
        self.model_path = "training/best.pt"
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"Model loaded: {self.model_path}")
            # Store class names for filtering
            self.class_names = self.model.names
        except Exception as e:
            self.get_logger().error(f"Model load failed: {e}")
            raise SystemExit

        # Current target object name
        self.target_object = None

        # Camera image subscriber
        self.create_subscription(
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

        # Target command subscriber
        self.create_subscription(
            String,
            '/perception/set_target',
            self.set_target_callback,
            10
        )

        self.bridge = CvBridge()
        self.get_logger().info("Object segmentation node started (waiting for target)")

    def set_target_callback(self, msg):
        # Set or clear target object
        command = msg.data.lower()
        if command == "stop" or command == "":
            self.target_object = None
            self.get_logger().info("Segmentation paused")
        else:
            self.target_object = command
            self.get_logger().info(f"Target set: {self.target_object}")

    def listener_callback(self, msg):
        # Convert ROS image to OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

        # No target: publish empty mask and show image
        if self.target_object is None:
            self.publish_empty_mask(cv_image.shape, msg.header)
            cv2.imshow("YOLO Segmentation", cv_image)
            cv2.waitKey(1)
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
        found_target = False

        # Process masks if detections exist
        if result.masks is not None and result.boxes is not None:
            try:
                h, w = cv_image.shape[:2]
                combined_mask = np.zeros((h, w), dtype=np.uint8)

                # Filter detections by target class
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names[cls_id].lower()

                    if cls_name == self.target_object:
                        mask_data = result.masks.data[i].cpu().numpy()

                        mask_resized = cv2.resize(
                            mask_data,
                            (w, h),
                            interpolation=cv2.INTER_NEAREST
                        )

                        combined_mask = np.maximum(
                            combined_mask,
                            (mask_resized > 0.5).astype(np.uint8) * 255
                        )
                        found_target = True

                if found_target:
                    mask_msg = self.bridge.cv2_to_imgmsg(
                        combined_mask,
                        encoding="mono8"
                    )
                    mask_msg.header = msg.header
                    self.mask_pub.publish(mask_msg)
                    final_mask = combined_mask
                else:
                    self.publish_empty_mask(cv_image.shape, msg.header)

            except Exception as e:
                self.get_logger().error(f"Mask processing error: {e}")
                self.publish_empty_mask(cv_image.shape, msg.header)
        else:
            self.publish_empty_mask(cv_image.shape, msg.header)

        # Visualization
        annotated_frame = result.plot()

        # Display current target
        cv2.putText(
            annotated_frame,
            f"TARGET: {self.target_object}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("YOLO Segmentation", annotated_frame)

        if final_mask is not None:
            cv2.imshow("Binary Mask", final_mask)
        else:
            h, w = cv_image.shape[:2]
            cv2.imshow("Binary Mask", np.zeros((h, w), dtype=np.uint8))

        cv2.waitKey(1)

    def publish_empty_mask(self, shape, header):
        # Publish a zero mask
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
