import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class ObjectSegmentationNode(Node):
    def __init__(self):
        super().__init__('object_segmentation_node')

        # Load YOLOv8 segmentation model
        self.model = YOLO("../runs/segment/franka_seg/weights/best.pt") 
        
        # Subscribe to RGB camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.listener_callback,
            10)
        
        self.bridge = CvBridge()
        self.get_logger().info("YOLO Segmentation Node started.")

    def listener_callback(self, msg):
        # Convert ROS Image to OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        # Run YOLO segmentation
        results = self.model(cv_image, verbose=False, conf=0.5)  

        # Draw predictions
        annotated_frame = results[0].plot()
        
        # Log detected objects
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])       # Class ID
                conf = float(box.conf[0])      # Confidence
                name = results[0].names[cls_id] # Class name
                self.get_logger().info(f"Detected: {name} ({conf:.2f})")
        else:
            self.get_logger().info("Nothing detected.")

        # Show segmentation result
        cv2.imshow("YOLO Segmentation", annotated_frame)
        cv2.waitKey(1)
        

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