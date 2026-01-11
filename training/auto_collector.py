import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import os
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive

# Dataset output
SAVE_DIR = "dataset_v2/images"
os.makedirs(SAVE_DIR, exist_ok=True)

class AutoDataCollector(Node):
    def __init__(self):
        super().__init__('auto_data_collector')

        # MoveIt action client
        self._action_client = ActionClient(self, MoveGroup, 'move_action')

        # Camera subscription
        self.subscription = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.bridge = CvBridge()
        self.latest_image = None
        self.image_count = 0

    def image_callback(self, msg):
        # Store latest frame
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def save_snapshot(self, suffix):
        # Save current image
        if self.latest_image is None:
            return

        filename = os.path.join(
            SAVE_DIR,
            f"img_{self.image_count:04d}_{suffix}.jpg"
        )
        cv2.imwrite(filename, self.latest_image)
        self.image_count += 1
        time.sleep(0.5)

    def go_to_pose(self, x, y, z):
        # Send MoveIt goal
        goal = MoveGroup.Goal()

        goal.request.workspace_parameters.header.frame_id = "panda_link0"
        goal.request.workspace_parameters.min_corner.x = -1.0
        goal.request.workspace_parameters.min_corner.y = -1.0
        goal.request.workspace_parameters.min_corner.z = -1.0
        goal.request.workspace_parameters.max_corner.x = 1.0
        goal.request.workspace_parameters.max_corner.y = 1.0
        goal.request.workspace_parameters.max_corner.z = 1.0

        goal.request.start_state.is_diff = True
        goal.request.group_name = "panda_arm"
        goal.request.allowed_planning_time = 5.0

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)

        # Fixed downward orientation
        pose.orientation.x = 0.924
        pose.orientation.y = -0.382
        pose.orientation.z = 0.0
        pose.orientation.w = 0.0

        constraints = Constraints()
        constraints.name = "goal"

        pc = PositionConstraint()
        pc.header.frame_id = "panda_link0"
        pc.link_name = "panda_link8"
        pc.constraint_region.primitives.append(
            SolidPrimitive(
                type=SolidPrimitive.SPHERE,
                dimensions=[0.01]
            )
        )
        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0
        constraints.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = "panda_link0"
        oc.link_name = "panda_link8"
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0
        constraints.orientation_constraints.append(oc)

        goal.request.goal_constraints.append(constraints)

        self._action_client.wait_for_server()
        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result:
            res_future = result.get_result_async()
            rclpy.spin_until_future_complete(self, res_future)
            time.sleep(1.0)
            return True

        return False


def main(args=None):
    rclpy.init(args=args)
    node = AutoDataCollector()

    # Camera viewpoints
    poses = [
        (0.3, 0.0, 0.6, "top_center"),
        (0.3, 0.2, 0.5, "top_left"),
        (0.3, -0.2, 0.5, "top_right"),
        (0.4, 0.0, 0.4, "mid_front"),
        (0.4, 0.2, 0.4, "mid_left"),
        (0.4, -0.2, 0.4, "mid_right"),
        (0.5, 0.0, 0.35, "low_front"),
        (0.25, 0.0, 0.6, "top_back"),
    ]

    try:
        time.sleep(2)

        for batch in range(1, 4):
            for x, y, z, name in poses:
                node.go_to_pose(x, y, z)
                rclpy.spin_once(node, timeout_sec=0.1)
                node.save_snapshot(f"batch{batch}_{name}")

            if batch < 3:
                input("Move objects and press ENTER to continue...")

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
