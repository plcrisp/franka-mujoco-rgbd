#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, CollisionObject
from shape_msgs.msg import SolidPrimitive
import time

class MoveItMenu(Node):
    def __init__(self):
        super().__init__('moveit_menu_node')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self._scene_pub = self.create_publisher(CollisionObject, '/collision_object', 10)

    def add_scene_objects(self):
        self.get_logger().info("Syncing planning scene")

        # Table
        table = CollisionObject()
        table.header.frame_id = "panda_link0"
        table.id = "table"
        box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.0, 1.5, 0.04])
        table_pose = Pose()
        table_pose.position.x = 0.3
        table_pose.position.y = 0.0
        table_pose.position.z = -0.02
        table.primitives.append(box)
        table.primitive_poses.append(table_pose)
        table.operation = CollisionObject.ADD

        # Mug
        mug = CollisionObject()
        mug.header.frame_id = "panda_link0"
        mug.id = "mug"
        mbox = SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[0.12, 0.05])
        mpose = Pose()
        mpose.position.x = 0.3
        mpose.position.y = 0.2
        mpose.position.z = 0.06
        mug.primitives.append(mbox)
        mug.primitive_poses.append(mpose)
        mug.operation = CollisionObject.ADD

        # Bottle
        bottle = CollisionObject()
        bottle.header.frame_id = "panda_link0"
        bottle.id = "bottle"
        bbox = SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[0.15, 0.03])
        bpose = Pose()
        bpose.position.x = 0.3
        bpose.position.y = -0.2
        bpose.position.z = 0.075
        bottle.primitives.append(bbox)
        bottle.primitive_poses.append(bpose)
        bottle.operation = CollisionObject.ADD

        # Hammer
        hammer = CollisionObject()
        hammer.header.frame_id = "panda_link0"
        hammer.id = "hammer"
        hbox = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.15, 0.05, 0.03])
        hpose = Pose()
        hpose.position.x = 0.5
        hpose.position.y = 0.0
        hpose.position.z = 0.015
        hammer.primitives.append(hbox)
        hammer.primitive_poses.append(hpose)
        hammer.operation = CollisionObject.ADD

        for _ in range(5):
            self._scene_pub.publish(table)
            self._scene_pub.publish(mug)
            self._scene_pub.publish(bottle)
            self._scene_pub.publish(hammer)
            time.sleep(0.1)

        print("Scene loaded")

    def go_to_pose(self, x, y, z):
        print(f"Moving to: {x:.2f}, {y:.2f}, {z:.2f}")

        goal_msg = MoveGroup.Goal()
        goal_msg.request.workspace_parameters.header.frame_id = "panda_link0"
        goal_msg.request.workspace_parameters.min_corner.x = -1.0
        goal_msg.request.workspace_parameters.min_corner.y = -1.0
        goal_msg.request.workspace_parameters.min_corner.z = -1.0
        goal_msg.request.workspace_parameters.max_corner.x = 1.0
        goal_msg.request.workspace_parameters.max_corner.y = 1.0
        goal_msg.request.workspace_parameters.max_corner.z = 1.0

        goal_msg.request.start_state.is_diff = True
        goal_msg.request.group_name = "panda_arm"
        goal_msg.request.allowed_planning_time = 5.0

        target_pose = Pose()
        target_pose.position.x = float(x)
        target_pose.position.y = float(y)
        target_pose.position.z = float(z)
        target_pose.orientation.x = 0.924
        target_pose.orientation.y = -0.382
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 0.0

        c = Constraints()
        c.name = "goal"

        pc = PositionConstraint()
        pc.header.frame_id = "panda_link0"
        pc.link_name = "panda_link8"
        pc.constraint_region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.005])
        )
        pc.constraint_region.primitive_poses.append(target_pose)
        pc.weight = 1.0
        c.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = "panda_link0"
        oc.link_name = "panda_link8"
        oc.orientation = target_pose.orientation
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0
        c.orientation_constraints.append(oc)

        goal_msg.request.goal_constraints.append(c)

        self._action_client.wait_for_server()
        future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            print("Planning rejected")
            return

        res_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        result = res_future.result().result

        if result.error_code.val == 1:
            print("Done")
        else:
            print(f"Failed ({result.error_code.val})")

def main():
    rclpy.init()
    node = MoveItMenu()

    node.add_scene_objects()
    time.sleep(1)

    while True:
        print("\n--- PANDA CONTROL ---")
        print("1. Home")
        print("2. Mug")
        print("3. Bottle")
        print("4. Hammer")
        print("5. Manual")
        print("0. Exit")

        choice = input("Option: ")

        if choice == '0':
            break

        elif choice == '1':
            node.go_to_pose(0.3, 0.0, 0.8)

        elif choice == '2':
            node.go_to_pose(0.3, 0.2, 0.25)

        elif choice == '3':
            node.go_to_pose(0.3, -0.2, 0.30)

        elif choice == '4':
            node.go_to_pose(0.5, 0.0, 0.20)

        elif choice == '5':
            try:
                x = float(input("X: "))
                y = float(input("Y: "))
                z = float(input("Z: "))
                node.go_to_pose(x, y, z)
            except ValueError:
                print("Invalid input")

        else:
            print("Invalid option")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
