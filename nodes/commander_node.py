#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import (
    Constraints, JointConstraint,
    PositionConstraint, OrientationConstraint,
    CollisionObject, PlanningSceneComponents
)
from shape_msgs.msg import SolidPrimitive

import threading
import time


class MoveItCommander(Node):
    def __init__(self):
        super().__init__('commander_node')

        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self._scene_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self.scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')

        self.create_subscription(PoseStamped, '/grasp_pose', self.grasp_callback, 10)

        self.latest_grasp = None
        self.grasp_time = 0.0

        self.get_logger().info("Commander ready (DEBUG mode)")

    def grasp_callback(self, msg):
        self.latest_grasp = msg.pose
        self.grasp_time = time.time()
        self.get_logger().info("Grasp received")

    def wait_for_object(self, object_name, timeout=2.0):
        """Checks if MoveIt knows about the object."""
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        
        start = time.time()
        while (time.time() - start) < timeout:
            if not self.scene_client.service_is_ready():
                time.sleep(0.1)
                continue
                
            future = self.scene_client.call_async(req)
            # Wait briefly for result
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
            
            if future.result():
                if object_name in [co.id for co in future.result().scene.world.collision_objects]:
                    return True
            time.sleep(0.2)
        return False

    def add_scene_objects(self):
        self.get_logger().info("Loading planning scene...")
        objects_to_add = []

        # 1. Table
        table = CollisionObject()
        table.header.frame_id = "panda_link0"; table.id = "table"
        table.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.0, 1.5, 0.04]))
        
        p_table = Pose()
        p_table.position.x = 0.3; p_table.position.z = -0.02
        table.primitive_poses.append(p_table)
        
        table.operation = CollisionObject.ADD
        objects_to_add.append(table)

        # Publish and Verify
        for obj in objects_to_add:
            self._scene_pub.publish(obj)
            if self.wait_for_object(obj.id):
                self.get_logger().info(f"Synced: {obj.id}")
            else:
                self.get_logger().warn(f"Retry syncing: {obj.id}")
                self._scene_pub.publish(obj) # Retry once

    def close_gripper(self):
        self.get_logger().info("Closing gripper")
        goal = MoveGroup.Goal()
        goal.request.group_name = "hand"
        c = Constraints(name="close_hand")
        for joint in ["panda_finger_joint1", "panda_finger_joint2"]:
            c.joint_constraints.append(JointConstraint(
                joint_name=joint, position=0.01, tolerance_above=0.01, tolerance_below=0.01, weight=1.0))
        goal.request.goal_constraints.append(c)
        self.send_moveit_goal(goal)

    def open_gripper(self):
        self.get_logger().info("Opening gripper")
        goal = MoveGroup.Goal()
        goal.request.group_name = "hand"
        c = Constraints()
        for joint in ["panda_finger_joint1", "panda_finger_joint2"]:
            c.joint_constraints.append(JointConstraint(
                joint_name=joint, position=0.04, tolerance_above=0.01, tolerance_below=0.01, weight=1.0))
        goal.request.goal_constraints.append(c)
        self.send_moveit_goal(goal)

    def execute_grasp_sequence(self):
        if not self.latest_grasp or (time.time() - self.grasp_time > 60.0):
            self.get_logger().warn("No valid grasp available")
            return

        target = self.latest_grasp
        pre_grasp_z = target.position.z + 0.15

        self.get_logger().info("1. Approach")
        self.open_gripper()
        
        # Go to pre-grasp
        if self.go_to_pose(target.position.x, target.position.y, pre_grasp_z, orientation=target.orientation):
            self.get_logger().info("2. Ready for descent")
            # To descend, uncomment:
            # self.go_to_pose(target.position.x, target.position.y, target.position.z, orientation=target.orientation)

    def send_moveit_goal(self, goal):
        self._action_client.wait_for_server()
        future = self._action_client.send_goal_async(goal)
        while not future.done(): time.sleep(0.1)
        res = future.result().get_result_async()
        while not res.done(): time.sleep(0.1)
        return res.result().result.error_code.val == 1

    def go_to_pose(self, x, y, z, orientation=None):
        self.get_logger().info(f"Moving to: [{x:.2f}, {y:.2f}, {z:.2f}]")

        goal = MoveGroup.Goal()
        goal.request.group_name = "panda_arm"
        goal.request.start_state.is_diff = True
        goal.request.allowed_planning_time = 5.0
        
        # Workspace bounds
        wp = goal.request.workspace_parameters
        wp.header.frame_id = "panda_link0"
        wp.min_corner.x = -1.0; wp.min_corner.y = -1.0; wp.min_corner.z = -1.0
        wp.max_corner.x = 1.0; wp.max_corner.y = 1.0; wp.max_corner.z = 1.0

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)

        if orientation:
            pose.orientation = orientation
        else:
            # CORREÇÃO: Orientação 'Neutra' do Panda.
            # Alinha os dedos com o eixo Y e evita torção do punho.
            pose.orientation.x = 0.9239
            pose.orientation.y = -0.3827
            pose.orientation.z = 0.0
            pose.orientation.w = 0.0

        c = Constraints(name="goal")
        
        # Position Constraint
        pc = PositionConstraint()
        pc.header.frame_id = "panda_link0"; pc.link_name = "panda_link8"
        pc.constraint_region.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01]))
        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0
        c.position_constraints.append(pc)

        # Orientation Constraint
        oc = OrientationConstraint()
        oc.header.frame_id = "panda_link0"; oc.link_name = "panda_link8"
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0
        c.orientation_constraints.append(oc)

        goal.request.goal_constraints.append(c)
        return self.send_moveit_goal(goal)


def main():
    rclpy.init()
    node = MoveItCommander()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    node.add_scene_objects()
    time.sleep(1)

    while True:
        print("\n--- PANDA CONTROL ---")
        print("1. Home (Vertical)")
        print("2. Mug (Vertical)")
        print("3. Bottle (Vertical)")
        print("4. Hammer (Vertical)")
        print("5. Grasp (Vision)")
        print("0. Exit")

        choice = input("Option: ")

        if choice == '0': break
        # Added height to Z to ensure safety with vertical gripper
        elif choice == '1': node.go_to_pose(0.3, 0.0, 0.6) 
        elif choice == '2': node.go_to_pose(0.3, 0.2, 0.45)
        elif choice == '3': node.go_to_pose(0.3, -0.2, 0.45)
        elif choice == '4': node.go_to_pose(0.5, 0.0, 0.45)
        elif choice == '5': node.execute_grasp_sequence()
        else: print("Invalid")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()