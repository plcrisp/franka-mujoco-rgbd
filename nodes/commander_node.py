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
from std_msgs.msg import String, Empty

import threading
import time


class MoveItCommander(Node):
    def __init__(self):
        super().__init__('commander_node')

        # MoveIt clients
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self._scene_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self.scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')

        # Perception control publishers
        self.target_pub = self.create_publisher(String, '/perception/set_target', 10)
        self.reset_grasp_pub = self.create_publisher(Empty, '/grasp/reset', 10)

        # Grasp subscriber
        self.create_subscription(PoseStamped, '/grasp_pose', self.grasp_callback, 10)

        # Internal state
        self.latest_grasp = None
        self.grasp_event = threading.Event()

        self.get_logger().info("Commander ready (interactive mode)")

    def grasp_callback(self, msg):
        # Receive grasp from vision node
        self.latest_grasp = msg.pose
        self.get_logger().info("Grasp received from vision node")
        self.grasp_event.set()

    # --- PERCEPTION COMMANDS ---
    def set_perception_target(self, object_name):
        # Reset previous grasp and set new target
        self.latest_grasp = None
        self.grasp_event.clear()

        self.reset_grasp_pub.publish(Empty())

        msg = String()
        msg.data = object_name
        self.target_pub.publish(msg)
        self.get_logger().info(f"Searching for object: {object_name}")

    def stop_perception(self):
        # Stop perception pipeline
        msg = String()
        msg.data = "STOP"
        self.target_pub.publish(msg)

    # --- PLANNING SCENE ---
    def wait_for_object(self, object_name, timeout=2.0):
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        start = time.time()

        while (time.time() - start) < timeout:
            if not self.scene_client.service_is_ready():
                time.sleep(0.1)
                continue

            future = self.scene_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)

            if future.result():
                ids = [co.id for co in future.result().scene.world.collision_objects]
                if object_name in ids:
                    return True

            time.sleep(0.2)

        return False

    def add_scene_objects(self):
        # Add static objects to the planning scene
        self.get_logger().info("Loading planning scene")
        objects_to_add = []

        table = CollisionObject()
        table.header.frame_id = "panda_link0"
        table.id = "table"
        table.primitives.append(
            SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.0, 1.5, 0.04])
        )
        p_table = Pose()
        p_table.position.x = 0.3
        p_table.position.z = -0.02
        table.primitive_poses.append(p_table)
        table.operation = CollisionObject.ADD
        objects_to_add.append(table)

        for obj in objects_to_add:
            self._scene_pub.publish(obj)
            if self.wait_for_object(obj.id):
                self.get_logger().info(f"Scene synced: {obj.id}")
            else:
                self._scene_pub.publish(obj)

    # --- GRIPPER CONTROL ---
    def close_gripper(self):
        self.get_logger().info("Closing gripper")
        goal = MoveGroup.Goal()
        goal.request.group_name = "hand"

        c = Constraints(name="close_hand")
        for joint in ["panda_finger_joint1", "panda_finger_joint2"]:
            c.joint_constraints.append(
                JointConstraint(
                    joint_name=joint,
                    position=0.01,
                    tolerance_above=0.01,
                    tolerance_below=0.01,
                    weight=1.0
                )
            )

        goal.request.goal_constraints.append(c)
        self.send_moveit_goal(goal)

    def open_gripper(self):
        self.get_logger().info("Opening gripper")
        goal = MoveGroup.Goal()
        goal.request.group_name = "hand"

        c = Constraints()
        for joint in ["panda_finger_joint1", "panda_finger_joint2"]:
            c.joint_constraints.append(
                JointConstraint(
                    joint_name=joint,
                    position=0.04,
                    tolerance_above=0.01,
                    tolerance_below=0.01,
                    weight=1.0
                )
            )

        goal.request.goal_constraints.append(c)
        self.send_moveit_goal(goal)

    def send_moveit_goal(self, goal):
        # Send goal and wait for result
        self._action_client.wait_for_server()
        future = self._action_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.1)

        result_future = future.result().get_result_async()
        while not result_future.done():
            time.sleep(0.1)

        return result_future.result().result.error_code.val == 1

    def go_to_pose(self, x, y, z, orientation=None):
        self.get_logger().info(f"Moving to [{x:.2f}, {y:.2f}, {z:.2f}]")

        goal = MoveGroup.Goal()
        goal.request.group_name = "panda_arm"
        goal.request.start_state.is_diff = True
        goal.request.allowed_planning_time = 5.0

        # Workspace limits
        wp = goal.request.workspace_parameters
        wp.header.frame_id = "panda_link0"
        wp.min_corner.x = wp.min_corner.y = wp.min_corner.z = -1.0
        wp.max_corner.x = wp.max_corner.y = wp.max_corner.z = 1.0

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)

        if orientation:
            pose.orientation = orientation
        else:
            pose.orientation.x = 0.9239
            pose.orientation.y = -0.3827
            pose.orientation.z = 0.0
            pose.orientation.w = 0.0

        c = Constraints(name="goal")

        pc = PositionConstraint()
        pc.header.frame_id = "panda_link0"
        pc.link_name = "panda_link8"
        pc.constraint_region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01])
        )
        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0
        c.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = "panda_link0"
        oc.link_name = "panda_link8"
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0
        c.orientation_constraints.append(oc)

        goal.request.goal_constraints.append(c)
        return self.send_moveit_goal(goal)

    # --- PICK SEQUENCE ---
    def execute_full_pick(self):
        # Open -> pre-grasp -> approach -> close -> lift
        if not self.latest_grasp:
            self.get_logger().error("Pick requested without a valid grasp")
            return

        target = self.latest_grasp
        pre_grasp_z = target.position.z + 0.15
        approach_z = target.position.z + 0.02

        self.get_logger().info("Opening gripper")
        self.open_gripper()

        self.get_logger().info("Moving to pre-grasp")
        if not self.go_to_pose(
            target.position.x,
            target.position.y,
            pre_grasp_z,
            orientation=target.orientation
        ):
            self.get_logger().error("Pre-grasp failed")
            return

        self.get_logger().info("Approaching object")
        if not self.go_to_pose(
            target.position.x,
            target.position.y,
            approach_z,
            orientation=target.orientation
        ):
            self.get_logger().error("Approach failed")
            return

        self.get_logger().info("Closing gripper")
        self.close_gripper()
        time.sleep(1.0)

        self.get_logger().info("Lifting object")
        self.go_to_pose(
            target.position.x,
            target.position.y,
            0.5,
            orientation=target.orientation
        )


# --- USER INTERFACE ---
def user_interface(node):
    time.sleep(2)
    node.add_scene_objects()

    while rclpy.ok():
        print("\n" + "=" * 40)
        print(" PANDA COMMANDER - MAIN MENU")
        print("=" * 40)
        print("1. Select object to grasp")
        print("2. Go to home position")
        print("0. Exit")

        opt = input(">> Option: ")

        if opt == '0':
            node.stop_perception()
            break

        elif opt == '2':
            node.stop_perception()
            node.go_to_pose(0.3, 0.0, 0.6)

        elif opt == '1':
            while True:
                print("\n[SEARCH] Select object:")
                print("1. mug")
                print("2. bottle")
                print("3. hammer")
                print("0. back")

                choice = input(">> Option: ").strip()

                if choice == '0':
                    break
                elif choice == '1':
                    obj_name = "mug"
                elif choice == '2':
                    obj_name = "bottle"
                elif choice == '3':
                    obj_name = "hammer"
                else:
                    print("[ERROR] Invalid option")
                    continue

                print(f"[SYSTEM] Searching for '{obj_name}'")
                node.set_perception_target(obj_name)

                print("[SYSTEM] Waiting for grasp (timeout 20s)")
                found = node.grasp_event.wait(timeout=20.0)

                if not found:
                    print(f"[ERROR] Timeout, no grasp found for '{obj_name}'")
                    retry = input("Try again? (y/n): ").lower()
                    if retry != 'y':
                        break
                    continue

                print(
                    f"[SUCCESS] Grasp at X={node.latest_grasp.position.x:.2f}, "
                    f"Y={node.latest_grasp.position.y:.2f}"
                )

                action = input("[E]xecute, [T]ry another, [C]ancel: ").lower()

                if action == 'e':
                    print("Executing pick sequence")
                    node.execute_full_pick()
                    node.stop_perception()
                    print("Done, returning to menu")
                    break

                elif action == 't':
                    print("Retrying grasp")
                    continue

                elif action == 'c':
                    print("Operation cancelled")
                    node.stop_perception()
                    break



def main():
    rclpy.init()
    node = MoveItCommander()

    # Spin in separate thread to allow blocking input()
    spinner_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )
    spinner_thread.start()

    try:
        user_interface(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
