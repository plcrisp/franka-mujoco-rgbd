#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, PoseStamped
# Adicionado ExecuteTrajectory e GetCartesianPath
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import GetPlanningScene, GetCartesianPath
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

        # --- MOVEIT CLIENTS ---
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        
        self._execute_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')
        
        self._cartesian_srv = self.create_client(GetCartesianPath, 'compute_cartesian_path')

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
        self.latest_grasp = msg.pose
        self.get_logger().info("Grasp received from vision node")
        self.grasp_event.set()

    # --- PERCEPTION COMMANDS ---
    def set_perception_target(self, object_name):
        self.latest_grasp = None
        self.grasp_event.clear()
        self.reset_grasp_pub.publish(Empty())
        msg = String()
        msg.data = object_name
        self.target_pub.publish(msg)
        self.get_logger().info(f"Searching for object: {object_name}")

    def stop_perception(self):
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
            while not future.done():
                time.sleep(0.05)
            
            if future.result():
                ids = [co.id for co in future.result().scene.world.collision_objects]
                if object_name in ids:
                    return True
            time.sleep(0.2)
        return False

    def add_scene_objects(self):
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
        goal = MoveGroup.Goal()
        goal.request.group_name = "hand"
        c = Constraints(name="close_hand")
        for joint in ["panda_finger_joint1", "panda_finger_joint2"]:
            c.joint_constraints.append(
                JointConstraint(
                    joint_name=joint, position=0.05, tolerance_above=0.01, tolerance_below=0.01, weight=1.0
                )
            )
        goal.request.goal_constraints.append(c)
        self.send_moveit_goal(goal)

    def open_gripper(self):
        goal = MoveGroup.Goal()
        goal.request.group_name = "hand"
        c = Constraints()
        for joint in ["panda_finger_joint1", "panda_finger_joint2"]:
            c.joint_constraints.append(
                JointConstraint(
                    joint_name=joint, position=0.1, tolerance_above=0.01, tolerance_below=0.01, weight=1.0
                )
            )
        goal.request.goal_constraints.append(c)
        self.send_moveit_goal(goal)

    def send_moveit_goal(self, goal):
        self._action_client.wait_for_server()
        future = self._action_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.05)
        
        res_future = future.result().get_result_async()
        while not res_future.done():
            time.sleep(0.05)
            
        return res_future.result().result.error_code.val == 1

    def go_to_pose(self, x, y, z, orientation=None):
        self.get_logger().info(f"Moving to [{x:.2f}, {y:.2f}, {z:.2f}] (Free)")

        goal = MoveGroup.Goal()
        goal.request.group_name = "panda_arm"
        goal.request.start_state.is_diff = True
        goal.request.allowed_planning_time = 5.0
        
        # Workspace params
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
        pc.constraint_region.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01]))
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

    def go_to_pose_cartesian(self, target_pose):
        self.get_logger().info("Planning Cartesian path (Straight Line)...")

        req = GetCartesianPath.Request()
        req.header.frame_id = "panda_link0"
        req.group_name = "panda_arm"
        req.link_name = "panda_link8"
        
        req.waypoints = [target_pose]
        
        req.max_step = 0.01       
        req.jump_threshold = 0.0 
        req.avoid_collisions = True

        if not self._cartesian_srv.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Cartesian service not available")
            return False

        future = self._cartesian_srv.call_async(req)
        while not future.done():
            time.sleep(0.05)
        
        result = future.result()
        
        if result.fraction < 0.90:
            self.get_logger().warn(f"Cartesian path incomplete! Fraction: {result.fraction}")
            return False

        self.get_logger().info(f"Cartesian path computed ({len(result.solution.joint_trajectory.points)} points). Executing...")

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = result.solution
        
        self._execute_client.wait_for_server()
        exe_future = self._execute_client.send_goal_async(goal)
        
        while not exe_future.done():
            time.sleep(0.05)
            
        exe_result_future = exe_future.result().get_result_async()
        while not exe_result_future.done():
            time.sleep(0.05)
            
        final_res = exe_result_future.result()
        if final_res.result.error_code.val == 1:
            self.get_logger().info("Cartesian move SUCCESS")
            return True
        else:
            self.get_logger().error(f"Cartesian move FAILED: {final_res.result.error_code.val}")
            return False

    # --- PICK SEQUENCE ---
    def execute_full_pick(self):
        # Open -> pre-grasp -> descend (LINEAR) -> close -> lift (LINEAR)
        if not self.latest_grasp:
            self.get_logger().error("Pick requested without a valid grasp")
            return

        target = self.latest_grasp
        
        pre_grasp_z = target.position.z + 0.35 
        
        grasp_height_z = target.position.z + 0.13          

        # --- 1. PRE-GRASP ---
        self.get_logger().info("1. Moving to PRE-GRASP (Air)")
        success = self.go_to_pose(
            target.position.x,
            target.position.y,
            pre_grasp_z,
            orientation=target.orientation
        )
        if not success:
            self.get_logger().error("Pre-grasp failed")
            return
        
        # --- 2. OPEN GRIPPER ---
        self.get_logger().info("2. Opening gripper...")
        self.open_gripper()
        
        time.sleep(3.0)

        # --- 3. DESCENT (Linear Cartesian Movement) ---
        self.get_logger().info("3. Descending to GRASP pose (Linear)")
        
        grasp_pose = Pose()
        grasp_pose.position.x = target.position.x
        grasp_pose.position.y = target.position.y
        grasp_pose.position.z = grasp_height_z 
        grasp_pose.orientation = target.orientation

        if not self.go_to_pose_cartesian(grasp_pose):
            self.get_logger().error("Descent failed / Path obstructed")
            return

        time.sleep(0.5)

        # --- 4. CLOSE GRIPPER ---
        self.get_logger().info("4. Closing gripper...")
        self.close_gripper()
        
        time.sleep(3.0) 


        # --- 5. LIFT (Linear Cartesian Movement) ---
        self.get_logger().info("5. Lifting object (Linear)")
        
        lift_pose = Pose()
        lift_pose.position.x = target.position.x
        lift_pose.position.y = target.position.y
        lift_pose.position.z = target.position.z + 0.40 
        lift_pose.orientation = target.orientation
        
        self.go_to_pose_cartesian(lift_pose)


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