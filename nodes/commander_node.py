#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import GetPlanningScene, GetCartesianPath
from moveit_msgs.msg import (
    Constraints, JointConstraint,
    PositionConstraint, OrientationConstraint,
    CollisionObject, PlanningSceneComponents
)
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String, Empty
from geometry_msgs.msg import Quaternion

import threading
import time


# top-down
top_down_orientation = Quaternion()
top_down_orientation.x = 1.0
top_down_orientation.y = 0.0
top_down_orientation.z = 0.0
top_down_orientation.w = 0.0


class MoveItCommander(Node):
    def __init__(self):
        super().__init__('commander_node')

        # --- MOVEIT CLIENTS ---
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self._execute_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')
        self._cartesian_srv = self.create_client(GetCartesianPath, 'compute_cartesian_path')
        self._scene_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self.scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')

        # Perception control
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
        self.get_logger().info("Grasp received from vision")
        self.grasp_event.set()

    # --- PERCEPTION ---
    def set_perception_target(self, object_name):
        self.latest_grasp = None
        self.grasp_event.clear()
        self.reset_grasp_pub.publish(Empty())
        time.sleep(0.2)

        msg = String()
        msg.data = object_name
        self.target_pub.publish(msg)
        self.get_logger().info(f"Searching for: {object_name}")

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

        table = CollisionObject()
        table.header.frame_id = "panda_link0"
        table.id = "table"
        table.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.0, 1.5, 0.04]))

        pose = Pose()
        pose.position.x = 0.3
        pose.position.z = -0.02

        table.primitive_poses.append(pose)
        table.operation = CollisionObject.ADD

        self._scene_pub.publish(table)
        self.wait_for_object("table")

    # --- GRIPPER ---
    def close_gripper(self):
        self.get_logger().info("Closing gripper")

        goal = MoveGroup.Goal()
        goal.request.group_name = "hand"
        c = Constraints(name="close_hand")

        for joint in ["panda_finger_joint1", "panda_finger_joint2"]:
            c.joint_constraints.append(
                JointConstraint(
                    joint_name=joint,
                    position=0.001,
                    tolerance_above=0.002,
                    tolerance_below=0.001,
                    weight=1.0
                )
            )

        goal.request.goal_constraints.append(c)
        self.send_moveit_goal(goal)

    def open_gripper(self, position=0.04):
        self.get_logger().info("Opening gripper")

        goal = MoveGroup.Goal()
        goal.request.group_name = "hand"
        c = Constraints(name="open_hand")

        for joint in ["panda_finger_joint1", "panda_finger_joint2"]:
            c.joint_constraints.append(
                JointConstraint(
                    joint_name=joint,
                    position=position,
                    tolerance_above=0.002,
                    tolerance_below=0.002,
                    weight=1.0
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

    # --- BASIC MOTION ---
    def go_to_pose(self, x, y, z, orientation=None):
        self.get_logger().info(f"Moving to [{x:.2f}, {y:.2f}, {z:.2f}] (PTP)")

        goal = MoveGroup.Goal()
        goal.request.group_name = "panda_arm"
        goal.request.start_state.is_diff = True
        goal.request.allowed_planning_time = 5.0

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

    def go_to_pose_cartesian(self, target_pose, speed_scale=3.0):
        req = GetCartesianPath.Request()
        req.header.frame_id = "panda_link0"
        req.group_name = "panda_arm"
        req.link_name = "panda_link8"
        req.waypoints = [target_pose]
        req.max_step = 0.01 
        req.jump_threshold = 0.0
        req.avoid_collisions = True

        if not self._cartesian_srv.wait_for_service(timeout_sec=2.0):
            return False

        future = self._cartesian_srv.call_async(req)
        while not future.done():
            time.sleep(0.05)

        result = future.result()

        if result.fraction < 0.90:
            self.get_logger().warn(f"Cartesian path incomplete: {result.fraction}")
            return False

        trajectory = result.solution.joint_trajectory
        
        if speed_scale > 1.0:
            for point in trajectory.points:
                time_total_sec = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                new_time_total = time_total_sec * speed_scale
                
                point.time_from_start.sec = int(new_time_total)
                point.time_from_start.nanosec = int((new_time_total - int(new_time_total)) * 1e9)
                
                new_velocities = []
                new_accelerations = []
                
                for v in point.velocities:
                    new_velocities.append(v / speed_scale)
                    
                for a in point.accelerations:
                    new_accelerations.append(a / (speed_scale * speed_scale))
                    
                point.velocities = new_velocities
                point.accelerations = new_accelerations

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = result.solution

        self._execute_client.wait_for_server()
        exe_future = self._execute_client.send_goal_async(goal)

        while not exe_future.done():
            time.sleep(0.05)

        exe_result_future = exe_future.result().get_result_async()
        while not exe_result_future.done():
            time.sleep(0.05)

        return exe_result_future.result().result.error_code.val == 1

    # --- PICK PHASES ---
    def perform_approach(self, grasp_msg):
        if not grasp_msg:
            return False

        target = grasp_msg
        pre_grasp_z = target.position.z + 0.40
        grasp_height_z = target.position.z + 0.13

        self.get_logger().info("Going to pre-grasp")
        success = self.go_to_pose(
            target.position.x,
            target.position.y,
            pre_grasp_z,
            orientation=top_down_orientation
        )
        if not success:
            return False

        self.open_gripper()
        time.sleep(3.0)

        self.get_logger().info("Descending linearly")
        grasp_pose = Pose()
        grasp_pose.position = target.position
        grasp_pose.position.z = grasp_height_z
        grasp_pose.orientation = top_down_orientation

        return self.go_to_pose_cartesian(grasp_pose)

    def perform_realignment_maneuver(self):
        self.get_logger().info("Realigning (up 27cm)")

        if not self.latest_grasp:
            return False

        retreat_pose = Pose()
        retreat_pose.position = self.latest_grasp.position
        retreat_pose.position.z += 0.27
        retreat_pose.orientation = top_down_orientation

        return self.go_to_pose_cartesian(retreat_pose)

    def perform_finish_pick(self):
        self.close_gripper()
        time.sleep(5.0)

        self.get_logger().info("Lifting object")
        if not self.latest_grasp:
            return

        lift_pose = Pose()
        lift_pose.position = self.latest_grasp.position
        lift_pose.position.z += 0.40
        lift_pose.orientation = top_down_orientation

        self.go_to_pose_cartesian(lift_pose)


# --- USER INTERFACE ---
def user_interface(node):
    time.sleep(2)
    node.add_scene_objects()

    while rclpy.ok():
        print("\n" + "=" * 40)
        print(" PANDA COMMANDER")
        print("=" * 40)
        print("1. Select object")
        print("2. Go home")
        print("0. Exit")

        opt = input(">> ")

        if opt == '0':
            node.stop_perception()
            break

        elif opt == '2':
            node.stop_perception()
            node.go_to_pose(0.3, 0.0, 0.6, orientation=top_down_orientation)
            node.close_gripper()

        elif opt == '1':
            while True:
                print("\nWhich object?")
                print("1. mug")
                print("2. bottle")
                print("0. back")

                choice = input(">> ").strip()
                if choice == '0':
                    break

                obj_name = "mug" if choice == '1' else "bottle" if choice == '2' else None
                if not obj_name:
                    continue

                print(f"Searching for '{obj_name}'")
                node.set_perception_target(obj_name)

                while True:
                    print("Waiting for grasp (20s)")
                    found = node.grasp_event.wait(timeout=20.0)

                    if not found:
                        print("Grasp not found")
                        if input("Try again? (y/n): ").lower() == 'y':
                            continue
                        else:
                            break

                    print(f"Grasp: X={node.latest_grasp.position.x:.2f}, Y={node.latest_grasp.position.y:.2f}")
                    action = input("Approach? [Y]es, [N]ew grasp, [C]ancel: ").lower()

                    if action == 'n':
                        node.set_perception_target(obj_name)
                        continue

                    elif action == 'c':
                        node.stop_perception()
                        break

                    elif action != 'y':
                        continue

                    success = node.perform_approach(node.latest_grasp)
                    if not success:
                        print("Approach failed")
                        node.go_to_pose(0.3, 0.0, 0.6, orientation=top_down_orientation)
                        node.close_gripper()
                        break

                    print("\nRobot at target")
                    print("[F]inish pick")
                    print("[R]etry alignment")
                    print("[C]ancel")

                    decision = input(">> ").lower()

                    if decision == 'f':
                        node.perform_finish_pick()
                        node.stop_perception()
                        break

                    elif decision == 'r':
                        node.perform_realignment_maneuver()
                        node.set_perception_target(obj_name)
                        node.close_gripper()
                        continue

                    elif decision == 'c':
                        node.go_to_pose(0.3, 0.0, 0.6, orientation=top_down_orientation)
                        node.close_gripper()
                        node.stop_perception()
                        break

                break


def main():
    rclpy.init()
    node = MoveItCommander()

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
