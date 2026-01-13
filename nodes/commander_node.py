#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, CollisionObject
from shape_msgs.msg import SolidPrimitive
import threading
import time
from scipy.spatial.transform import Rotation as R  # Adicionado para logar graus

class MoveItCommander(Node):
    def __init__(self):
        super().__init__('commander_node')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self._scene_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self.create_subscription(PoseStamped, '/grasp_pose', self.grasp_callback, 10)
        
        self.latest_grasp = None
        self.grasp_time = 0.0
        self.get_logger().info("Commander pronto (MODO DEBUG: Só aproximação).")

    def grasp_callback(self, msg):
        self.latest_grasp = msg.pose
        self.grasp_time = time.time()
        # Log imediato quando recebe
        self.get_logger().info("Grasp recebido e armazenado na memória.")

    def add_scene_objects(self):
        self.get_logger().info("Syncing planning scene")
        table = CollisionObject()
        table.header.frame_id = "panda_link0"
        table.id = "table"
        box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.0, 1.5, 0.04])
        table_pose = Pose()
        table_pose.position.x = 0.3; table_pose.position.y = 0.0; table_pose.position.z = -0.02
        table.primitives.append(box); table.primitive_poses.append(table_pose); table.operation = CollisionObject.ADD

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

    def close_gripper(self):
        print(">>> Fechando a garra...")
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "hand"
        c = Constraints(); c.name = "close_hand"
        for joint in ["panda_finger_joint1", "panda_finger_joint2"]:
            jc = JointConstraint(); jc.joint_name = joint; jc.position = 0.01; jc.weight = 1.0
            jc.tolerance_above = 0.01; jc.tolerance_below = 0.01; c.joint_constraints.append(jc)
        goal_msg.request.goal_constraints.append(c)
        self.send_moveit_goal(goal_msg)

    def open_gripper(self):
        print(">>> Abrindo a garra...")
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "hand"
        c = Constraints()
        for joint in ["panda_finger_joint1", "panda_finger_joint2"]:
            jc = JointConstraint(); jc.joint_name = joint; jc.position = 0.04; jc.weight = 1.0
            jc.tolerance_above = 0.01; jc.tolerance_below = 0.01; c.joint_constraints.append(jc)
        goal_msg.request.goal_constraints.append(c)
        self.send_moveit_goal(goal_msg)

    def execute_grasp_sequence(self):
        if self.latest_grasp is None: 
            print(">>> ERRO: Nada detectado.")
            return

        # Timeout generoso de 60 segundos
        if (time.time() - self.grasp_time) > 60.0: 
            print(">>> ERRO: Dado velho (> 60s). Capture novamente.")
            return

        target_pose = self.latest_grasp

        pre_grasp_z = target_pose.position.z + 0.15
        
        print(f">>> [1/2] APROXIMANDO (Pre-Grasp)...")
        self.open_gripper()
        
        # Vai para cima do objeto (mantendo orientação do grasp)
        success = self.go_to_pose(
            target_pose.position.x, 
            target_pose.position.y, 
            pre_grasp_z, 
            orientation=target_pose.orientation
        )
        
        if success:
             print(">>> [2/2] INDO PARA O PEGA...")
             # Descomente para testar a descida final
             # self.go_to_pose(
             #    target_pose.position.x, 
             #    target_pose.position.y, 
             #    target_pose.position.z, 
             #    orientation=target_pose.orientation
             # )
        
    def send_moveit_goal(self, goal_msg):
        self._action_client.wait_for_server()
        future = self._action_client.send_goal_async(goal_msg)
        while not future.done(): time.sleep(0.1)
        res = future.result().get_result_async()
        while not res.done(): time.sleep(0.1)
        return res.result().result.error_code.val == 1

    def go_to_pose(self, x, y, z, orientation=None):
        print(f"Planejando movimento...")

        goal_msg = MoveGroup.Goal()
        goal_msg.request.workspace_parameters.header.frame_id = "panda_link0"
        goal_msg.request.workspace_parameters.min_corner.x = -1.0; goal_msg.request.workspace_parameters.min_corner.y = -1.0; goal_msg.request.workspace_parameters.min_corner.z = -1.0
        goal_msg.request.workspace_parameters.max_corner.x = 1.0; goal_msg.request.workspace_parameters.max_corner.y = 1.0; goal_msg.request.workspace_parameters.max_corner.z = 1.0

        goal_msg.request.start_state.is_diff = True
        goal_msg.request.group_name = "panda_arm"
        goal_msg.request.allowed_planning_time = 10.0 # Aumentei um pouco o tempo de planejamento

        target_pose = Pose()
        target_pose.position.x = float(x)
        target_pose.position.y = float(y)
        target_pose.position.z = float(z)
        
        if orientation is not None:
            target_pose.orientation = orientation
        else:
            # Default
            target_pose.orientation.x = 0.924
            target_pose.orientation.y = -0.382
            target_pose.orientation.z = 0.0
            target_pose.orientation.w = 0.0

        c = Constraints(); c.name = "goal"
        pc = PositionConstraint()
        pc.header.frame_id = "panda_link0"; pc.link_name = "panda_link8"
        pc.constraint_region.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01]))
        pc.constraint_region.primitive_poses.append(target_pose); pc.weight = 1.0
        c.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = "panda_link0"; oc.link_name = "panda_link8"
        oc.orientation = target_pose.orientation
        oc.absolute_x_axis_tolerance = 0.1; oc.absolute_y_axis_tolerance = 0.1; oc.absolute_z_axis_tolerance = 0.1; oc.weight = 1.0
        c.orientation_constraints.append(oc)

        goal_msg.request.goal_constraints.append(c)
        return self.send_moveit_goal(goal_msg)

def main():
    rclpy.init()
    node = MoveItCommander()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True); thread.start()
    node.add_scene_objects(); time.sleep(1)

    while True:
        print("\n--- PANDA CONTROL (DEBUG MODE) ---")
        print("1. Home")
        print("6. >> TESTAR ROTAÇÃO E APROXIMAÇÃO <<")
        print("0. Sair")
        choice = input("Opção: ")
        if choice == '0': break
        elif choice == '1': node.go_to_pose(0.3, 0.0, 0.6); node.open_gripper()
        elif choice == '6': node.execute_grasp_sequence()
        else: print("Opção inválida")
    node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()