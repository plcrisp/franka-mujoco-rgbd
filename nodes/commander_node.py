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
        self.get_logger().info(">>> Grasp received from vision node")
        self.grasp_event.set()

    # --- PERCEPTION COMMANDS ---
    def set_perception_target(self, object_name):
        self.latest_grasp = None
        self.grasp_event.clear()
        self.reset_grasp_pub.publish(Empty())
        
        # Delay para garantir reset
        time.sleep(0.2) 
        
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
                time.sleep(0.1); continue
            future = self.scene_client.call_async(req)
            while not future.done(): time.sleep(0.05)
            
            if future.result():
                ids = [co.id for co in future.result().scene.world.collision_objects]
                if object_name in ids: return True
            time.sleep(0.2)
        return False

    def add_scene_objects(self):
        self.get_logger().info("Loading planning scene")
        objects_to_add = []

        table = CollisionObject()
        table.header.frame_id = "panda_link0"
        table.id = "table"
        table.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.0, 1.5, 0.04]))
        p_table = Pose(); p_table.position.x = 0.3; p_table.position.z = -0.02
        table.primitive_poses.append(p_table)
        table.operation = CollisionObject.ADD
        objects_to_add.append(table)

        for obj in objects_to_add:
            self._scene_pub.publish(obj)
            self.wait_for_object(obj.id)

    # --- GRIPPER CONTROL ---
    # --- GRIPPER CONTROL CORRIGIDO ---
    def close_gripper(self):
        self.get_logger().info("Closing gripper...")

        goal = MoveGroup.Goal()
        goal.request.group_name = "hand"
        c = Constraints(name="close_hand")

        # Para fechar, a tolerância deve ser pequena, mas não zero absoluto
        # para evitar falha se o objeto impedir o fechamento total.
        for joint in ["panda_finger_joint1", "panda_finger_joint2"]:
            c.joint_constraints.append(
                JointConstraint(
                    joint_name=joint,
                    position=0.02,       # Posição fechada
                    tolerance_above=0.002, # Aceita até 2mm aberto
                    tolerance_below=0.001, # Não pode ir abaixo de 0
                    weight=1.0
                )
            )

        goal.request.goal_constraints.append(c)
        self.send_moveit_goal(goal)


    def open_gripper(self):
        self.get_logger().info("Opening gripper...")

        goal = MoveGroup.Goal()
        goal.request.group_name = "hand"
        c = Constraints(name="open_hand")

        # Aumentamos o alvo para 0.035 (3.5cm) para ser bem visível
        target_open = 0.02 

        for joint in ["panda_finger_joint1", "panda_finger_joint2"]:
            c.joint_constraints.append(
                JointConstraint(
                    joint_name=joint,
                    position=target_open,
                    tolerance_above=0.002, # Tolerância apertada (2mm)
                    tolerance_below=0.002, # Tolerância apertada (2mm)
                    weight=1.0
                )
            )

        goal.request.goal_constraints.append(c)
        self.send_moveit_goal(goal)


    def send_moveit_goal(self, goal):
        self._action_client.wait_for_server()
        future = self._action_client.send_goal_async(goal)
        while not future.done(): time.sleep(0.05)
        res_future = future.result().get_result_async()
        while not res_future.done(): time.sleep(0.05)
        return res_future.result().result.error_code.val == 1

    # --- MOVIMENTOS BÁSICOS ---
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
        pose.position.x = float(x); pose.position.y = float(y); pose.position.z = float(z)
        if orientation: pose.orientation = orientation
        else: 
            pose.orientation.x = 0.9239; pose.orientation.y = -0.3827
            pose.orientation.z = 0.0; pose.orientation.w = 0.0

        c = Constraints(name="goal")
        pc = PositionConstraint()
        pc.header.frame_id = "panda_link0"; pc.link_name = "panda_link8"
        pc.constraint_region.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01]))
        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0
        c.position_constraints.append(pc)
        
        oc = OrientationConstraint()
        oc.header.frame_id = "panda_link0"; oc.link_name = "panda_link8"
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.1; oc.absolute_y_axis_tolerance = 0.1; oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0
        c.orientation_constraints.append(oc)

        goal.request.goal_constraints.append(c)
        return self.send_moveit_goal(goal)

    def go_to_pose_cartesian(self, target_pose):
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
        while not future.done(): time.sleep(0.05)
        result = future.result()
        
        if result.fraction < 0.90:
            self.get_logger().warn(f"Cartesian path incomplete! Fraction: {result.fraction}")
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = result.solution
        
        self._execute_client.wait_for_server()
        exe_future = self._execute_client.send_goal_async(goal)
        while not exe_future.done(): time.sleep(0.05)
        exe_result_future = exe_future.result().get_result_async()
        while not exe_result_future.done(): time.sleep(0.05)
            
        return exe_result_future.result().result.error_code.val == 1

    # --- ETAPAS DO PICK (Modularizadas) ---

    def perform_approach(self, grasp_msg):
        """1. Vai para Pre-Grasp, 2. Abre a Garra, 3. Desce até o objeto"""
        if not grasp_msg: return False

        target = grasp_msg
        
        # Alturas de Segurança
        pre_grasp_z = target.position.z + 0.40  # 40cm acima
        grasp_height_z = target.position.z + 0.15 # Altura final (ajustável)

        # 1. Movimento Livre até o Pre-Grasp (PTP)
        self.get_logger().info(">>> Indo para PRE-GRASP...")
        success = self.go_to_pose(
            target.position.x, target.position.y, pre_grasp_z, orientation=target.orientation
        )
        if not success: return False

        # 2. Abrir Garra
        self.get_logger().info(">>> Abrindo Garra...")
        self.open_gripper()
        time.sleep(3.0) 
        
        # 3. Descida Linear (Cartesiana)
        self.get_logger().info(">>> Descendo para o objeto (Linear)...")
        grasp_pose = Pose()
        grasp_pose.position = target.position
        grasp_pose.position.z = grasp_height_z
        grasp_pose.orientation = target.orientation
        

        if not self.go_to_pose_cartesian(grasp_pose):
            self.get_logger().error("Falha na descida linear!")
            return False

        
        
        return True

    def perform_realignment_maneuver(self):
        """Sobe 20cm (Linear) para permitir re-scan"""
        self.get_logger().info(">>> Realinhando: Subindo 20cm...")
        
        if not self.latest_grasp: return False
        
        retreat_pose = Pose()
        retreat_pose.position = self.latest_grasp.position
        retreat_pose.position.z = self.latest_grasp.position.z + 0.35 # Altura segura
        retreat_pose.orientation = self.latest_grasp.orientation
        
        return self.go_to_pose_cartesian(retreat_pose)

    def perform_finish_pick(self):
        """Fecha a garra e levanta o objeto"""
        # 1. Fechar
        self.get_logger().info(">>> Fechando Garra...")
        self.close_gripper()
        time.sleep(5.0) # Espera firmar
        
        # 2. Levantar
        self.get_logger().info(">>> Levantando (Linear)...")
        if not self.latest_grasp: return

        lift_pose = Pose()
        lift_pose.position = self.latest_grasp.position
        lift_pose.position.z += 0.40 # Sobe 40cm
        lift_pose.orientation = self.latest_grasp.orientation

        self.go_to_pose_cartesian(lift_pose)
        


# --- INTERFACE CORRIGIDA ---
def user_interface(node):
    time.sleep(2)
    node.add_scene_objects()

    while rclpy.ok():
        print("\n" + "=" * 40)
        print(" PANDA COMMANDER - MENU")
        print("=" * 40)
        print("1. Escolher Objeto")
        print("2. Ir para Home")
        print("0. Sair")

        opt = input(">> Opção: ")

        if opt == '0':
            node.stop_perception()
            break
        elif opt == '2':
            node.stop_perception()
            node.go_to_pose(0.3, 0.0, 0.6)

        elif opt == '1':
            # --- SELEÇÃO DE OBJETO ---
            while True:
                print("\n[BUSCA] Qual objeto?")
                print("1. mug")
                print("2. bottle")
                print("0. voltar")
                choice = input(">> ").strip()
                
                if choice == '0': break
                obj_name = "mug" if choice == '1' else "bottle" if choice == '2' else None
                if not obj_name: continue

                # Inicia Busca
                print(f"[SYSTEM] Buscando '{obj_name}'...")
                node.set_perception_target(obj_name)
                
                # --- LOOP DE GRASP & REFINAMENTO ---
                while True:
                    print("[SYSTEM] Aguardando Grasp do GPD (20s)...")
                    found = node.grasp_event.wait(timeout=20.0)

                    if not found:
                        print(f"[ERRO] Grasp não encontrado.")
                        if input("Tentar de novo? (s/n): ").lower() == 's': continue
                        else: break 

                    # ---------------------------------------------------------
                    # DECISÃO 1: CONFIRMAR SE QUER DESCER (O que faltava antes)
                    # ---------------------------------------------------------
                    print(f"\n[GRASP ENCONTRADO] X={node.latest_grasp.position.x:.2f}, Y={node.latest_grasp.position.y:.2f}")
                    acao_inicial = input(">> Executar Aproximação? [S]im, [N]ão (outro grasp), [C]ancelar: ").lower()
                    
                    if acao_inicial == 'n':
                        print("Descartando grasp. Buscando outro...")
                        node.set_perception_target(obj_name) # Reinicia busca
                        continue
                    
                    elif acao_inicial == 'c':
                        print("Cancelando.")
                        node.stop_perception()
                        break 
                    
                    elif acao_inicial != 's':
                        continue # Input inválido, repete

                    # --- EXECUTAR APROXIMAÇÃO (Se escolheu 'S') ---
                    print("Executando aproximação...")
                    success = node.perform_approach(node.latest_grasp)
                    
                    if not success:
                        print("[ERRO] Falha ao aproximar.")
                        node.go_to_pose(0.3, 0.0, 0.6) 
                        break

                    # ---------------------------------------------------------
                    # DECISÃO 2: JÁ ESTÁ EM BAIXO. FECHAR OU MELHORAR?
                    # ---------------------------------------------------------
                    print("\n" + "!"*40)
                    print(" ROBÔ NO ALVO (Garra Aberta).")
                    print("!"*40)
                    print("[F]echar garra e levantar (Concluir)")
                    print("[T]entar Melhorar (Sobe 20cm e busca de novo)")
                    print("[C]ancelar e ir para Home")
                    
                    decisao_final = input(">> Ação: ").lower()

                    if decisao_final == 'f':
                        node.perform_finish_pick()
                        print("[FIM] Pick concluído.")
                        node.stop_perception()
                        break 

                    elif decisao_final == 't':
                        print("[RETRY] Subindo para realinhar...")
                        node.perform_realignment_maneuver()
                        
                        # Limpa grasp e busca de novo (agora estando mais perto)
                        node.set_perception_target(obj_name)
                        print("[SYSTEM] Recalculando grasp de cima...")
                        continue # Volta para o inicio do While True (esperar grasp)

                    elif decisao_final == 'c':
                        print("[ABORT] Cancelando...")
                        node.go_to_pose(0.3, 0.0, 0.6) 
                        node.stop_perception()
                        break 

                break # Sai do loop de seleção e volta menu principal

def main():
    rclpy.init()
    node = MoveItCommander()
    spinner_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
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