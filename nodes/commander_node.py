#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, CollisionObject
from shape_msgs.msg import SolidPrimitive
import sys
import time

class MoveItMenu(Node):
    def __init__(self):
        super().__init__('moveit_menu_node')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self._scene_pub = self.create_publisher(CollisionObject, '/collision_object', 10)

    def add_scene_objects(self):
        self.get_logger().info("--- SINCRONIZANDO CENÁRIO (RVIZ <-> MUJOCO) ---")
        
        # 1. Mesa (Ajustada para a base do robô)
        # O robô está em cima da mesa. Então o topo da mesa é Z=0.
        # Como a mesa tem 0.04m de altura, o centro dela fica em -0.02m.
        table = CollisionObject()
        table.header.frame_id = "panda_link0"
        table.id = "table"
        # Dimensões Totais: [X=1.0, Y=1.5, Z=0.04]
        box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.0, 1.5, 0.04])
        table_pose = Pose()
        table_pose.position.x = 0.3  # Centro da mesa (Mundo 0.5 - Robô 0.2)
        table_pose.position.y = 0.0
        table_pose.position.z = -0.02 # Centro geométrico da madeira
        table.primitives.append(box); table.primitive_poses.append(table_pose)
        table.operation = CollisionObject.ADD
        
        # 2. Caneca (Mug)
        # Altura = 0.12m. Para a base estar em 0, o centro deve ser 0.06m.
        mug = CollisionObject()
        mug.header.frame_id = "panda_link0"
        mug.id = "mug"
        # Cylinder: [Altura, Raio] -> [0.12, 0.05]
        mbox = SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[0.12, 0.05])
        mpose = Pose()
        mpose.position.x = 0.3
        mpose.position.y = 0.2
        mpose.position.z = 0.06 # Exatamente metade da altura
        mug.primitives.append(mbox); mug.primitive_poses.append(mpose)
        mug.operation = CollisionObject.ADD

        # 3. Garrafa (Bottle)
        # Altura = 0.15m. Para a base estar em 0, o centro deve ser 0.075m.
        bottle = CollisionObject()
        bottle.header.frame_id = "panda_link0"
        bottle.id = "bottle"
        # Cylinder: [Altura, Raio] -> [0.15, 0.03]
        bbox = SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[0.15, 0.03])
        bpose = Pose()
        bpose.position.x = 0.3
        bpose.position.y = -0.2
        bpose.position.z = 0.075 # Exatamente metade da altura
        bottle.primitives.append(bbox); bottle.primitive_poses.append(bpose)
        bottle.operation = CollisionObject.ADD

        # 4. Martelo (Hammer)
        # Altura (Z) = 0.03m. Para a base estar em 0, o centro deve ser 0.015m.
        hammer = CollisionObject()
        hammer.header.frame_id = "panda_link0"
        hammer.id = "hammer"
        hbox = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.15, 0.05, 0.03])
        hpose = Pose()
        hpose.position.x = 0.5
        hpose.position.y = 0.0
        hpose.position.z = 0.015 # Exatamente metade da altura (Z)
        hammer.primitives.append(hbox); hammer.primitive_poses.append(hpose)
        hammer.operation = CollisionObject.ADD

        # Publica repetidamente
        for _ in range(5):
            self._scene_pub.publish(table)
            self._scene_pub.publish(mug)
            self._scene_pub.publish(bottle)
            self._scene_pub.publish(hammer)
            time.sleep(0.1)
        print("✅ Cenário Carregado!")

    def go_to_pose(self, x, y, z):
        print(f"🚀 Indo para: X={x}, Y={y}, Z={z}")
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
        
        # Cria Pose Alvo
        target_pose = Pose()
        target_pose.position.x = float(x)
        target_pose.position.y = float(y)
        target_pose.position.z = float(z)
        # Orientação fixa (Mão para baixo)
        target_pose.orientation.x = 0.924
        target_pose.orientation.y = -0.382
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 0.0

        # Restrições
        c = Constraints(); c.name = "goal"
        
        pc = PositionConstraint()
        pc.header.frame_id = "panda_link0"
        pc.link_name = "panda_link8"
        pc.constraint_region.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.005]))
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
            print("❌ O Planejador rejeitou (Colisão ou fora de alcance).")
            return

        print("⏳ Movendo...")
        res_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        result = res_future.result().result
        
        if result.error_code.val == 1:
            print("✅ CHEGOU!")
        else:
            print(f"❌ Falhou. Erro: {result.error_code.val}")

def main():
    rclpy.init()
    node = MoveItMenu()
    
    # 1. Carrega cena primeiro
    node.add_scene_objects()
    time.sleep(1)

    while True:
        print("\n" + "="*30)
        print("   CONTROLE DO ROBÔ PANDA")
        print("="*30)
        print("1. 🏠 Home (Inicial)")
        print("2. ☕ Caneca (Azul)")
        print("3. 🍾 Garrafa (Verde)")
        print("4. 🔨 Martelo (Vermelho)")
        print("5. 📍 Coordenadas Manuais")
        print("0. ❌ Sair")
        
        escolha = input("\nEscolha uma opção: ")

        if escolha == '0':
            print("Saindo...")
            break
        
        elif escolha == '1': # Home
            node.go_to_pose(0.3, 0.0, 0.5)

        elif escolha == '2': # Caneca
            # Relativo ao robô (Mundo 0.5, 0.2 -> Relativo 0.3, 0.2)
            node.go_to_pose(0.3, 0.2, 0.25) 

        elif escolha == '3': # Garrafa
            # Relativo ao robô (Mundo 0.5, -0.2 -> Relativo 0.3, -0.2)
            node.go_to_pose(0.3, -0.2, 0.30)

        elif escolha == '4': # Martelo
             # Relativo ao robô (Mundo 0.7, 0.0 -> Relativo 0.5, 0.0)
            node.go_to_pose(0.5, 0.0, 0.20)

        elif escolha == '5': # Manual
            try:
                print("\nDigite as coordenadas (Relativas à base do robô!)")
                x = float(input("X (frente/trás): "))
                y = float(input("Y (esq/dir): "))
                z = float(input("Z (altura): "))
                node.go_to_pose(x, y, z)
            except ValueError:
                print("❌ Entrada inválida! Use números (ex: 0.5).")

        else:
            print("Opção inválida.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()