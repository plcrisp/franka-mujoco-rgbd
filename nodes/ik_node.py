import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
import mujoco
import numpy as np
import time

# --- Configurações ---
XML_PATH = "../model/scene.xml"  # O mesmo arquivo que a simulação usa
EndEffector_Name = "hand"        # Nome do corpo da garra no XML (verifique se é 'hand' ou 'link7')

class MujocoIKNode(Node):
    def __init__(self):
        super().__init__('ik_node')
        
        # 1. Carrega o modelo do MuJoCo apenas para matemática (sem janela)
        try:
            self.m = mujoco.MjModel.from_xml_path(XML_PATH)
            self.d = mujoco.MjData(self.m)
            self.get_logger().info(f"Modelo MuJoCo carregado para cálculos IK: {XML_PATH}")
        except Exception as e:
            self.get_logger().error(f"Erro ao carregar XML: {e}")
            return

        # 2. Identifica o ID da garra
        try:
            self.ee_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, EndEffector_Name)
            if self.ee_id == -1: 
                # Tenta nomes comuns caso não seja 'hand'
                self.ee_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "panda_link7")
            self.get_logger().info(f"End Effector ID: {self.ee_id}")
        except:
            self.get_logger().error("Não achei o corpo da garra. Verifique o XML.")

        # 3. Comunicação ROS
        self.pub_joints = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.sub_target = self.create_subscription(Point, '/target_position', self.calculate_ik_callback, 10)
        
        # Posição inicial de busca (Home Pose)
        self.q_ref = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

    def calculate_ik_callback(self, msg):
        target_pos = np.array([msg.x, msg.y, msg.z])
        self.get_logger().info(f"Calculando IK para: {target_pos}")

        # --- ALGORITMO IK NUMÉRICO (Damped Least Squares) ---
        # Reseta o modelo interno para a posição de referência (para não começar do zero)
        # Nota: O Panda tem 7 juntas + 2 da garra. Vamos focar nas 7 primeiras.
        
        # Copia a referência atual para o simulador interno
        self.d.qpos[:7] = self.q_ref
        mujoco.mj_forward(self.m, self.d)
        
        # Loop de convergência (Gradient Descent)
        # Tenta chegar no alvo em no máximo 50 passos
        for i in range(50):
            # 1. Onde está a garra agora?
            current_pos = self.d.xpos[self.ee_id]
            
            # 2. Erro (distância até o alvo)
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            # Se chegou perto o suficiente (1cm), para.
            if error_norm < 0.01:
                break

            # 3. Calcula o Jacobiano (Matriz que diz como cada junta afeta a posição)
            # jacp = jacobiano de posição (3 linhas x N juntas)
            jacp = np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.d, jacp, None, current_pos, self.ee_id)
            
            # Pegamos apenas as colunas das 7 juntas do braço (ignora garra)
            J = jacp[:, :7]
            
            # 4. Calcula a correção dos ângulos (Delta Q)
            # Fórmula: dq = J_pseudo_inversa * erro
            # alpha é a "velocidade" de aprendizado
            alpha = 0.5
            dq = alpha * np.linalg.pinv(J) @ error
            
            # 5. Aplica e atualiza
            self.d.qpos[:7] += dq
            mujoco.mj_forward(self.m, self.d)

        # --- FIM DO CÁLCULO ---
        
        # Pega o resultado final
        result_joints = self.d.qpos[:7].copy()
        
        # Atualiza a referência para a próxima vez ser mais rápida
        self.q_ref = result_joints
        
        # Publica
        msg_cmd = Float64MultiArray()
        msg_cmd.data = result_joints.tolist()
        self.pub_joints.publish(msg_cmd)
        
        self.get_logger().info("Comando enviado!")

def main(args=None):
    rclpy.init(args=args)
    node = MujocoIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()