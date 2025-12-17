import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import mujoco
import mujoco.viewer
import numpy as np
import os
import threading
import time

# --- CONFIG ---
DEFAULT_XML_PATH = "model/scene.xml"

class MoveItBridgeNode(Node):
    def __init__(self):
        super().__init__('moveit_bridge_node')
        
        # 1. Carrega MuJoCo
        final_path = DEFAULT_XML_PATH
        if not os.path.exists(final_path):
             if os.path.exists("model/scene.xml"):
                 final_path = "model/scene.xml"
             else:
                 self.get_logger().error("XML não encontrado!")
                 raise SystemExit
        
        self.m = mujoco.MjModel.from_xml_path(final_path)
        self.d = mujoco.MjData(self.m)
        
        # 2. ROS Setup
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        # --- MAPEAMENTO DE NOMES (A CORREÇÃO ESTÁ AQUI) ---
        # Nomes que o ROS/MoveIt usa (sempre panda_jointX)
        self.ros_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        
        # Nomes que o MuJoCo usa (vamos tentar descobrir)
        self.mujoco_joint_ids = []
        
        self.get_logger().info("--- MAPEANDO JUNTAS ---")
        
        for i, ros_name in enumerate(self.ros_joint_names):
            # 1ª Tentativa: Nome igual ao do ROS (panda_joint1)
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, ros_name)
            
            # 2ª Tentativa: Nome curto (joint1) - Padrão comum no MuJoCo
            if jid == -1:
                short_name = f"joint{i+1}"
                jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, short_name)
                if jid != -1:
                     self.get_logger().info(f"✅ Traduzido: ROS '{ros_name}' -> MuJoCo '{short_name}' (ID: {jid})")
            else:
                 self.get_logger().info(f"✅ Nome exato: ROS '{ros_name}' -> MuJoCo '{ros_name}' (ID: {jid})")

            # Se ainda for -1, aí temos um problema real
            if jid == -1:
                self.get_logger().error(f"❌ CRÍTICO: Não achei nem '{ros_name}' nem 'joint{i+1}' no XML!")
            
            self.mujoco_joint_ids.append(jid)
        
        # Dedos (Geralmente chamam 'finger1' e 'finger2' ou 'panda_finger_...')
        self.ros_finger_names = ["panda_finger_joint1", "panda_finger_joint2"]
        self.mujoco_finger_ids = []
        for name in self.ros_finger_names:
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
            # Tentativa fallback para dedos (finger1, finger2)
            if jid == -1:
                short_name = name.replace("panda_finger_joint", "finger") 
                jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, short_name)
            self.mujoco_finger_ids.append(jid)

        # Buffer
        self.target_arm_qpos = np.zeros(7)
        # Inicializa com a posição atual do robô para evitar "pulo"
        for i, jid in enumerate(self.mujoco_joint_ids):
            if jid != -1:
                self.target_arm_qpos[i] = self.d.qpos[self.m.jnt_qposadr[jid]]

        self.target_finger_qpos = np.array([0.0, 0.0])

        self.get_logger().info("Ponte Pronta! Verifique se apareceram os ✅ verdes acima.")

    def joint_callback(self, msg):
        # Braço
        for i, ros_name in enumerate(self.ros_joint_names):
            if ros_name in msg.name:
                idx = msg.name.index(ros_name)
                self.target_arm_qpos[i] = msg.position[idx]

        # Dedos
        for i, ros_name in enumerate(self.ros_finger_names):
             if ros_name in msg.name:
                idx = msg.name.index(ros_name)
                if i < 2: self.target_finger_qpos[i] = msg.position[idx]

def run_simulation(node):
    with mujoco.viewer.launch_passive(node.m, node.d) as viewer:
        while viewer.is_running() and rclpy.ok():
            step_start = time.time()

            # Aplica posições (Só se o ID for válido)
            for i, mj_id in enumerate(node.mujoco_joint_ids):
                if mj_id != -1:
                    addr = node.m.jnt_qposadr[mj_id]
                    node.d.qpos[addr] = node.target_arm_qpos[i]
            
            for i, mj_id in enumerate(node.mujoco_finger_ids):
                if mj_id != -1:
                    addr = node.m.jnt_qposadr[mj_id]
                    node.d.qpos[addr] = node.target_finger_qpos[i]

            mujoco.mj_step(node.m, node.d)
            viewer.sync()

            time_until_next_step = node.m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

def main(args=None):
    rclpy.init(args=args)
    node = MoveItBridgeNode()
    ros_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    ros_thread.start()
    try:
        run_simulation(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()