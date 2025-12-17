import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, CameraInfo
from cv_bridge import CvBridge
import mujoco
import mujoco.viewer
import numpy as np
import os
import threading
import time

# --- CONFIG ---
XML_PATH = "model/scene.xml" # Ajuste o caminho se necessario
CAM_NAME = "end_effector_camera" # Nome da camera no XML
WIDTH, HEIGHT = 640, 480         # Resolução da imagem

class SimulationNode(Node):
    def __init__(self):
        super().__init__('mujoco_simulation_node')
        
        # 1. Carrega MuJoCo
        if not os.path.exists(XML_PATH):
             self.get_logger().error(f"XML não encontrado: {XML_PATH}")
             # Tenta fallback
             if os.path.exists("model/scene.xml"):
                 path = "model/scene.xml"
             else:
                 raise SystemExit
        else:
             path = XML_PATH

        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)
        
        # 2. Configura Câmera e Renderizador
        self.renderer = mujoco.Renderer(self.m, HEIGHT, WIDTH)
        self.bridge = CvBridge()
        
        # Publishers (Sensores)
        self.pub_rgb = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        
        # Subscribers (Controle do MoveIt)
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        # 3. Mapeamento de Juntas (Lógica do MoveIt Node)
        self.ros_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        self.mujoco_joint_ids = []
        self.ros_finger_names = ["panda_finger_joint1", "panda_finger_joint2"]
        self.mujoco_finger_ids = []

        self._map_joints() # Função auxiliar abaixo
        
        # Buffers de posição
        self.target_arm_qpos = np.zeros(7)
        # Inicializa com a posição atual para não dar "pulo"
        for i, jid in enumerate(self.mujoco_joint_ids):
            if jid != -1: self.target_arm_qpos[i] = self.d.qpos[self.m.jnt_qposadr[jid]]
            
        self.target_finger_qpos = np.array([0.0, 0.0])

        # 4. Configura Intrínsecos da Câmera (Matemática)
        self._setup_camera_intrinsics()

        self.get_logger().info("Simulação + Câmera Iniciada!")

    def _map_joints(self):
        # Mapeia braço
        for i, ros_name in enumerate(self.ros_joint_names):
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, ros_name)
            if jid == -1: # Tenta nome curto
                jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            self.mujoco_joint_ids.append(jid)
        
        # Mapeia dedos
        for name in self.ros_finger_names:
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                 jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "finger1" if "1" in name else "finger2")
            self.mujoco_finger_ids.append(jid)

    def _setup_camera_intrinsics(self):
        cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
        if cam_id == -1:
            self.get_logger().warn(f"Câmera '{CAM_NAME}' não encontrada no XML!")
            return

        fovy = self.m.cam_fovy[cam_id]
        f = 0.5 * HEIGHT / np.tan(0.5 * fovy * np.pi/180)
        cx, cy = WIDTH / 2, HEIGHT / 2
        
        self.cam_info = CameraInfo()
        self.cam_info.width, self.cam_info.height = WIDTH, HEIGHT
        self.cam_info.k = [f, 0., cx, 0., f, cy, 0., 0., 1.]
        self.cam_info.p = [f, 0., cx, 0., 0., f, cy, 0., 0., 0., 1., 0.]

    def joint_callback(self, msg):
        # Recebe do MoveIt e guarda no buffer
        for i, ros_name in enumerate(self.ros_joint_names):
            if ros_name in msg.name:
                self.target_arm_qpos[i] = msg.position[msg.name.index(ros_name)]
        
        for i, ros_name in enumerate(self.ros_finger_names):
            if ros_name in msg.name:
                val = msg.position[msg.name.index(ros_name)]
                if i < 2: self.target_finger_qpos[i] = val

    def run(self):
        # Loop Principal
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            while viewer.is_running() and rclpy.ok():
                step_start = time.time()

                # A. Aplica Posições do MoveIt (Teletransporte controlado)
                for i, jid in enumerate(self.mujoco_joint_ids):
                    if jid != -1: self.d.qpos[self.m.jnt_qposadr[jid]] = self.target_arm_qpos[i]
                for i, jid in enumerate(self.mujoco_finger_ids):
                    if jid != -1: self.d.qpos[self.m.jnt_qposadr[jid]] = self.target_finger_qpos[i]

                # B. Avança Física
                mujoco.mj_step(self.m, self.d)
                viewer.sync()

                # C. Renderiza e Publica Câmera (A cada X passos para não pesar)
                # Renderizar a 60FPS pesa muito. Vamos tentar ~15-30FPS
                self.publish_camera()

                # D. Mantém tempo real
                time_until_next = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

    def publish_camera(self):
        # Atualiza cena do renderizador
        self.renderer.update_scene(self.d, camera=CAM_NAME)
        
        # RGB
        rgb = self.renderer.render()
        
        # Depth
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()

        timestamp = self.get_clock().now().to_msg()
        
        # Mensagem RGB
        msg_rgb = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        msg_rgb.header.stamp = timestamp
        msg_rgb.header.frame_id = 'camera_optical_frame'
        self.pub_rgb.publish(msg_rgb)

        # Mensagem Depth
        msg_depth = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        msg_depth.header.stamp = timestamp
        msg_depth.header.frame_id = 'camera_optical_frame'
        self.pub_depth.publish(msg_depth)

        # Mensagem Info
        self.cam_info.header.stamp = timestamp
        self.pub_info.publish(self.cam_info)

def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()
    
    # Thread separada para o ROS spin (callbacks)
    thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    thread.start()

    try:
        node.run() # Roda o loop de física/render na thread principal
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()