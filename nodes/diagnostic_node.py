#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R
import time

class VisionDiagnostic(Node):
    def __init__(self):
        super().__init__('vision_diagnostic')
        self.bridge = CvBridge()
        
        # Variável para controlar o tempo do print
        self.last_print_time = 0.0
        
        # Assina tópicos de visão
        self.sub_rgb = message_filters.Subscriber(self, Image, '/camera/rgb/image_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.sub_mask = message_filters.Subscriber(self, Image, '/perception/mask')
        
        # Sincroniza
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_mask], 10, 0.1)
        self.ts.registerCallback(self.callback)

        # TF Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.cam_info = None
        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_cb, 10)
        
        print("\n=== DIAGNÓSTICO DE VISÃO INICIADO ===")
        print("Aponte o robô para a GARRAFA e aguarde (prints a cada 3s)...\n")

    def info_cb(self, msg):
        self.cam_info = msg

    def callback(self, rgb, depth, mask):
        if self.cam_info is None: return

        # --- TIMER: Só processa se passou 3 segundos desde o último print ---
        current_time = time.time()
        if (current_time - self.last_print_time) < 3.0:
            return
        
        # Atualiza o relógio para o próximo ciclo
        self.last_print_time = current_time

        # 1. Processa imagem
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(depth, "32FC1")
            cv_mask = self.bridge.imgmsg_to_cv2(mask, "mono8")
        except: return

        # 2. Pega apenas pixels da garrafa
        v, u = np.where(cv_mask > 100)
        if len(v) < 50: 
            print(">> Nenhuma garrafa detectada na máscara.")
            return

        # 3. Pega profundidade média
        z_vals = cv_depth[v, u]
        z_vals = z_vals[(z_vals > 0.1) & (z_vals < 2.0)] # Filtra erros
        if len(z_vals) == 0: return
        
        # --- CÁLCULO NA CÂMERA (Bruto) ---
        fx = self.cam_info.k[0]; fy = self.cam_info.k[4]
        cx = self.cam_info.k[2]; cy = self.cam_info.k[5]
        
        z_cam = np.mean(z_vals)
        u_mean = np.mean(u)
        v_mean = np.mean(v)
        
        x_cam = (u_mean - cx) * z_cam / fx
        y_cam = (v_mean - cy) * z_cam / fy
        
        point_cam = np.array([x_cam, y_cam, z_cam])

        # --- CÁLCULO NO ROBÔ (Transformado) ---
        try:
            trans = self.tf_buffer.lookup_transform(
                "panda_link0", "camera_optical_frame", rclpy.time.Time())
            
            rot = R.from_quat([
                trans.transform.rotation.x, trans.transform.rotation.y,
                trans.transform.rotation.z, trans.transform.rotation.w
            ]).as_matrix()
            t = np.array([
                trans.transform.translation.x, trans.transform.translation.y,
                trans.transform.translation.z
            ])
            
            point_robot = rot @ point_cam + t
            
            # --- GABARITO (Bottle do Commander Node) ---
            # Baseado no seu commander_node.py: X=0.3, Y=-0.2
            gabarito = np.array([0.30, -0.20, 0.15]) 
            
            erro = point_robot - gabarito
            
            print(f"--- LEITURA ATUAL ---")
            print(f"CÂMERA (Bruto): [X={point_cam[0]:.2f}, Y={point_cam[1]:.2f}, Z={point_cam[2]:.2f}]")
            print(f"ROBÔ (Calculado):  [X={point_robot[0]:.2f}, Y={point_robot[1]:.2f}, Z={point_robot[2]:.2f}]")
            print(f"GABARITO ESPERADO: {gabarito}")
            print(f"DIFERENÇA (Erro):  {erro}")
            print("-" * 30)

        except Exception as e:
            print(f"Erro de TF: {e}")

def main():
    rclpy.init()
    node = VisionDiagnostic()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()