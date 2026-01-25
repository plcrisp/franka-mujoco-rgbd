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


# Configuration
XML_PATH = "model/scene.xml"
CAM_NAME = "end_effector_camera"
WIDTH, HEIGHT = 640, 480

USE_SCALING = False


class SimulationNode(Node):
    def __init__(self):
        super().__init__('mujoco_simulation_node')

        # Load MuJoCo model
        if not os.path.exists(XML_PATH):
            self.get_logger().error(f"XML not found: {XML_PATH}")
            if os.path.exists("model/scene.xml"):
                path = "model/scene.xml"
            else:
                raise SystemExit
        else:
            path = XML_PATH

        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)

        # Rendering and camera
        self.renderer = mujoco.Renderer(self.m, HEIGHT, WIDTH)
        self.bridge = CvBridge()

        # ROS publishers
        self.pub_rgb = self.create_publisher(
            Image, '/camera/rgb/image_raw', 10
        )
        self.pub_depth = self.create_publisher(
            Image, '/camera/depth/image_raw', 10
        )
        self.pub_info = self.create_publisher(
            CameraInfo, '/camera/camera_info', 10
        )

        # ROS subscribers
        self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )

        # Joint mapping
        self.ros_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        self.ros_finger_names = [
            "panda_finger_joint1",
            "panda_finger_joint2"
        ]

        self.mujoco_joint_ids = []
        self.mujoco_finger_ids = []
        self.current_gripper_val = 0.0

        self.m = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.m)

        self._map_joints()

        # Target buffers
        self.target_arm_qpos = np.zeros(7)
        for i, jid in enumerate(self.mujoco_joint_ids):
            if jid != -1:
                self.target_arm_qpos[i] = self.d.qpos[
                    self.m.jnt_qposadr[jid]
                ]

        self.target_finger_qpos = np.array([0.0, 0.0])

        # Camera intrinsics
        self._setup_camera_intrinsics()

        self.get_logger().info("Simulation started")
        self.get_logger().info(f"DEBUG MODE: Scaling is {'ON (255)' if USE_SCALING else 'OFF (Direct Mapping)'}")

    # Joint mapping
    def _map_joints(self):
        for i, ros_name in enumerate(self.ros_joint_names):
            jid = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_JOINT, ros_name
            )
            if jid == -1:
                jid = mujoco.mj_name2id(
                    self.m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}"
                )
            self.mujoco_joint_ids.append(jid)

        self.gripper_actuator_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_act"
        )

        for name in ["finger_joint1", "finger_joint2"]:
            jid = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            self.mujoco_finger_ids.append(jid)

        if self.gripper_actuator_id == -1:
            self.get_logger().error(
                "ERROR: 'gripper_act' not found in XML"
            )
        else:
            self.get_logger().info(
                f"Gripper actuator found (ID {self.gripper_actuator_id})"
            )

    # Camera intrinsics
    def _setup_camera_intrinsics(self):
        cam_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME
        )
        if cam_id == -1:
            self.get_logger().warn(f"Camera '{CAM_NAME}' not found")
            return

        fovy = self.m.cam_fovy[cam_id]
        f = 0.5 * HEIGHT / np.tan(0.5 * fovy * np.pi / 180)
        cx, cy = WIDTH / 2, HEIGHT / 2

        self.cam_info = CameraInfo()
        self.cam_info.width = WIDTH
        self.cam_info.height = HEIGHT
        self.cam_info.k = [f, 0., cx, 0., f, cy, 0., 0., 1.]
        self.cam_info.p = [
            f, 0., cx, 0.,
            0., f, cy, 0.,
            0., 0., 1., 0.
        ]

    # ROS callback
    def joint_callback(self, msg):
        # Update Arm
        for i, ros_name in enumerate(self.ros_joint_names):
            if ros_name in msg.name:
                self.target_arm_qpos[i] = msg.position[
                    msg.name.index(ros_name)
                ]
        # Update Gripper
        for i, ros_name in enumerate(self.ros_finger_names):
            if ros_name in msg.name:
                self.target_finger_qpos[i] = msg.position[
                    msg.name.index(ros_name)
                ]

    # Main simulation loop
    def run(self):
        log_counter = 0
        
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            while viewer.is_running() and rclpy.ok():
                step_start = time.time()

                # Arm joints (teleport)
                for i, jid in enumerate(self.mujoco_joint_ids):
                    if jid != -1:
                        self.d.qpos[self.m.jnt_qposadr[jid]] = self.target_arm_qpos[i]

                # --- GRIPPER LOGIC & DEBUG ---
                ros_target = self.target_finger_qpos[0] 

                # 1. Alvo final (Mapeamento 1:1 em Metros)
                target_mujoco = ros_target
                target_mujoco = max(0.0, min(0.04, target_mujoco))

                # 2. Fator de Suavização (Alpha)
                # 0.05 = Muito lento/suave (como hidráulica)
                # 0.20 = Rápido mas fluido
                # 1.00 = Instantâneo (sem suavização)
                alpha = 0.08  

                # 3. Aplica o filtro (Média Exponencial)
                # O valor atual "caminha" 8% da distância em direção ao alvo a cada loop
                diff = target_mujoco - self.current_gripper_val
                self.current_gripper_val += alpha * diff
                
                # Envia para o MuJoCo
                if self.gripper_actuator_id != -1:
                    self.d.ctrl[self.gripper_actuator_id] = self.current_gripper_val

                # LOGGING (A cada 100 loops para não travar o terminal)
                log_counter += 1
                if log_counter % 100 == 0:
                    self.get_logger().info(
                        f"[GRIPPER DEBUG] ROS: {ros_target:.4f} | MuJoCo Target: {target_mujoco:.4f} | Atual: {self.current_gripper_val:.4f}"
                    )

                mujoco.mj_step(self.m, self.d)
                viewer.sync()

                self.publish_camera()

                dt = self.m.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)

    # Camera publishing
    def publish_camera(self):
        self.renderer.update_scene(self.d, camera=CAM_NAME)

        rgb = self.renderer.render()

        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()

        stamp = self.get_clock().now().to_msg()

        msg_rgb = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        msg_rgb.header.stamp = stamp
        msg_rgb.header.frame_id = 'camera_optical_frame'
        self.pub_rgb.publish(msg_rgb)

        msg_depth = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        msg_depth.header.stamp = stamp
        msg_depth.header.frame_id = 'camera_optical_frame'
        self.pub_depth.publish(msg_depth)

        self.cam_info.header.stamp = stamp
        self.pub_info.publish(self.cam_info)


# Main
def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()

    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(node),
        daemon=True
    )
    spin_thread.start()

    try:
        node.run()

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
