import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
import mujoco
import numpy as np
import os
from scipy.optimize import minimize

# --- CONFIG ---
XML_PATH = "../model/scene.xml"
EE_NAME = "hand"

class OptimizationIKNode(Node):
    def __init__(self):
        super().__init__('optimization_ik_node')
        
        # Load MuJoCo model
        if not os.path.exists(XML_PATH):
            self.get_logger().error(f"XML not found: {XML_PATH}")
            raise SystemExit
        
        self.m = mujoco.MjModel.from_xml_path(XML_PATH)
        self.d = mujoco.MjData(self.m)
        
        # End-effector ID
        self.ee_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, EE_NAME)
        if self.ee_id == -1:
            self.get_logger().error(f"End effector '{EE_NAME}' not found")
            raise SystemExit

        # Joint limits
        self.joint_limits = [self.m.jnt_range[i] for i in range(7)]

        # ROS setup
        self.pub_joints = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.sub_target = self.create_subscription(Point, '/target_position', self.solve_ik, 10)
        
        # Fixed initial seed pose for optimization
        self.seed_pose = np.array([0.0, 0.5, 0.0, -2.0, 0.0, 2.5, 0.785])
        
        self.get_logger().info("IK node ready.")

    def objective_function(self, x, target_pos):
        # Update internal simulation
        self.d.qpos[:7] = x
        mujoco.mj_kinematics(self.m, self.d)
        
        # Position error
        current_pos = self.d.xpos[self.ee_id]
        dist_error = np.linalg.norm(current_pos - target_pos)
        
        # Orientation error (match Z-axis)
        current_z_axis = self.d.xmat[self.ee_id][6:9]
        target_z_axis = np.array([0, 0, -1])
        orient_error = np.linalg.norm(current_z_axis - target_z_axis)
        
        return dist_error + 2.0 * orient_error

    def solve_ik(self, msg):
        target_pos = np.array([msg.x, msg.y, msg.z])
        self.get_logger().info(f"Computing IK from scratch for target: {target_pos}")

        result = minimize(
            fun=self.objective_function,
            x0=self.seed_pose,
            args=(target_pos,),
            method='SLSQP',
            bounds=self.joint_limits,
            tol=1e-4,
            options={'maxiter': 200}
        )

        if result.success or result.fun < 0.05:
            self.get_logger().info(f"IK success (error: {result.fun:.4f})")
            
            msg_cmd = Float64MultiArray()
            msg_cmd.data = result.x.tolist()
            self.pub_joints.publish(msg_cmd)
        else:
            self.get_logger().warn(f"IK failed (error: {result.fun:.4f})")

def main(args=None):
    rclpy.init(args=args)
    node = OptimizationIKNode()
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
