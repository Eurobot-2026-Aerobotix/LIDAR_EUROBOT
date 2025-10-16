#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class RobotPoseTFBroadcaster(Node):
    def __init__(self):
        super().__init__('robot_pose_tf_broadcaster')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.latest_pose = [0.0, 0.0, 0.0]
        self.got_pose = False

        self.subscription = self.create_subscription(
            Float32MultiArray,
            'robot_pose_raw',
            self.pose_callback,
            10
        )
        # Publish TF at 10Hz regardless of movement
        self.timer = self.create_timer(0.1, self.publish_tf)

    def pose_callback(self, msg):
        if len(msg.data) < 3:
            self.get_logger().warn("Invalid pose message, expected [x, y, theta]")
            return
        self.latest_pose = msg.data[:3]
        self.got_pose = True

    def publish_tf(self):
        if not self.got_pose:
            return
        x, y, theta = self.latest_pose
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        # Yaw to quaternion
        t.transform.rotation.z = math.sin(theta / 2.0)
        t.transform.rotation.w = math.cos(theta / 2.0)
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = RobotPoseTFBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()