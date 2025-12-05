#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from pynput import keyboard
import math

class RobotSimulation(Node):
    def __init__(self):
        super().__init__('robot_simulation')
        self.publisher_ = self.create_publisher(Vector3, 'robot_pose_raw', 10)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.step = 0.1
        self.turn = math.radians(10)

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

    def publish_pose(self):
        pose = Vector3()
        pose.x = float(self.x)
        pose.y = float(self.y)
        pose.z = float(self.theta)
        self.publisher_.publish(pose)

    def on_press(self, key):
        try:
            if key.char == 'z':
                self.x += self.step * math.cos(self.theta)
                self.y += self.step * math.sin(self.theta)
            elif key.char == 's':
                self.x -= self.step * math.cos(self.theta)
                self.y -= self.step * math.sin(self.theta)
            elif key.char == 'd':
                self.theta += self.turn
            elif key.char == 'q':
                self.theta -= self.turn
            self.publish_pose()
        except AttributeError:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = RobotSimulation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()