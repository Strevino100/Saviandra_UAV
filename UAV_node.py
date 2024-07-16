#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class UAV(Node):

    def __init__(self):
        super().__init__("uav")
        self.get_logger().info("UAV is Awesome") 

def main(args=None):
    rclpy.init(args=args)
    node = UAV()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
