import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class TofNode(Node):
    """
    CygLiDAR D2의 2D LaserScan('scan' 토픽)에서
    가장 가까운 거리값을 뽑아 /tof/distance 로 재발행하는 노드.
    """

    def __init__(self):
        super().__init__('tof_node')

        self.pub = self.create_publisher(Float32, '/tof/distance', 10)
        self.create_subscription(LaserScan, 'scan', self.scan_cb, 10)

        self.get_logger().info('ToF 노드 시작 (CygLiDAR 2D scan에서 근거리 추출)')

    def scan_cb(self, msg: LaserScan):
        valid_ranges = [
            r for r in msg.ranges
            if math.isfinite(r) and msg.range_min <= r <= msg.range_max
        ]

        if not valid_ranges:
            return

        min_dist = min(valid_ranges)

        out = Float32()
        out.data = min_dist
        self.pub.publish(out)


def main():
    rclpy.init()
    node = TofNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

