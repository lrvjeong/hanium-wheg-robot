#!/usr/bin/env python3
"""
CygLiDAR D2의 /scan (sensor_msgs/LaserScan)을 구독해서
연속된 스캔 사이의 거리 급변(단차 edge)을 감지하고,
/step_detected (std_msgs/Bool) 토픽으로 발행합니다.

단차 감지 로직:
    - LaserScan.ranges 배열에서 인접한 두 지점의 거리 차이가
      STEP_HEIGHT_THRESHOLD_M 이상이면 "단차"로 판단
    - 단순 거리 임계값 방식이라 튜닝이 필요합니다.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


class StepDetectorNode(Node):
    def __init__(self):
        super().__init__('step_detector_node')

        # ------------------- 튜닝 파라미터 -------------------
        self.declare_parameter('step_height_threshold_m', 0.05)  # 단차로 판단할 거리 차이 (m)
        self.declare_parameter('min_valid_range_m', 0.05)
        self.declare_parameter('max_valid_range_m', 2.0)
        # ------------------------------------------------------

        self.threshold = self.get_parameter('step_height_threshold_m').value
        self.min_range = self.get_parameter('min_valid_range_m').value
        self.max_range = self.get_parameter('max_valid_range_m').value

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.publisher = self.create_publisher(Bool, '/step_detected', 10)

        self.get_logger().info(
            f'단차 감지 시작 (임계값: {self.threshold} m)'
        )

    def scan_callback(self, msg: LaserScan):
        ranges = msg.ranges
        step_found = False

        for i in range(1, len(ranges)):
            r1, r2 = ranges[i - 1], ranges[i]

            # 유효하지 않은 값(0, inf, nan)은 건너뜀
            if not (self.min_range < r1 < self.max_range):
                continue
            if not (self.min_range < r2 < self.max_range):
                continue

            if abs(r2 - r1) >= self.threshold:
                step_found = True
                self.get_logger().info(
                    f'단차 감지! index {i}: {r1:.3f}m -> {r2:.3f}m (차이 {abs(r2 - r1):.3f}m)'
                )
                break

        msg_out = Bool()
        msg_out.data = step_found
        self.publisher.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    node = StepDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
