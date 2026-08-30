#!/usr/bin/env python3
"""
/step_detected (std_msgs/Bool)를 구독해서
    True  -> 다이나믹셀 정지 + 서보로 다리 펼치기 (deploy_legs)
    False -> 다이나믹셀 주행 (drive_forward)
를 실행하는 노드.

dynamixel_servo_combined_control.py의 함수들을 그대로 재사용합니다.
같은 motor_pkg 안에 두고 import 하세요.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from motor_pkg.dynamixel_servo_combined_control import (
    init_dynamixel,
    init_arduino,
    dxl_drive_forward,
    deploy_legs,
    retract_legs,
    dxl_stop,
    DXL_IDS,
    ADDR_TORQUE_ENABLE,
)


class LegControlNode(Node):
    def __init__(self):
        super().__init__('leg_control_node')

        self.dxl_port, self.dxl_packet = init_dynamixel()
        self.arduino = init_arduino()

        self.legs_deployed = False  # 현재 다리 상태 추적 (중복 명령 방지)

        self.subscription = self.create_subscription(
            Bool, '/step_detected', self.step_callback, 10
        )

        self.get_logger().info('다리 제어 노드 시작. 기본 주행 상태로 시작.')
        dxl_drive_forward(self.dxl_port, self.dxl_packet)

    def step_callback(self, msg: Bool):
        if msg.data and not self.legs_deployed:
            self.get_logger().info('단차 감지됨 -> 다리 펼치기')
            deploy_legs(self.dxl_port, self.dxl_packet, self.arduino)
            self.legs_deployed = True

        elif not msg.data and self.legs_deployed:
            self.get_logger().info('단차 통과 -> 다리 접고 주행 재개')
            retract_legs(self.dxl_port, self.dxl_packet, self.arduino)
            self.legs_deployed = False

    def destroy_node(self):
        dxl_stop(self.dxl_port, self.dxl_packet)
        for dxl_id in DXL_IDS:
            self.dxl_packet.write1ByteTxRx(self.dxl_port, dxl_id, ADDR_TORQUE_ENABLE, 0)
        self.dxl_port.closePort()
        self.arduino.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LegControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
