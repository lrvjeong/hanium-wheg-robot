import rclpy
from rclpy.node import Node
from robot_msgs.msg import RobotMode
from std_msgs.msg import Float32MultiArray


class MotorInterfaceNode(Node):
    def __init__(self):
        super().__init__('motor_interface_node')

        # ── 속도/토크 파라미터 (실물 테스트 후 조정) ──────────────
        self.PLANAR_SPEED       = 0.5   # 평지 일반 주행
        self.HIGH_TORQUE_SPEED  = 0.9   # 1~3cm 단차 - 속도 업, 서보 제어 없음
        self.WHEG_SPEED         = 0.8   # 3~6cm 단차 와다다 등반
        self.STOP_SPEED         = 0.0   # 정지
        self.REVERSE_SPEED      = -0.5  # 후진 (블락 회피, 실물 보고 조정)

        # 서보 각도 (도 단위, 실물 보고 조정)
        self.WHEEL_MODE_ANGLE   = 0.0    # 바퀴 모드 (Wheg 완전 접힘)
        self.LEG_MODE_ANGLE     = 90.0   # 다리 모드 (Wheg 완전 전개)

        self.WHEG_HOLD_HZ = 5.0  # 휘그 모드 중 서보 각도 주기적 재전송 (유지용)
        self.wheg_hold_timer = None

        # ── 구독/발행 ───────────────────────────────────────────
        self.create_subscription(RobotMode, '/robot/mode', self.mode_cb, 10)

        self.dc_pub = self.create_publisher(
            Float32MultiArray, '/motor/dc_cmd', 10
        )
        self.servo_pub = self.create_publisher(
            Float32MultiArray, '/motor/servo_cmd', 10
        )

        self.current_mode = RobotMode.PLANAR
        self.get_logger().info('모터 인터페이스 노드 시작')

    def mode_cb(self, msg: RobotMode):
        if msg.state == self.current_mode:
            return

        self.current_mode = msg.state

        # WHEG 모드에서 빠져나가면 유지 타이머 정지
        if msg.state != RobotMode.WHEG and self.wheg_hold_timer is not None:
            self.wheg_hold_timer.cancel()
            self.wheg_hold_timer = None

        labels = {
            0: 'PLANAR', 1: 'HIGH_TORQUE', 2: 'WHEG',
            3: 'BLOCKED', 4: 'SAFETY_STOP', 5: 'STEP_STOP'
        }
        self.get_logger().info(
            f'모드 수신: {labels.get(msg.state, "UNKNOWN")}'
        )

        if msg.state == RobotMode.PLANAR:
            self.set_dc(self.PLANAR_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)

        elif msg.state == RobotMode.STEP_STOP:
            self.set_dc(self.STOP_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)
            self.get_logger().info('단차 인식 — 정지, 높이 판별 중')

        elif msg.state == RobotMode.HIGH_TORQUE:
            # a: 서보 제어 없이 속도만 업 (다리는 토크로 자연스럽게 살짝 펼쳐짐)
            self.set_dc(self.HIGH_TORQUE_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)
            self.get_logger().info('고토크 모드 — 고속으로 단차 밀고 넘기')

        elif msg.state == RobotMode.WHEG:
            # b: 다리 전개 + 구동 동시 시작, 이후 유지 타이머로 각도 계속 재전송
            self.set_servo(self.LEG_MODE_ANGLE)
            self.set_dc(self.WHEG_SPEED)
            self.get_logger().info('Wheg 모드 — 다리 전개 후 등반')
            if self.wheg_hold_timer is None:
                self.wheg_hold_timer = self.create_timer(
                    1.0 / self.WHEG_HOLD_HZ, self._wheg_hold_cb
                )

        elif msg.state == RobotMode.BLOCKED:
            # c: 후진으로 회피
            self.set_dc(self.REVERSE_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)
            self.get_logger().warn('단차 극복 불가 — 후진 회피')

        elif msg.state == RobotMode.SAFETY_STOP:
            self.set_dc(self.STOP_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)
            self.get_logger().error('전복 위험 — 긴급 정지!')

    def _wheg_hold_cb(self):
        self.set_servo(self.LEG_MODE_ANGLE)

    def set_dc(self, speed: float):
        msg = Float32MultiArray()
        msg.data = [speed, speed]
        self.dc_pub.publish(msg)
        self.get_logger().info(f'DC 모터: 좌={speed}, 우={speed}')

    def set_servo(self, angle: float):
        msg = Float32MultiArray()
        msg.data = [angle, angle, angle, angle]
        self.servo_pub.publish(msg)
        self.get_logger().info(f'서보: {angle}도')


def main():
    rclpy.init()
    node = MotorInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('모터 인터페이스 노드 종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
