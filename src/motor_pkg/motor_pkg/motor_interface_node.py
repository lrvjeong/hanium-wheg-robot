import rclpy
from rclpy.node import Node
from robot_msgs.msg import RobotMode
from std_msgs.msg import Float32MultiArray

class MotorInterfaceNode(Node):
    def __init__(self):
        super().__init__('motor_interface_node')

        # ── 속도/토크 파라미터 (실물 테스트 후 조정) ──────────────
        self.PLANAR_SPEED       = 0.5   # 평지 일반 주행
        self.HIGH_TORQUE_SPEED  = 0.7   # 3cm 이하 단차 밀고 넘기
        self.WHEG_SPEED         = 0.8   # 3~10cm 단차 와다다 등반
        self.STOP_SPEED         = 0.0   # 정지

        # 서보 각도 (도 단위, 실물 보고 조정)
        self.WHEEL_MODE_ANGLE   = 0.0   # 바퀴 모드 (Wheg 접힘)
        self.LEG_MODE_ANGLE     = 90.0  # 다리 모드 (Wheg 전개)

        # ── 구독/발행 ───────────────────────────────────────────
        self.create_subscription(RobotMode, '/robot/mode', self.mode_cb, 10)

        # DC 모터 명령 (좌/우 속도)
        self.dc_pub   = self.create_publisher(
            Float32MultiArray, '/motor/dc_cmd', 10
        )
        # 서보 명령 (4개 각도)
        self.servo_pub = self.create_publisher(
            Float32MultiArray, '/motor/servo_cmd', 10
        )

        self.current_mode = RobotMode.PLANAR
        self.get_logger().info('모터 인터페이스 노드 시작')

    def mode_cb(self, msg: RobotMode):
        if msg.state == self.current_mode:
            return  # 같은 모드면 무시

        self.current_mode = msg.state
        labels = {
            0: 'PLANAR',
            1: 'HIGH_TORQUE',
            2: 'WHEG',
            3: 'BLOCKED',
            4: 'SAFETY_STOP'
            5: 'STEP_STOP'
        }
        self.get_logger().info(
            f'모드 수신: {labels.get(msg.state, "UNKNOWN")}'
        )

        if msg.state == RobotMode.PLANAR:
            self.set_dc(self.PLANAR_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)

        elif msg.state == RobotMode.HIGH_TORQUE:
            # 고토크 + 속도 유지 → 3cm 이하 단차 그냥 밀고 넘기
            self.set_dc(self.HIGH_TORQUE_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)  # 바퀴 모드 유지
            self.get_logger().info('고토크 모드 — 단차 밀고 넘기')

        elif msg.state == RobotMode.WHEG:
            # 다리 전개 후 고속 등반
            self.set_servo(self.LEG_MODE_ANGLE)    # 다리 먼저 전개
            self.set_dc(self.WHEG_SPEED)           # 와다다 올라가기
            self.get_logger().info('Wheg 모드 — 다리 전개 후 등반')

        elif msg.state == RobotMode.BLOCKED:
            # 못 넘는 단차 → 정지 (나중에 회피 로직 추가)
            self.set_dc(self.STOP_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)
            self.get_logger().warn('단차 극복 불가 — 정지')

        elif msg.state == RobotMode.SAFETY_STOP:
            # 전복 위험 → 즉시 전체 정지
            self.set_dc(self.STOP_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)
            self.get_logger().error('전복 위험 — 긴급 정지!')
        elif msg.state == RobotMode.STEP_STOP:
            self.set_dc(self.STOP_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)
            self.get_logger().info('30cm 단차 인식 — 정지')

    def set_dc(self, speed: float):
        """DC 모터 좌/우 속도 설정 (0.0~1.0)"""
        msg = Float32MultiArray()
        msg.data = [speed, speed]  # [좌, 우]
        self.dc_pub.publish(msg)
        self.get_logger().info(f'DC 모터: 좌={speed}, 우={speed}')

    def set_servo(self, angle: float):
        """서보 모터 4개 각도 설정 (도 단위)"""
        msg = Float32MultiArray()
        msg.data = [angle, angle, angle, angle]  # 4개 동일 각도
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
