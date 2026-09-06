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
        self.WHEG_SPEED         = 0.8   # 3~8cm 단차 와다다 등반
        self.STOP_SPEED         = 0.0   # 정지
        self.REVERSE_SPEED      = -0.5  # 후진 (블락 회피, 실물 보고 조정)

        # BLOCKED 회피 시퀀스: 후진 → 제자리 방향전환 → 정지
        # (blocked_mode.py의 reverse→turn_one_wheel 시퀀스를 이 노드 안으로 이식)
        self.REVERSE_DURATION_SEC = 2.0   # 후진 지속 시간(초), 실측 후 조정
        self.TURN_DURATION_SEC    = 2.5   # 방향전환 지속 시간(초), 실측 후 조정
        self.TURN_SPEED           = 0.6   # 방향전환 시 좌우 반대로 줄 속도 크기
        self._blocked_reverse_timer = None
        self._blocked_turn_timer = None

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

        # BLOCKED 모드에서 빠져나가면(예: SAFETY_STOP이 끼어든 경우) 회피 시퀀스 중단
        if msg.state != RobotMode.BLOCKED:
            self._cancel_blocked_timers()

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
            # c: 후진 → 제자리 방향전환 순서로 회피 (blocked_mode.py 시퀀스 이식)
            self.set_servo(self.WHEEL_MODE_ANGLE)
            self.get_logger().warn('단차 극복 불가 — 후진 후 방향전환 회피 시작')
            self._start_blocked_avoid_sequence()

        elif msg.state == RobotMode.SAFETY_STOP:
            self.set_dc(self.STOP_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)
            self.get_logger().error('전복 위험 — 긴급 정지!')

    def _wheg_hold_cb(self):
        self.set_servo(self.LEG_MODE_ANGLE)

    # ── BLOCKED 회피 시퀀스: 후진 → 방향전환 → 정지 ──────────────
    def _start_blocked_avoid_sequence(self):
        self._cancel_blocked_timers()  # 혹시 이전 시퀀스가 남아있으면 정리

        self.get_logger().info(f'[회피] 후진 시작 ({self.REVERSE_DURATION_SEC}초)')
        self.set_dc(self.REVERSE_SPEED)

        self._blocked_reverse_timer = self.create_timer(
            self.REVERSE_DURATION_SEC, self._blocked_start_turn
        )

    def _blocked_start_turn(self):
        if self._blocked_reverse_timer is not None:
            self._blocked_reverse_timer.cancel()
            self._blocked_reverse_timer = None

        self.get_logger().info(f'[회피] 방향전환 시작 ({self.TURN_DURATION_SEC}초)')
        # 좌우를 반대 부호로 줘서 제자리 회전에 가깝게 방향 전환
        self.set_dc(-self.TURN_SPEED, self.TURN_SPEED)

        self._blocked_turn_timer = self.create_timer(
            self.TURN_DURATION_SEC, self._blocked_finish
        )

    def _blocked_finish(self):
        if self._blocked_turn_timer is not None:
            self._blocked_turn_timer.cancel()
            self._blocked_turn_timer = None

        self.set_dc(self.STOP_SPEED)
        self.get_logger().info('[회피] 완료 — 정지, 재주행 판단 대기')

    def _cancel_blocked_timers(self):
        if self._blocked_reverse_timer is not None:
            self._blocked_reverse_timer.cancel()
            self._blocked_reverse_timer = None
        if self._blocked_turn_timer is not None:
            self._blocked_turn_timer.cancel()
            self._blocked_turn_timer = None

    def set_dc(self, left_speed: float, right_speed: float = None):
        """좌우 속도를 따로 줄 수 있도록 확장 (right_speed 생략 시 좌우 동일)"""
        if right_speed is None:
            right_speed = left_speed
        msg = Float32MultiArray()
        msg.data = [left_speed, right_speed]
        self.dc_pub.publish(msg)
        self.get_logger().info(f'DC 모터: 좌={left_speed}, 우={right_speed}')

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
