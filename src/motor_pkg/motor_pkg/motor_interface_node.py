import rclpy
from rclpy.node import Node
from robot_msgs.msg import RobotMode
from std_msgs.msg import Float32MultiArray


class MotorInterfaceNode(Node):
    def __init__(self):
        super().__init__('motor_interface_node')
        # ── 속도/토크 파라미터 (실물 테스트 후 조정) ──────────────
        self.PLANAR_SPEED       = 0.5
        self.HIGH_TORQUE_SPEED  = 0.9
        self.WHEG_SPEED         = 0.8
        self.STOP_SPEED         = 0.0
        self.REVERSE_SPEED      = -0.5

        # BLOCKED 회피 시퀀스
        self.REVERSE_DURATION_SEC = 2.0
        self.TURN_DURATION_SEC    = 2.5
        self.TURN_SPEED           = 0.6
        self._blocked_reverse_timer = None
        self._blocked_turn_timer = None

        # WHEG 진입/이탈 시 서보+구동모터 동시 역방향 회전 보조
        self.WHEG_DEPLOY_DURATION_SEC  = 5.0
        self.WHEG_DEPLOY_SPEED         = 0.3   # 다이나믹셀 raw velocity 60 (0.3 * 200)
        self.WHEG_RETRACT_DURATION_SEC = 5.0
        self.WHEG_RETRACT_SPEED        = 0.3   # 다이나믹셀 raw velocity 60 (0.3 * 200)
        self._wheg_deploy_timer = None
        self._wheg_retract_timer = None
        self._pending_state_after_retract = None

        # (추가) SAFETY_STOP 중 다리 접기 보조 회전 타이머
        self._safety_fold_timer = None

        # 서보 각도
        self.WHEEL_MODE_ANGLE   = 0.0
        self.LEG_MODE_ANGLE     = 90.0
        self.WHEG_HOLD_HZ = 5.0
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
        prev_mode = self.current_mode
        self.current_mode = msg.state

        # WHEG/SAFETY 관련 타이머는 모드가 바뀌는 순간 전부 정지 (필요하면 아래서 다시 시작)
        if self.wheg_hold_timer is not None:
            self.wheg_hold_timer.cancel()
            self.wheg_hold_timer = None
        if self._wheg_deploy_timer is not None:
            self._wheg_deploy_timer.cancel()
            self._wheg_deploy_timer = None
        if self._wheg_retract_timer is not None:
            self._wheg_retract_timer.cancel()
            self._wheg_retract_timer = None
        if self._safety_fold_timer is not None:
            self._safety_fold_timer.cancel()
            self._safety_fold_timer = None
        self._pending_state_after_retract = None

        # BLOCKED 모드에서 빠져나가면 회피 시퀀스 중단
        if msg.state != RobotMode.BLOCKED:
            self._cancel_blocked_timers()

        labels = {
            0: 'PLANAR', 1: 'HIGH_TORQUE', 2: 'WHEG',
            3: 'BLOCKED', 4: 'SAFETY_STOP', 5: 'STEP_STOP'
        }
        self.get_logger().info(
            f'모드 수신: {labels.get(msg.state, "UNKNOWN")}'
        )

        # SAFETY_STOP: 접기 시퀀스와 무관하게 항상 최우선으로 즉시 전체 정지부터
        if msg.state == RobotMode.SAFETY_STOP:
            self.set_dc(self.STOP_SPEED)
            self.get_logger().error('전복 위험 — 긴급 정지!')

            if prev_mode == RobotMode.WHEG:
                # 다리가 펼쳐진 상태였으면 접기부터 (서보+모터 동시 역방향 회전 필요)
                self.set_servo(self.WHEEL_MODE_ANGLE)
                self.set_dc(self.WHEG_RETRACT_SPEED, -self.WHEG_RETRACT_SPEED)
                self.get_logger().warn(
                    f'전복 위험 — 다리 접기 보조 회전 시작 '
                    f'({self.WHEG_RETRACT_DURATION_SEC}초) 후 후진 예정'
                )
                self._safety_fold_timer = self.create_timer(
                    self.WHEG_RETRACT_DURATION_SEC, self._safety_reverse_after_fold
                )
            else:
                # 이미 바퀴 모드였으면 접기 기다릴 필요 없이 바로 후진
                self.set_servo(self.WHEEL_MODE_ANGLE)
                self.set_dc(self.REVERSE_SPEED)
                self.get_logger().warn('전복 위험 — 후진 중')
            return

        # WHEG에서 다른 모드로 빠져나가는 경우: 접기 보조 회전부터 먼저 실행
        if prev_mode == RobotMode.WHEG:
            self._start_wheg_retract(msg.state)
            return

        self._apply_state_action(msg.state)

    def _apply_state_action(self, state):
        if state == RobotMode.PLANAR:
            self.set_dc(self.PLANAR_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)
        elif state == RobotMode.STEP_STOP:
            self.set_dc(self.STOP_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)
            self.get_logger().info('단차 인식 — 정지, 높이 판별 중')
        elif state == RobotMode.HIGH_TORQUE:
            self.set_dc(self.HIGH_TORQUE_SPEED)
            self.set_servo(self.WHEEL_MODE_ANGLE)
            self.get_logger().info('고토크 모드 — 고속으로 단차 밀고 넘기')
        elif state == RobotMode.WHEG:
            self.set_servo(self.LEG_MODE_ANGLE)
            self.set_dc(-self.WHEG_DEPLOY_SPEED, self.WHEG_DEPLOY_SPEED)
            self.get_logger().info(
                f'Wheg 모드 — 다리 전개 보조 회전 시작 ({self.WHEG_DEPLOY_DURATION_SEC}초)'
            )
            self._wheg_deploy_timer = self.create_timer(
                self.WHEG_DEPLOY_DURATION_SEC, self._wheg_deploy_finish
            )
        elif state == RobotMode.BLOCKED:
            self.set_servo(self.WHEEL_MODE_ANGLE)
            self.get_logger().warn('단차 극복 불가 — 후진 후 방향전환 회피 시작')
            self._start_blocked_avoid_sequence()

    def _wheg_deploy_finish(self):
        if self._wheg_deploy_timer is not None:
            self._wheg_deploy_timer.cancel()
            self._wheg_deploy_timer = None
        if self.current_mode != RobotMode.WHEG:
            return

        self.set_dc(self.WHEG_SPEED)
        self.get_logger().info('Wheg 모드 — 다리 전개 완료, 등반 주행 시작')
        if self.wheg_hold_timer is None:
            self.wheg_hold_timer = self.create_timer(
                1.0 / self.WHEG_HOLD_HZ, self._wheg_hold_cb
            )

    def _wheg_hold_cb(self):
        self.set_servo(self.LEG_MODE_ANGLE)

    def _start_wheg_retract(self, pending_state):
        self._pending_state_after_retract = pending_state
        self.set_servo(self.WHEEL_MODE_ANGLE)
        self.set_dc(self.WHEG_RETRACT_SPEED, -self.WHEG_RETRACT_SPEED)
        self.get_logger().info(
            f'Wheg 다리 접기 보조 회전 시작 ({self.WHEG_RETRACT_DURATION_SEC}초) '
            f'→ 완료 후 {pending_state} 모드로 전환 예정'
        )
        self._wheg_retract_timer = self.create_timer(
            self.WHEG_RETRACT_DURATION_SEC, self._wheg_retract_finish
        )

    def _wheg_retract_finish(self):
        if self._wheg_retract_timer is not None:
            self._wheg_retract_timer.cancel()
            self._wheg_retract_timer = None

        pending_state = self._pending_state_after_retract
        self._pending_state_after_retract = None

        if pending_state is None or self.current_mode != pending_state:
            return

        self.get_logger().info('Wheg 다리 접기 완료 — 목표 모드로 전환')
        self._apply_state_action(pending_state)

    def _safety_reverse_after_fold(self):
        """SAFETY_STOP 진입 시 다리 접기 보조 회전이 끝난 뒤 후진 시작."""
        if self._safety_fold_timer is not None:
            self._safety_fold_timer.cancel()
            self._safety_fold_timer = None
        if self.current_mode != RobotMode.SAFETY_STOP:
            return  # 그 사이 위험 해제되어 다른 모드로 바뀌었으면 후진 취소

        self.set_dc(self.REVERSE_SPEED)
        self.get_logger().warn('전복 위험 — 다리 접기 완료, 후진 중')

    # ── BLOCKED 회피 시퀀스: 후진 → 방향전환 → 정지 ──────────────
    def _start_blocked_avoid_sequence(self):
        self._cancel_blocked_timers()
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
