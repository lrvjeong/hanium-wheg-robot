import math
import rclpy
from rclpy.node import Node
from robot_msgs.msg import TerrainInfo, RobotMode
from sensor_msgs.msg import Imu


class ModeFsmNode(Node):
    def __init__(self):
        super().__init__('mode_fsm_node')
        self.state = RobotMode.PLANAR
        self.stop_dist      = 0.10   # 10cm 이내 단차 인식 → 무조건 정지
        self.declare_parameter('stop_hold_sec', 8.0)
        self.stop_hold_sec = self.get_parameter('stop_hold_sec').get_parameter_value().double_value

        # 단차 판별 기준
        self.no_step_h      = 0.01   # 1cm 미만 → 단차로 안 침 (바닥 인식 오차)
        self.high_torque_h  = 0.03   # 1~3cm → 고토크
        self.wheg_h          = 0.08   # 3~8cm → 휘그
                                       # 8cm 이상 → 블락
        self.stop_entered_time = None

        # STEP_STOP 중 step_height 샘플 버퍼 (노이즈 방지용 평균 판정)
        self.stop_height_samples = []

        # HIGH_TORQUE/WHEG/BLOCKED → PLANAR 복귀 조건용 클리어 타이머
        self.clear_hold_sec = 0.5
        self.clear_start_time = None

        # 단차 안 보임 + IMU 안정화(수평 복귀) 둘 다 만족해야 PLANAR로 복귀
        self.stable_pitch_deg = 5.0     # 이 각도 이내면 "수평/안정화"로 간주
        self.latest_pitch_deg = 0.0

        # (수정) 전복 위험 판단 임계각: WHEG 모드는 원래 몸체가 더 기울어진
        # 상태(다리 전개 + 등반 중)라서 평지 기준(25도)을 그대로 쓰면 정상적인
        # 등반 중에도 오탐으로 SAFETY_STOP에 빠질 수 있음. WHEG 중에는 임계각을
        # 40도로 높여서 잡고, 그 외 모드에서는 기존 25도 그대로 유지.
        self.safety_enter_deg_normal = 25.0
        self.safety_enter_deg_wheg   = 40.0

        self.safety_clear_hold_sec = 1.0
        self.safety_clear_start_time = None

        self.create_subscription(TerrainInfo, '/terrain/info', self.terrain_cb, 10)
        self.create_subscription(Imu, '/imu/data', self.imu_cb, 10)
        self.mode_pub = self.create_publisher(RobotMode, '/robot/mode', 10)

        self.get_logger().info(
            f'FSM 노드 시작 — 초기 상태: PLANAR (stop_hold_sec={self.stop_hold_sec}초)'
        )

    def terrain_cb(self, msg: TerrainInfo):
        prev = self.state
        if self.state == RobotMode.PLANAR:
            if msg.step_detected and msg.distance_to_step <= self.stop_dist:
                self.state = RobotMode.STEP_STOP
                self.stop_entered_time = self.get_clock().now()
                self.stop_height_samples = []
                self.get_logger().info(
                    f'단차 인식 (거리 {msg.distance_to_step*100:.1f}cm) → 정지, 높이 판별 대기'
                )
        elif self.state == RobotMode.STEP_STOP:
            if msg.step_detected:
                self.stop_height_samples.append(msg.step_height)
            elapsed = (self.get_clock().now() - self.stop_entered_time).nanoseconds / 1e9
            if elapsed >= self.stop_hold_sec:
                if self.stop_height_samples:
                    avg_height = sum(self.stop_height_samples) / len(self.stop_height_samples)
                    self.get_logger().info(
                        f'단차 높이 판정: 평균 {avg_height*100:.1f}cm '
                        f'(샘플 {len(self.stop_height_samples)}개 기준)'
                    )
                    self.clear_start_time = None
                    if avg_height < self.no_step_h:
                        self.state = RobotMode.PLANAR
                    elif avg_height < self.high_torque_h:
                        self.state = RobotMode.HIGH_TORQUE
                    elif avg_height < self.wheg_h:
                        self.state = RobotMode.WHEG
                    else:
                        self.state = RobotMode.BLOCKED
                else:
                    self.state = RobotMode.PLANAR
                self.stop_height_samples = []
        elif self.state in (
            RobotMode.HIGH_TORQUE,
            RobotMode.WHEG,
            RobotMode.BLOCKED
        ):
            step_cleared = not msg.step_detected
            imu_stable = abs(self.latest_pitch_deg) < self.stable_pitch_deg
            if step_cleared and imu_stable:
                if self.clear_start_time is None:
                    self.clear_start_time = self.get_clock().now()
                clear_elapsed = (
                    self.get_clock().now() - self.clear_start_time
                ).nanoseconds / 1e9
                if clear_elapsed >= self.clear_hold_sec:
                    self.state = RobotMode.PLANAR
                    self.clear_start_time = None
            else:
                self.clear_start_time = None

        if prev != self.state:
            labels = {
                0: 'PLANAR', 1: 'HIGH_TORQUE', 2: 'WHEG',
                3: 'BLOCKED', 4: 'SAFETY_STOP', 5: 'STEP_STOP'
            }
            self.get_logger().info(
                f'상태 전환: {labels[prev]} → {labels[self.state]}'
            )
        self.publish_mode()

    def imu_cb(self, msg: Imu):
        pitch_deg = self.get_pitch_deg(msg.orientation)
        self.latest_pitch_deg = pitch_deg

        # (수정) WHEG 모드 중엔 임계각을 40도로, 그 외에는 25도로 판단
        enter_threshold = (
            self.safety_enter_deg_wheg
            if self.state == RobotMode.WHEG
            else self.safety_enter_deg_normal
        )

        if abs(pitch_deg) > enter_threshold:
            self.safety_clear_start_time = None  # 위험 지속 중이면 복귀타이머 리셋
            if self.state != RobotMode.SAFETY_STOP:
                self.get_logger().warn(
                    f'전복 위험 감지 (pitch={pitch_deg:.1f}°, 기준={enter_threshold}°) → SAFETY_STOP'
                )
                self.state = RobotMode.SAFETY_STOP
                self.publish_mode()
        elif self.state == RobotMode.SAFETY_STOP and abs(pitch_deg) < self.stable_pitch_deg:
            if self.safety_clear_start_time is None:
                self.safety_clear_start_time = self.get_clock().now()
            elapsed = (self.get_clock().now() - self.safety_clear_start_time).nanoseconds / 1e9
            if elapsed >= self.safety_clear_hold_sec:
                self.get_logger().info(
                    f'위험 해제 (pitch={pitch_deg:.1f}°) → PLANAR 복귀'
                )
                self.state = RobotMode.PLANAR
                self.safety_clear_start_time = None
                self.publish_mode()

    def get_pitch_deg(self, q):
        sinp = 2 * (q.w * q.y - q.z * q.x)
        sinp = max(-1.0, min(1.0, sinp))
        return math.degrees(math.asin(sinp))

    def publish_mode(self):
        msg = RobotMode()
        msg.state = self.state
        self.mode_pub.publish(msg)


def main():
    rclpy.init()
    node = ModeFsmNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
