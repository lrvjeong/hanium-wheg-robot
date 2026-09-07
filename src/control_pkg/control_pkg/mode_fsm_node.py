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

        # 단차 판별 기준
        self.no_step_h      = 0.01   # 1cm 미만 → 단차로 안 침 (바닥 인식 오차)
        self.high_torque_h  = 0.03   # 1~3cm → 고토크
        self.wheg_h          = 0.08   # 3~8cm → 휘그
                                       # 8cm 이상 → 블락

        self.stop_entered_time = None

        # (수정) STEP_STOP으로 정지해있는 동안 들어오는 step_height를 계속 쌓아뒀다가,
        # 1.5초가 다 됐을 때 "그 순간 도착한 메시지 하나"가 아니라 정지 구간 전체의
        # 평균으로 모드를 판정하기 위한 버퍼. 노이즈 튄 값 한 프레임 때문에
        # 엉뚱한 모드로 확정되는 걸 방지.
        self.stop_height_samples = []

        # (추가) HIGH_TORQUE/WHEG/BLOCKED 상태에서 "단차를 다 넘었다"고 판단해
        # PLANAR로 복귀하는 조건도, STEP_STOP과 비슷하게 노이즈 한 프레임 때문에
        # 잘못 풀리지 않도록 방어함. step_detected=False가 연속으로
        # clear_hold_sec 동안 유지돼야만 실제로 PLANAR로 복귀시킴 - 중간에
        # step_detected=True가 한 번이라도 오면(아직 단차 위) 타이머 리셋.
        self.clear_hold_sec = 0.5
        self.clear_start_time = None

        # (추가) "단차 안 보임" 뿐 아니라 IMU pitch도 평지 수준으로 안정화됐을
        # 때만 PLANAR로 풀리게 함. 단차를 넘는 도중엔 몸체가 기울어져 있다가
        # 다 넘고 나서야 수평으로 돌아오니까, 이 조건까지 같이 봐야 "진짜로
        # 다 넘었다"고 판단할 수 있음 (안 그러면 센서상으로는 단차가 안
        # 보여도 몸은 아직 기울어진 채 공중에 떠 있는 상태에서 PLANAR로
        # 오판할 수 있음).
        self.stable_pitch_deg = 5.0     # 이 각도 이내면 "수평/안정화"로 간주
        self.latest_pitch_deg = 0.0
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
                self.stop_height_samples = []  # (수정) STEP_STOP 진입 시 버퍼 초기화
                self.get_logger().info(
                    f'단차 인식 (거리 {msg.distance_to_step*100:.1f}cm) → 정지, 높이 판별 대기'
                )

        elif self.state == RobotMode.STEP_STOP:
            # (수정) 프레임 하나가 필터를 못 통과해서 step_detected=False로 튀어도
            # 그 즉시 PLANAR로 되돌리지 않음. 그냥 그 프레임만 샘플에서 빼고
            # 계속 정지 상태를 유지하다가, 시간이 다 되면 그때까지 모인
            # 유효한 샘플들의 평균으로만 판정함. (일시적 노이즈 프레임 때문에
            # 판정 자체가 무산되는 문제를 없앰)
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

                    # (추가) climbing 상태로 새로 진입하는 거니까 이전에 남아있을 수
                    # 있는 clear_start_time 타이머를 깨끗이 리셋해줌 (혹시 몰라 방어).
                    self.clear_start_time = None

                    if avg_height < self.no_step_h:
                        # 1cm 미만 → 단차 아님, 그냥 평지로 취급
                        self.state = RobotMode.PLANAR
                    elif avg_height < self.high_torque_h:
                        self.state = RobotMode.HIGH_TORQUE
                    elif avg_height < self.wheg_h:
                        self.state = RobotMode.WHEG
                    else:
                        self.state = RobotMode.BLOCKED
                else:
                    # 유효 샘플이 하나도 없으면(계속 미검출) 평지로 판단
                    self.state = RobotMode.PLANAR

                self.stop_height_samples = []  # (수정) 판정 끝났으니 버퍼 정리

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
                # 단차가 아직 감지되거나(계속 넘는 중), IMU가 아직 안 안정화됐으면
                # (몸체가 기울어진 채 공중에 있는 중) 타이머 리셋
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

        if abs(pitch_deg) > 25.0:
            self.safety_clear_start_time = None  # 위험 지속 중이면 복귀타이머 리셋
            if self.state != RobotMode.SAFETY_STOP:
                self.get_logger().warn(
                    f'전복 위험 감지 (pitch={pitch_deg:.1f}°) → SAFETY_STOP'
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
