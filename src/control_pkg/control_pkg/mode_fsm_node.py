import math
import rclpy
from rclpy.node import Node
from robot_msgs.msg import TerrainInfo, RobotMode
from sensor_msgs.msg import Imu

class ModeFsmNode(Node):
    def __init__(self):
        super().__init__('mode_fsm_node')
        self.state = RobotMode.PLANAR

        # 단차 판별 기준
        self.approach_dist  = 0.10  # 단차 10cm 이내 접근 시 모드 전환
        self.high_torque_h  = 0.03  # 3cm 이하 → 고토크
        self.wheg_h         = 0.10  # 3~10cm → 휠레그
                                    # 10cm 초과 → 못 넘음

        self.create_subscription(TerrainInfo, '/terrain/info', self.terrain_cb, 10)
        self.create_subscription(Imu, '/imu/data', self.imu_cb, 10)
        self.mode_pub = self.create_publisher(RobotMode, '/robot/mode', 10)

        self.get_logger().info('FSM 노드 시작 — 초기 상태: PLANAR')

    def terrain_cb(self, msg: TerrainInfo):
        prev = self.state

        if self.state == RobotMode.PLANAR:
            # 단차 감지 + 10cm 이내 접근했을 때만 모드 전환
            if msg.step_detected and msg.distance_to_step <= self.approach_dist:
                if msg.step_height < self.high_torque_h:
                    # 3cm 미만 → 고토크로 그냥 넘기
                    self.state = RobotMode.HIGH_TORQUE
                elif msg.step_height < self.wheg_h:
                    # 3cm 이상 10cm 미만 → 휠레그 전개
                    self.state = RobotMode.WHEG
                else:
                    # 10cm 이상 → 못 넘음
                    self.state = RobotMode.BLOCKED

        elif self.state in (
            RobotMode.HIGH_TORQUE,
            RobotMode.WHEG,
            RobotMode.BLOCKED
        ):
            # 단차 없어지면 평지로 복귀
            if not msg.step_detected:
                self.state = RobotMode.PLANAR

        # 상태 바뀌었을 때만 로그 출력
        if prev != self.state:
            labels = {
                0: 'PLANAR',
                1: 'HIGH_TORQUE',
                2: 'WHEG',
                3: 'BLOCKED',
                4: 'SAFETY_STOP'
            }
            self.get_logger().info(
                f'상태 전환: {labels[prev]} → {labels[self.state]}'
            )

        self.publish_mode()

    def imu_cb(self, msg: Imu):
        pitch_deg = self.get_pitch_deg(msg.orientation)
        if abs(pitch_deg) > 30.0:
            if self.state != RobotMode.SAFETY_STOP:
                self.get_logger().warn(
                    f'전복 위험 감지 (pitch={pitch_deg:.1f}°) → SAFETY_STOP'
                )
                self.state = RobotMode.SAFETY_STOP
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

