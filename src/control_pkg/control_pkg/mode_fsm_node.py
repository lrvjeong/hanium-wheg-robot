import math
import rclpy
from rclpy.node import Node
from robot_msgs.msg import TerrainInfo, RobotMode
from sensor_msgs.msg import Imu


class ModeFsmNode(Node):
    def __init__(self):
        super().__init__('mode_fsm_node')
        self.state = RobotMode.PLANAR

        self.stop_dist      = 0.30   # 30cm 이내 단차 인식 → 무조건 정지
        self.stop_hold_sec  = 1.5    # 정지 유지 시간(초)

        # 단차 판별 기준
        self.no_step_h      = 0.01   # 1cm 미만 → 단차로 안 침 (바닥 인식 오차)
        self.high_torque_h  = 0.03   # 1~3cm → 고토크
        self.wheg_h          = 0.06   # 3~6cm → 휘그
                                       # 6cm 이상 → 블락

        self.stop_entered_time = None

        self.create_subscription(TerrainInfo, '/terrain/info', self.terrain_cb, 10)
        self.create_subscription(Imu, '/imu/data', self.imu_cb, 10)
        self.mode_pub = self.create_publisher(RobotMode, '/robot/mode', 10)

        self.get_logger().info('FSM 노드 시작 — 초기 상태: PLANAR')

    def terrain_cb(self, msg: TerrainInfo):
        prev = self.state

        if self.state == RobotMode.PLANAR:
            if msg.step_detected and msg.distance_to_step <= self.stop_dist:
                self.state = RobotMode.STEP_STOP
                self.stop_entered_time = self.get_clock().now()
                self.get_logger().info(
                    f'단차 인식 (거리 {msg.distance_to_step*100:.1f}cm) → 정지, 높이 판별 대기'
                )

        elif self.state == RobotMode.STEP_STOP:
            if not msg.step_detected:
                self.state = RobotMode.PLANAR
            else:
                elapsed = (self.get_clock().now() - self.stop_entered_time).nanoseconds / 1e9
                if elapsed >= self.stop_hold_sec:
                    if msg.step_height < self.no_step_h:
                        # 1cm 미만 → 단차 아님, 그냥 평지로 취급
                        self.state = RobotMode.PLANAR
                    elif msg.step_height < self.high_torque_h:
                        self.state = RobotMode.HIGH_TORQUE
                    elif msg.step_height < self.wheg_h:
                        self.state = RobotMode.WHEG
                    else:
                        self.state = RobotMode.BLOCKED

        elif self.state in (
            RobotMode.HIGH_TORQUE,
            RobotMode.WHEG,
            RobotMode.BLOCKED
        ):
            if not msg.step_detected:
                self.state = RobotMode.PLANAR

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
