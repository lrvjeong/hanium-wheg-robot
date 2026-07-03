import rclpy
from rclpy.node import Node
from robot_msgs.msg import TerrainInfo, RobotMode
from sensor_msgs.msg import Imu

class ModeFsmNode(Node):
    def __init__(self):
        super().__init__('mode_fsm_node')
        self.state = RobotMode.PLANAR

        self.create_subscription(TerrainInfo, '/terrain/info', self.terrain_cb, 10)
        self.create_subscription(Imu, '/imu/data', self.imu_cb, 10)
        self.mode_pub = self.create_publisher(RobotMode, '/robot/mode', 10)

        self.get_logger().info('FSM 노드 시작 — 초기 상태: PLANAR')

    def terrain_cb(self, msg: TerrainInfo):
        prev = self.state

        if self.state == RobotMode.PLANAR:
            if msg.step_detected and msg.step_height < 0.10:
                self.state = RobotMode.APPROACH
            elif msg.step_detected and msg.step_height >= 0.10:
                self.state = RobotMode.CLIMBING

        elif self.state in (RobotMode.APPROACH, RobotMode.CLIMBING):
            if not msg.step_detected:
                self.state = RobotMode.PLANAR

        if prev != self.state:
            labels = {0:'PLANAR', 1:'APPROACH', 2:'CLIMBING', 3:'SAFETY_STOP'}
            self.get_logger().info(
                f'상태 전환: {labels[prev]} → {labels[self.state]}'
            )

        self.publish_mode()

    def imu_cb(self, msg: Imu):
        pitch = msg.linear_acceleration.x
        if abs(pitch) > 9.0:
            if self.state != RobotMode.SAFETY_STOP:
                self.get_logger().warn('전복 위험 감지 → SAFETY_STOP')
                self.state = RobotMode.SAFETY_STOP
                self.publish_mode()

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
