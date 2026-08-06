import rclpy
from rclpy.node import Node
from robot_msgs.msg import TerrainInfo

class DummyTerrainPub(Node):
    def __init__(self):
        super().__init__('dummy_terrain_pub')
        self.pub = self.create_publisher(TerrainInfo, '/terrain/info', 10)
        self.timer = self.create_timer(2.0, self.publish_scenario)
        self.step = 0
        self.get_logger().info('더미 terrain 발행 시작')

    def publish_scenario(self):
        msg = TerrainInfo()
        scenarios = [
            (False, 0.00, 1.00, '평지 주행 중'),
            (True,  0.02, 0.08, '2cm 단차, 8cm 거리 → HIGH_TORQUE'),
            (False, 0.00, 1.00, '단차 통과 → PLANAR'),
            (True,  0.06, 0.08, '6cm 단차, 8cm 거리 → WHEG'),
            (False, 0.00, 1.00, '단차 통과 → PLANAR'),
            (True,  0.15, 0.08, '15cm 단차, 8cm 거리 → BLOCKED'),
            (False, 0.00, 1.00, '단차 회피 → PLANAR'),
        ]
        s = scenarios[self.step % len(scenarios)]
        msg.step_detected    = s[0]
        msg.step_height      = s[1]
        msg.distance_to_step = s[2]
        self.get_logger().info(f'발행: {s[3]}')
        self.pub.publish(msg)
        self.step += 1

def main():
    rclpy.init()
    node = DummyTerrainPub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
