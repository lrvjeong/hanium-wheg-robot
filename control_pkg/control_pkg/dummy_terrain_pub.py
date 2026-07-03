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
            (False, 0.0,  0.0, '평지 주행 중'),
            (True,  0.05, 2.0, '낮은 단차 감지 (5cm) → APPROACH'),
            (True,  0.15, 1.5, '높은 단차 감지 (15cm) → CLIMBING'),
            (False, 0.0,  0.0, '단차 통과 → PLANAR 복귀'),
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
