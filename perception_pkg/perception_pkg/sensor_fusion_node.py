import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from robot_msgs.msg import TerrainInfo

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # 파라미터 (실제 장착 위치에 맞게 조정)
        self.lidar_height     = 0.15  # 라이다가 지면에서 몇 m 높이에 있는지
        self.detect_range     = 1.0   # 전방 몇 m까지 감지할지
        self.step_threshold   = 0.03  # 단차로 판별할 최소 높이 (3cm)
        self.ground_threshold = 0.02  # 바닥으로 판별할 최대 높이 (2cm)
        self.side_limit       = 0.3   # 좌우 ±몇 m 범위만 볼지

        # 구독: /scan_3D (PointCloud2)
        self.create_subscription(
            PointCloud2, '/scan_3D', self.scan3d_cb, 10
        )

        # 발행: /terrain/info
        self.terrain_pub = self.create_publisher(TerrainInfo, '/terrain/info', 10)

        self.get_logger().info('센서 퓨전 노드 시작 (3D 모드)')

    def scan3d_cb(self, msg: PointCloud2):
        terrain = TerrainInfo()

        # PointCloud2 → (x, y, z) 리스트로 변환
        points = list(pc2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True
        ))

        if not points:
            terrain.step_detected = False
            self.terrain_pub.publish(terrain)
            return

        # 전방 범위 필터링
        # x: 전방 거리 (5cm~1m)
        # y: 좌우 (±30cm)
        front_points = [
            p for p in points
            if 0.05 < p[0] < self.detect_range
            and abs(p[1]) < self.side_limit
        ]

        if not front_points:
            terrain.step_detected = False
            self.terrain_pub.publish(terrain)
            return

        # z값에 라이다 장착 높이 보정
        # (라이다가 지면에서 15cm 위에 있으면 z=0이 실제로 -15cm)
        z_corrected = [p[2] + self.lidar_height for p in front_points]

        # 바닥 vs 단차 분리
        # ground: z가 거의 0 (바닥)
        # step: z가 step_threshold 이상 (단차)
        step_z = [z for z in z_corrected if z >= self.step_threshold]

        if step_z:
            # 단차 높이 = 감지된 포인트 중 최대 z값
            step_height = max(step_z)
            # 단차까지 거리 = 가장 가까운 포인트의 x값
            min_dist    = min(p[0] for p in front_points)

            terrain.step_detected    = True
            terrain.step_height      = float(step_height)
            terrain.distance_to_step = float(min_dist)
            terrain.slope_deg        = 0.0  # IMU 연결 후 업데이트

            self.get_logger().info(
                f'단차 감지! '
                f'거리: {min_dist:.2f}m | '
                f'높이: {step_height * 100:.1f}cm'
            )
        else:
            terrain.step_detected = False

        self.terrain_pub.publish(terrain)


def main():
    rclpy.init()
    node = SensorFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
