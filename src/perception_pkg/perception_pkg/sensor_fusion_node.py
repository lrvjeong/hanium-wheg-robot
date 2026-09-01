import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Float32  # ToF 더미용
from robot_msgs.msg import TerrainInfo

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # ── 거리 경계 파라미터 ─────────────────────────────────
        self.lidar_reliable_min = 0.30   # 30cm 이상에서만 라이다 신뢰
        self.lidar_height       = 0.05   # 라이다 장착 높이 (m)
        self.detect_range       = 0.35    # 전방 감지 거리 (m)
        self.step_threshold     = 0.005   # 최소 단차 높이
        self.side_limit         = 0.04    # 좌우 범위 (±4cm)

        # ToF 경계
        self.tof_near_limit     = 0.10   # 10cm 이하는 ToF만 사용

        # 최신 ToF 값 저장
        self.latest_tof_dist    = 9999.0
        self.latest_pitch       = 0.0

        # ── 높이값 노이즈 완화용 지수이동평균(EMA) ────────────────
        self.smoothed_height = None
        self.ema_alpha = 0.3   # 클수록 최신 프레임 비중 큼(빠른 반응, 노이즈 많음)
                                 # 작을수록 더 부드러움(느린 반응, 노이즈 적음)

        # ── 구독 ───────────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.create_subscription(
            PointCloud2, '/scan_3D', self.scan3d_cb, qos
        )
        self.create_subscription(
            Float32, '/tof/distance', self.tof_cb, 10
        )

        # ── 발행 ───────────────────────────────────────────────
        self.terrain_pub = self.create_publisher(
            TerrainInfo, '/terrain/info', 10
        )

        self.get_logger().info('센서 퓨전 노드 시작')
        self.get_logger().info(
            f'라이다 신뢰 구간: {self.lidar_reliable_min}m 이상'
        )

    def tof_cb(self, msg: Float32):
        self.latest_tof_dist = msg.data

    def smooth_height(self, raw_height: float) -> float:
        """여러 프레임에 걸쳐 높이값을 부드럽게 만드는 EMA 필터"""
        if self.smoothed_height is None:
            self.smoothed_height = raw_height
        else:
            self.smoothed_height = (
                self.ema_alpha * raw_height
                + (1 - self.ema_alpha) * self.smoothed_height
            )
        return self.smoothed_height

    def scan3d_cb(self, msg: PointCloud2):
        terrain = TerrainInfo()

        points = list(pc2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True
        ))

        if not points:
            terrain.step_detected = False
            self.smoothed_height = None   # 단차 없음 → 필터 초기화
            self.terrain_pub.publish(terrain)
            return

        front_points = [
            p for p in points
            if 0.15 < p[0] < self.detect_range
            and abs(p[1]) < self.side_limit
        ]
        self.get_logger().info(
            f'[DEBUG] front_points 개수: {len(front_points)}'
            + (f', x범위: {min(p[0] for p in front_points):.2f}~{max(p[0] for p in front_points):.2f}, '
               f'y범위: {min(p[1] for p in front_points):.2f}~{max(p[1] for p in front_points):.2f}'
               if front_points else '')
        )

        if not front_points:
            terrain.step_detected = False
            self.smoothed_height = None
            self.terrain_pub.publish(terrain)
            return

        min_dist = min(p[0] for p in front_points)

        # [구간 1] 10cm 이하 → ToF만 사용
        if self.latest_tof_dist < self.tof_near_limit:
            terrain.tof_active    = True
            terrain.tof_distance  = self.latest_tof_dist
            terrain.step_detected = True
            terrain.step_height   = 0.0
            terrain.distance_to_step = self.latest_tof_dist
            self.get_logger().info(
                f'[ToF 모드] 거리: {self.latest_tof_dist*100:.1f}cm'
            )

        # [구간 2] 10~30cm → ToF 거리 + 라이다 높이 병행
        elif min_dist < self.lidar_reliable_min:
            z_corrected = [p[2] + self.lidar_height for p in front_points]
            step_z = [z for z in z_corrected if z >= self.step_threshold]

            terrain.tof_active    = True
            terrain.tof_distance  = self.latest_tof_dist

            if step_z:
                step_z_sorted = sorted(step_z, reverse=True)
                top_n = max(1, len(step_z_sorted) // 10)
                raw_height = sum(step_z_sorted[:top_n]) / top_n
                raw_height += 0.07

                step_height = self.smooth_height(raw_height)

                terrain.step_detected    = True
                terrain.step_height      = float(step_height)
                terrain.distance_to_step = float(self.latest_tof_dist)
                terrain.slope_deg        = float(self.latest_pitch)
                self.get_logger().info(
                    f'[병행 모드] 거리(ToF): {self.latest_tof_dist*100:.1f}cm | '
                    f'높이(라이다, 필터전): {raw_height*100:.1f}cm | '
                    f'높이(필터후): {step_height*100:.1f}cm'
                )
            else:
                terrain.step_detected = False
                self.smoothed_height = None

        # [구간 3] 30cm 이상 → 라이다만 사용
        else:
            z_corrected = [p[2] + self.lidar_height for p in front_points]
            step_z = [z for z in z_corrected if z >= self.step_threshold]

            terrain.tof_active = False

            if step_z:
                step_z_sorted = sorted(step_z, reverse=True)
                top_n = max(1, len(step_z_sorted) // 10)
                raw_height = sum(step_z_sorted[:top_n]) / top_n
                raw_height += 0.07

                step_height = self.smooth_height(raw_height)

                terrain.step_detected    = True
                terrain.step_height      = float(step_height)
                terrain.distance_to_step = float(min_dist)
                terrain.slope_deg        = float(self.latest_pitch)
                self.get_logger().info(
                    f'[라이다 모드] 거리: {min_dist:.2f}m | '
                    f'높이(필터전): {raw_height*100:.1f}cm | '
                    f'높이(필터후): {step_height*100:.1f}cm'
                )
            else:
                terrain.step_detected = False
                self.smoothed_height = None

        self.terrain_pub.publish(terrain)

def main():
    rclpy.init()
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('센서 퓨전 노드 종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
