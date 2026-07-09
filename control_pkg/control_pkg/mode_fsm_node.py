import math
import rclpy
from rclpy.node import Node
from robot_msgs.msg import TerrainInfo, RobotMode
from sensor_msgs.msg import Imu

class ModeFsmNode(Node):
    def __init__(self):
        super().__init__('mode_fsm_node')
        
        # 1. 초기 상태 설정 (평지 주행 모드)
        self.state = RobotMode.PLANAR
        
        # 상태 ID와 문자열 매핑 (로그 가독성용)
        self.labels = {
            RobotMode.PLANAR: 'PLANAR',
            RobotMode.HIGH_TORQUE_PLANAR: 'HIGH_TORQUE_PLANAR',
            RobotMode.APPROACH: 'APPROACH',
            RobotMode.CLIMBING: 'CLIMBING',
            RobotMode.SAFETY_STOP: 'SAFETY_STOP'
        }
        
        # 2. 구독자(Subscription) 설정
        self.create_subscription(TerrainInfo, '/terrain/info', self.terrain_cb, 10)
        self.create_subscription(Imu, '/imu/data', self.imu_cb, 10)
        
        # 3. 발행자(Publisher) 설정
        self.mode_pub = self.create_publisher(RobotMode, '/robot/mode', 10)
        
        self.get_logger().info('FSM 노드 가동 시작 — 초기 상태: PLANAR')

    def terrain_cb(self, msg: TerrainInfo):
        """ 지형 인지 토픽 수신 시 높이와 거리를 모두 고려한 정밀 상태 천이 제어 """
        prev_state = self.state
        
        # [조건 1] 현재 일반 주행(PLANAR) 또는 고토크 주행(HIGH_TORQUE_PLANAR) 상태일 때
        if self.state in (RobotMode.PLANAR, RobotMode.HIGH_TORQUE_PLANAR):
            if msg.step_detected:
                # 1. 10cm 이상의 극복 불가능한 높은 장애물 -> 즉시 정지 및 우회 판정
                if msg.step_height >= 0.10:
                    self.state = RobotMode.SAFETY_STOP
                    self.get_logger().error(f'❌ 10cm 초과 높은 단차 감지 ({msg.step_height*100:.1f}cm) -> SAFETY_STOP (우회 유도)')
                
                # 2. 3cm 미만의 아주 낮은 단차 -> 형태 변형 없이 힘으로 밀고 가도록 고토크 발동
                elif msg.step_height < 0.03:
                    self.state = RobotMode.HIGH_TORQUE_PLANAR
                
                # 3. 3cm 이상 10cm 미만의 등반 가능한 계단 발견 시
                elif 0.03 <= msg.step_height < 0.10:
                    # 장애물과의 거리가 10cm(0.10m) 미만으로 좁혀졌을 때만 변형 준비(APPROACH) 상태로 전환
                    # (TerrainInfo 메시지에 장애물과의 거리 필드가 있다는 가정 하에 msg.distance 활용)
                    if hasattr(msg, 'distance') and msg.distance < 0.10:
                        self.state = RobotMode.APPROACH
                        self.get_logger().info(f'🏃 계단 근접 감지 ({msg.distance*100:.1f}cm) -> 다리 변형 준비 (APPROACH)')
                    else:
                        # 10cm보다 멀리 있다면 감지는 했으나 계단 앞까지는 일반 주행(PLANAR) 유지
                        self.state = RobotMode.PLANAR
            else:
                # 장애물이 감지되지 않으면 일반 주행 유지
                self.state = RobotMode.PLANAR
                
        # [조건 2] 변형 준비(APPROACH) 상태일 때
        elif self.state == RobotMode.APPROACH:
            if msg.step_detected:
                if msg.step_height >= 0.10:
                    self.state = RobotMode.SAFETY_STOP
                else:
                    # 변형 준비 단계에서 메커니즘 변형 신호를 하달하며 본격적인 등반(CLIMBING) 상태로 천이
                    self.state = RobotMode.CLIMBING
            else:
                self.state = RobotMode.PLANAR

        # [조건 3] 본격적인 등반(CLIMBING) 상태일 때
        elif self.state == RobotMode.CLIMBING:
            # 계단을 완전히 다 극복해서 더 이상 센서에 단차가 걸리지 않으면 평지 주행으로 복귀
            if not msg.step_detected:
                self.get_logger().info('🎉 계단 등반 완료! 일반 주행 모드로 복귀합니다.')
                self.state = RobotMode.PLANAR

        # 상태가 실제로 변경되었을 때만 퍼블리시 및 로그 출력
        if prev_state != self.state:
            self.get_logger().info(f'상태 전환: {self.labels[prev_state]} → {self.labels[self.state]}')
            self.publish_mode()

    def imu_cb(self, msg: Imu):
        """ IMU 토픽 수신 시 고주파 전복 위험 감지 및 안전 제어 """
        pitch_deg = self.get_pitch_deg(msg.orientation)

        # 30도 이상 기울어지면 지형 조건 무시하고 비상 정지
        if abs(pitch_deg) > 30.0:  
            if self.state != RobotMode.SAFETY_STOP:
                self.get_logger().warn(f'🚨 전복 위험 감지 (Pitch: {pitch_deg:.1f}°) → SAFETY_STOP 강제 전환!')
                self.state = RobotMode.SAFETY_STOP
                self.publish_mode()
        else:
            # 자세가 안정을 찾고 기존에 SAFETY_STOP 상태였다면 자가 복귀
            if self.state == RobotMode.SAFETY_STOP and abs(pitch_deg) < 10.0:
                self.get_logger().info(f'✅ 위험 상황 해제 (Pitch: {pitch_deg:.1f}°) → PLANAR 모드 자동 복귀')
                self.state = RobotMode.PLANAR
                self.publish_mode()

    def get_pitch_deg(self, q):
        """ 쿼터니언 데이터를 활용해 오일러 각도 중 Pitch(앞뒤 기울기, 단위: 도) 계산 """
        sinp = 2 * (q.w * q.y - q.z * q.x)
        sinp = max(-1.0, min(1.0, sinp))  
        pitch_rad = math.asin(sinp)
        return math.degrees(pitch_rad)

    def publish_mode(self):
        """ 현재 FSM의 최신 상태(State)를 /robot/mode 토픽으로 발행 """
        msg = RobotMode()
        msg.state = self.state
        self.mode_pub.publish(msg)

def main():
    rclpy.init()
    node = ModeFsmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('사용자에 의해 FSM 노드가 종료되었습니다.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
