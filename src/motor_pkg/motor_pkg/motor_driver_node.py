import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from dynamixel_sdk import PortHandler, PacketHandler

try:
    import pigpio
except ImportError:
    pigpio = None


DEVICENAME       = '/dev/ttyUSB0'
BAUDRATE         = 57600
PROTOCOL_VERSION = 2.0

LEFT_MOTOR_ID  = 1
RIGHT_MOTOR_ID = 2

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE  = 64
ADDR_GOAL_VELOCITY  = 104

OPERATING_MODE_VELOCITY = 1

MAX_VELOCITY_UNIT = 200

SERVO_GPIO_PINS = [5, 6, 13, 19]
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500


class MotorDriverNode(Node):
    def __init__(self):
        super().__init__('motor_driver_node')

        self.port_handler = PortHandler(DEVICENAME)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.dxl_ready = False

        try:
            port_opened = self.port_handler.openPort() and self.port_handler.setBaudRate(BAUDRATE)
        except Exception as e:
            self.get_logger().warn(f'다이나믹셀 포트 연결 안 됨 (하드웨어 미연결): {e}')
            port_opened = False

        if port_opened:
            self.get_logger().info(f'다이나믹셀 포트 연결됨: {DEVICENAME}')
            self.dxl_ready = True
            for motor_id in (LEFT_MOTOR_ID, RIGHT_MOTOR_ID):
                self._setup_motor(motor_id)
        else:
            self.get_logger().warn('다이나믹셀 없이 서보/노드 기능만 테스트 모드로 진행')

        self.pi = None
        if pigpio is not None:
            candidate = pigpio.pi()
            if candidate.connected:
                self.pi = candidate
                self.get_logger().info('pigpio 연결됨 (서보 제어 가능)')
            else:
                self.get_logger().error(
                    'pigpiod 데몬 연결 실패 — sudo systemctl start pigpiod 확인'
                )
        else:
            self.get_logger().warn('pigpio 모듈 없음 — 서보 제어 비활성화')

        self.create_subscription(
            Float32MultiArray, '/motor/dc_cmd', self.dc_cmd_cb, 10
        )
        self.create_subscription(
            Float32MultiArray, '/motor/servo_cmd', self.servo_cmd_cb, 10
        )

        self.get_logger().info('모터 드라이버 노드 시작')

    def _setup_motor(self, motor_id: int):
        self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 0
        )
        self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, ADDR_OPERATING_MODE, OPERATING_MODE_VELOCITY
        )
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 1
        )
        if dxl_comm_result != 0:
            self.get_logger().error(
                f'모터 {motor_id} 토크 켜기 실패: '
                f'{self.packet_handler.getTxRxResult(dxl_comm_result)}'
            )
        else:
            self.get_logger().info(f'모터 {motor_id} 초기화 완료 (속도 제어 모드)')

    def dc_cmd_cb(self, msg: Float32MultiArray):
        if not self.dxl_ready:
            return
        if len(msg.data) < 2:
            self.get_logger().warn('dc_cmd 데이터 부족 (좌/우 2개 필요)')
            return

        left_speed, right_speed = msg.data[0], msg.data[1]
        left_vel = int(left_speed * MAX_VELOCITY_UNIT)
        right_vel = int(right_speed * MAX_VELOCITY_UNIT)

        self.packet_handler.write4ByteTxRx(
            self.port_handler, LEFT_MOTOR_ID, ADDR_GOAL_VELOCITY, self._to_uint32(left_vel)
        )
        self.packet_handler.write4ByteTxRx(
            self.port_handler, RIGHT_MOTOR_ID, ADDR_GOAL_VELOCITY, self._to_uint32(right_vel)
        )

        self.get_logger().info(f'다이나믹셀 속도 설정: 좌={left_vel}, 우={right_vel}')

    def servo_cmd_cb(self, msg: Float32MultiArray):
        if self.pi is None:
            return
        for i, angle in enumerate(msg.data):
            if i >= len(SERVO_GPIO_PINS):
                break
            pulse = self._angle_to_pulse(angle)
            self.pi.set_servo_pulsewidth(SERVO_GPIO_PINS[i], pulse)
        self.get_logger().info(f'서보 각도 적용: {list(msg.data)}')

    @staticmethod
    def _angle_to_pulse(angle_deg: float) -> int:
        angle_deg = max(0.0, min(180.0, angle_deg))
        return int(
            SERVO_MIN_PULSE + (angle_deg / 180.0) * (SERVO_MAX_PULSE - SERVO_MIN_PULSE)
        )

    @staticmethod
    def _to_uint32(value: int) -> int:
        return value & 0xFFFFFFFF

    def destroy_node(self):
        if self.dxl_ready:
            for motor_id in (LEFT_MOTOR_ID, RIGHT_MOTOR_ID):
                self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 0
                )
            self.port_handler.closePort()
        if self.pi is not None:
            for pin in SERVO_GPIO_PINS:
                self.pi.set_servo_pulsewidth(pin, 0)
            self.pi.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = MotorDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('모터 드라이버 노드 종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
