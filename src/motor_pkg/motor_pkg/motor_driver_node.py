#!/usr/bin/env python3

import math
import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu

from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, COMM_SUCCESS

DXL_DEVICENAME = "/dev/ttyUSB2"
DXL_BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

DXL_ID_1 = 1   # 왼쪽
DXL_ID_2 = 2   # 오른쪽
DXL_IDS = [DXL_ID_1, DXL_ID_2]
DIR_1, DIR_2 = +1, -1        # 좌우 모터가 반대로 장착돼 있어서 방향 미러링
TRIM_1, TRIM_2 = 1.0, 1.0    # 좌우 속도 미세보정 (실물 테스트 후 조정)

ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_VELOCITY = 104
LEN_GOAL_VELOCITY = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
OPERATING_MODE_VELOCITY = 1

# motor_interface_node가 보내는 -1.0~1.0 속도값을 raw velocity로 환산하는 배율
MAX_VELOCITY_UNIT = 200

ARDUINO_DEVICENAME = "/dev/ttyUSB0"
ARDUINO_BAUDRATE = 115200


class MotorDriverNode(Node):
    def __init__(self):
        super().__init__('motor_driver_node')

        self.dxl_port, self.dxl_packet = self.init_dynamixel()
        self.arduino = self.init_arduino()

        self.create_subscription(Float32MultiArray, '/motor/dc_cmd', self.dc_cmd_cb, 10)
        self.create_subscription(Float32MultiArray, '/motor/servo_cmd', self.servo_cmd_cb, 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)

        self._serial_buffer = b''
        self.imu_poll_timer = self.create_timer(0.02, self._poll_arduino_serial)  # 50Hz

        self.get_logger().info('모터 하드웨어 드라이버 노드 시작 (다이나믹셀 + 아두이노 서보/IMU)')

    def init_dynamixel(self):
        port_handler = PortHandler(DXL_DEVICENAME)
        packet_handler = PacketHandler(PROTOCOL_VERSION)

        try:
            if not port_handler.openPort():
                raise IOError("다이나믹셀 포트를 열 수 없습니다.")
            if not port_handler.setBaudRate(DXL_BAUDRATE):
                raise IOError("다이나믹셀 Baudrate 설정에 실패했습니다.")

            for dxl_id in DXL_IDS:
                packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE_VELOCITY)
                packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        except Exception as e:
            self.get_logger().error(f'다이나믹셀 초기화 실패: {e}')
            return None, None

        return port_handler, packet_handler

    def init_arduino(self):
        try:
            return serial.Serial(ARDUINO_DEVICENAME, ARDUINO_BAUDRATE, timeout=0.1)
        except serial.SerialException as e:
            self.get_logger().error(f'아두이노 시리얼 연결 실패: {e}')
            return None

    def dc_cmd_cb(self, msg: Float32MultiArray):
        if self.dxl_port is None or len(msg.data) < 2:
            return
        left_speed, right_speed = msg.data[0], msg.data[1]
        left_vel = int(left_speed * MAX_VELOCITY_UNIT * TRIM_1)
        right_vel = int(right_speed * MAX_VELOCITY_UNIT * TRIM_2)
        self.sync_write_velocity({
            DXL_ID_1: DIR_1 * left_vel,
            DXL_ID_2: DIR_2 * right_vel,
        })

    def servo_cmd_cb(self, msg: Float32MultiArray):
        if self.arduino is None or len(msg.data) == 0:
            return
        angle = int(max(0.0, min(180.0, msg.data[0])))
        try:
            self.arduino.write(f"S:{angle}\n".encode())
        except serial.SerialException as e:
            self.get_logger().warn(f'아두이노 서보 명령 전송 실패: {e}')

    # ── 아두이노 -> Pi: pitch 스트림 파싱해서 /imu/data 로 발행 ──────────
    def _poll_arduino_serial(self):
        if self.arduino is None:
            return
        try:
            waiting = self.arduino.in_waiting
        except Exception as e:
            self.get_logger().warn(f'아두이노 시리얼 상태 확인 실패: {e}')
            return

        if waiting:
            self._serial_buffer += self.arduino.read(waiting)

        while b'\n' in self._serial_buffer:
            line, self._serial_buffer = self._serial_buffer.split(b'\n', 1)
            self._handle_arduino_line(line.decode(errors='ignore').strip())

    def _handle_arduino_line(self, line: str):
        if not line.startswith('PITCH:'):
            return
        try:
            pitch_deg = float(line.split(':', 1)[1])
        except ValueError:
            return
        self._publish_imu(pitch_deg)

    def _publish_imu(self, pitch_deg: float):
        pitch = math.radians(pitch_deg)
        qx, qy, qz, qw = self._euler_to_quaternion(0.0, pitch, 0.0)

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        self.imu_pub.publish(msg)

    @staticmethod
    def _euler_to_quaternion(roll, pitch, yaw):
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw

    def sync_write_velocity(self, id_to_velocity: dict):
        if self.dxl_port is None:
            return
        group_sync_write = GroupSyncWrite(
            self.dxl_port, self.dxl_packet, ADDR_GOAL_VELOCITY, LEN_GOAL_VELOCITY
        )
        for dxl_id, velocity in id_to_velocity.items():
            v = velocity & 0xFFFFFFFF
            param = [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF]
            if not group_sync_write.addParam(dxl_id, param):
                self.get_logger().warn(f'ID {dxl_id} 파라미터 추가 실패')

        dxl_comm_result = group_sync_write.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            self.get_logger().warn(
                f'다이나믹셀 통신 오류: {self.dxl_packet.getTxRxResult(dxl_comm_result)}'
            )
        group_sync_write.clearParam()

    def destroy_node(self):
        if self.dxl_port is not None:
            self.sync_write_velocity({DXL_ID_1: 0, DXL_ID_2: 0})
            for dxl_id in DXL_IDS:
                self.dxl_packet.write1ByteTxRx(self.dxl_port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            self.dxl_port.closePort()
        if self.arduino is not None:
            self.arduino.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
