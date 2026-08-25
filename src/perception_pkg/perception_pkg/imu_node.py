import math
import serial
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

SERIAL_PORT = '/dev/imu_arduino'   # 실물 연결 후 ls /dev/ttyUSB* 로 확인해서 수정
BAUDRATE = 9600                 # WT901 기본값 (모듈에 따라 115200일 수도 있음, 실물 보고 조정)

FRAME_HEADER = 0x55
TYPE_ANGLE = 0x53   # WT901이 roll/pitch/yaw를 자체 계산해서 보내주는 패킷 타입


class ImuNode(Node):
    def __init__(self):
        super().__init__('imu_node')

        self.pub = self.create_publisher(Imu, '/imu/data', 10)

        self.ser = None
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
            self.get_logger().info(f'IMU 시리얼 포트 연결됨: {SERIAL_PORT}')
        except Exception as e:
            self.get_logger().warn(f'IMU 포트 연결 실패 (하드웨어 미연결): {e}')

        self.buffer = bytearray()
        self.timer = self.create_timer(0.02, self.read_serial)  # 50Hz 폴링

    def read_serial(self):
        if self.ser is None:
            return

        try:
            waiting = self.ser.in_waiting
        except Exception as e:
            self.get_logger().warn(f'IMU 시리얼 읽기 실패: {e}')
            return

        if waiting:
            self.buffer.extend(self.ser.read(waiting))

        while len(self.buffer) >= 11:
            if self.buffer[0] != FRAME_HEADER:
                self.buffer.pop(0)
                continue

            packet = self.buffer[:11]
            checksum = sum(packet[0:10]) & 0xFF
            if checksum != packet[10]:
                self.buffer.pop(0)
                continue

            if packet[1] == TYPE_ANGLE:
                self.handle_angle_packet(packet)

            del self.buffer[:11]

    def handle_angle_packet(self, packet):
        roll_raw  = self._to_int16(packet[2], packet[3])
        pitch_raw = self._to_int16(packet[4], packet[5])
        yaw_raw   = self._to_int16(packet[6], packet[7])

        roll  = math.radians(roll_raw  / 32768.0 * 180.0)
        pitch = math.radians(pitch_raw / 32768.0 * 180.0)
        yaw   = math.radians(yaw_raw   / 32768.0 * 180.0)

        qx, qy, qz, qw = self._euler_to_quaternion(roll, pitch, yaw)

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw

        self.pub.publish(msg)

    @staticmethod
    def _to_int16(low, high):
        val = (high << 8) | low
        if val >= 32768:
            val -= 65536
        return val

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

    def destroy_node(self):
        if self.ser is not None:
            self.ser.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = ImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
