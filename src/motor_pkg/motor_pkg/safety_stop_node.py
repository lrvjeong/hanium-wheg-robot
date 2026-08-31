#!/usr/bin/env python3
"""
[3-d 단계] safety stop: IMU로 전복 위험 감지 시 후진

Arduino(leg_servo_imu_controller.ino)가 WT901 IMU를 직접 감시하다가
위험 각도를 넘으면 즉시 자체적으로 서보를 접고, Pi에는
    "TILT_WARNING:<pitch>\n"  -> 위험 시작
    "TILT_CLEAR\n"            -> 위험 해제
를 보냅니다. 이 노드는 그 신호를 받아 구동 다이나믹셀을 후진/정지시킵니다.

서보(다리) 자체는 Arduino가 이미 접었으므로, 이 노드는 구동모터만 다룹니다.
"""

import time
import threading
import serial
import rclpy
from rclpy.node import Node

from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, COMM_SUCCESS

# ------------------- Dynamixel 설정 -------------------
DXL_DEVICENAME = "/dev/ttyUSB0"
DXL_BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

DXL_ID_1 = 1
DXL_ID_2 = 2
DXL_IDS = [DXL_ID_1, DXL_ID_2]
DIR_1, DIR_2 = +1, -1

ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_VELOCITY = 104
LEN_GOAL_VELOCITY = 4
OPERATING_MODE_VELOCITY = 1

REVERSE_SPEED = 80
# --------------------------------------------------------

ARDUINO_DEVICENAME = "/dev/ttyUSB1"
ARDUINO_BAUDRATE = 115200


class SafetyStopNode(Node):
    def __init__(self):
        super().__init__('safety_stop_node')

        self.dxl_port, self.dxl_packet = self.init_dynamixel()
        self.arduino = serial.Serial(ARDUINO_DEVICENAME, ARDUINO_BAUDRATE, timeout=1)
        time.sleep(2)

        self.tilt_active = False

        # Arduino 시리얼을 블로킹으로 계속 읽어야 하므로 별도 스레드에서 처리
        self.reader_thread = threading.Thread(target=self.read_arduino_loop, daemon=True)
        self.reader_thread.start()

        self.get_logger().info('전복 위험 감시 시작 (Arduino TILT_WARNING / TILT_CLEAR 대기)')

    # ---------------- Dynamixel 제어 ----------------
    def init_dynamixel(self):
        port_handler = PortHandler(DXL_DEVICENAME)
        packet_handler = PacketHandler(PROTOCOL_VERSION)
        if not port_handler.openPort():
            raise IOError("포트를 열 수 없습니다.")
        if not port_handler.setBaudRate(DXL_BAUDRATE):
            raise IOError("Baudrate 설정 실패.")

        for dxl_id in DXL_IDS:
            packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, 0)
            packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE_VELOCITY)
            packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, 1)

        return port_handler, packet_handler

    def sync_write_velocity(self, id_to_velocity: dict):
        group_sync_write = GroupSyncWrite(
            self.dxl_port, self.dxl_packet, ADDR_GOAL_VELOCITY, LEN_GOAL_VELOCITY
        )
        for dxl_id, velocity in id_to_velocity.items():
            v = velocity & 0xFFFFFFFF
            param = [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF]
            group_sync_write.addParam(dxl_id, param)
        result = group_sync_write.txPacket()
        if result != COMM_SUCCESS:
            self.get_logger().warn(f'통신 오류: {self.dxl_packet.getTxRxResult(result)}')
        group_sync_write.clearParam()

    def drive_forward(self):
        self.sync_write_velocity({DXL_ID_1: DIR_1 * REVERSE_SPEED, DXL_ID_2: DIR_2 * REVERSE_SPEED})

    def drive_reverse(self):
        self.sync_write_velocity({DXL_ID_1: -DIR_1 * REVERSE_SPEED, DXL_ID_2: -DIR_2 * REVERSE_SPEED})

    def stop_driving(self):
        self.sync_write_velocity({DXL_ID_1: 0, DXL_ID_2: 0})

    # ---------------- Arduino 시리얼 읽기 (별도 스레드) ----------------
    def read_arduino_loop(self):
        while rclpy.ok():
            try:
                line = self.arduino.readline().decode(errors='ignore').strip()
            except serial.SerialException:
                continue

            if not line:
                continue

            if line.startswith("TILT_FOLD_START"):
                # wheg가 접히는 중 -> 구동모터는 '정방향' (서보 접힘과 짝이 맞는 방향)
                self.get_logger().warn(f'전복 위험 감지! ({line}) -> wheg 접는 중, 구동모터 정방향')
                self.tilt_active = True
                self.drive_forward()

            elif line.startswith("TILT_REVERSE"):
                # wheg 접기 완료 -> 이제 진짜 회피용 후진
                self.get_logger().warn('wheg 접힘 완료 -> 후진 시작')
                self.drive_reverse()

            elif line.startswith("TILT_CLEAR"):
                if self.tilt_active:
                    self.get_logger().info('전복 위험 해제 -> 정지')
                    self.tilt_active = False
                self.stop_driving()

    def destroy_node(self):
        self.stop_driving()
        for dxl_id in DXL_IDS:
            self.dxl_packet.write1ByteTxRx(self.dxl_port, dxl_id, ADDR_TORQUE_ENABLE, 0)
        self.dxl_port.closePort()
        self.arduino.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SafetyStopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
