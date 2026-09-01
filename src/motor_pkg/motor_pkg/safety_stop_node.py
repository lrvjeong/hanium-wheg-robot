#!/usr/bin/env python3
"""
motor_pkg/safety_stop_node.py

다이나믹셀(구동모터)과 아두이노(서보+IMU)를 이 노드 하나가 전담합니다.

평소:
    /motor/dc_cmd    구독 -> 다이나믹셀 속도 명령
    /motor/servo_cmd 구독 -> 아두이노에 "S:<angle>\n" 시리얼 명령

전복 위험 시 (Arduino가 TILT_FOLD_START / TILT_REVERSE 보냄):
    ROS 주행/서보 명령 무시하고 이 노드가 직접 다이나믹셀 후진시킴.
    TILT_CLEAR 오면 평소 모드(ROS 명령 따름)로 복귀.
"""

import time
import threading
import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, COMM_SUCCESS

DXL_DEVICENAME = "/dev/ttyUSB1"
DXL_BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

DXL_ID_1 = 1
DXL_ID_2 = 2
DXL_IDS = [DXL_ID_1, DXL_ID_2]
DIR_1, DIR_2 = +1, -1   # 좌우 모터가 반대로 장착돼 있어서 방향 미러링

ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_VELOCITY = 104
LEN_GOAL_VELOCITY = 4
OPERATING_MODE_VELOCITY = 1

MAX_VELOCITY_UNIT = 200
REVERSE_SPEED = 80

ARDUINO_DEVICENAME = "/dev/ttyUSB1"
ARDUINO_BAUDRATE = 115200


class SafetyStopNode(Node):
    def __init__(self):
        super().__init__('safety_stop_node')

        self.dxl_port, self.dxl_packet = self.init_dynamixel()
        self.arduino = self.init_arduino()

        self.tilt_active = False

        self.create_subscription(Float32MultiArray, '/motor/dc_cmd', self.dc_cmd_cb, 10)
        self.create_subscription(Float32MultiArray, '/motor/servo_cmd', self.servo_cmd_cb, 10)

        self.reader_thread = threading.Thread(target=self.read_arduino_loop, daemon=True)
        self.reader_thread.start()

        self.get_logger().info('모터/안전 통합 노드 시작 (다이나믹셀 + 아두이노 서보/IMU)')

    def init_dynamixel(self):
        port_handler = PortHandler(DXL_DEVICENAME)
        packet_handler = PacketHandler(PROTOCOL_VERSION)
        try:
            if not port_handler.openPort():
                raise IOError("다이나믹셀 포트를 열 수 없습니다.")
            if not port_handler.setBaudRate(DXL_BAUDRATE):
                raise IOError("Baudrate 설정 실패.")
            for dxl_id in DXL_IDS:
                packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, 0)
                packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE_VELOCITY)
                packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, 1)
            self.get_logger().info(f'다이나믹셀 연결됨: {DXL_DEVICENAME}')
        except Exception as e:
            self.get_logger().warn(f'다이나믹셀 연결 실패 (하드웨어 미연결): {e}')
        return port_handler, packet_handler

    def init_arduino(self):
        try:
            ser = serial.Serial(ARDUINO_DEVICENAME, ARDUINO_BAUDRATE, timeout=1)
            time.sleep(2)
            self.get_logger().info(f'아두이노 연결됨: {ARDUINO_DEVICENAME}')
            return ser
        except Exception as e:
            self.get_logger().warn(f'아두이노 연결 실패 (하드웨어 미연결): {e}')
            return None

    def dc_cmd_cb(self, msg: Float32MultiArray):
        if self.tilt_active:
            return
        if len(msg.data) < 2:
            return
        left_speed, right_speed = msg.data[0], msg.data[1]
        left_vel = int(left_speed * MAX_VELOCITY_UNIT)
        right_vel = int(right_speed * MAX_VELOCITY_UNIT)
        self.sync_write_velocity({
            DXL_ID_1: DIR_1 * left_vel,
            DXL_ID_2: DIR_2 * right_vel,
        })

    def servo_cmd_cb(self, msg: Float32MultiArray):
        if self.tilt_active:
            return
        if self.arduino is None or len(msg.data) == 0:
            return
        angle = int(max(0.0, min(180.0, msg.data[0])))
        try:
            self.arduino.write(f"S:{angle}\n".encode())
        except serial.SerialException as e:
            self.get_logger().warn(f'아두이노 서보 명령 전송 실패: {e}')

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

    def read_arduino_loop(self):
        if self.arduino is None:
            return
        while rclpy.ok():
            try:
                line = self.arduino.readline().decode(errors='ignore').strip()
            except serial.SerialException:
                continue
            if not line:
                continue

            if line.startswith("TILT_FOLD_START"):
                self.get_logger().warn(f'전복 위험 감지! ({line}) -> wheg 접는 중, 구동모터 정방향')
                self.tilt_active = True
                self.drive_forward()
            elif line.startswith("TILT_REVERSE"):
                self.get_logger().warn('wheg 접힘 완료 -> 후진 시작')
                self.drive_reverse()
            elif line.startswith("TILT_CLEAR"):
                if self.tilt_active:
                    self.get_logger().info('전복 위험 해제 -> 정지, 평소 모드로 복귀')
                    self.tilt_active = False
                self.stop_driving()

    def destroy_node(self):
        self.stop_driving()
        for dxl_id in DXL_IDS:
            self.dxl_packet.write1ByteTxRx(self.dxl_port, dxl_id, ADDR_TORQUE_ENABLE, 0)
        self.dxl_port.closePort()
        if self.arduino is not None:
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
