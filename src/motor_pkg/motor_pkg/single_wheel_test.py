#!/usr/bin/env python3
"""
한쪽 바퀴만 테스트: Dynamixel 1개(주행) + 서보 1개(다리)만 동작시킵니다.
어느 ID/포트를 쓸지는 아래 설정값만 바꾸면 됩니다.

사전 준비:
    pip install dynamixel-sdk pyserial --break-system-packages
"""

import time
import serial
from dynamixel_sdk import PortHandler, PacketHandler

# ------------------- 테스트할 쪽 설정 -------------------
DXL_DEVICENAME = "/dev/ttyUSB0"
DXL_BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

DXL_ID = 1              # 테스트할 모터 ID (반대쪽은 2로 바꿔서 재사용)
DXL_DIRECTION = +1      # 이 바퀴의 전진 방향 부호 (ID1=+, ID2=-)

ARDUINO_DEVICENAME = "/dev/ttyUSB1"
ARDUINO_BAUDRATE = 115200

ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_VELOCITY = 104
OPERATING_MODE_VELOCITY = 1

FORWARD_SPEED = 100
LEG_DEPLOYED_ANGLE = 120
LEG_RETRACTED_ANGLE = 0
# --------------------------------------------------------


def init_dynamixel():
    port_handler = PortHandler(DXL_DEVICENAME)
    packet_handler = PacketHandler(PROTOCOL_VERSION)
    if not port_handler.openPort():
        raise IOError("포트를 열 수 없습니다.")
    if not port_handler.setBaudRate(DXL_BAUDRATE):
        raise IOError("Baudrate 설정 실패.")

    packet_handler.write1ByteTxRx(port_handler, DXL_ID, ADDR_TORQUE_ENABLE, 0)
    packet_handler.write1ByteTxRx(port_handler, DXL_ID, ADDR_OPERATING_MODE, OPERATING_MODE_VELOCITY)
    packet_handler.write1ByteTxRx(port_handler, DXL_ID, ADDR_TORQUE_ENABLE, 1)
    return port_handler, packet_handler


def dxl_set_velocity(port_handler, packet_handler, velocity):
    v = velocity & 0xFFFFFFFF
    packet_handler.write4ByteTxRx(port_handler, DXL_ID, ADDR_GOAL_VELOCITY, v)


def init_arduino():
    ser = serial.Serial(ARDUINO_DEVICENAME, ARDUINO_BAUDRATE, timeout=1)
    time.sleep(2)  # 아두이노 리셋 대기
    return ser


def servo_set_angle(ser, angle):
    ser.write(f"S:{angle}\n".encode())
    response = ser.readline().decode().strip()
    print(f"  서보 응답: {response}")


if __name__ == "__main__":
    print(f"=== ID {DXL_ID} 쪽 바퀴만 테스트 ===")

    dxl_port, dxl_packet = init_dynamixel()
    arduino = init_arduino()

    try:
        print("1. 전진 3초")
        dxl_set_velocity(dxl_port, dxl_packet, DXL_DIRECTION * FORWARD_SPEED)
        time.sleep(3)

        print("2. 정지")
        dxl_set_velocity(dxl_port, dxl_packet, 0)
        time.sleep(1)

        print("3. 다리 펼치기")
        servo_set_angle(arduino, LEG_DEPLOYED_ANGLE)
        time.sleep(2)

        print("4. 다리 접기")
        servo_set_angle(arduino, LEG_RETRACTED_ANGLE)

    except KeyboardInterrupt:
        print("\n중단")

    finally:
        dxl_set_velocity(dxl_port, dxl_packet, 0)
        dxl_packet.write1ByteTxRx(dxl_port, DXL_ID, ADDR_TORQUE_ENABLE, 0)
        dxl_port.closePort()
        arduino.close()
        print("종료")
