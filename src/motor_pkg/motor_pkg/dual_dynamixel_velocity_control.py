#!/usr/bin/env python3
"""
Raspberry Pi + U2D2 로 XC430 Dynamixel 모터 2개를 Velocity Control Mode로
동시에 무한 회전시키는 예제.

Position Control Mode와 달리 목표 '위치'가 아니라 목표 '속도'를 주기 때문에
회전 각도 제한 없이 계속 돕니다. Ctrl+C 로 멈출 때까지 계속 회전합니다.

사전 준비:
    pip install dynamixel-sdk --break-system-packages
"""

import time
from dynamixel_sdk import (
    PortHandler,
    PacketHandler,
    GroupSyncWrite,
    COMM_SUCCESS,
)

# ------------------- 사용자 설정 -------------------
DEVICENAME = "/dev/ttyUSB0"
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

DXL_ID_1 = 1
DXL_ID_2 = 2
DXL_IDS = [DXL_ID_1, DXL_ID_2]

ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_VELOCITY = 104
LEN_GOAL_VELOCITY = 4

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

OPERATING_MODE_VELOCITY = 1   # Velocity Control Mode

# 목표 속도. XC430 기준 단위는 0.229 rev/min 당 1.
# 값이 클수록 빨리 돕니다. 음수면 반대 방향.
# 최대치는 약 265~330 부근 (모델/전압에 따라 다름) 이니 너무 크게 잡지 마세요.
GOAL_VELOCITY_1 = 100
GOAL_VELOCITY_2 = -100
# ----------------------------------------------------


def to_uint32(value: int) -> int:
    """음수 속도값을 4바이트 부호 있는 정수로 올바르게 변환"""
    return value & 0xFFFFFFFF


def init_dynamixel():
    port_handler = PortHandler(DEVICENAME)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        raise IOError("포트를 열 수 없습니다.")
    if not port_handler.setBaudRate(BAUDRATE):
        raise IOError("Baudrate 설정에 실패했습니다.")

    for dxl_id in DXL_IDS:
        packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE_VELOCITY)
        packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

    return port_handler, packet_handler


def sync_write_velocity(port_handler, packet_handler, id_to_velocity: dict):
    group_sync_write = GroupSyncWrite(
        port_handler, packet_handler, ADDR_GOAL_VELOCITY, LEN_GOAL_VELOCITY
    )

    for dxl_id, velocity in id_to_velocity.items():
        v = to_uint32(velocity)
        param = [
            v & 0xFF,
            (v >> 8) & 0xFF,
            (v >> 16) & 0xFF,
            (v >> 24) & 0xFF,
        ]
        if not group_sync_write.addParam(dxl_id, param):
            print(f"[경고] ID {dxl_id} 파라미터 추가 실패")

    dxl_comm_result = group_sync_write.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[통신 오류] {packet_handler.getTxRxResult(dxl_comm_result)}")

    group_sync_write.clearParam()


def stop_motors(port_handler, packet_handler):
    """속도 0으로 정지 후 토크 끄기"""
    sync_write_velocity(port_handler, packet_handler, {DXL_ID_1: 0, DXL_ID_2: 0})
    time.sleep(0.2)
    for dxl_id in DXL_IDS:
        packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)


if __name__ == "__main__":
    port_handler, packet_handler = init_dynamixel()

    try:
        print(f"두 모터를 속도 {GOAL_VELOCITY_1}, {GOAL_VELOCITY_2} 로 동시에 무한 회전시킵니다.")
        print("Ctrl+C 로 정지하세요.")

        sync_write_velocity(
            port_handler, packet_handler,
            {DXL_ID_1: GOAL_VELOCITY_1, DXL_ID_2: GOAL_VELOCITY_2}
        )

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n정지합니다.")

    finally:
        stop_motors(port_handler, packet_handler)
        port_handler.closePort()
