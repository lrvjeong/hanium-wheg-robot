#!/usr/bin/env python3
"""
디스크형 휠을 주행 방향과 반대로 순간적으로 세게 돌렸다가 다시 정방향으로
복귀시켜서, 그 반동으로 슬롯을 따라 다리(leg)가 펼쳐지도록 하는 코드.

좌우 모터가 기구적으로 미러링되어 있어서 방향 부호가 반대입니다:
    - ID1 (예: 왼쪽) : 전진 = 양수(+)
    - ID2 (예: 오른쪽): 전진 = 음수(-)

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
ADDR_PROFILE_ACCELERATION = 108
ADDR_HARDWARE_ERROR_STATUS = 70
LEN_GOAL_VELOCITY = 4

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
OPERATING_MODE_VELOCITY = 1

# 방향 부호: ID1은 +가 전진, ID2는 -가 전진 (미러링된 배치)
DIR_1 = +1
DIR_2 = -1

FORWARD_SPEED = 100        # 평소 주행 속도
REVERSE_SPEED = 200        # 다리 펼칠 때 역방향 펄스 세기 (필요시 튜닝)
REVERSE_DURATION = 0.3     # 역방향으로 유지할 시간(초) - 슬롯 걸리는 정도 보며 튜닝
# ----------------------------------------------------


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
        # Profile Acceleration = 0 -> 가속 제한 없음 (가장 급격한 반전 가능)
        packet_handler.write4ByteTxRx(port_handler, dxl_id, ADDR_PROFILE_ACCELERATION, 0)
        packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

    return port_handler, packet_handler


def to_uint32(value: int) -> int:
    return value & 0xFFFFFFFF


def sync_write_velocity(port_handler, packet_handler, id_to_velocity: dict):
    group_sync_write = GroupSyncWrite(
        port_handler, packet_handler, ADDR_GOAL_VELOCITY, LEN_GOAL_VELOCITY
    )
    for dxl_id, velocity in id_to_velocity.items():
        v = to_uint32(velocity)
        param = [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF]
        if not group_sync_write.addParam(dxl_id, param):
            print(f"[경고] ID {dxl_id} 파라미터 추가 실패")

    result = group_sync_write.txPacket()
    if result != COMM_SUCCESS:
        print(f"[통신 오류] {packet_handler.getTxRxResult(result)}")
    group_sync_write.clearParam()


def check_errors(port_handler, packet_handler):
    """역방향 펄스 후 과부하 등 에러가 안 떴는지 체크"""
    for dxl_id in DXL_IDS:
        status, _, _ = packet_handler.read1ByteTxRx(port_handler, dxl_id, ADDR_HARDWARE_ERROR_STATUS)
        if status != 0:
            print(f"[경고] ID {dxl_id} 에러 발생 (raw={status}) - 즉시 정지 권장")
            return True
    return False


def drive_forward(port_handler, packet_handler, speed=FORWARD_SPEED):
    sync_write_velocity(
        port_handler, packet_handler,
        {DXL_ID_1: DIR_1 * speed, DXL_ID_2: DIR_2 * speed}
    )


def deploy_legs(port_handler, packet_handler):
    """
    주행 방향과 반대로 순간 역회전 -> 다시 정방향 복귀.
    이 반동으로 슬롯을 타고 다리가 펼쳐지는 것을 노림.
    """
    print(">> 다리 펼치기: 역방향 펄스 시작")
    # 역방향 = 평소 전진 부호의 반대
    sync_write_velocity(
        port_handler, packet_handler,
        {DXL_ID_1: -DIR_1 * REVERSE_SPEED, DXL_ID_2: -DIR_2 * REVERSE_SPEED}
    )
    time.sleep(REVERSE_DURATION)

    if check_errors(port_handler, packet_handler):
        # 에러 나면 바로 정지시키고 중단
        sync_write_velocity(port_handler, packet_handler, {DXL_ID_1: 0, DXL_ID_2: 0})
        return False

    print(">> 다리 펼치기: 정방향 복귀")
    drive_forward(port_handler, packet_handler)
    return True


def stop_motors(port_handler, packet_handler):
    sync_write_velocity(port_handler, packet_handler, {DXL_ID_1: 0, DXL_ID_2: 0})
    time.sleep(0.2)
    for dxl_id in DXL_IDS:
        packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)


if __name__ == "__main__":
    port_handler, packet_handler = init_dynamixel()

    try:
        print("평소 주행 시작 (Ctrl+C 전까지 계속 전진)")
        drive_forward(port_handler, packet_handler)
        time.sleep(1)  # 평지 주행 예시 - 실제로는 LiDAR가 계단 감지했을 때 아래 함수 호출

        deploy_legs(port_handler, packet_handler)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n정지합니다.")

    finally:
        stop_motors(port_handler, packet_handler)
        port_handler.closePort()
