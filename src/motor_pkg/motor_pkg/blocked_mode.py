#!/usr/bin/env python3
"""
dynamixel_reverse_turn.py

기존 무한회전 예제를 기반으로,
1) 뒤로 약 10cm 후진
2) 정지
3) 한쪽 바퀴만 돌려서 방향 전환 (제자리 회전에 가깝게)
4) 정지
순서로 동작하는 스크립트.

사전 준비:
    pip install dynamixel-sdk --break-system-packages

주의:
    - WHEEL_RADIUS_CM 은 실제 구동 바퀴(또는 다리) 반지름으로 꼭 바꿔줘야
      10cm 후진 거리가 정확해짐. 모르면 대충 재서 넣고, 실제로 돌려보면서
      REVERSE_DURATION_SEC 을 미세 조정해도 됨.
    - TURN_DURATION_SEC 은 "얼마나 돌지"를 시간으로 조절하는 값이라,
      실제로 돌려보면서 원하는 회전각이 나올 때까지 튜닝 필요.
"""
import math
import time
from dynamixel_sdk import (
    PortHandler,
    PacketHandler,
    GroupSyncWrite,
    COMM_SUCCESS,
)

# ------------------- 사용자 설정 -------------------
DEVICENAME = "/dev/ttyUSB2"
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

DXL_ID_1 = 1   # 왼쪽
DXL_ID_2 = 2   # 오른쪽
DXL_IDS = [DXL_ID_1, DXL_ID_2]

ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_VELOCITY = 104
LEN_GOAL_VELOCITY = 4

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
OPERATING_MODE_VELOCITY = 1   # Velocity Control Mode

# 기존 예제와 동일: 두 모터가 서로 반대 부호일 때 같은 방향(전진)으로 굴러가는 배치
FORWARD_VELOCITY_1 = 100
FORWARD_VELOCITY_2 = -100

# 바퀴(또는 다리) 반지름 (cm) - 실측해서 꼭 바꿔줄 것
WHEEL_RADIUS_CM = 6.0

# 후진 거리
REVERSE_DISTANCE_CM = 20.0

# 방향전환 시 회전 시간 (초) - 실측하면서 튜닝
TURN_DURATION_SEC = 2.5
TURN_VELOCITY = 100   # 방향전환에 쓸 속도 크기

# 방향전환 후 다시 전진할 거리 (cm)
FORWARD_AGAIN_DISTANCE_CM = 150.0

VELOCITY_UNIT_TO_RPM = 0.229   # XC430 Goal Velocity 1 unit = 0.229 rev/min
# ----------------------------------------------------


def to_uint32(value: int) -> int:
    """음수 속도값을 4바이트 부호 있는 정수로 올바르게 변환"""
    return value & 0xFFFFFFFF


def velocity_unit_to_linear_cm_per_sec(velocity_unit: int, radius_cm: float) -> float:
    """Goal Velocity raw unit -> 선속도(cm/s) 환산"""
    rpm = abs(velocity_unit) * VELOCITY_UNIT_TO_RPM
    rev_per_sec = rpm / 60.0
    return rev_per_sec * 2 * math.pi * radius_cm


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


def stop_motors_soft(port_handler, packet_handler):
    """속도 0으로만 정지 (토크는 유지)"""
    sync_write_velocity(port_handler, packet_handler, {DXL_ID_1: 0, DXL_ID_2: 0})


def stop_motors_and_disable(port_handler, packet_handler):
    """속도 0으로 정지 후 토크까지 끄기"""
    stop_motors_soft(port_handler, packet_handler)
    time.sleep(0.2)
    for dxl_id in DXL_IDS:
        packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)


def reverse(port_handler, packet_handler, distance_cm=REVERSE_DISTANCE_CM):
    """뒤로 distance_cm 만큼 후진 (전진 속도의 부호 반대)"""
    reverse_v1 = -FORWARD_VELOCITY_1
    reverse_v2 = -FORWARD_VELOCITY_2

    linear_speed = velocity_unit_to_linear_cm_per_sec(FORWARD_VELOCITY_1, WHEEL_RADIUS_CM)
    duration_sec = distance_cm / linear_speed

    print(f"[후진] {distance_cm}cm, 예상 소요시간: {duration_sec:.2f}초")
    sync_write_velocity(port_handler, packet_handler, {DXL_ID_1: reverse_v1, DXL_ID_2: reverse_v2})
    time.sleep(duration_sec)
    stop_motors_soft(port_handler, packet_handler)
    time.sleep(0.3)  # 관성으로 밀리는 것 방지용 소강 시간


def turn_one_wheel(port_handler, packet_handler, turn_left=True):
    """한쪽 바퀴만 돌려서 방향 전환.
    turn_left=True  -> 왼쪽(ID1) 바퀴만 돌아서 왼쪽으로 회전
    turn_left=False -> 오른쪽(ID2) 바퀴만 돌아서 오른쪽으로 회전
    """
    if turn_left:
        v1, v2 = TURN_VELOCITY, TURN_VELOCITY
        print(f"[방향전환] 왼쪽 바퀴만 구동, {TURN_DURATION_SEC}초")
    else:
        v1, v2 = -TURN_VELOCITY, -TURN_VELOCITY
        print(f"[방향전환] 오른쪽 바퀴만 구동, {TURN_DURATION_SEC}초")

    sync_write_velocity(port_handler, packet_handler, {DXL_ID_1: v1, DXL_ID_2: v2})
    time.sleep(TURN_DURATION_SEC)
    stop_motors_soft(port_handler, packet_handler)
    time.sleep(0.3)  # 관성으로 밀리는 것 방지용 소강 시간


def drive_forward(port_handler, packet_handler, distance_cm=FORWARD_AGAIN_DISTANCE_CM):
    """방향전환 이후 다시 앞으로 distance_cm 만큼 주행"""
    linear_speed = velocity_unit_to_linear_cm_per_sec(FORWARD_VELOCITY_1, WHEEL_RADIUS_CM)
    duration_sec = distance_cm / linear_speed

    print(f"[전진] {distance_cm}cm, 예상 소요시간: {duration_sec:.2f}초")
    sync_write_velocity(port_handler, packet_handler, {DXL_ID_1: FORWARD_VELOCITY_1, DXL_ID_2: FORWARD_VELOCITY_2})
    time.sleep(duration_sec)
    stop_motors_soft(port_handler, packet_handler)


if __name__ == "__main__":
    port_handler, packet_handler = init_dynamixel()
    try:
        reverse(port_handler, packet_handler, REVERSE_DISTANCE_CM)
        turn_one_wheel(port_handler, packet_handler, turn_left=True)
        drive_forward(port_handler, packet_handler, FORWARD_AGAIN_DISTANCE_CM)
        print("완료.")
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        stop_motors_and_disable(port_handler, packet_handler)
        port_handler.closePort()
