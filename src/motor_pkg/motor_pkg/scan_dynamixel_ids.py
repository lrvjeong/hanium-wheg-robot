#!/usr/bin/env python3
"""
U2D2에 연결된 Dynamixel들의 ID를 스캔합니다.
ID가 1개만 나오면 -> 두 모터가 같은 ID를 쓰고 있다는 뜻입니다.
"""

from dynamixel_sdk import PortHandler, PacketHandler

DEVICENAME = "/dev/ttyUSB0"
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

port_handler = PortHandler(DEVICENAME)
packet_handler = PacketHandler(PROTOCOL_VERSION)

if not port_handler.openPort():
    raise IOError("포트를 열 수 없습니다.")
if not port_handler.setBaudRate(BAUDRATE):
    raise IOError("Baudrate 설정 실패.")

print("ID 스캔 중... (0~20번까지 확인)")
found_ids = []
for dxl_id in range(0, 21):
    model_number, comm_result, error = packet_handler.ping(port_handler, dxl_id)
    if comm_result == 0:  # COMM_SUCCESS
        print(f"  -> ID {dxl_id} 발견 (모델 번호: {model_number})")
        found_ids.append(dxl_id)

print(f"\n총 {len(found_ids)}개의 모터 발견: {found_ids}")

if len(found_ids) < 2:
    print("경고: 모터가 2개 연결되어 있는데 ID가 1개만 잡혔다면,")
    print("두 모터의 ID가 겹쳐있는 것입니다. Dynamixel Wizard 2.0으로")
    print("한쪽 모터만 남기고 ID를 바꿔주세요.")

port_handler.closePort()
