/*
  leg_servo_imu_controller.ino

  기존 서보 제어(leg_servo_controller.ino)에 WT901 IMU 감시를 추가.

  - 초기 상태: wheg가 이미 180도로 펼쳐진 상태로 시작 (처음부터 높은 단차에
    wheg를 끼워놓고 시작하는 시나리오 가정)
  - 평소: Pi로부터 "S:<angle>\n" 명령을 받아 서보 각도 제어 (기존과 동일)
  - WT901의 Pitch(또는 Roll) 각도가 TILT_THRESHOLD_DEG를 넘으면:
        1. 즉시 서보를 0도(접힘)로 되돌려 후진 가능한 상태로 만듦 (안전 최우선, Pi 응답 기다리지 않음)
        2. Pi에 "TILT_WARNING:<pitch>\n" 을 보내서 후진하라고 알림
    이후 Pitch가 정상 범위로 돌아오면 "TILT_CLEAR\n" 을 보내 후진을 멈추게 함

  배선 (WT901 <-> Arduino, UART 모드):
      WT901 ①VCC -> Arduino 3.3V  (5V 아님! 3.3V로 연결)
      WT901 ⑤GND -> Arduino GND
      WT901 ③TXD -> Arduino D2 (imuSerial RX)   // IMU가 보내는 데이터를 Arduino가 받음
      WT901 ④RXD -> Arduino D3 (imuSerial TX)   // Arduino가 보내는 설정 명령을 IMU가 받음
  * WT901을 I2C로 연결하셨다면 이 코드는 그대로 못 쓰고 I2C 버전으로 바꿔야 합니다.
  * WT901 기본 출력 프로토콜(0x55 헤더, 11바이트 패킷)을 파싱합니다.
  * 주의: Arduino Uno의 디지털 핀은 5V 로직입니다. WT901이 5V 입력에 안전하다고
    명시되어 있지 않다면(데이터시트 확인 필요), D3(Arduino->IMU RX) 쪽에 레벨시프터나
    전압분배 저항을 추가하는 것이 안전합니다. IMU->Arduino 방향(D2)은 보통 문제없습니다.
*/

#include <Servo.h>
#include <SoftwareSerial.h>

Servo legServo;
const int SERVO_PIN = 9;

SoftwareSerial imuSerial(2, 3);  // RX, TX (WT901과 연결)

const float TILT_THRESHOLD_DEG = 25.0;  // 이 각도(절대값) 넘으면 전복 위험으로 판단, 실측 후 튜닝 필요
const int SERVO_DEPLOYED_ANGLE = 180;   // 시작 시 이미 펼쳐진 상태로 가정하는 각도
const int SERVO_FOLDED_ANGLE = 0;       // 위험 감지 시 후진을 위해 접는 각도
const unsigned long FOLD_DURATION_MS = 1500;  // 서보가 180->0 까지 물리적으로 접히는 데 걸리는 시간 (튜닝 필요)

float currentPitch = 0.0;
bool tiltWarningActive = false;

uint8_t imuBuffer[11];
int imuBufferIndex = 0;

void setup() {
  Serial.begin(115200);       // Pi와 통신
  imuSerial.begin(9600);      // WT901 기본 baudrate

  legServo.attach(SERVO_PIN);
  legServo.write(SERVO_DEPLOYED_ANGLE);  // 초기값: 이미 wheg가 펼쳐진 상태로 시작
                                          // (처음부터 높은 단차에 wheg를 끼워놓고 시작하는 시나리오)
}

void loop() {
  readImu();
  checkTiltSafety();
  handlePiCommands();
}

void readImu() {
  while (imuSerial.available() > 0) {
    uint8_t b = imuSerial.read();

    if (imuBufferIndex == 0 && b != 0x55) {
      continue;  // 패킷 헤더 찾을 때까지 버림
    }

    imuBuffer[imuBufferIndex++] = b;

    if (imuBufferIndex >= 11) {
      // 두 번째 바이트가 0x53이면 각도 출력 패킷 (Roll, Pitch, Yaw)
      if (imuBuffer[1] == 0x53) {
        int16_t rawPitch = (imuBuffer[5] << 8) | imuBuffer[4];
        currentPitch = (rawPitch / 32768.0) * 180.0;
      }
      imuBufferIndex = 0;
    }
  }
}

void checkTiltSafety() {
  bool tiltDangerous = (abs(currentPitch) >= TILT_THRESHOLD_DEG);

  if (tiltDangerous && !tiltWarningActive) {
    tiltWarningActive = true;

    // 1단계: 접기 시작 알림 -> Pi는 이 신호에 구동모터를 '정방향'으로 움직여야 함
    //         (서보가 펼칠 때 구동모터 역방향과 짝이었으니, 접을 때는 반대로 정방향과 짝)
    Serial.print("TILT_FOLD_START:");
    Serial.println(currentPitch);

    legServo.write(SERVO_FOLDED_ANGLE);   // wheg 접기 시작 (180 -> 0)
    delay(FOLD_DURATION_MS);              // 서보가 물리적으로 다 접힐 때까지 대기

    // 2단계: 접기 완료 -> 이제부터 진짜 회피용 후진 시작
    Serial.println("TILT_REVERSE");
  }
  else if (!tiltDangerous && tiltWarningActive) {
    // 위험 해제
    tiltWarningActive = false;
    Serial.println("TILT_CLEAR");
  }
}

void handlePiCommands() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    // 위험 상태에서는 Pi가 펼치라고 명령해도 무시 (안전 우선)
    if (tiltWarningActive) {
      Serial.println("IGNORED:TILT_ACTIVE");
      return;
    }

    if (line.startsWith("S:")) {
      int angle = line.substring(2).toInt();
      angle = constrain(angle, 0, 180);
      legServo.write(angle);
      Serial.print("OK:");
      Serial.println(angle);
    }
  }
}