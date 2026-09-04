/*
  leg_servo_imu_controller_dual.ino
  기존 leg_servo_imu_controller.ino를 서보 2개(좌/우) 동시 제어로 확장.

  - 서보 2개는 좌우 대칭으로 장착되어 있어서, 같은 "펼침/접힘" 동작을 만들려면
    한쪽은 정방향, 한쪽은 역방향으로 돌아야 함.
    -> L 서보를 기준(canonical) 각도로 삼고, R 서보는 항상 (180 - L각도)로 반전 명령.
  - 초기 상태: 두 서보 모두 이미 펼쳐진 상태로 시작
    (L=180도 / R은 반전이므로 0도가 "펼쳐진" 상태)
  - 평소: Pi로부터 "S:<angle>\n" 명령을 받아 두 서보를 동시에 제어 (angle은 L 기준 각도)
  - WT901의 Pitch 각도가 TILT_THRESHOLD_DEG를 넘으면:
        1. 즉시 두 서보를 모두 접힘 상태로 되돌려 후진 가능한 상태로 만듦
           (안전 최우선, Pi 응답 기다리지 않음)
        2. Pi에 "TILT_FOLD_START:<pitch>\n" -> 접기 완료 후 "TILT_REVERSE\n" 전송
    이후 Pitch가 정상 범위로 돌아오면 "TILT_CLEAR\n" 을 보내 후진을 멈추게 함

  배선 (WT901 <-> Arduino, UART 모드):
      WT901 ①VCC -> Arduino 3.3V  (5V 아님! 3.3V로 연결)
      WT901 ⑤GND -> Arduino GND
      WT901 ③TXD -> Arduino D2 (imuSerial RX)
      WT901 ④RXD -> Arduino D3 (imuSerial TX)

  서보 배선:
      L 서보 신호선 -> Arduino D9
      R 서보 신호선 -> Arduino D10
      두 서보 전원(+/-)은 각각 독립된 배터리팩에서 공급하고,
      GND는 Arduino GND와 반드시 공통으로 묶을 것.

  * WT901을 I2C로 연결하셨다면 이 코드는 그대로 못 쓰고 I2C 버전으로 바꿔야 합니다.
  * WT901 기본 출력 프로토콜(0x55 헤더, 11바이트 패킷)을 파싱합니다.
  * 주의: Arduino Uno의 디지털 핀은 5V 로직입니다. WT901이 5V 입력에 안전하다고
    명시되어 있지 않다면(데이터시트 확인 필요), D3(Arduino->IMU RX) 쪽에 레벨시프터나
    전압분배 저항을 추가하는 것이 안전합니다. IMU->Arduino 방향(D2)은 보통 문제없습니다.
*/
#include <Servo.h>
#include <SoftwareSerial.h>

Servo legServoL;
Servo legServoR;
const int SERVO_L_PIN = 9;
const int SERVO_R_PIN = 10;

SoftwareSerial imuSerial(2, 3);  // RX, TX (WT901과 연결)

const float TILT_THRESHOLD_DEG = 25.0;  // 이 각도(절대값) 넘으면 전복 위험으로 판단, 실측 후 튜닝 필요
const int SERVO_DEPLOYED_ANGLE = 180;   // L 기준: 시작 시 이미 펼쳐진 상태로 가정하는 각도
const int SERVO_FOLDED_ANGLE = 0;       // L 기준: 위험 감지 시 후진을 위해 접는 각도
const unsigned long FOLD_DURATION_MS = 1500;  // 서보가 180->0 까지 물리적으로 접히는 데 걸리는 시간 (튜닝 필요)

float currentPitch = 0.0;
bool tiltWarningActive = false;
uint8_t imuBuffer[11];
int imuBufferIndex = 0;

// L 기준 각도를 받아서 두 서보에 동시에 반영 (R은 반전)
void writeBothServos(int angleL) {
  angleL = constrain(angleL, 0, 180);
  int angleR = 180 - angleL;   // 좌우 대칭 반전
  legServoL.write(angleL);
  legServoR.write(angleR);
}

void setup() {
  Serial.begin(115200);       // Pi와 통신
  imuSerial.begin(9600);      // WT901 기본 baudrate
  legServoL.attach(SERVO_L_PIN);
  legServoR.attach(SERVO_R_PIN);
  writeBothServos(SERVO_DEPLOYED_ANGLE);  // 초기값: 이미 wheg가 펼쳐진 상태로 시작
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
    Serial.print("TILT_FOLD_START:");
    Serial.println(currentPitch);
    writeBothServos(SERVO_FOLDED_ANGLE);  // 두 wheg 모두 접기 시작
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
      writeBothServos(angle);   // L=angle, R=180-angle 자동 반영
      Serial.print("OK:");
      Serial.println(angle);
    }
  }
}
