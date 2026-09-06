/*
  leg_servo_imu_controller.ino

  - 서보 2개(좌/우) 동시 제어. L 서보를 기준(canonical) 각도로 삼고,
    R 서보는 항상 (180 - L각도)로 반전 명령 (좌우 대칭 장착이라서).
  - WT901 IMU가 이 아두이노에 직결되어 있음. pitch를 계산해서
    "PITCH:<deg>\n" 형식으로 Pi에 50Hz로 스트리밍만 함.
    전복 위험 판단/조치(진입 25도, 해제 15도 히스테리시스)는 더 이상
    아두이노가 하지 않고, Pi 쪽 mode_fsm_node가 전담함. 이 아두이노는
    순수 센서+서보 구동 역할만 함.
  - Pi로부터 "S:<angle>\n" 명령을 받아 두 서보를 동시에 제어 (angle은 L 기준 각도)

  배선 (WT901 <-> Arduino, UART 모드): 기존과 동일
      WT901 ①VCC -> Arduino 3.3V  (5V 아님! 3.3V로 연결)
      WT901 ⑤GND -> Arduino GND
      WT901 ③TXD -> Arduino D2 (imuSerial RX)
      WT901 ④RXD -> Arduino D3 (imuSerial TX)

  서보 배선: 기존과 동일
      L 서보 신호선 -> Arduino D9
      R 서보 신호선 -> Arduino D10
      두 서보 전원(+/-)은 각각 독립된 배터리팩에서 공급하고,
      GND는 Arduino GND와 반드시 공통으로 묶을 것.

  * WT901을 I2C로 연결하셨다면 이 코드는 그대로 못 쓰고 I2C 버전으로 바꿔야 합니다.
  * WT901 기본 출력 프로토콜(0x55 헤더, 11바이트 패킷)을 파싱합니다.
*/
#include <Servo.h>
#include <SoftwareSerial.h>

Servo legServoL;
Servo legServoR;
const int SERVO_L_PIN = 9;
const int SERVO_R_PIN = 10;

SoftwareSerial imuSerial(2, 3);  // RX, TX (WT901과 연결)

const int SERVO_DEPLOYED_ANGLE = 180;  // L 기준: 시작 시 이미 펼쳐진 상태로 가정하는 각도

const unsigned long PITCH_REPORT_INTERVAL_MS = 20;  // Pi로 pitch 전송 주기 (약 50Hz)
unsigned long lastPitchReportMs = 0;

float currentPitch = 0.0;
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
  reportPitch();
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

void reportPitch() {
  unsigned long now = millis();
  if (now - lastPitchReportMs >= PITCH_REPORT_INTERVAL_MS) {
    lastPitchReportMs = now;
    Serial.print("PITCH:");
    Serial.println(currentPitch, 2);
  }
}

void handlePiCommands() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.startsWith("S:")) {
      int angle = line.substring(2).toInt();
      writeBothServos(angle);   // L=angle, R=180-angle 자동 반영
    }
  }
}
