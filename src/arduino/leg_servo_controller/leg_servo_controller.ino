/*
  Pi로부터 USB 시리얼로 명령을 받아 다리(leg) 서보를 움직이는 스케치.
  명령 형식: "S:<angle>\n"  예) "S:120\n" -> 서보를 120도로 이동
            "S:0\n"         -> 다리 접기(0도)로 가정

  IMU(WT901)도 같은 Arduino가 담당한다고 하셨으니, 필요하면
  loop() 안에서 IMU 읽는 코드와 나란히 두시면 됩니다.
*/

#include <Servo.h>

Servo legServo;
const int SERVO_PIN = 9;

void setup() {
  Serial.begin(115200);
  legServo.attach(SERVO_PIN);
  legServo.write(0);  // 초기값: 다리 접힘
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.startsWith("S:")) {
      int angle = line.substring(2).toInt();
      angle = constrain(angle, 0, 180);
      legServo.write(angle);

      Serial.print("OK:");
      Serial.println(angle);
    }
  }

  // 여기에 IMU(WT901) 읽는 코드를 같이 넣으시면 됩니다.
}
