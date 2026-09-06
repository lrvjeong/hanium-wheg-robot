#include <Servo.h>

Servo myServo;
const int SERVO_PIN = 9;   // 주황(신호)선 연결한 핀

void setup() {
  myServo.attach(SERVO_PIN);
  Serial.begin(9600);
  myServo.write(180);  // 시작을 180도로
}

void loop() {
  // 180도 -> 0도로 부드럽게 이동
  for (int angle = 180; angle >= 0; angle -= 1) {
    myServo.write(angle);
    delay(15);
  }

  delay(2000);   // 0도에서 2초 유지

  // 0도 -> 180도로 부드럽게 복귀
  for (int angle = 0; angle <= 180; angle += 1) {
    myServo.write(angle);
    delay(15);
  }

  delay(2000);   // 180도에서 2초 유지
}
