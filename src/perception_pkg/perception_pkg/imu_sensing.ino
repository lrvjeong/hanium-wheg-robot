#include <SoftwareSerial.h>
#include "JY901.h"

SoftwareSerial imuSerial(2, 3);  // WT901과 통신용

void setup() {
  Serial.begin(9600);      // 라즈베리파이와 통신용(USB)
  imuSerial.begin(9600);   // WT901과 통신용
}

void loop() {
  while (imuSerial.available()) {
    JY901.CopeSerialData(imuSerial.read());
  }

  float roll  = (float)JY901.stcAngle.Angle[0] / 32768 * 180;
  float pitch = (float)JY901.stcAngle.Angle[1] / 32768 * 180;
  float yaw   = (float)JY901.stcAngle.Angle[2] / 32768 * 180;

  // 각속도(자이로) - 단위: deg/s
  float rollRate  = (float)JY901.stcGyro.w[0] / 32768 * 2000;
  float pitchRate = (float)JY901.stcGyro.w[1] / 32768 * 2000;
  float yawRate   = (float)JY901.stcGyro.w[2] / 32768 * 2000;

  // 라즈베리파이로 CSV 형식(콤마구분)으로 전송
  // 형식: pitch,pitchRate,roll,rollRate,yaw,yawRate
  Serial.print(pitch, 3);   Serial.print(",");
  Serial.print(pitchRate, 3); Serial.print(",");
  Serial.print(roll, 3);    Serial.print(",");
  Serial.print(rollRate, 3);  Serial.print(",");
  Serial.print(yaw, 3);     Serial.print(",");
  Serial.println(yawRate, 3);

  delay(20);  // 약 50Hz
}