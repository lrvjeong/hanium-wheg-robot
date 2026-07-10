# -*- coding: utf-8 -*-
"""
demo_climb.py — 자동 등반 데모 v2

변경점 (v1 → v2):
  * 스포크 전개 시점: 전방 10cm 감지 시 '즉시' 전개 (기존 55cm)
    → 갈고리 발로 평지를 한참 걸어가며 튀던 현상(비보잉) 제거
  * DEPLOY 상태 추가: 감지 → 제자리 정지 → 스포크가 실제로 다
    펴진 것을 관절각으로 확인한 뒤에 등반 토크 인가
  * 계단 근접 시 감속 접근, 몸이 크게 들리면 토크 컷(뒤집힘 방지)
  * 오므림 판정에 자이로 추가: 몸이 완전히 잠잠해야 평지 복귀

실행:  python3 demo_climb.py   (climber_scene.py와 같은 폴더에)
"""
import math
import time

import mujoco
import mujoco.viewer

from climber_scene import build_xml, DEPLOY_MAX

# ===== 튜닝 상수 =====
DETECT_DIST     = 0.10   # ★ 전방 이 거리에서 즉시 스포크 전개 [m]
DIAG_DETECT     = 0.15   # 사선 ToF 보조 감지 — 낮은 단차 대비 [m]
APPROACH_DIST   = 0.35   # 이 거리부터 감속 접근 [m]
FLAT_TORQUE     = 1.4    # 평지 주행 토크 [N·m]
APPROACH_TORQUE = 0.7    # 접근 토크 [N·m]
CLIMB_TORQUE    = 3.4    # 등반 토크 [N·m]
DEPLOY_DONE     = 0.85   # 목표각의 85% 이상 펴지면 등반 시작
DEPLOY_TIMEOUT  = 1.5    # 전개 최대 대기 시간 [s]
PITCH_FLAT      = 0.12   # 이보다 수평이면 '평지' 판정 [rad]
PITCH_CUT       = 0.60   # 이보다 들리면 토크 축소(뒤집힘 방지) [rad]
GYRO_CALM       = 0.5    # 피치 각속도가 이보다 작아야 '잠잠' [rad/s]
CLEAR_TIME      = 1.2    # 조건 만족이 이만큼 지속되면 오므림 [s]


def rf(data, name):
    """rangefinder 읽기. 미검출(-1)은 5.0m로 처리."""
    v = float(data.sensor(name).data[0])
    return 5.0 if v < 0 else v


def pitch_of(data):
    """framequat(w,x,y,z) → pitch [rad] (앞이 들리면 +)"""
    w, x, y, z = data.sensor("imu_quat").data
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return math.asin(s)


class Controller:
    """FLAT → DEPLOY → CLIMB → FLAT 상태기계.

    실기에서는 이 로직이 라즈베리파이에서 돌고, 나중에 CLIMB의
    토크 출력을 강화학습 정책이 대신하게 됩니다.
    """

    def __init__(self, dt):
        self.dt = dt
        self.mode = "FLAT"
        self.t_in_mode = 0.0
        self.clear_t = 0.0

    def _go(self, mode, why):
        print(f"    >>> {self.mode} → {mode}  ({why})")
        self.mode, self.t_in_mode, self.clear_t = mode, 0.0, 0.0

    def step(self, data):
        self.t_in_mode += self.dt
        d_front = rf(data, "rf_front")
        d_diag = rf(data, "rf_diag")
        pitch = pitch_of(data)
        pitch_rate = float(data.sensor("imu_gyro").data[1])
        wall_near = (d_front < DETECT_DIST) or (d_diag < DIAG_DETECT)

        if self.mode == "FLAT":
            deploy = 0.0
            torque = APPROACH_TORQUE if d_front < APPROACH_DIST else FLAT_TORQUE
            if wall_near:
                self._go("DEPLOY", f"전방 {min(d_front, d_diag):.2f}m 감지")

        elif self.mode == "DEPLOY":
            deploy, torque = DEPLOY_MAX, 0.0  # 제자리에서 변신
            opened = min(float(data.joint("spoke_l0").qpos[0]),
                         float(data.joint("spoke_r0").qpos[0]))
            if opened > DEPLOY_MAX * DEPLOY_DONE:
                self._go("CLIMB", f"스포크 전개 완료 ({opened:.2f} rad)")
            elif self.t_in_mode > DEPLOY_TIMEOUT:
                self._go("CLIMB", "전개 대기 시간 초과, 등반 강행")

        else:  # CLIMB
            deploy, torque = DEPLOY_MAX, CLIMB_TORQUE
            if abs(pitch) > PITCH_CUT:  # 몸이 크게 들리면 토크 컷
                torque *= 0.35
            calm = ((not wall_near) and abs(pitch) < PITCH_FLAT
                    and abs(pitch_rate) < GYRO_CALM)
            self.clear_t = self.clear_t + self.dt if calm else 0.0
            if self.clear_t > CLEAR_TIME:
                self._go("FLAT", "전방 클리어 + 몸체 수평/안정")

        data.actuator("drive_l").ctrl = torque
        data.actuator("drive_r").ctrl = torque
        data.actuator("deploy_l").ctrl = deploy
        data.actuator("deploy_r").ctrl = deploy
        return d_front, pitch


def main():
    model = mujoco.MjModel.from_xml_string(build_xml())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl = Controller(model.opt.timestep)
    last_print = 0.0
    reached = False

    print(f"{'t[s]':>6} {'x[m]':>6} {'z[m]':>6} {'pitch[deg]':>10} "
          f"{'rf_front[m]':>11}  mode")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()

            d_front, pitch = ctrl.step(data)
            mujoco.mj_step(model, data)
            viewer.sync()

            x, _, z = data.body("chassis").xpos
            if data.time - last_print >= 0.5:
                last_print = data.time
                print(f"{data.time:6.1f} {x:6.2f} {z:6.2f} "
                      f"{math.degrees(pitch):10.1f} {d_front:11.2f}  {ctrl.mode}")

            if not reached and z > 0.5 and ctrl.mode == "FLAT":
                reached = True
                print(">>> 꼭대기 도달! 스포크를 오므리고 평지 주행으로 복귀했습니다.")

            leftover = model.opt.timestep - (time.time() - t0)
            if leftover > 0:
                time.sleep(leftover)


if __name__ == "__main__":
    main()
