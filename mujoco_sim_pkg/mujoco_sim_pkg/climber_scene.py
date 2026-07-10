# -*- coding: utf-8 -*-
"""
climber_scene.py — 계단 등반 로봇 프록시 모델 생성기 (MuJoCo MJCF)

구성:
  * 파라미터로 만드는 계단 (단높이/단깊이/단수 조절 가능)
  * 3-스포크 변신 바퀴 로봇 (프록시):
      - 오므린 상태: 호(arc) 모양 스포크가 림을 이뤄 거의 원형으로 굴러감
      - 펼친 상태: 스포크가 갈고리처럼 바깥으로 펴져 단차를 딛고 오름
  * 센서: rangefinder(ToF/라이다 대용) 3개, IMU(자이로+가속도+자세)

실행:
  python3 climber_scene.py
  → climber.xml 저장 + 뷰어 실행 (오른쪽 Control 패널에서 직접 조종)

나중에 URDF가 오면: 로봇 부분(robot_xml)만 교체하면
계단/센서/학습 파이프라인은 그대로 재사용할 수 있습니다.
"""
import math

# ============================================================
# 튜닝 상수 (여기 숫자만 바꿔가며 실험하세요)
# ============================================================
# --- 계단 ---
N_STEPS   = 5       # 단수
STEP_H    = 0.13    # 단높이 [m]  (한국 계단 보통 0.15~0.18, 처음엔 낮게)
STEP_D    = 0.30    # 단깊이 [m]
STEP_W    = 1.6     # 계단 폭 [m]
STAIR_X0  = 1.10    # 로봇 시작점에서 첫 단까지 거리 [m]

# --- 바퀴 / 스포크 ---
HUB_R      = 0.080  # 바퀴 허브 반지름 [m]
ARC_R      = 0.088  # 스포크 호의 반지름 (오므렸을 때 림이 되는 원)
ARC_SPAN   = math.radians(95)   # 스포크 호가 덮는 각도
SPOKE_RAD  = 0.009  # 스포크 캡슐 굵기 [m]
ARC_DIR    = +1     # 훅이 감기는 방향. 등반 시 갈고리가 계단에 안 걸리면 -1로!
DEPLOY_MAX = 1.85   # 스포크 전개 각도 [rad] (~106도)
WHEEL_X    = 0.08   # 차체 기준 바퀴 축 위치 (앞쪽 +x)
WHEEL_Y    = 0.130  # 차체 중심에서 바퀴까지 좌우 거리
SPOKE_YOFF = 0.026  # 스포크를 허브 바깥면 쪽으로 빼는 오프셋

# --- 차체 / 구동 ---
CHASSIS_HALF = (0.14, 0.09, 0.035)  # 차체 박스 절반 치수 (x, y, z)
CHASSIS_MASS = 2.0                  # [kg]
DRIVE_MAX    = 5.0                  # 바퀴 모터 최대 토크 [N·m]
START_Z      = 0.105                # 초기 차체 높이

# ============================================================
def _f(*vals):
    return " ".join(f"{v:.4f}" for v in vals)


def _spoke_body(prefix: str, k: int, y_off: float) -> str:
    """스포크 1개(호를 캡슐 2개로 근사)를 XML로 생성.

    피벗은 허브 림 위 (반지름 HUB_R, 각도 phi)에 있고,
    스포크 조인트가 0이면 호가 림을 따라 눕고(오므림),
    DEPLOY_MAX가 되면 바깥으로 펴져 갈고리가 됩니다.
    """
    phi = 2.0 * math.pi * k / 3.0
    px, pz = HUB_R * math.cos(phi), HUB_R * math.sin(phi)

    # 호 위의 3점 (시작 / 중간 / 끝) — 피벗 기준 로컬 좌표
    angs = [phi, phi + ARC_DIR * ARC_SPAN / 2.0, phi + ARC_DIR * ARC_SPAN]
    pts = [(ARC_R * math.cos(a) - px, ARC_R * math.sin(a) - pz) for a in angs]

    # ARC_DIR에 따라 '양수 조인트각 = 전개'가 되도록 축 부호를 맞춤
    axis = f"0 {ARC_DIR} 0"
    name = f"spoke_{prefix}{k}"
    cap1 = f'fromto="{_f(pts[0][0], y_off, pts[0][1])} {_f(pts[1][0], y_off, pts[1][1])}"'
    cap2 = f'fromto="{_f(pts[1][0], y_off, pts[1][1])} {_f(pts[2][0], y_off, pts[2][1])}"'
    return f"""
        <body name="{name}" pos="{_f(px, 0, pz)}">
          <joint name="{name}" type="hinge" axis="{axis}" range="0 {DEPLOY_MAX}"
                 damping="0.4" armature="0.001"/>
          <geom class="spoke" {cap1}/>
          <geom class="spoke" {cap2}/>
        </body>"""


def _wheel(prefix: str, side: int) -> str:
    """변신 바퀴 1개. side: 왼쪽 +1 / 오른쪽 -1 (스포크 오프셋 방향만 다름)"""
    y = side * WHEEL_Y
    spokes = "".join(_spoke_body(prefix, k, side * SPOKE_YOFF) for k in range(3))
    return f"""
      <body name="wheel_{prefix}" pos="{_f(WHEEL_X, y, 0)}">
        <joint name="wheel_{prefix}" type="hinge" axis="0 1 0"
               damping="0.05" armature="0.002"/>
        <geom name="hub_{prefix}" type="cylinder" size="{_f(HUB_R, 0.014)}"
              zaxis="0 1 0" mass="0.30" friction="1.0 0.005 0.0001"
              rgba="0.25 0.28 0.33 1"/>{spokes}
      </body>"""


def make_stairs() -> str:
    """계단 XML. 각 단은 바닥부터 올라오는 통짜 박스."""
    parts = []
    for i in range(N_STEPS):
        cx = STAIR_X0 + (i + 0.5) * STEP_D
        hz = (i + 1) * STEP_H / 2.0
        shade = 0.55 + 0.06 * (i % 2)
        parts.append(
            f'    <geom name="step{i}" type="box" '
            f'size="{_f(STEP_D/2, STEP_W/2, hz)}" pos="{_f(cx, 0, hz)}" '
            f'friction="1.1 0.005 0.0001" rgba="{shade} {shade} {shade+0.08} 1"/>'
        )
    # 꼭대기 평지
    top_h = N_STEPS * STEP_H
    cx = STAIR_X0 + N_STEPS * STEP_D + 0.55
    parts.append(
        f'    <geom name="platform" type="box" '
        f'size="{_f(0.55, STEP_W/2, top_h/2)}" pos="{_f(cx, 0, top_h/2)}" '
        f'friction="1.1 0.005 0.0001" rgba="0.5 0.55 0.6 1"/>'
    )
    return "\n".join(parts)


def build_xml() -> str:
    front = CHASSIS_HALF[0]
    equalities = "\n".join(
        f'    <joint joint1="spoke_{p}{k}" joint2="spoke_{p}0" polycoef="0 1 0 0 0"/>'
        for p in ("l", "r") for k in (1, 2)
    )
    return f"""<mujoco model="stair_climber_proxy">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <statistic extent="1.6" center="{_f(STAIR_X0 + 0.5, 0, 0.4)}"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.70 0.90"
             rgb2="0.90 0.93 0.97" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.35 0.40"
             rgb2="0.38 0.43 0.48" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.05"/>
  </asset>

  <default>
    <default class="spoke">
      <geom type="capsule" size="{SPOKE_RAD:.4f}"
            friction="1.3 0.005 0.0001" rgba="0.95 0.40 0.10 1"/>
    </default>
  </default>

  <worldbody>
    <light pos="1 -1 3" dir="-0.3 0.3 -1" directional="true" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="14 4 0.1" material="grid"/>

{make_stairs()}

    <body name="chassis" pos="0 0 {START_Z:.4f}">
      <freejoint/>
      <geom name="body_box" type="box" size="{_f(*CHASSIS_HALF)}"
            mass="{CHASSIS_MASS}" rgba="0.16 0.45 0.75 1"/>

      <!-- 꼬리: 등반 시 뒤로 넘어가는 것 방지 (수동, 저마찰) -->
      <geom name="tail" type="capsule" size="0.012"
            fromto="{_f(-0.13, 0, -0.02)} {_f(-0.38, 0, -0.055)}"
            friction="0.15 0.005 0.0001" rgba="0.3 0.3 0.3 1"/>
      <geom name="tail_ball" type="sphere" size="0.020"
            pos="{_f(-0.38, 0, -0.055)}"
            friction="0.15 0.005 0.0001" rgba="0.2 0.2 0.2 1"/>

      <!-- 센서 사이트: rangefinder는 사이트의 +Z축 방향으로 광선을 쏨 -->
      <site name="imu" pos="0 0 0" size="0.008" rgba="1 1 0 0.5"/>
      <site name="rf_front" pos="{_f(front + 0.015, 0, 0.02)}"
            zaxis="1 0 0" size="0.007" rgba="1 0 0 0.6"/>
      <site name="rf_diag" pos="{_f(front + 0.015, 0, 0.01)}"
            zaxis="0.766 0 -0.643" size="0.007" rgba="1 0.5 0 0.6"/>
      <site name="rf_down" pos="{_f(front + 0.015, 0, -0.02)}"
            zaxis="0 0 -1" size="0.007" rgba="1 0 1 0.6"/>

      <!-- 로봇을 따라다니는 카메라 (뷰어에서 [ ] 키로 전환) -->
      <camera name="follow" mode="trackcom" pos="0 -2.2 0.75"
              xyaxes="1 0 0 0 0.32 0.95"/>
{_wheel("l", +1)}
{_wheel("r", -1)}
    </body>
  </worldbody>

  <!-- 스포크 3개를 기계적으로 연동: 액추에이터 1개로 동시에 개폐 -->
  <equality>
{equalities}
  </equality>

  <actuator>
    <motor    name="drive_l"  joint="wheel_l"  gear="1" ctrlrange="-{DRIVE_MAX} {DRIVE_MAX}"/>
    <motor    name="drive_r"  joint="wheel_r"  gear="1" ctrlrange="-{DRIVE_MAX} {DRIVE_MAX}"/>
    <position name="deploy_l" joint="spoke_l0" kp="40" forcerange="-30 30"
              ctrlrange="0 {DEPLOY_MAX}"/>
    <position name="deploy_r" joint="spoke_r0" kp="40" forcerange="-30 30"
              ctrlrange="0 {DEPLOY_MAX}"/>
  </actuator>

  <sensor>
    <rangefinder name="rf_front" site="rf_front" cutoff="4"/>
    <rangefinder name="rf_diag"  site="rf_diag"  cutoff="4"/>
    <rangefinder name="rf_down"  site="rf_down"  cutoff="4"/>
    <gyro          name="imu_gyro" site="imu"/>
    <accelerometer name="imu_acc"  site="imu"/>
    <framequat     name="imu_quat" objtype="site" objname="imu"/>
  </sensor>

  <keyframe>
    <key name="home" qpos="0 0 {START_Z:.4f} 1 0 0 0   0 0 0 0   0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def main():
    xml = build_xml()
    with open("climber.xml", "w") as f:
        f.write(xml)
    print("climber.xml 저장 완료.")
    print("뷰어 조작법:")
    print("  - 오른쪽 Control 패널: drive_l / drive_r (바퀴 토크),")
    print("    deploy_l / deploy_r (스포크 전개, 0=오므림 ~ 1.85=펼침)")
    print("  - 마우스 왼쪽 드래그 = 회전, 휠 = 줌, Space = 일시정지,")
    print("    Backspace = 리셋, [ ] = 카메라 전환")

    import mujoco
    import mujoco.viewer
    model = mujoco.MjModel.from_xml_string(xml)
    mujoco.viewer.launch(model)


if __name__ == "__main__":
    main()
