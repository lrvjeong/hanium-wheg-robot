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
N_STEPS   = 1       # 단수 — 단차 하나만 (5단 계단 대신 단일 층 단차)
STEP_H    = 0.035   # 단높이 [m] — 목표 성능: 무변형 2cm 통과, 2~5cm는 wheg 변형으로 극복
                     # (기본값은 도메인 랜덤화 학습 범위(2~5cm)의 중간값 정도로 설정)
STEP_D    = 0.30    # 단깊이 [m]
STEP_W    = 1.6     # 계단 폭 [m]
STAIR_X0  = 1.10    # 로봇 시작점에서 첫 단까지 거리 [m]

# --- 3D LiDAR (CygLiDAR 등 소형 3D ToF 모듈 근사) ---
LIDAR_ROWS       = 5     # 세로 방향 광선 수
LIDAR_COLS       = 5     # 가로 방향 광선 수
LIDAR_FOV_H      = math.radians(60)   # 수평 화각
LIDAR_FOV_V      = math.radians(45)   # 수직 화각
LIDAR_MOUNT_TILT = math.radians(25)   # 정면 기준 아래로 기울인 각도 (계단 인식용)
LIDAR_MIN_RANGE  = 0.05   # 실제 부품 최소 인식거리 50mm — 이보다 가까우면 판독 불가로 처리
LIDAR_MAX_RANGE  = 2.5    # 최대 인식거리 [m] (부품 스펙에 맞춰 조정하세요)

# --- 바퀴 / 스포크 ---
HUB_R      = 0.045  # 바퀴 허브 반지름 [m] — 실제 로봇 디스크 반지름(45mm)과 일치.
                     # 무변형으로 2cm까지는 통과 가능(HUB_R > 2cm 여유)
HUB_HALF_T = 0.022  # 허브 원판 반두께 [m]
ARC_R      = 0.065  # 스포크 호의 반지름 — 훅 전개 시 5cm 단차까지 닿을 수 있는 여유 반영
ARC_SPAN   = math.radians(95)   # 스포크 호가 덮는 각도
ARC_SEGMENTS = 5    # 호를 근사하는 캡슐 개수 (기존 2 → 부드러운 곡선을 위해 증가)
SPOKE_RAD_BASE = 0.013  # 훅 밑동(허브 쪽) 굵기 [m] — 더 두껍게
SPOKE_RAD_TIP  = 0.006  # 훅 끝(발톱 쪽) 굵기 [m] — 가늘게 테이퍼
ARC_DIR    = +1     # 훅이 감기는 방향. 등반 시 갈고리가 계단에 안 걸리면 -1로!
DEPLOY_MAX = 1.85   # 스포크 전개 각도 [rad] (~106도)
WHEEL_X    = 0.045  # 차체 기준 바퀴 축 위치 (앞쪽 +x, 축소된 차체에 맞춰 비례 조정)
WHEEL_Y    = 0.120  # 차체 중심에서 바퀴까지 좌우 거리 (작아진 허브에 맞춰 재조정)
SPOKE_YOFF = 0.030  # 스포크를 허브 바깥면 쪽으로 빼는 오프셋 (허브 두께에 맞춰 조정)

# --- 차체 / 구동 ---
CHASSIS_HALF = (0.069, 0.09, 0.035)  # 차체 박스 절반 치수 (x, y, z) — 실측 도면(137.6mm 전장) 반영
CHASSIS_MASS = 2.0                  # [kg]
DRIVE_MAX    = 5.0                  # 바퀴 모터 최대 토크 [N·m] — 평지 주행/등반 시 기본값
DRIVE_MAX_BOOST = 12.0              # 고토크 모드 최대 토크 [N·m] — 1~2cm 무변형 단차를
                                     # 스포크 전개 없이 힘으로 밀고 올라갈 때 사용
START_Z      = 0.050                # 초기 차체 높이 (HUB_R=0.045에 맞춰 재조정 — 바퀴가 딱 지면에 닿도록)

# ============================================================
def _f(*vals):
    return " ".join(f"{v:.4f}" for v in vals)


def _spoke_body(prefix: str, k: int, y_off: float) -> str:
    """스포크 1개(호를 캡슐 ARC_SEGMENTS개로 근사)를 XML로 생성.

    피벗은 허브 림 위 (반지름 HUB_R, 각도 phi)에 있고,
    스포크 조인트가 0이면 호가 림을 따라 눕고(오므림),
    DEPLOY_MAX가 되면 바깥으로 펴져 갈고리가 됩니다.

    v2: 세그먼트를 늘려 각진 2-캡슐 근사 대신 부드러운 곡선을 만들고,
    밑동(SPOKE_RAD_BASE)에서 끝(SPOKE_RAD_TIP)으로 갈수록 가늘어지는
    테이퍼를 줘서 더 유기적인 갈고리/발톱 느낌을 냅니다.
    """
    phi = 2.0 * math.pi * k / 3.0
    px, pz = HUB_R * math.cos(phi), HUB_R * math.sin(phi)

    # 호 위의 (ARC_SEGMENTS+1)개 점 — 피벗 기준 로컬 좌표
    n = ARC_SEGMENTS
    angs = [phi + ARC_DIR * ARC_SPAN * (i / n) for i in range(n + 1)]
    pts = [(ARC_R * math.cos(a) - px, ARC_R * math.sin(a) - pz) for a in angs]

    # ARC_DIR에 따라 '양수 조인트각 = 전개'가 되도록 축 부호를 맞춤
    axis = f"0 {ARC_DIR} 0"
    name = f"spoke_{prefix}{k}"

    geoms = []
    for i in range(n):
        a, b = pts[i], pts[i + 1]
        # 세그먼트 중간 지점 반지름을 선형 보간해 테이퍼 형성
        t_mid = (i + 0.5) / n
        r = SPOKE_RAD_BASE + (SPOKE_RAD_TIP - SPOKE_RAD_BASE) * t_mid
        fromto = f'fromto="{_f(a[0], y_off, a[1])} {_f(b[0], y_off, b[1])}"'
        geoms.append(f'<geom class="spoke" size="{r:.4f}" {fromto}/>')
    geoms_xml = "\n          ".join(geoms)

    return f"""
        <body name="{name}" pos="{_f(px, 0, pz)}">
          <joint name="{name}" type="hinge" axis="{axis}" range="0 {DEPLOY_MAX}"
                 damping="0.4" armature="0.001"/>
          {geoms_xml}
        </body>"""


def _wheel(prefix: str, side: int) -> str:
    """변신 바퀴 1개. side: 왼쪽 +1 / 오른쪽 -1 (스포크 오프셋 방향만 다름)"""
    y = side * WHEEL_Y
    spokes = "".join(_spoke_body(prefix, k, side * SPOKE_YOFF) for k in range(3))
    return f"""
      <body name="wheel_{prefix}" pos="{_f(WHEEL_X, y, 0)}">
        <joint name="wheel_{prefix}" type="hinge" axis="0 1 0"
               damping="0.05" armature="0.002"/>
        <geom name="hub_{prefix}" type="cylinder" size="{_f(HUB_R, HUB_HALF_T)}"
              zaxis="0 1 0" mass="0.30" friction="1.0 0.005 0.0001"
              rgba="0.25 0.28 0.33 1"/>{spokes}
      </body>"""


def make_stairs(step_h: float = STEP_H) -> str:
    """계단 XML. 각 단은 바닥부터 올라오는 통짜 박스.
    step_h를 인자로 받아서, 매 에피소드 다른 높이로 재생성할 수 있게 함
    (도메인 랜덤화: 학습 중 계단 높이를 무작위로 바꿔가며 강건한 정책 학습)."""
    parts = []
    for i in range(N_STEPS):
        cx = STAIR_X0 + (i + 0.5) * STEP_D
        hz = (i + 1) * step_h / 2.0
        shade = 0.55 + 0.06 * (i % 2)
        parts.append(
            f'    <geom name="step{i}" type="box" '
            f'size="{_f(STEP_D/2, STEP_W/2, hz)}" pos="{_f(cx, 0, hz)}" '
            f'friction="1.1 0.005 0.0001" rgba="{shade} {shade} {shade+0.08} 1"/>'
        )
    # 꼭대기 평지
    top_h = N_STEPS * step_h
    cx = STAIR_X0 + N_STEPS * STEP_D + 0.55
    parts.append(
        f'    <geom name="platform" type="box" '
        f'size="{_f(0.55, STEP_W/2, top_h/2)}" pos="{_f(cx, 0, top_h/2)}" '
        f'friction="1.1 0.005 0.0001" rgba="0.5 0.55 0.6 1"/>'
    )
    return "\n".join(parts)


def _lidar_sites(front: float) -> str:
    """차체 전방에 장착된 3D LiDAR를 rangefinder 격자(LIDAR_ROWS x LIDAR_COLS)로 근사.
    각 광선의 방향은 LIDAR_MOUNT_TILT(아래로 기운 각도)를 중심으로
    수평/수직 화각 안에서 격자로 퍼짐."""
    lines = []
    base_pos = (front + 0.02, 0, 0.03)
    for r in range(LIDAR_ROWS):
        v = -LIDAR_FOV_V / 2 + LIDAR_FOV_V * (r / max(LIDAR_ROWS - 1, 1))
        pitch = LIDAR_MOUNT_TILT + v  # 아래로 기울수록 +
        for c in range(LIDAR_COLS):
            h = -LIDAR_FOV_H / 2 + LIDAR_FOV_H * (c / max(LIDAR_COLS - 1, 1))
            zx = math.cos(pitch) * math.cos(h)
            zy = math.cos(pitch) * math.sin(h)
            zz = -math.sin(pitch)
            name = f"lidar_{r}_{c}"
            lines.append(
                f'      <site name="{name}" pos="{_f(*base_pos)}" '
                f'zaxis="{_f(zx, zy, zz)}" size="0.005" rgba="0 1 1 0.35"/>'
            )
    return "\n".join(lines)


def _lidar_sensors() -> str:
    lines = []
    for r in range(LIDAR_ROWS):
        for c in range(LIDAR_COLS):
            name = f"lidar_{r}_{c}"
            lines.append(
                f'    <rangefinder name="{name}" site="{name}" cutoff="{LIDAR_MAX_RANGE}"/>'
            )
    return "\n".join(lines)


def build_xml(step_h: float = STEP_H) -> str:
    front = CHASSIS_HALF[0]
    rear = -CHASSIS_HALF[0]
    tail_x0 = rear + 0.005          # 후미 살짝 안쪽에서 시작
    tail_x1 = tail_x0 - 0.1875      # 기존 꼬리 돌출 길이(0.1875m) 유지
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
      <geom type="capsule"
            friction="1.3 0.005 0.0001" rgba="0.95 0.40 0.10 1"/>
    </default>
  </default>

  <worldbody>
    <light pos="1 -1 3" dir="-0.3 0.3 -1" directional="true" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="14 4 0.1" material="grid"/>

{make_stairs(step_h)}

    <body name="chassis" pos="0 0 {START_Z:.4f}">
      <freejoint/>
      <geom name="body_box" type="box" size="{_f(*CHASSIS_HALF)}"
            mass="{CHASSIS_MASS}" rgba="0.16 0.45 0.75 1"/>

      <!-- 꼬리: 등반 시 뒤로 넘어가는 것 방지 (수동, 저마찰). 차체 후미(-CHASSIS_HALF[0])
           기준으로 위치 계산 — 차체 크기가 바뀌어도 자동으로 따라감 -->
      <geom name="tail" type="capsule" size="0.012"
            fromto="{_f(tail_x0, 0, -0.02)} {_f(tail_x1, 0, -0.04625)}"
            friction="0.15 0.005 0.0001" rgba="0.3 0.3 0.3 1"/>
      <geom name="tail_ball" type="sphere" size="0.020"
            pos="{_f(tail_x1, 0, -0.04625)}"
            friction="0.15 0.005 0.0001" rgba="0.2 0.2 0.2 1"/>

      <!-- 3D LiDAR 격자 (실제 부품 근사: LIDAR_ROWS x LIDAR_COLS 개의 rangefinder) -->
      <site name="imu" pos="0 0 0" size="0.008" rgba="1 1 0 0.5"/>
{_lidar_sites(front)}

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
    <motor    name="drive_l"  joint="wheel_l"  gear="1" ctrlrange="-{DRIVE_MAX_BOOST} {DRIVE_MAX_BOOST}"/>
    <motor    name="drive_r"  joint="wheel_r"  gear="1" ctrlrange="-{DRIVE_MAX_BOOST} {DRIVE_MAX_BOOST}"/>
    <position name="deploy_l" joint="spoke_l0" kp="40" forcerange="-30 30"
              ctrlrange="0 {DEPLOY_MAX}"/>
    <position name="deploy_r" joint="spoke_r0" kp="40" forcerange="-30 30"
              ctrlrange="0 {DEPLOY_MAX}"/>
  </actuator>

  <sensor>
{_lidar_sensors()}
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
