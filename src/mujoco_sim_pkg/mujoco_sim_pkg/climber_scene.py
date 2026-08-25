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
HUB_HALF_T = 0.003  # 허브 원판 반두께 [m] — 실측 STL 기준(디스크 두께 6mm)
SPOKE_RAD = 0.004  # 훅 굵기 [m] — 실측 STL(다리/스포크 부품, 두께 8mm) 그대로 반영.
                    # 캡슐 size는 반지름이라 8mm 두께 = 반지름 4mm.

WHEEL_X    = 0.024  # 차체 기준 바퀴 축 위치 (앞쪽 +x, 축소된 전후길이(73.5mm)에 비례 조정)
WHEEL_Y    = 0.1406 # 차체 중심에서 바퀴까지 좌우 거리 — 전체 좌우폭 절반(137.6mm)
                     # + 디스크 반두께(3mm, 실측)로 계산, 바퀴 안쪽면이 판 바깥면에 딱 붙도록

# --- 차체 / 구동 ---
# 차체 = 상판+하판 두 장 + 그 사이를 잇는 쇠막대 기둥 4개 (단일 박스 아님)
# 도면(137.6 x 73.5mm) 기준. 판 두께·기둥 정확한 위치는 도면에 명시 안 돼있어서
# 합리적인 가정값 사용 — 정확한 값 아시면 알려주세요.
PLATE_LEN    = 0.0735   # 판 1장의 전후 길이(X) [m] — 도면 73.5mm
PLATE_WID_1  = 0.1376   # 판 1장의 좌우 폭(Y) [m] — 도면 137.6mm
PLATE_GAP    = 0.0      # 두 판을 이어붙일 때 사이 틈 [m] — 맞닿는다고 가정(0)
PLATE_WID    = PLATE_WID_1 * 2 + PLATE_GAP  # 상판/하판 전체 폭(판 2장 이어붙인 길이)
PLATE_THICK  = 0.009    # 판 두께 [m] — 실측 STL(샷시.STL) 기준, 기존 3mm는 가정값이었음
ROD_HEIGHT   = 0.04     # 상판-하판 간격(기둥 높이) [m] — "약 4cm" 명시대로
ROD_RADIUS   = 0.004    # 기둥(쇠막대) 반지름 [m] — 가정값 (3/8" 볼트 구멍 기준 추정)
ROD_X_OFF    = 0.025    # 기둥 x위치(판 1장 내부, 중심 기준 대칭) [m] — 가정값
ROD_Y_OFF    = 0.057    # 기둥 y위치(판 1장 내부, 중심 기준 대칭) [m] — 도면 "12mm" 홀 오프셋 기반 추정
CHASSIS_HALF = (PLATE_LEN/2, PLATE_WID/2, ROD_HEIGHT/2 + PLATE_THICK)  # 센서 배치 등에 쓰는 전체 절반크기 근사
CHASSIS_MASS = 2.0                  # [kg] — 상판/하판/기둥에 분배

# --- CygLiDAR (실측 37.4 x 37.4 x 27mm, 상판/하판 사이 중앙에 전방을 보도록 장착) ---
LIDAR_BOX = (0.0374, 0.0374, 0.027)  # (가로, 세로, 높이) [m]

DRIVE_MAX    = 5.0                  # 바퀴 모터 최대 토크 [N·m] — 평지 주행/등반 시 기본값
DRIVE_MAX_BOOST = 12.0              # 고토크 모드 최대 토크 [N·m] — 1~2cm 무변형 단차를
                                     # 스포크 전개 없이 힘으로 밀고 올라갈 때 사용
START_Z      = 0.050                # 초기 차체 높이 — 디스크 반지름(45mm) 기준.
                                     # 평지에서는 디스크가 땅에 닿아 구르고,
                                     # 단차 만났을 때만 스포크가 펼쳐짐.

# ============================================================
def _f(*vals):
    return " ".join(f"{v:.4f}" for v in vals)


# 실측 STL에서 뽑은 다리 피벗 위치 (디스크 중심 기준, m)
# 실측 STL에서 뽑은 다리 피벗 위치 (디스크 중심 기준, m)
LEG_PIVOTS = {
    1: (-0.01354, 0.00280, 0.01946),
    2: (0.02366, 0.00280, 0.00199),
    3: (-0.01006, 0.00280, -0.02149),
}
LEG_MAX_ANGLE = math.radians(29)  # 실제 메시로 검증된 최대 회전각(28.6도 근처에서
                                   # 최대로 펼쳐지고, 그 이후는 다시 오므라듦)
LEG_YOFF = 0.001  # 다리 메시를 디스크보다 살짝 바깥쪽으로 — 안 밀면 차체 판과 겹침


def _leg_mesh_body(prefix: str, k: int, side: int) -> str:
    """다리 1개 — 실측 STL(leg1/2/3.stl) 그대로 사용. 오른쪽 바퀴는 Y축
    반전된 메시(leg{k}_r) 사용 — 미러링 없이 동일 메시를 양쪽에 쓰면
    오른쪽 다리가 반대 방향으로 튀어나와 차체 판과 심하게 간섭/잠기는
    문제가 실제로 확인됨 (미러링이 시각적 취향이 아니라 기능적으로 필수).
    남아있는 근소한 좌우 비대칭(주행 시 약간의 드리프트)은 접촉 계산의
    미세한 비선형성에서 오는 것으로 보이며, 실제 로봇도 완벽 대칭은
    아니므로 정책이 IMU 피드백으로 스스로 보정하도록 두는 게 합리적임."""
    px, py, pz = LEG_PIVOTS[k]
    leg_name = f"leg_{prefix}{k}"
    mesh_name = f"leg{k}" if prefix == "l" else f"leg{k}_r"
    py_use = (py if prefix == "l" else -py) + side * LEG_YOFF
    return f"""
        <body name="{leg_name}" pos="{_f(px, py_use, pz)}">
          <joint name="{leg_name}" type="hinge" axis="0 -1 0" range="0 {LEG_MAX_ANGLE}"
                 damping="0.3" armature="0.001"/>
          <geom type="mesh" mesh="{mesh_name}" class="spoke"/>
        </body>"""


def _wheel(prefix: str, side: int) -> str:
    """변신 바퀴 1개 — 실측 STL(disk.stl, leg1/2/3.stl) 그대로 조립.
    좌우 바퀴에 동일한 다리 메시(미러링 없음) 사용."""
    y = side * WHEEL_Y
    legs = "".join(_leg_mesh_body(prefix, k, side) for k in (1, 2, 3))
    return f"""
      <body name="wheel_{prefix}" pos="{_f(WHEEL_X, y, 0)}">
        <joint name="wheel_{prefix}" type="hinge" axis="0 1 0"
               damping="0.15" armature="0.006"/>
        <geom name="hub_{prefix}" type="mesh" mesh="disk" mass="0.30"
              friction="1.0 0.005 0.0001" rgba="0.25 0.28 0.33 1"/>
        {legs}
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


def _chassis_plates_and_rods() -> str:
    """상판 2장(좌우 이어붙임) + 하판 2장 + 기둥 4개(판 세트당 앞/뒤 1개씩, 좌우
    중앙정렬) 생성. 각 판은 도면 그대로(전후 73.5mm x 좌우 137.6mm),
    y=+-PLATE_WID_1/2에 나란히 배치해서 전체 좌우폭이 PLATE_WID(275.2mm)가
    되도록 함."""
    lines = []
    z_top = ROD_HEIGHT / 2 + PLATE_THICK / 2
    z_bot = -z_top
    for side, y_center in (("l", PLATE_WID_1 / 2), ("r", -PLATE_WID_1 / 2)):
        lines.append(
            f'      <geom name="plate_top_{side}" type="box" '
            f'size="{_f(PLATE_LEN/2, PLATE_WID_1/2, PLATE_THICK/2)}" '
            f'pos="{_f(0, y_center, z_top)}" '
            f'mass="{CHASSIS_MASS*0.2:.3f}" rgba="0.55 0.58 0.62 1"/>'
        )
        lines.append(
            f'      <geom name="plate_bottom_{side}" type="box" '
            f'size="{_f(PLATE_LEN/2, PLATE_WID_1/2, PLATE_THICK/2)}" '
            f'pos="{_f(0, y_center, z_bot)}" '
            f'mass="{CHASSIS_MASS*0.2:.3f}" rgba="0.55 0.58 0.62 1"/>'
        )
        # 판마다 앞/뒤 기둥 1개씩 (좌우 중앙정렬, 총 4개)
        for xi, xs in (("f", 1), ("b", -1)):
            lines.append(
                f'      <geom name="rod_{side}_{xi}" type="cylinder" '
                f'size="{_f(ROD_RADIUS, ROD_HEIGHT/2)}" '
                f'pos="{_f(xs*ROD_X_OFF, y_center, 0)}" '
                f'mass="{CHASSIS_MASS*0.05:.3f}" rgba="0.3 0.3 0.32 1"/>'
            )
    return "\n".join(lines)


def _lidar_sites(front: float) -> str:
    """차체 전방에 장착된 3D LiDAR를 rangefinder 격자(LIDAR_ROWS x LIDAR_COLS)로 근사.
    각 광선의 방향은 LIDAR_MOUNT_TILT(아래로 기운 각도)를 중심으로
    수평/수직 화각 안에서 격자로 퍼짐."""
    lines = []
    base_pos = (front + 0.002, 0, 0)  # CygLiDAR 실장 위치: 상판-하판 사이 중앙(z=0), 전방 정중앙
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
    # ToF(근접센서): 정면 정중앙, 라이다와 같은 자리에서 수평으로 짧은 거리만 측정
    # (실측 sensor_fusion_node.py의 tof_near_limit=0.10m 근거리 폴백용)
    lines.append(
        f'      <site name="tof_0" pos="{_f(*base_pos)}" '
        f'zaxis="1 0 0" size="0.005" rgba="1 1 0 0.35"/>'
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
    lines.append(f'    <rangefinder name="tof_0" site="tof_0" cutoff="0.15"/>')
    return "\n".join(lines)


def build_xml(step_h: float = STEP_H) -> str:
    front = CHASSIS_HALF[0]
    rear = -CHASSIS_HALF[0]
    # 다리 3개를 하나로 동기화 (액추에이터 1개로 동시 구동)
    equalities = "\n".join(
        f'    <joint joint1="leg_{p}{k}" joint2="leg_{p}1" polycoef="0 1 0 0 0"/>'
        for p in ("l", "r") for k in (2, 3)
    )
    return f"""<mujoco model="stair_climber_proxy">
  <compiler angle="radian" inertiafromgeom="true" meshdir="mesh_assets"/>
  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81" iterations="150" noslip_iterations="5" cone="elliptic"/>
  <statistic extent="1.6" center="{_f(STAIR_X0 + 0.5, 0, 0.4)}"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.70 0.90"
             rgb2="0.90 0.93 0.97" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.35 0.40"
             rgb2="0.38 0.43 0.48" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.05"/>
    <mesh name="tail_plate" file="tail.stl"/>
    <mesh name="disk" file="disk.stl"/>
    <mesh name="leg1" file="leg1.stl"/>
    <mesh name="leg2" file="leg2.stl"/>
    <mesh name="leg3" file="leg3.stl"/>
    <mesh name="leg1_r" file="leg1.stl" scale="1 -1 1"/>
    <mesh name="leg2_r" file="leg2.stl" scale="1 -1 1"/>
    <mesh name="leg3_r" file="leg3.stl" scale="1 -1 1"/>
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
      <!-- 상판 2장(좌우 이어붙임) + 하판 2장(좌우 이어붙임) + 기둥 8개(판 세트당 4개)
           로 구성된 실제 샷시 구조. z=0을 상판-하판 사이 정중앙으로 둠. -->
{_chassis_plates_and_rods()}
      <!-- CygLiDAR (37.4 x 37.4 x 27mm) — 상판/하판 사이 중앙, 전방 정중앙 -->
      <geom name="cyglidar_box" type="box"
            size="{_f(LIDAR_BOX[0]/2, LIDAR_BOX[1]/2, LIDAR_BOX[2]/2)}"
            pos="{_f(front - LIDAR_BOX[0]/2 - 0.005, 0, 0)}"
            mass="0.03" rgba="0.15 0.15 0.15 1"/>

      <!-- 꼬리: 실제 STL 평판(160mm 폭 x 200mm 길이 x 5mm 두께). 하판 후방 중앙에 장착. -->
      <geom name="tail" type="mesh" mesh="tail_plate"
            friction="0.15 0.005 0.0001" rgba="0.3 0.3 0.3 1"/>

      <!-- IMU 2개: 양쪽 바퀴 축 중심에 각각 장착 (차체에 고정, 바퀴 자체가 아님 —
           베어링을 통해 축 자체는 회전하지 않으므로 차체 프레임의 site로 배치).
           imu_l이 기존 코드 호환용 기본 센서, imu_r은 예비/이중화용으로 추가만 해둠
           (둘 다 같은 강체(차체)에 고정돼 있어 자세값 자체는 imu_l과 동일하게 나옴). -->
      <site name="imu_l" pos="{_f(WHEEL_X, WHEEL_Y, 0)}" size="0.008" rgba="1 1 0 0.5"/>
      <site name="imu_r" pos="{_f(WHEEL_X, -WHEEL_Y, 0)}" size="0.008" rgba="1 0.6 0 0.5"/>
{_lidar_sites(front)}

      <!-- 로봇을 따라다니는 카메라 (뷰어에서 [ ] 키로 전환) -->
      <camera name="follow" mode="trackcom" pos="0 -2.2 0.75"
              xyaxes="1 0 0 0 0.32 0.95"/>
{_wheel("l", +1)}
{_wheel("r", -1)}
    </body>
  </worldbody>

  <!-- 같은 바퀴의 다리끼리는 충돌 계산 제외 — 실제로는 볼트로 결합되는
       부분이라 메시가 서로 겹쳐있는 게 정상(부품 경계에서 살짝 겹침) -->
  <contact>
    <exclude body1="leg_l1" body2="leg_l2"/>
    <exclude body1="leg_l1" body2="leg_l3"/>
    <exclude body1="leg_l2" body2="leg_l3"/>
    <exclude body1="leg_r1" body2="leg_r2"/>
    <exclude body1="leg_r1" body2="leg_r3"/>
    <exclude body1="leg_r2" body2="leg_r3"/>
  </contact>

  <!-- 스포크 3개를 기계적으로 연동: 액추에이터 1개로 동시에 개폐 -->
  <equality>
{equalities}
  </equality>

  <actuator>
    <motor    name="drive_l"  joint="wheel_l"  gear="1" ctrlrange="-{DRIVE_MAX_BOOST} {DRIVE_MAX_BOOST}"/>
    <motor    name="drive_r"  joint="wheel_r"  gear="1" ctrlrange="-{DRIVE_MAX_BOOST} {DRIVE_MAX_BOOST}"/>
    <position name="deploy_l" joint="leg_l1" kp="40" forcerange="-30 30"
              ctrlrange="0 {LEG_MAX_ANGLE}"/>
    <position name="deploy_r" joint="leg_r1" kp="40" forcerange="-30 30"
              ctrlrange="0 {LEG_MAX_ANGLE}"/>
  </actuator>

  <sensor>
{_lidar_sensors()}
    <gyro          name="imu_gyro" site="imu_l"/>
    <accelerometer name="imu_acc"  site="imu_l"/>
    <framequat     name="imu_quat" objtype="site" objname="imu_l"/>
    <gyro          name="imu_r_gyro" site="imu_r"/>
    <accelerometer name="imu_r_acc"  site="imu_r"/>
    <framequat     name="imu_r_quat" objtype="site" objname="imu_r"/>
  </sensor>

  <keyframe>
    <key name="home" qpos="0 0 {START_Z:.4f} 1 0 0 0   0 0   0 0 0 0 0 0"/>
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
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)  # home keyframe 적용 (안 하면 z=0에서 시작해 땅에 파묻힘)
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
