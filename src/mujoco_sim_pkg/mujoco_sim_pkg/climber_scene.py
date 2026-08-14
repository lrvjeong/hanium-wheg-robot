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
SPOKE_RAD = 0.004  # 훅 굵기 [m] — 실측 STL(다리/스포크 부품, 두께 8mm) 그대로
                    # 반영. 캡슐 size는 반지름이라 8mm 두께 = 반지름 4mm.
                    # 밑동-끝 테이퍼는 실측 근거 없는 임의 추가였어서 제거,
                    # 실제 부품처럼 두께 균일하게 처리.
ARC_DIR    = +1     # 훅이 감기는 방향. 등반 시 갈고리가 계단에 안 걸리면 -1로!

# --- 홈-슬라이더(슬라이더-크랭크) 메커니즘 ---
# 실제 구조: 디스크의 직선 홈을 따라 슬라이더(핀)가 움직이면, 별도 고정 피벗에
# 달린 다리가 링크로 연결되어 수동으로 회전(열림/닫힘)함. 정확한 홈 각도·링크
# 길이는 실측값이 없어서, MuJoCo에서 직접 시뮬레이션 검증(막힘/특이점 없이
# 매끄럽게 0~120도 넘게 움직이는 조합)을 거쳐 확정한 값입니다.
PIVOT_R      = 0.029              # 다리 고정 피벗 반지름 (실측: 원주에서 16mm 안쪽)
GROOVE_THETA = math.radians(55)   # 홈(그루브) 방향 — 피벗 각도 기준 오프셋
LINK_LEN     = 0.024              # 피벗~커플러(슬라이더 연결점) 링크 길이
LEG_BETA     = math.radians(275)  # 다리 초기(접힘) 방향 — 접힘 시 허브 반지름
                                   # 안쪽으로 들어오도록 재계산 (기존 30도는 접힌
                                   # 상태에서도 훅이 32mm나 더 튀어나와 바닥에
                                   # 걸리며 바퀴가 안 굴러가는 원인이었음)
SLIDE_R0     = 0.03839            # 슬라이더 기준 반경(qpos=0일 때, 다리 거의 접힘)
SLIDE_MIN    = -0.025             # 슬라이더 이동범위 하한 [m] (시각 장식용)
SLIDE_MAX    = 0.0                # 슬라이더 이동범위 상한 [m] (시각 장식용)
LEG_MAX_ANGLE = math.radians(100) # 다리 힌지 최대 회전각 — 직접 구동, 실측 전 가정값
BRACKET_RAD   = 0.005             # 중앙 브래킷(고정 팔) 굵기 [m] — 가정값
HOOK_LEN     = 0.035              # 커플러 지점 너머 갈고리 곡선 길이 [m] (실측 다리 전체
                                   # 길이 ~60mm에서 LINK_LEN을 뺀 나머지 반영)
HOOK_CURVE   = math.radians(70)   # 갈고리가 안쪽으로 말리는 각도

DEPLOY_MAX = 1.85   # 스포크 전개 각도 [rad] (~106도) — 현재 미사용(참고용, 실제 범위는 SLIDE_MIN/MAX)
WHEEL_X    = 0.024  # 차체 기준 바퀴 축 위치 (앞쪽 +x, 축소된 전후길이(73.5mm)에 비례 조정)
WHEEL_Y    = 0.160  # 차체 중심에서 바퀴까지 좌우 거리 — 전체 좌우폭 절반(137.6mm)
                     # + 바퀴두께(HUB_HALF_T 22mm)로 계산, 바퀴 안쪽면이 판 바깥면에 딱 붙도록
SPOKE_YOFF = 0.030  # 스포크를 허브 바깥면 쪽으로 빼는 오프셋 (허브 두께에 맞춰 조정)

# --- 차체 / 구동 ---
# 차체 = 상판+하판 두 장 + 그 사이를 잇는 쇠막대 기둥 4개 (단일 박스 아님)
# 도면(137.6 x 73.5mm) 기준. 판 두께·기둥 정확한 위치는 도면에 명시 안 돼있어서
# 합리적인 가정값 사용 — 정확한 값 아시면 알려주세요.
PLATE_LEN    = 0.0735   # 판 1장의 전후 길이(X) [m] — 도면 73.5mm
PLATE_WID_1  = 0.1376   # 판 1장의 좌우 폭(Y) [m] — 도면 137.6mm
PLATE_GAP    = 0.0      # 두 판을 이어붙일 때 사이 틈 [m] — 맞닿는다고 가정(0)
PLATE_WID    = PLATE_WID_1 * 2 + PLATE_GAP  # 상판/하판 전체 폭(판 2장 이어붙인 길이)
PLATE_THICK  = 0.003    # 판 두께 [m] — 가정값 (미도시)
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
START_Z      = 0.050                # 초기 차체 높이 (HUB_R=0.045에 맞춰 재조정 — 바퀴가 딱 지면에 닿도록)

# ============================================================
def _f(*vals):
    return " ".join(f"{v:.4f}" for v in vals)


def _leg_slider_body(prefix: str, k: int, y_off: float) -> str:
    """다리 1개를 XML로 생성. 회전힌지로 서보가 직접 구동(안정적, 검증됨).

    '핀(파란 구슬)'은 별도 몸체가 아니라 다리(주황 스포크) 자신의 커플러
    지점(훅이 시작되는 곳)에 그냥 얹힌 장식용 geom으로 처리함 — 이러면
    핀이 실제로 스포크에 붙어서 같이 움직이는 게 시각적으로 명확해짐.
    다만 다리가 회전하면 이 점은 피벗을 중심으로 호(원)를 그리며 움직이지,
    초록색 홈 직선을 따라 미끄러지듯 움직이진 않음 — 그렇게 하려면 진짜
    슬라이더-크랭크 폐루프 물리가 필요한데, 실측 없이는 계속 특이점에
    걸려서(여러 조합 실제 검증 반복) 안정적인 방식으로 단순화한 것.
    """
    phi = 2.0 * math.pi * k / 3.0
    px, pz = PIVOT_R * math.cos(phi), PIVOT_R * math.sin(phi)

    beta_abs = phi + ARC_DIR * LEG_BETA
    link_x = LINK_LEN * math.cos(beta_abs)
    link_z = LINK_LEN * math.sin(beta_abs)

    leg_name = f"leg_{prefix}{k}"

    # 갈고리처럼 안쪽으로 말리는 곡선 3세그먼트
    n_hook = 3
    hook_geoms = []
    prev_x, prev_z = link_x, link_z
    for i in range(1, n_hook + 1):
        t = i / n_hook
        ang = beta_abs - ARC_DIR * HOOK_CURVE * t
        seg_len = HOOK_LEN * t
        hx = link_x + seg_len * math.cos(ang)
        hz = link_z + seg_len * math.sin(ang)
        hook_geoms.append(
            f'<geom class="spoke" size="{SPOKE_RAD:.4f}" '
            f'fromto="{_f(prev_x, 0, prev_z)} {_f(hx, 0, hz)}"/>'
        )
        prev_x, prev_z = hx, hz
    hook_xml = "\n          ".join(hook_geoms)

    return f"""
        <body name="{leg_name}" pos="{_f(px, y_off, pz)}">
          <joint name="{leg_name}" type="hinge" axis="0 1 0" range="0 {LEG_MAX_ANGLE}"
                 damping="0.3" armature="0.001"/>
          <geom class="spoke" size="{SPOKE_RAD:.4f}"
                fromto="0 0 0 {_f(link_x, 0, link_z)}"/>
          {hook_xml}
          <geom type="sphere" size="0.004" pos="{_f(link_x, 0, link_z)}"
                rgba="0.1 0.6 0.9 1" contype="0" conaffinity="0"/>
        </body>"""


def _groove_visual(k: int, y_off: float) -> str:
    """디스크에 파인 홈(그루브)을 시각적으로 표시 (충돌 없음, 장식용)."""
    phi = 2.0 * math.pi * k / 3.0
    gtheta_abs = phi + ARC_DIR * GROOVE_THETA
    r_a = SLIDE_R0 + SLIDE_MIN
    r_b = SLIDE_R0 + SLIDE_MAX
    x0, z0 = r_a * math.cos(gtheta_abs), r_a * math.sin(gtheta_abs)
    x1, z1 = r_b * math.cos(gtheta_abs), r_b * math.sin(gtheta_abs)
    return (f'<geom type="capsule" size="0.0015" '
            f'fromto="{_f(x0, y_off, z0)} {_f(x1, y_off, z1)}" '
            f'rgba="0.15 0.7 0.15 0.5" contype="0" conaffinity="0"/>')


def _bracket_arm(k: int, y_off: float) -> str:
    """중앙 브래킷(스포크 팔): 디스크 중심에서 다리 피벗 지점까지 이어지는
    고정된(회전하지 않는, 디스크와 함께 도는) 팔. 실제 CAD 이미지처럼
    다리가 허브 림에서 바로 튀어나오는 게 아니라, 이 중앙 브래킷 끝의
    핀(피벗)에서 회전하도록 시각적 구조를 맞춤."""
    phi = 2.0 * math.pi * k / 3.0
    px, pz = PIVOT_R * math.cos(phi), PIVOT_R * math.sin(phi)
    return (f'<geom type="capsule" size="{BRACKET_RAD:.4f}" '
            f'fromto="0 {y_off:.4f} 0 {_f(px, y_off, pz)}" '
            f'rgba="0.35 0.38 0.42 1"/>')


def _wheel(prefix: str, side: int) -> str:
    """변신 바퀴 1개. side: 왼쪽 +1 / 오른쪽 -1 (스포크 오프셋 방향만 다름)"""
    y = side * WHEEL_Y
    legs = "".join(_leg_slider_body(prefix, k, side * SPOKE_YOFF) for k in range(3))
    grooves = "".join(_groove_visual(k, side * SPOKE_YOFF) for k in range(3))
    brackets = "".join(_bracket_arm(k, side * SPOKE_YOFF) for k in range(3))
    return f"""
      <body name="wheel_{prefix}" pos="{_f(WHEEL_X, y, 0)}">
        <joint name="wheel_{prefix}" type="hinge" axis="0 1 0"
               damping="0.05" armature="0.002"/>
        <geom name="hub_{prefix}" type="cylinder" size="{_f(HUB_R, HUB_HALF_T)}"
              zaxis="0 1 0" mass="0.30" friction="1.0 0.005 0.0001"
              rgba="0.25 0.28 0.33 1"/>
        {brackets}{grooves}{legs}
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
    # 다리 3개를 하나로 동기화 (액추에이터 1개로 동시 구동)
    equalities = "\n".join(
        f'    <joint joint1="leg_{p}{k}" joint2="leg_{p}0" polycoef="0 1 0 0 0"/>'
        for p in ("l", "r") for k in (1, 2)
    )
    return f"""<mujoco model="stair_climber_proxy">
  <compiler angle="radian" inertiafromgeom="true" meshdir="mesh_assets"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81" iterations="100" noslip_iterations="5"/>
  <statistic extent="1.6" center="{_f(STAIR_X0 + 0.5, 0, 0.4)}"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.70 0.90"
             rgb2="0.90 0.93 0.97" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.35 0.40"
             rgb2="0.38 0.43 0.48" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.05"/>
    <mesh name="tail_plate" file="tail.stl"/>
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

      <!-- 꼬리: 실제 STL 평판(160mm 폭 x 200mm 길이 x 5mm 두께). 하판 후방 중앙에 장착
           (도면상 "샷시 하판 후방 중앙"). STL 로컬좌표: x=폭, y=길이(뒤로 뻗는 방향), z=두께.
           Z축 +90도 회전, 폭은 중앙정렬. -->
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

  <!-- 스포크 3개를 기계적으로 연동: 액추에이터 1개로 동시에 개폐 -->
  <equality>
{equalities}
  </equality>

  <actuator>
    <motor    name="drive_l"  joint="wheel_l"  gear="1" ctrlrange="-{DRIVE_MAX_BOOST} {DRIVE_MAX_BOOST}"/>
    <motor    name="drive_r"  joint="wheel_r"  gear="1" ctrlrange="-{DRIVE_MAX_BOOST} {DRIVE_MAX_BOOST}"/>
    <position name="deploy_l" joint="leg_l0" kp="40" forcerange="-30 30"
              ctrlrange="0 {LEG_MAX_ANGLE}"/>
    <position name="deploy_r" joint="leg_r0" kp="40" forcerange="-30 30"
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
    mujoco.viewer.launch(model)


if __name__ == "__main__":
    main()
