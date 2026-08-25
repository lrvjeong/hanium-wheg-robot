# -*- coding: utf-8 -*-
"""
wheg_env.py — climber_scene.py(MuJoCo MJCF)를 감싸는 Gymnasium 환경

v5 변경점 (상태기계 세분화):
  * 전개 상태기계를 요청하신 순서대로 재구성:
    DRIVE(주행) -> MEASURE(높이 측정) -> STOP(정지) -> DEPLOY(스포크 펼치기)
    -> CLIMB(등반) -> RETRACT(원래대로 복귀) -> DRIVE(주행)
  * MEASURE/STOP/DEPLOY/RETRACT 구간에서는 RL의 drive 명령을 무시하고 강제로
    0으로 처리함 (그 구간의 거동은 정책이 학습할 필요 없이 결정적으로 동작).
    RL은 DRIVE(평지 주행), CLIMB(등반 중 추진력) 구간에서만 drive_l/drive_r을
    실제로 결정함.
  * MEASURE 단계는 현재는 감지 후 잠깐 정지하는 자리만 잡아둔 상태이고,
    실제 "높이를 계산해서 필요하면 전개를 건너뛴다" 같은 로직은 TODO로 남겨둠
    (원하시면 다음에 추가해드릴 수 있어요).

주의: HUB_R, ARC_R, CHASSIS_HALF 등 로봇 본체/디스크 치수는 절대 수정하지
않습니다 (climber_scene.py 쪽에 고정값으로 명시되어 있음).

Observation (LIDAR_ROWS*LIDAR_COLS + 7 차원, 기본 5x5+7=32D):
    [0 .. R*C-1]  3D LiDAR 격자 거리값 [m]
    [-7] pitch, [-6] pitch_rate, [-5] roll, [-4] roll_rate,
    [-3] yaw(상대), [-2] yaw_rate, [-1] deploy_angle

Action (2D, [-1, 1] 정규화 → 내부에서 실제 범위로 스케일):
    [0] drive_l_cmd   왼쪽 바퀴 토크 명령 (DRIVE/CLIMB 상태에서만 실제 반영)
    [1] drive_r_cmd   오른쪽 바퀴 토크 명령 (DRIVE/CLIMB 상태에서만 실제 반영)
"""
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco

from climber_scene import (
    build_xml, DRIVE_MAX, DRIVE_MAX_BOOST, N_STEPS, STEP_H,
    STEP_D, STAIR_X0, LEG_MAX_ANGLE,
    LIDAR_ROWS, LIDAR_COLS, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE,
    LIDAR_FOV_H, LIDAR_FOV_V, LIDAR_MOUNT_TILT, START_Z,
)

# ===== 튜닝 상수 =====
PITCH_FLIP = 1.2
CLIMB_PITCH_LIMIT = 0.7  # 등반 중 높이 보상을 받을 수 있는 최대 피치[rad](~40도).
                          # 정상적인 등반은 어느 정도 앞으로/뒤로 기울지만, 이걸
                          # 넘는 과도한 피치(뒤로 확 젖혀지는 웰리 동작 등)는
                          # 진짜 등반이 아니라 보상 편법일 가능성이 높아 배제.
ROLL_FLIP = 1.0
CTRL_COST_W = 0.005   # 기존 0.01 -> 하향 (토크 쓰는 걸 너무 겁내지 않도록)
PITCH_PENALTY_W = 0.3
ROLL_PENALTY_W = 0.5
YAW_PENALTY_W = 2.5  # 기존 0.5 -> 5배 상향. 직진 유지가 최우선 과제로 확인돼서 강화
FORWARD_REWARD_W = 15.0  # 기존 8.0 -> 상향 (전진을 훨씬 매력적으로)
IDLE_PENALTY_W = 2.0     # 움직여야 하는 구간(DRIVE/CLIMB/PUSH)에서 실제로
                          # 거의 안 움직이면 매 스텝 부과되는 페널티
ACTION_RATE_PENALTY_W = 0.3  # 직전 스텝 액션 대비 급격한 변화에 부과되는
                              # 페널티 - 좌우 바퀴가 들쭉날쭉하게 튀는
                              # 부산스러운 움직임을 줄이기 위함
IDLE_DIST_THRESH = 0.001  # 이 정도(1mm) 이하 전진이면 "정지 중"으로 간주
APPROACH_SPEED_LIMIT = 0.004   # 계단 30cm 이내(DRIVE)에서 스텝당 허용 전진거리 상한 [m]
APPROACH_SPEED_PENALTY_W = 40.0  # 이 상한을 넘는 초과분에 대한 페널티 가중치 —
                                   # 감지 직전 관성을 줄여서 MEASURE 중 라이다
                                   # 높이 추정이 부정확해지는 것을 방지

# --- 센서 노이즈 (도메인 랜덤화, sim-to-real 견고성용) ---
LIDAR_NOISE_STD = 0.01   # 라이다 거리 측정 노이즈 표준편차 [m]
IMU_ANGLE_NOISE_STD = 0.01  # IMU 각도(pitch/roll) 노이즈 표준편차 [rad]
IMU_RATE_NOISE_STD = 0.03   # IMU 각속도 노이즈 표준편차 [rad/s]

HEIGHT_REWARD_W = 20.0
HEIGHT_REWARD_CAP = 0.008  # 스텝당 높이 보상에 반영되는 dz 상한 [m] — 정상적인
                            # 등반 속도라면 스텝(0.01~0.02s)당 8mm도 충분히 넉넉함.
                            # 이게 없으면 순간적으로 붕 뜨는 dz까지 그대로 보상됨.
STOP_DIST_BONUS_W = 3.0     # 목표 정지거리(계단 6cm 앞)에 가까울수록 주는 보너스 가중치
STOP_DIST_TOLERANCE = 0.08  # 이 범위[m] 안이면 부분 보너스, 벗어나면 0
BACKSLIDE_PENALTY_W = 30.0  # 계단을 넘은 뒤 다시 뒤로 밀리는 것에 대한 추가 페널티 가중치
FLIP_PENALTY = 100.0  # 기존 20 -> 대폭 상향. 기존 값이 너무 작아서, 500스텝
                        # 내내 어설프게 버티며 쌓이는 자잘한 페널티(idle/pitch/roll)
                        # 총합보다 "빨리 넘어져서 에피소드를 조기종료"하는 쪽이
                        # 오히려 덜 손해였음 -> 정책이 이걸 역이용해 일부러
                        # 격렬하게 움직이다 넘어지는 행동을 학습하는 부작용 발견.
SUCCESS_BONUS = 50.0
# 목표 계단 높이의 이 비율 이상 실제로 상승해야 "성공"으로 인정
# (고정 마진을 쓰면 계단이 낮을 때 마진이 0에 가까워져서 리셋 직후 미세한
# 물리 흔들림만으로 성공 오탐이 나던 버그가 있었음 -> 비율 기반으로 수정)
SUCCESS_HEIGHT_FRACTION = 0.8
# 계단까지 실제로 이동했는지 확인하는 최소 전진거리 [m] — climber_scene.py의
# STAIR_X0(로봇 시작점~첫 계단 거리, 1.10m)를 감안해 정함. 이게 없으면
# 제자리에서 위아래로 튀기만 해도 height_gained 조건은 만족시킬 수 있어서
# 반드시 같이 걸어야 함.
# 계단까지 실제로 이동했는지 확인하는 최소 전진거리 [m] — climber_scene.py의
# STAIR_X0(로봇 시작점~첫 계단 거리, 1.10m) + 꼬리가 차체 뒤로 나온 길이(실측
# 232mm)를 감안해서 정함. 이게 짧으면 꼬리가 아직 계단에 걸쳐있는데도(차체만
# 이미 계단 위) 성공 처리될 수 있음 — 실제로 이 문제가 있었어서 1.0 -> 1.4로 상향.
SUCCESS_MIN_FORWARD_DIST = 1.4
SUCCESS_PITCH_TOLERANCE = 0.25  # 성공 판정용 pitch 허용치(~14도) — CLIMB->RETRACT
                                  # 전환에 쓰는 PITCH_FLAT(0.12, ~7도)보다 넉넉하게
                                  # 잡음. 등반 직후 완벽 수평이 아니어도(약간 흔들리는
                                  # 채로도) 실질적으로 다 올라왔으면 성공 인정 —
                                  # 너무 엄격하면 성공 판정이 거의 안 나서 정책이
                                  # 계속 무의미하게 전진만 하는 문제가 있었음.
N_SUBSTEPS = 10
MAX_EPISODE_STEPS = 500

# ----- 계단 높이 도메인 랜덤화 (실측 성능 기준: 2~5cm) -----
STEP_H_MIN = 0.01    # 학습 시 계단 높이 랜덤 범위 하한 [m] — 무변형 통과 구간 포함
STEP_H_MAX = 0.05    # 학습 시 계단 높이 랜덤 범위 상한 [m] — wheg 변형이 필요한 최대 높이

# 이 높이 이하는 스포크 전개 없이 고토크로 밀고 올라감 (변형 없는 등반)
LOW_STEP_THRESH = 0.02
# 라이다 추정치가 실제보다 낮게 나오는 경향(꼭대기 대신 옆면에 걸리는 행이
# 있을 수 있음)이 있어서, 애매한 경우 안전한 쪽(DEPLOY)으로 가도록 여유를 둠
HEIGHT_SAFETY_MARGIN = 0.005  # 기존 0.012 -> 하향, PUSH 분류가 더 자주 나오게

# ----- 상태기계 타이밍 -----
MEASURE_DWELL = 0.30      # MEASURE 상태 유지 시간 [s] — 여러 프레임 스캔해서 누적
STOP_DWELL = 0.20         # STOP 상태 유지 시간 [s] — 완전히 멈추는 걸 보장
DEPLOY_DONE = 0.85        # 목표각의 85% 이상 펴지면 CLIMB로 전환
DEPLOY_TIMEOUT = 1.5      # DEPLOY 최대 대기 시간 [s]
PITCH_FLAT = 0.12
GYRO_CALM = 0.5
CLEAR_TIME = 1.2          # CLIMB에서 이 시간만큼 안정되면 RETRACT로 전환
CLIMB_TIMEOUT = 4.0       # CLIMB 최대 유지 시간 [s] — 이게 없으면 안정(calm)
STUCK_WATCHDOG_TIME = 1.0  # DRIVE 상태에서 계단 근처인데 이 시간[s] 이상
                             # 정체되면 강제로 MEASURE 진입 (감지 실패 안전장치)
                           # 조건을 계속 못 만족할 때(예: 계속 흔들리거나 튐)
                           # 에피소드 끝까지 CLIMB에 갇혀서, 매 스텝 조금씩
                           # 튀는 것만으로 높이 보상을 계속 누적하는 부작용이 있었음.
RETRACT_DONE = 0.1        # 다리가 이 각도[rad] 이하로 접히면 RETRACT 완료
RETRACT_TIMEOUT = 1.5     # RETRACT 최대 대기 시간 [s]

N_LIDAR = LIDAR_ROWS * LIDAR_COLS
_LIDAR_NAMES = [f"lidar_{r}_{c}" for r in range(LIDAR_ROWS) for c in range(LIDAR_COLS)]


def _lidar_scan(data):
    out = np.empty(N_LIDAR, dtype=np.float32)
    for i, name in enumerate(_LIDAR_NAMES):
        v = float(data.sensor(name).data[0])
        if v < 0:
            v = LIDAR_MAX_RANGE
        elif v < LIDAR_MIN_RANGE:
            v = LIDAR_MIN_RANGE
        out[i] = v
    return out


# ----- 실측 sensor_fusion_node.py와 동일한 지형 감지 파이프라인 -----
# 시뮬레이션의 25개 rangefinder + ToF 1개를, 실제 로봇의 PointCloud2 + ToF
# 처리 로직과 최대한 똑같이 재구성. 좌표계는 각 광선의 원점/방향이 이미
# 차체(로봇 기준) 프레임으로 정의되어 있어서(climber_scene.py의 _lidar_sites),
# 별도 좌표변환 없이 원점+거리*방향으로 바로 (x,y,z) 포인트를 얻을 수 있음.
LIDAR_RELIABLE_MIN = 0.30   # 실측 sensor_fusion_node.py와 동일
LIDAR_HEIGHT = 0.05          # 실측과 동일 (라이다 장착 높이)
DETECT_RANGE = 0.35          # 전방 감지 거리
STEP_THRESHOLD = 0.005       # 최소 단차 높이로 인정하는 하한
SIDE_LIMIT = 0.04            # 좌우 범위
TOF_NEAR_LIMIT = 0.10        # 이 이하는 ToF만 사용
STEP_HEIGHT_OFFSET = 0.07    # 실측 코드의 경험적 보정값 그대로 반영

# 각 라이다 광선의 (원점, 방향) — climber_scene.py의 _lidar_sites와 동일 공식
_LIDAR_RAY_DIRS = []
for _r in range(LIDAR_ROWS):
    _v = -LIDAR_FOV_V / 2 + LIDAR_FOV_V * (_r / max(LIDAR_ROWS - 1, 1))
    _pitch = LIDAR_MOUNT_TILT + _v
    for _c in range(LIDAR_COLS):
        _h = -LIDAR_FOV_H / 2 + LIDAR_FOV_H * (_c / max(LIDAR_COLS - 1, 1))
        _zx = math.cos(_pitch) * math.cos(_h)
        _zy = math.cos(_pitch) * math.sin(_h)
        _zz = -math.sin(_pitch)
        _LIDAR_RAY_DIRS.append((_zx, _zy, _zz))


def _lidar_to_points(lidar_flat):
    """25개 raw 거리값 -> (x,y,z) 포인트 리스트. sensor_fusion_node.py가
    받는 PointCloud2를 흉내냄 (원점은 라이다 장착 위치 기준 0,0,0으로 봄 —
    실제 x범위 필터(0.15~0.35)도 이 기준이라 그대로 맞음)."""
    points = []
    for dist, (dx, dy, dz) in zip(lidar_flat, _LIDAR_RAY_DIRS):
        if dist >= LIDAR_MAX_RANGE:  # cutoff에 걸려 무한대 취급된 광선은 제외
            continue
        points.append((dist * dx, dist * dy, dist * dz))
    return points


def get_terrain_info(lidar_flat, tof_dist):
    """sensor_fusion_node.py의 scan3d_cb()와 동일한 로직.
    반환: dict(step_detected, step_height, distance_to_step, tof_active, tof_distance)
    """
    points = _lidar_to_points(lidar_flat)
    front_points = [p for p in points if 0.15 < p[0] < DETECT_RANGE and abs(p[1]) < SIDE_LIMIT]

    result = dict(step_detected=False, step_height=0.0, distance_to_step=0.0,
                  tof_active=False, tof_distance=tof_dist)

    if not front_points:
        return result

    min_dist = min(p[0] for p in front_points)

    if tof_dist < TOF_NEAR_LIMIT:
        result.update(tof_active=True, step_detected=True, step_height=0.0,
                       distance_to_step=tof_dist)
        return result

    z_corrected = [p[2] + LIDAR_HEIGHT for p in front_points]
    step_z = [z for z in z_corrected if z >= STEP_THRESHOLD]

    if min_dist < LIDAR_RELIABLE_MIN:
        result["tof_active"] = True
        if step_z:
            step_z_sorted = sorted(step_z, reverse=True)
            top_n = max(1, len(step_z_sorted) // 10)
            step_height = sum(step_z_sorted[:top_n]) / top_n + STEP_HEIGHT_OFFSET
            result.update(step_detected=True, step_height=float(step_height),
                           distance_to_step=float(tof_dist))
        return result

    # 30cm 이상: 라이다만 사용
    if step_z:
        step_z_sorted = sorted(step_z, reverse=True)
        top_n = max(1, len(step_z_sorted) // 10)
        step_height = sum(step_z_sorted[:top_n]) / top_n + STEP_HEIGHT_OFFSET
        result.update(step_detected=True, step_height=float(step_height),
                       distance_to_step=float(min_dist))
    return result


_STEP_CX = STAIR_X0 + 0.5 * STEP_D          # step0 x위치 (계단 높이와 무관, 고정)
_PLATFORM_CX = STAIR_X0 + N_STEPS * STEP_D + 0.55  # platform x위치 (고정)


def _set_step_height(model, step_h):
    """모델을 통째로 다시 빌드하지 않고, step0/platform geom의 크기·위치만
    그 자리에서 바꿔서 계단 높이를 변경. 뷰어가 model 객체를 계속 붙들고
    있어도(재생성 없이) 문제없이 반영됨 -> 매 에피소드 모델을 새로 만들면서
    생기던 '뷰어가 옛날 객체를 계속 그림' 문제, 그리고 그걸 고치려고
    에피소드마다 뷰어를 새로 열었다 닫으면서 생기던 GLX 크래시 문제를
    둘 다 근본적으로 없앰."""
    step_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "step0")
    platform_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "platform")

    hz = step_h / 2.0
    model.geom_size[step_id][2] = hz
    model.geom_pos[step_id][2] = hz
    model.geom_pos[step_id][0] = _STEP_CX

    top_h = N_STEPS * step_h
    model.geom_size[platform_id][2] = top_h / 2.0
    model.geom_pos[platform_id][2] = top_h / 2.0
    model.geom_pos[platform_id][0] = _PLATFORM_CX


def _quat_to_euler(quat):
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class WhegEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, randomize_terrain=True, domain_randomize=None):
        super().__init__()
        self.render_mode = render_mode
        self.randomize_terrain = randomize_terrain
        # domain_randomize: 질량/마찰/센서노이즈 랜덤화 (계단 높이 랜덤화와는
        # 별개 스위치). 기본값은 randomize_terrain을 따라가되, 필요하면 독립
        # 지정 가능 — 재생(watch) 스크립트에서 "계단 높이는 다양하게 보고
        # 싶은데, 질량이 매 에피소드 바뀌면서 생기는 재생 버벅임은 피하고
        # 싶을 때" 이렇게 따로 끌 수 있게 분리함.
        self.domain_randomize = randomize_terrain if domain_randomize is None else domain_randomize

        self._step_h = STEP_H
        self._xml = build_xml(self._step_h)
        self.model = mujoco.MjModel.from_xml_string(self._xml)
        self.data = mujoco.MjData(self.model)
        self._base_body_mass = self.model.body_mass.copy()  # 도메인 랜덤화 기준값
                                                               # (매 에피소드 이 값에서
                                                               # 다시 스케일링, 누적 곱셈 방지)

        # 관측값 구조 (실측 sensor_fusion_node.py의 TerrainInfo 메시지와 동일하게
        # 맞춤 — 기존엔 25개 raw 라이다값이었는데, 실제 로봇은 그 원시값을 안
        # 주고 이미 요약된 지형정보(step_detected, step_height, distance_to_step,
        # tof_active, tof_distance)만 주기 때문에 정책도 이 형태로 다시 학습함):
        #   [step_detected, step_height, distance_to_step, tof_active, tof_distance,
        #    pitch, pitch_rate, roll, roll_rate, yaw_rel, yaw_rate, deploy_angle]
        obs_low = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0,
             -math.pi, -50.0, -math.pi, -50.0, -math.pi, -50.0, 0.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [1.0, 0.5, LIDAR_MAX_RANGE, 1.0, 0.5,
             math.pi, 50.0, math.pi, 50.0, math.pi, 50.0, LEG_MAX_ANGLE],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._viewer = None
        self._step_count = 0
        self._prev_x = 0.0
        self._prev_z = 0.0
        self._start_x = 0.0
        self._overshot_stop_x = False
        self._stuck_t = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._start_z = 0.0
        self._measure_max_h = 0.0
        self._start_yaw = 0.0
        self._last_terrain = dict(step_detected=False, step_height=0.0,
                                    distance_to_step=0.0, tof_active=False,
                                    tof_distance=0.15)

        # 상태기계: DRIVE -> MEASURE -> STOP -> DEPLOY -> CLIMB -> RETRACT -> DRIVE
        self._mode = "DRIVE"
        self._last_estimated_h = 0.0
        self._t_in_mode = 0.0
        self._clear_t = 0.0
        self._dt = self.model.opt.timestep * N_SUBSTEPS

    # ------------------------------------------------------------
    def _get_obs(self):
        lidar = _lidar_scan(self.data)
        tof_dist = float(self.data.sensor("tof_0").data[0])
        if tof_dist < 0:
            tof_dist = 0.15  # cutoff 밖(장애물 없음) -> ToF 최대치로 취급
        roll, pitch, yaw = _quat_to_euler(self.data.sensor("imu_quat").data)
        yaw_rel = yaw - self._start_yaw
        pitch_rate = float(self.data.sensor("imu_gyro").data[1])
        roll_rate = float(self.data.sensor("imu_gyro").data[0])
        yaw_rate = float(self.data.sensor("imu_gyro").data[2])
        deploy = float(self.data.joint("leg_l1").qpos[0])

        if self.domain_randomize:
            lidar = lidar + self.np_random.normal(0, LIDAR_NOISE_STD, size=lidar.shape)
            lidar = np.clip(lidar, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE)
            tof_dist += self.np_random.normal(0, LIDAR_NOISE_STD)
            pitch += self.np_random.normal(0, IMU_ANGLE_NOISE_STD)
            roll += self.np_random.normal(0, IMU_ANGLE_NOISE_STD)
            pitch_rate += self.np_random.normal(0, IMU_RATE_NOISE_STD)
            roll_rate += self.np_random.normal(0, IMU_RATE_NOISE_STD)
            yaw_rate += self.np_random.normal(0, IMU_RATE_NOISE_STD)

        # 실측 파이프라인과 동일한 지형정보 산출 (raw 라이다는 이 함수를
        # 거쳐서만 정책에게 전달됨 — 25개 원시값 자체는 관측값에 안 들어감)
        terrain = get_terrain_info(lidar, tof_dist)
        terrain_obs = np.array([
            1.0 if terrain["step_detected"] else 0.0,
            terrain["step_height"],
            terrain["distance_to_step"],
            1.0 if terrain["tof_active"] else 0.0,
            terrain["tof_distance"],
        ], dtype=np.float32)

        imu_obs = np.array(
            [pitch, pitch_rate, roll, roll_rate, yaw_rel, yaw_rate, deploy],
            dtype=np.float32,
        )
        # 상태기계에서도 재사용할 수 있도록 저장
        self._last_terrain = terrain
        return terrain_obs, imu_obs

    def _update_mode(self, terrain_obs, pitch, pitch_rate):
        """DRIVE -> MEASURE -> (PUSH | STOP -> DEPLOY -> CLIMB -> RETRACT) -> DRIVE
        상태기계. (allow_drive, deploy_target, drive_max) 반환.
        allow_drive=False인 구간은 RL의 drive 명령을 강제로 0 처리.

        wall_near/높이 추정 모두 get_terrain_info()(실측 sensor_fusion_node.py와
        동일한 포인트클라우드+ToF 융합 로직)만 사용.
        MEASURE에서 추정 높이 기준으로 분기:
          - LOW_STEP_THRESH(2cm) 이하로 추정: PUSH — 스포크 안 펴고
            고토크(DRIVE_MAX_BOOST)로 그냥 밀고 올라감 (무변형 등반)
          - 그보다 높게 추정: 기존 STOP->DEPLOY->CLIMB->RETRACT 시퀀스로 등반
        """
        self._t_in_mode += self._dt
        wall_near = self._last_terrain["step_detected"]
        live_height = self._last_terrain["step_height"]
        opened = float(self.data.joint("leg_l1").qpos[0])  # 다리 각도 기준 (0=접힘, 양수=열림)

        if self._mode == "DRIVE":
            allow_drive, deploy_target, drive_max = True, 0.0, DRIVE_MAX
            if wall_near:
                self._mode, self._t_in_mode = "MEASURE", 0.0
                self._measure_max_h = live_height
            else:
                # 감시(watchdog): 계단 근처(30cm 이내)인데 감지(wall_near)가
                # 안 걸린 채로 오랫동안 거의 안 움직이면(예: 너무 빨리 접근하다
                # 계단에 부딪혀 튕겨나온 뒤 애매한 위치에 걸린 경우) 강제로
                # MEASURE로 진입시켜서 "벽에 막힌 것처럼 영원히 멈춰있는" 상황을
                # 방지함. 정상적인 감지 로직을 우회하는 게 아니라, 그 로직이
                # 어떤 이유로든 안 걸렸을 때의 안전장치임.
                cur_x = self.data.body("chassis").xpos[0]
                near_stair = cur_x > STAIR_X0 - 0.30
                if near_stair and (cur_x - self._prev_x) < IDLE_DIST_THRESH:
                    self._stuck_t += self._dt
                else:
                    self._stuck_t = 0.0
                if self._stuck_t > STUCK_WATCHDOG_TIME:
                    self._mode, self._t_in_mode = "MEASURE", 0.0
                    self._measure_max_h = live_height
                    self._stuck_t = 0.0

        elif self._mode == "MEASURE":
            allow_drive, deploy_target, drive_max = False, 0.0, DRIVE_MAX
            # 구간 내내 계속 스캔해서 최댓값 누적 (단일 순간만 보면 로봇 위치/
            # 자세에 따라 놓칠 수 있음)
            self._measure_max_h = max(self._measure_max_h, live_height)
            if self._t_in_mode > MEASURE_DWELL:
                self._last_estimated_h = self._measure_max_h
                if self._measure_max_h <= LOW_STEP_THRESH - HEIGHT_SAFETY_MARGIN:
                    self._mode, self._t_in_mode, self._clear_t = "PUSH", 0.0, 0.0
                else:
                    self._mode, self._t_in_mode = "STOP", 0.0

        elif self._mode == "PUSH":
            # 저단차: 스포크 안 펴고 고토크로 그냥 통과
            allow_drive, deploy_target, drive_max = True, 0.0, DRIVE_MAX_BOOST
            calm = (not wall_near) and abs(pitch) < PITCH_FLAT and abs(pitch_rate) < GYRO_CALM
            self._clear_t = self._clear_t + self._dt if calm else 0.0
            if self._clear_t > CLEAR_TIME:
                self._mode, self._t_in_mode = "DRIVE", 0.0

        elif self._mode == "STOP":
            allow_drive, deploy_target, drive_max = False, 0.0, DRIVE_MAX
            if self._t_in_mode > STOP_DWELL:
                self._mode, self._t_in_mode = "DEPLOY", 0.0

        elif self._mode == "DEPLOY":
            allow_drive, deploy_target, drive_max = False, LEG_MAX_ANGLE, DRIVE_MAX
            if opened > LEG_MAX_ANGLE * DEPLOY_DONE or self._t_in_mode > DEPLOY_TIMEOUT:
                self._mode, self._t_in_mode, self._clear_t = "CLIMB", 0.0, 0.0

        elif self._mode == "CLIMB":
            allow_drive, deploy_target, drive_max = True, LEG_MAX_ANGLE, DRIVE_MAX
            calm = (not wall_near) and abs(pitch) < PITCH_FLAT and abs(pitch_rate) < GYRO_CALM
            self._clear_t = self._clear_t + self._dt if calm else 0.0
            if self._clear_t > CLEAR_TIME or self._t_in_mode > CLIMB_TIMEOUT:
                self._mode, self._t_in_mode = "RETRACT", 0.0

        else:  # RETRACT
            allow_drive, deploy_target, drive_max = False, 0.0, DRIVE_MAX
            if opened < RETRACT_DONE or self._t_in_mode > RETRACT_TIMEOUT:
                self._mode, self._t_in_mode = "DRIVE", 0.0

        return allow_drive, deploy_target, drive_max

    def _apply_action(self, action, terrain_obs, pitch, pitch_rate):
        action = np.clip(action, -1.0, 1.0)
        allow_drive, deploy_target, drive_max = self._update_mode(terrain_obs, pitch, pitch_rate)

        if allow_drive:
            drive_l = float(action[0]) * drive_max
            drive_r = float(action[1]) * drive_max
        else:
            drive_l = drive_r = 0.0

        self.data.actuator("drive_l").ctrl = drive_l
        self.data.actuator("drive_r").ctrl = drive_r
        self.data.actuator("deploy_l").ctrl = deploy_target
        self.data.actuator("deploy_r").ctrl = deploy_target
        return drive_l, drive_r, drive_max, allow_drive

    # ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)

        if self.randomize_terrain:
            self._step_h = float(self.np_random.uniform(STEP_H_MIN, STEP_H_MAX))
            _set_step_height(self.model, self._step_h)

        # 도메인 랜덤화: 마찰/질량을 매 에피소드 살짝씩 바꿔서, 실제 로봇의
        # 제조 공차·바닥재 차이·배선 무게 편차 등에도 정책이 견고하게
        # 작동하도록 함 (특정 고정값에만 과적합되는 것 방지). 계단 높이
        # 랜덤화(randomize_terrain)와는 별개 스위치 — 재생(watch) 스크립트
        # 에서는 계단 높이는 다양하게 보되, 질량이 매 에피소드 바뀌면서
        # 생기는 재생 버벅임(모델-기록 불일치)은 피하려고 이렇게 분리함.
        if self.domain_randomize:
            floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            base_friction = self.np_random.uniform(0.85, 1.25)  # 바닥 마찰 랜덤화
            self.model.geom_friction[floor_id, 0] = base_friction
            for name in ("hub_l", "hub_r"):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                self.model.geom_friction[gid, 0] = self.np_random.uniform(0.85, 1.15)
            mass_scale = self.np_random.uniform(0.9, 1.1)  # 전체 질량 +-10%
            self.model.body_mass[:] = self._base_body_mass * mass_scale
            mujoco.mj_setConst(self.model, self.data)  # 질량 변경 후 관성 등 재계산

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        noise = self.np_random.uniform(-0.01, 0.01, size=self.model.nq)
        self.data.qpos[:] += noise
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        x, _, z = self.data.body("chassis").xpos
        self._prev_x, self._prev_z = float(x), float(z)
        self._start_x = float(x)  # 성공 판정용 — 실제로 앞으로 이동했는지 확인
        self._overshot_stop_x = False  # 정지목표 지점을 지나쳤다가 후진으로 맞추는
                                        # 편법 방지용 플래그
        self._stuck_t = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._start_z = float(z)  # 성공 판정 기준 — 절대높이가 아니라 이 값 대비 상승분으로 판단
        _, _, yaw0 = _quat_to_euler(self.data.sensor("imu_quat").data)
        self._start_yaw = yaw0

        self._mode, self._t_in_mode, self._clear_t = "DRIVE", 0.0, 0.0
        self._measure_max_h = 0.0

        terrain_obs, imu_obs = self._get_obs()
        return np.concatenate([terrain_obs, imu_obs]), {
            "mode": self._mode,
            "step_h_true": self._step_h,
            "step_h_estimated": self._last_estimated_h,
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action_rate_penalty = ACTION_RATE_PENALTY_W * float(
            np.sum((action - self._prev_action) ** 2)
        )
        self._prev_action = action.copy()

        terrain_obs, imu_obs = self._get_obs()
        pitch, pitch_rate = float(imu_obs[0]), float(imu_obs[1])
        drive_l, drive_r, drive_max, allow_drive = self._apply_action(action, terrain_obs, pitch, pitch_rate)

        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        terrain_obs, imu_obs = self._get_obs()
        obs = np.concatenate([terrain_obs, imu_obs])
        pitch, roll, yaw_rel = float(imu_obs[0]), float(imu_obs[2]), float(imu_obs[4])

        x, _, z = self.data.body("chassis").xpos
        x, z = float(x), float(z)
        dx = x - self._prev_x
        dz = z - self._prev_z
        self._prev_x, self._prev_z = x, z

        ctrl_cost = CTRL_COST_W * 0.5 * (
            (drive_l / drive_max) ** 2 + (drive_r / drive_max) ** 2
        )
        pitch_penalty = PITCH_PENALTY_W * (pitch ** 2)
        roll_penalty = ROLL_PENALTY_W * (roll ** 2)
        yaw_penalty = YAW_PENALTY_W * (yaw_rel ** 2)

        # 계단 근처(DRIVE 상태, 계단까지 30cm 이내)에서 과속 페널티 — MEASURE
        # 상태 자체는 이미 토크=0으로 강제되지만, 그 직전까지 너무 빠른 속도로
        # 달려오면 관성 때문에 MEASURE 동안 계속 움직이면서 라이다 높이 추정이
        # 부정확해지고, 높은 계단도 낮게 오판(PUSH로 잘못 분기)하는 문제가
        # 실제로 관측됨. 감지 전부터 미리 감속하도록 유도해서 이를 방지.
        approach_zone = self._mode == "DRIVE" and x > STAIR_X0 - 0.30
        approach_speed_penalty = (
            APPROACH_SPEED_PENALTY_W * max(0.0, dx - APPROACH_SPEED_LIMIT)
            if approach_zone else 0.0
        )

        # 움직여도 되는 구간(DRIVE/CLIMB/PUSH)인데 실제로 거의 안 움직이면
        # 페널티 — "가만히 있기"가 매력적인 선택지가 되는 걸 막기 위함
        idle_penalty = IDLE_PENALTY_W if (allow_drive and dx < IDLE_DIST_THRESH) else 0.0

        # 높이 보상은 "등반 중" 상태(CLIMB/PUSH) + "실제로 전진" + "실제
        # 계단 위치 근처에 도달했을 때" + "자세가 정상적인 등반 범위일 때"만
        # 지급. 앞의 조건들만으로는, 실제로 앞으로 나아가지 않고 그 자리에서
        # 뒤로 확 젖혀서(꼬리가 들리고 본체가 뒤로 빠지는 "웰리"처럼) 순간적인
        # dx>0을 만들어내는 것만으로도 보상을 받는 편법이 있었음(실제 관측된
        # 문제). CLIMB_PITCH_LIMIT으로 정상적인 등반 범위를 넘는 과도한
        # 피치(뒤로 확 젖혀짐)일 때는 아예 보상을 안 주도록 막음. 계단의 실제
        # x위치(STAIR_X0)는 정책 입력(observation)에는 안 들어가고 리워드
        # 계산에만 쓰는 특권 정보라 문제 없음.
        near_step_x = STAIR_X0 - 0.05  # 계단 시작 살짝 이전부터 허용(차체 앞부분이
                                        # 중심보다 먼저 닿는 여유 반영)
        height_ok_mode = self._mode in ("CLIMB", "PUSH")
        height_gain_reward = (
            min(max(dz, 0.0), HEIGHT_REWARD_CAP)
            if (height_ok_mode and dx > 0.0 and x > near_step_x
                and abs(pitch) < CLIMB_PITCH_LIMIT) else 0.0
        )

        reward = FORWARD_REWARD_W * dx + HEIGHT_REWARD_W * height_gain_reward
        reward -= (ctrl_cost + pitch_penalty + roll_penalty + yaw_penalty
                   + idle_penalty + action_rate_penalty + approach_speed_penalty)

        # 목표 정지거리(계단 6cm 앞) 유도 보상 — STOP/DEPLOY 상태일 때, 이상적인
        # 정지 위치(STAIR_X0 - 0.06)에 가까울수록 작은 보너스를 줘서, 정책이
        # DRIVE/MEASURE 구간에서부터 접근 속도를 스스로 조절해 그 지점 근처에서
        # 서도록 유도함. 단, 목표 지점을 이미 "지나쳤다가 후진해서 맞추는" 것도
        # 방향 상관없이 보상받을 수 있었던 허점이 있었음(실제로 관측된 편법
        # 행동) — 지나친 적이 있으면 이번 에피소드 내내 보너스를 아예 안 주도록
        # 막아서, 처음부터 적절한 속도로 접근하는 것만 보상받게 함.
        target_stop_x = STAIR_X0 - 0.06
        if x > target_stop_x + 0.02:  # 목표보다 2cm 이상 더 가면 "지나침"으로 판정
            self._overshot_stop_x = True
        if self._mode in ("STOP", "DEPLOY") and not self._overshot_stop_x:
            stop_error = abs(x - target_stop_x)
            reward += STOP_DIST_BONUS_W * max(0.0, 1.0 - stop_error / STOP_DIST_TOLERANCE)

        # 계단을 이미 넘은(플랫폼 위에 올라선) 뒤에 뒤로 밀리는 것에 대한 추가
        # 페널티 — 등반 성공 후 다시 계단 아래로 미끄러지거나 후진하는 문제 방지.
        # FORWARD_REWARD_W*dx가 이미 음수 dx에 페널티를 주긴 하지만, 이 구간은
        # 특히 더 강하게 눌러서 "넘었으면 그 자리를 지키거나 더 전진"을 명확히 함.
        past_step_x = STAIR_X0 + STEP_D
        if x > past_step_x and dx < 0.0:
            reward -= BACKSLIDE_PENALTY_W * (-dx)

        terminated = False
        truncated = False
        info = {
            "mode": self._mode,
            "step_h_true": self._step_h,             # 검증용 정답값 (정책/제어 로직은 안 씀)
            "step_h_estimated": self._last_estimated_h,  # 라이다로 추정한 값 (실제 판단 기준)
        }

        if abs(pitch) > PITCH_FLIP or abs(roll) > ROLL_FLIP:
            reward -= FLIP_PENALTY
            terminated = True
            info["termination_reason"] = "flipped"

        height_gained = z - self._start_z
        forward_dist = x - self._start_x
        # 고정 마진(예: -1cm) 대신 목표 높이의 비율로 판정 — 계단 높이가
        # 1cm까지 낮아진 상황에서 고정 마진을 쓰면 마진이 0에 가까워져서
        # 리셋 직후 미세한 물리 흔들림만으로 성공 오탐이 나던 버그를 수정.
        # 최소 스텝 수 조건 + 최소 전진거리 조건도 추가로 걸어서, 제자리에서
        # 위아래로 튀기만 해도(전진 없이) 성공 처리되던 허점을 막음.
        # 자세(pitch) 조건은 완전히 제거함 — 이게 있으면(느슨하게 풀어도)
        # 등반 직후 완벽 수평이 아닌 경우가 실제로 흔해서 성공 판정 자체가
        # 거의 안 걸리고, 그 결과 에피소드가 안 끝나서 정책이 계속 무의미하게
        # 전진만 하는 문제가 반복적으로 관측됨. 전진거리(꼬리 길이 포함)와
        # 높이 조건만으로도 "계단을 실제로 넘었는지"는 충분히 판별 가능함.
        if (self._step_count > 10
                and height_gained > SUCCESS_HEIGHT_FRACTION * self._step_h
                and forward_dist > SUCCESS_MIN_FORWARD_DIST
                and not terminated):
            reward += SUCCESS_BONUS
            terminated = True
            info["termination_reason"] = "success"

        if self._step_count >= MAX_EPISODE_STEPS:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        import mujoco.viewer
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


if __name__ == "__main__":
    env = WhegEnv()
    obs, info = env.reset()
    print("obs space:", env.observation_space.shape)
    print("action space:", env.action_space)
    print("initial obs (lidar 앞부분 5개):", obs[:5],
          "... pitch/pitch_rate/roll/roll_rate/yaw/yaw_rate/deploy:", obs[-7:])

    print("\n계단 높이 랜덤화 확인 (reset 5번):")
    for _ in range(5):
        _, info = env.reset()
        print(f"  이번 에피소드 step_h_true = {info['step_h_true']:.3f} m")

    total_reward = 0.0
    modes_seen = set()
    for i in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        modes_seen.add(info["mode"])
        total_reward += reward
        if terminated or truncated:
            print(f"episode ended at step {i}: {info}")
            obs, info = env.reset()
    print("random rollout total reward (300 steps):", total_reward)
    print("이번 롤아웃에서 관측된 모드들:", modes_seen)

    # ----- 라이다 높이 추정 정확도 검증 -----
    print("\n=== 라이다 높이 추정 정확도 검증 (MEASURE 통과할 때마다 비교) ===")
    errors = []
    for ep in range(15):
        obs, info = env.reset()
        done = False
        prev_mode = "DRIVE"
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # MEASURE를 막 벗어난 시점(PUSH 또는 STOP 진입)에 추정치 기록됨
            if prev_mode == "MEASURE" and info["mode"] in ("PUSH", "STOP"):
                true_h = info["step_h_true"]
                est_h = info["step_h_estimated"]
                err = est_h - true_h
                errors.append(err)
                print(f"  ep{ep}: 실제={true_h:.4f}m  추정={est_h:.4f}m  "
                      f"오차={err:+.4f}m  분기={info['mode']}")
            prev_mode = info["mode"]
    if errors:
        errors = np.array(errors)
        print(f"\n평균 오차: {errors.mean():+.4f}m  |  평균 절대오차: {np.abs(errors).mean():.4f}m  "
              f"|  표본수: {len(errors)}")
    else:
        print("MEASURE 단계를 통과한 표본이 없었어요 (랜덤 행동이라 계단을 못 만났을 수 있음).")
