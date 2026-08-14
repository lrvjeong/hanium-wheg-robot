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
    build_xml, DEPLOY_MAX, DRIVE_MAX, DRIVE_MAX_BOOST, N_STEPS, STEP_H,
    STEP_D, STAIR_X0, LEG_MAX_ANGLE,
    LIDAR_ROWS, LIDAR_COLS, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE,
    LIDAR_FOV_V, LIDAR_MOUNT_TILT, START_Z,
)

# ===== 튜닝 상수 =====
PITCH_FLIP = 1.2
ROLL_FLIP = 1.0
CTRL_COST_W = 0.005   # 기존 0.01 -> 하향 (토크 쓰는 걸 너무 겁내지 않도록)
PITCH_PENALTY_W = 0.3
ROLL_PENALTY_W = 0.5
YAW_PENALTY_W = 0.5
FORWARD_REWARD_W = 15.0  # 기존 8.0 -> 상향 (전진을 훨씬 매력적으로)
IDLE_PENALTY_W = 2.0     # 움직여야 하는 구간(DRIVE/CLIMB/PUSH)에서 실제로
                          # 거의 안 움직이면 매 스텝 부과되는 페널티
ACTION_RATE_PENALTY_W = 0.3  # 직전 스텝 액션 대비 급격한 변화에 부과되는
                              # 페널티 - 좌우 바퀴가 들쭉날쭉하게 튀는
                              # 부산스러운 움직임을 줄이기 위함
IDLE_DIST_THRESH = 0.001  # 이 정도(1mm) 이하 전진이면 "정지 중"으로 간주
HEIGHT_REWARD_W = 20.0
FLIP_PENALTY = 20.0
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
# STAIR_X0(로봇 시작점~첫 계단 거리, 1.10m) + 꼬리가 차체 뒤로 나온 길이(~0.25m)
# 를 감안해서 정함. 이게 짧으면 꼬리가 아직 계단에 걸쳐있는데도(차체만 이미
# 계단 위) 성공 처리될 수 있음 — 실제로 이 문제가 있었어서 1.0 -> 1.4로 상향.
SUCCESS_MIN_FORWARD_DIST = 1.4
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


# ----- LiDAR 기반 장애물 감지 + 높이 추정 (삼각법, ground-truth 참조 없음) -----
# 라이다 장착 높이(지면 기준) 근사치: 차체 원점 높이(START_Z) + 라이다 z 오프셋(0.03)
# — climber_scene.py의 _lidar_sites base_pos z값(0.03)과 반드시 맞춰야 함
_SENSOR_HEIGHT = START_Z + 0.03
_HEIGHT_MARGIN = 0.010  # 노이즈 대비 여유 [m] — 이 이상 짧게 찍혀야 "장애물"로 판정
# 감지(wall_near) 판정에 걸리는 절대 최대 거리 [m] — 이게 없으면 수평에 가까운
# 행은 '장애물 없을 때 예상거리' 자체가 워낙 커서(1m 이상), 계단과 무관한 먼
# 지형에도 반응해 너무 일찍 멈춰서는 문제가 있었음. 실제로 계단 코앞까지
# 왔을 때만 반응하도록 근거리로 제한.
_MAX_DETECT_DIST = 0.18

# 각 행(row)의 하향각. climber_scene.py의 _lidar_sites와 동일한 공식
# (pitch = LIDAR_MOUNT_TILT + v). 행마다 화각이 달라서, 로봇-계단 거리에 따라
# 어느 행이 계단에 먼저 걸리는지가 달라짐 -> 모든 행을 스캔해야 함.
# (라이다 장착 높이가 목표 단차보다 높아서, 수평에 가까운 행은 낮은 계단
#  위로 그냥 지나가 버려 못 잡는다 -> 여러 행을 봐야 놓치지 않음)
_ROW_TILTS = [
    LIDAR_MOUNT_TILT + (-LIDAR_FOV_V / 2 + LIDAR_FOV_V * (r / max(LIDAR_ROWS - 1, 1)))
    for r in range(LIDAR_ROWS)
]


def _scan_obstruction(lidar_flat):
    """중앙 열의 각 행을 스캔해서, 그 행 고유의 하향각 기준 '장애물 없을 때
    예상 거리'보다 실제 거리가 짧게 찍힌 행이 있는지 확인하고, 있다면 그 중
    최댓값으로 높이를 추정. wall_near 감지와 높이 추정을 하나로 통합 —
    행 하나(예: 거의 수평인 행)만 보면 라이다 장착 높이보다 낮은 계단은
    가까이 가도 위로 그냥 지나쳐서 놓칠 수 있기 때문."""
    grid = lidar_flat.reshape(LIDAR_ROWS, LIDAR_COLS)
    mid_col = LIDAR_COLS // 2

    detected = False
    best_height = 0.0
    for r in range(LIDAR_ROWS):
        theta = _ROW_TILTS[r]
        if theta <= 0:
            continue  # 수평 이상은 지면 반사 기준 계산 불가 (스킵)
        d = float(grid[r, mid_col])
        expected_ground_dist = _SENSOR_HEIGHT / math.sin(theta)
        if d < expected_ground_dist - _HEIGHT_MARGIN and d < _MAX_DETECT_DIST:
            detected = True
            height = _SENSOR_HEIGHT - d * math.sin(theta)
            best_height = max(best_height, height)

    return detected, max(0.0, best_height)


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

    def __init__(self, render_mode=None, randomize_terrain=True):
        super().__init__()
        self.render_mode = render_mode
        self.randomize_terrain = randomize_terrain

        self._step_h = STEP_H
        self._xml = build_xml(self._step_h)
        self.model = mujoco.MjModel.from_xml_string(self._xml)
        self.data = mujoco.MjData(self.model)

        obs_low = np.concatenate([
            np.full(N_LIDAR, LIDAR_MIN_RANGE, dtype=np.float32),
            np.array([-math.pi, -50.0, -math.pi, -50.0, -math.pi, -50.0, 0.0], dtype=np.float32),
        ])
        obs_high = np.concatenate([
            np.full(N_LIDAR, LIDAR_MAX_RANGE, dtype=np.float32),
            np.array([math.pi, 50.0, math.pi, 50.0, math.pi, 50.0, LEG_MAX_ANGLE], dtype=np.float32),
        ])
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
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._start_z = 0.0
        self._measure_max_h = 0.0
        self._start_yaw = 0.0

        # 상태기계: DRIVE -> MEASURE -> STOP -> DEPLOY -> CLIMB -> RETRACT -> DRIVE
        self._mode = "DRIVE"
        self._last_estimated_h = 0.0
        self._t_in_mode = 0.0
        self._clear_t = 0.0
        self._dt = self.model.opt.timestep * N_SUBSTEPS

    # ------------------------------------------------------------
    def _get_obs(self):
        lidar = _lidar_scan(self.data)
        roll, pitch, yaw = _quat_to_euler(self.data.sensor("imu_quat").data)
        yaw_rel = yaw - self._start_yaw
        pitch_rate = float(self.data.sensor("imu_gyro").data[1])
        roll_rate = float(self.data.sensor("imu_gyro").data[0])
        yaw_rate = float(self.data.sensor("imu_gyro").data[2])
        deploy = float(self.data.joint("leg_l0").qpos[0])
        return lidar, np.array(
            [pitch, pitch_rate, roll, roll_rate, yaw_rel, yaw_rate, deploy],
            dtype=np.float32,
        )

    def _update_mode(self, lidar, pitch, pitch_rate):
        """DRIVE -> MEASURE -> (PUSH | STOP -> DEPLOY -> CLIMB -> RETRACT) -> DRIVE
        상태기계. (allow_drive, deploy_target, drive_max) 반환.
        allow_drive=False인 구간은 RL의 drive 명령을 강제로 0 처리.

        wall_near/높이 추정 모두 _scan_obstruction()(라이다 다중 행 스캔)만
        사용 — 시뮬레이션 내부 실제값을 몰래 참조하지 않음.
        MEASURE에서 추정 높이 기준으로 분기:
          - LOW_STEP_THRESH(2cm) 이하로 추정: PUSH — 스포크 안 펴고
            고토크(DRIVE_MAX_BOOST)로 그냥 밀고 올라감 (무변형 등반)
          - 그보다 높게 추정: 기존 STOP->DEPLOY->CLIMB->RETRACT 시퀀스로 등반
        """
        self._t_in_mode += self._dt
        wall_near, live_height = _scan_obstruction(lidar)
        opened = float(self.data.joint("leg_l0").qpos[0])  # 다리 각도 기준 (0=접힘, 양수=열림)

        if self._mode == "DRIVE":
            allow_drive, deploy_target, drive_max = True, 0.0, DRIVE_MAX
            if wall_near:
                self._mode, self._t_in_mode = "MEASURE", 0.0
                self._measure_max_h = live_height

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
            if self._clear_t > CLEAR_TIME:
                self._mode, self._t_in_mode = "RETRACT", 0.0

        else:  # RETRACT
            allow_drive, deploy_target, drive_max = False, 0.0, DRIVE_MAX
            if opened < RETRACT_DONE or self._t_in_mode > RETRACT_TIMEOUT:
                self._mode, self._t_in_mode = "DRIVE", 0.0

        return allow_drive, deploy_target, drive_max

    def _apply_action(self, action, lidar, pitch, pitch_rate):
        action = np.clip(action, -1.0, 1.0)
        allow_drive, deploy_target, drive_max = self._update_mode(lidar, pitch, pitch_rate)

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

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        noise = self.np_random.uniform(-0.01, 0.01, size=self.model.nq)
        self.data.qpos[:] += noise
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        x, _, z = self.data.body("chassis").xpos
        self._prev_x, self._prev_z = float(x), float(z)
        self._start_x = float(x)  # 성공 판정용 — 실제로 앞으로 이동했는지 확인
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._start_z = float(z)  # 성공 판정 기준 — 절대높이가 아니라 이 값 대비 상승분으로 판단
        _, _, yaw0 = _quat_to_euler(self.data.sensor("imu_quat").data)
        self._start_yaw = yaw0

        self._mode, self._t_in_mode, self._clear_t = "DRIVE", 0.0, 0.0
        self._measure_max_h = 0.0

        lidar, rest = self._get_obs()
        return np.concatenate([lidar, rest]), {
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

        lidar, rest = self._get_obs()
        pitch, pitch_rate = float(rest[0]), float(rest[1])
        drive_l, drive_r, drive_max, allow_drive = self._apply_action(action, lidar, pitch, pitch_rate)

        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        lidar, rest = self._get_obs()
        obs = np.concatenate([lidar, rest])
        pitch, roll, yaw_rel = float(rest[0]), float(rest[2]), float(rest[4])

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

        # 움직여도 되는 구간(DRIVE/CLIMB/PUSH)인데 실제로 거의 안 움직이면
        # 페널티 — "가만히 있기"가 매력적인 선택지가 되는 걸 막기 위함
        idle_penalty = IDLE_PENALTY_W if (allow_drive and dx < IDLE_DIST_THRESH) else 0.0

        reward = FORWARD_REWARD_W * dx + HEIGHT_REWARD_W * max(dz, 0.0)
        reward -= ctrl_cost + pitch_penalty + roll_penalty + yaw_penalty + idle_penalty + action_rate_penalty

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
        # + IMU(pitch) 기준 "다시 수평" 조건 추가 — 이게 없으면 앞부분만
        # 계단 위에 걸친 상태(꼬리는 아직 안 올라옴, 차체는 기울어진 채)에서도
        # height_gained/forward_dist 조건만으로 조기 성공 처리될 수 있었음.
        is_level = abs(pitch) < PITCH_FLAT
        if (self._step_count > 10
                and height_gained > SUCCESS_HEIGHT_FRACTION * self._step_h
                and forward_dist > SUCCESS_MIN_FORWARD_DIST
                and is_level
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
