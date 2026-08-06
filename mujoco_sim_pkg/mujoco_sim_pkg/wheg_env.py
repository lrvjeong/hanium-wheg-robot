# -*- coding: utf-8 -*-
"""
wheg_env.py — climber_scene.py(MuJoCo MJCF)를 감싸는 Gymnasium 환경

Observation (LIDAR_ROWS*LIDAR_COLS + 3 차원, 기본 5x5+3=28D):
    [0 .. R*C-1]  3D LiDAR 격자 거리값 [m] (실제 부품 근사: 최소 인식거리
                  LIDAR_MIN_RANGE=50mm 보다 가까우면 판독불가로 간주해 min값으로 clip,
                  미검출(범위 밖)은 LIDAR_MAX_RANGE로 처리)
    [-3] pitch         차체 피치각 [rad] (앞이 들리면 +)
    [-2] pitch_rate    피치 각속도 [rad/s]
    [-1] deploy_angle  스포크 전개각 [rad] (0=오므림 ~ DEPLOY_MAX=펼침)

Action (2D, [-1, 1] 정규화 → 내부에서 실제 범위로 스케일):
    [0] drive_cmd     양쪽 바퀴 동일 토크 명령 (-DRIVE_MAX ~ +DRIVE_MAX)
    [1] deploy_cmd    스포크 목표 전개각 명령 (0 ~ DEPLOY_MAX)

Reward / Termination: 이전 버전과 동일 (전진+높이 상승 보상, 전복 페널티, 성공 보너스)
"""
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco

from climber_scene import (
    build_xml, DEPLOY_MAX, DRIVE_MAX, N_STEPS, STEP_H,
    LIDAR_ROWS, LIDAR_COLS, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE,
)

# ===== 튜닝 상수 =====
PITCH_FLIP = 1.2
CTRL_COST_W = 0.01
PITCH_PENALTY_W = 0.3
FORWARD_REWARD_W = 8.0
HEIGHT_REWARD_W = 20.0
FLIP_PENALTY = 20.0
SUCCESS_BONUS = 50.0
N_SUBSTEPS = 10
MAX_EPISODE_STEPS = 500

N_LIDAR = LIDAR_ROWS * LIDAR_COLS
_LIDAR_NAMES = [f"lidar_{r}_{c}" for r in range(LIDAR_ROWS) for c in range(LIDAR_COLS)]


def _lidar_scan(data):
    """LiDAR 격자 전체를 읽어서 (N_LIDAR,) 배열로 반환.
    실제 부품처럼 LIDAR_MIN_RANGE보다 가까운 값은 판독불가로 보고 min range로 clip,
    미검출(-1, 즉 범위 밖)은 LIDAR_MAX_RANGE로 채움."""
    out = np.empty(N_LIDAR, dtype=np.float32)
    for i, name in enumerate(_LIDAR_NAMES):
        v = float(data.sensor(name).data[0])
        if v < 0:
            v = LIDAR_MAX_RANGE
        elif v < LIDAR_MIN_RANGE:
            v = LIDAR_MIN_RANGE
        out[i] = v
    return out


def _pitch(data):
    w, x, y, z = data.sensor("imu_quat").data
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return math.asin(s)


class WhegEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self._xml = build_xml()
        self.model = mujoco.MjModel.from_xml_string(self._xml)
        self.data = mujoco.MjData(self.model)

        obs_low = np.concatenate([
            np.full(N_LIDAR, LIDAR_MIN_RANGE, dtype=np.float32),
            np.array([-math.pi, -50.0, 0.0], dtype=np.float32),
        ])
        obs_high = np.concatenate([
            np.full(N_LIDAR, LIDAR_MAX_RANGE, dtype=np.float32),
            np.array([math.pi, 50.0, DEPLOY_MAX], dtype=np.float32),
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # action은 [-1, 1]로 정규화. step()에서 실제 범위로 스케일링.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._viewer = None
        self._step_count = 0
        self._prev_x = 0.0
        self._prev_z = 0.0
        self._top_z = N_STEPS * STEP_H  # 꼭대기 판정 높이 (계단 끝)

    # ------------------------------------------------------------
    def _get_obs(self):
        lidar = _lidar_scan(self.data)
        pitch = _pitch(self.data)
        pitch_rate = float(self.data.sensor("imu_gyro").data[1])
        deploy = float(self.data.joint("spoke_l0").qpos[0])
        return np.concatenate([
            lidar,
            np.array([pitch, pitch_rate, deploy], dtype=np.float32),
        ])

    def _apply_action(self, action):
        action = np.clip(action, -1.0, 1.0)
        drive = float(action[0]) * DRIVE_MAX
        deploy_target = (float(action[1]) + 1.0) / 2.0 * DEPLOY_MAX

        self.data.actuator("drive_l").ctrl = drive
        self.data.actuator("drive_r").ctrl = drive
        self.data.actuator("deploy_l").ctrl = deploy_target
        self.data.actuator("deploy_r").ctrl = deploy_target
        return drive, deploy_target

    # ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        if seed is not None:
            self.action_space.seed(seed)
        noise = self.np_random.uniform(-0.01, 0.01, size=self.model.nq)
        self.data.qpos[:] += noise
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        x, _, z = self.data.body("chassis").xpos
        self._prev_x, self._prev_z = float(x), float(z)

        return self._get_obs(), {}

    def step(self, action):
        drive, deploy_target = self._apply_action(action)

        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        pitch = float(obs[-3])

        x, _, z = self.data.body("chassis").xpos
        x, z = float(x), float(z)
        dx = x - self._prev_x
        dz = z - self._prev_z
        self._prev_x, self._prev_z = x, z

        ctrl_cost = CTRL_COST_W * (drive / DRIVE_MAX) ** 2
        pitch_penalty = PITCH_PENALTY_W * (pitch ** 2)

        reward = FORWARD_REWARD_W * dx + HEIGHT_REWARD_W * max(dz, 0.0)
        reward -= ctrl_cost + pitch_penalty

        terminated = False
        truncated = False
        info = {}

        if abs(pitch) > PITCH_FLIP:
            reward -= FLIP_PENALTY
            terminated = True
            info["termination_reason"] = "flipped"

        if z > self._top_z + 0.02 and not terminated:
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
    print("initial obs (lidar 앞부분 5개):", obs[:5], "... pitch/pitch_rate/deploy:", obs[-3:])
    total_reward = 0.0
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"episode ended at step {i}: {info}")
            obs, info = env.reset()
    print("random rollout total reward (20 steps):", total_reward)
