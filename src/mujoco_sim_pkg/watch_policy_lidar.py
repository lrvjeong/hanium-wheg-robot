# -*- coding: utf-8 -*-
"""
watch_policy_lidar.py — LiDAR 기반 WhegEnv 학습 모델을 MuJoCo 뷰어에서 재생

같은 폴더에 climber_scene.py, wheg_env.py가 있어야 합니다.

이 버전은 mujoco.viewer.launch_passive() 대신 launch()(블로킹, 메인스레드)
방식을 씁니다. WSL 환경에 따라 launch_passive()가 별도 스레드에서 창을 열려다
실패(화면이 아예 안 뜸)하는 경우가 있어서, 그 문제를 피하기 위함입니다.
대신 정책을 먼저 끝까지 돌려서 각 스텝의 자세(qpos)를 전부 기록해두고,
그 기록을 launch()로 그대로 재생하는 방식으로 바꿨습니다.

실행 예:
    python3 watch_policy_lidar.py --model ./checkpoints/sac_wheg_v1/best_model.zip
"""
import os
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC

from wheg_env import WhegEnv


def collect_episodes(model, env, n_episodes, deterministic):
    """정책을 뷰어 없이 먼저 끝까지 돌려서, 각 스텝의 qpos를 기록.
    반환: [(step_h_cm, [qpos, qpos, ...], reward, reason, steps, mode), ...]
    """
    episodes = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        frames = [env.data.qpos.copy()]
        ep_reward = 0.0
        step_count = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.data.qpos.copy())
            ep_reward += reward
            step_count += 1
            done = terminated or truncated
        reason = info.get("termination_reason", "max_steps")
        episodes.append((
            info["step_h_true"] * 100, frames, ep_reward, reason,
            step_count, info.get("mode"),
        ))
        print(f"[{ep + 1}/{n_episodes}] 시뮬레이션 완료 — "
              f"계단 {info['step_h_true']*100:.2f}cm, reward={ep_reward:.2f}, "
              f"reason={reason}, steps={step_count}")
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                         help="불러올 .zip 모델 경로")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--slowdown", type=float, default=3.0,
                         help="재생 속도 배율 (1.0=실제속도, 3.0=3배 느리게, 기본 3.0)")
    parser.add_argument("--pause-between", type=float, default=1.5,
                         help="에피소드 사이 정지 시간 [s]")
    args = parser.parse_args()

    print(f"모델 로드: {args.model}")
    model = SAC.load(args.model)

    env = WhegEnv()

    print("정책 실행 중 (뷰어 없이 먼저 전체 궤적 계산)...")
    episodes = collect_episodes(model, env, args.episodes, args.deterministic)

    print("\n재생을 시작합니다. 뷰어 창을 닫으면 종료됩니다.")
    frame_dt = env.model.opt.timestep * 10 * args.slowdown

    # launch()는 블로킹이라, 재생 로직은 컨트롤 콜백 안에서 처리.
    # 실제 물리를 다시 계산하는 게 아니라, 기록해둔 qpos를 그대로 밀어넣기만 함
    # (진짜 "재생"이지 "재시뮬레이션"이 아님 — 정책 계산과 렌더링을 분리).
    state = {"ep": 0, "frame": 0, "last_t": time.time()}

    def controller(m, d):
        if state["ep"] >= len(episodes):
            d.qvel[:] = 0.0  # 마지막 자세에서 멈춰있도록
            return
        step_h, frames, reward, reason, steps, mode = episodes[state["ep"]]
        now = time.time()
        if now - state["last_t"] >= frame_dt:
            state["last_t"] = now
            if state["frame"] == 0:
                print(f"\n=== 에피소드 {state['ep'] + 1}/{len(episodes)} "
                      f"(계단 높이: {step_h:.2f}cm) ===")
            state["frame"] += 1
            if state["frame"] >= len(frames):
                print(f"에피소드 종료: reward={reward:.2f}, reason={reason}, "
                      f"steps={steps}, mode={mode}")
                state["ep"] += 1
                state["frame"] = 0
                state["last_t"] = time.time() + args.pause_between
                return
        # 시간이 아직 안 지났어도(프레임 전진 안 해도) 매 콜백마다 항상
        # "현재 프레임" 자세로 고정 — 이걸 안 하면 그 사이 물리엔진이 자체
        # 중력/접촉 계산으로 자세를 계속 바꿔버려서(드리프트), 다음 프레임에
        # 갑자기 스냅되는 게 반복되며 제자리에서 떠는/도는 것처럼 보임.
        cur = min(state["frame"], len(frames) - 1)
        d.qpos[:] = frames[cur]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    mujoco.set_mjcb_control(controller)
    mujoco.viewer.launch(env.model, env.data)

    env.close()


if __name__ == "__main__":
    main()
