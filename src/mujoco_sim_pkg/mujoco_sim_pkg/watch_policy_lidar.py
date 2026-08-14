# -*- coding: utf-8 -*-
"""
watch_policy_lidar.py — LiDAR 기반 WhegEnv 학습 모델을 MuJoCo 뷰어에서 재생

같은 폴더에 climber_scene.py, wheg_env.py가 있어야 합니다.

실행 예:
    python3 watch_policy_lidar.py --model ./checkpoints/sac_wheg_v1/best_model.zip
"""
import os
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import argparse
import time

import mujoco
import mujoco.viewer
from stable_baselines3 import SAC

from wheg_env import WhegEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                         help="불러올 .zip 모델 경로")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--slowdown", type=float, default=3.0,
                         help="재생 속도 배율 (1.0=실제속도, 3.0=3배 느리게, 기본 3.0)")
    parser.add_argument("--pause-between", type=float, default=1.5,
                         help="에피소드 사이 정지 시간 [s] (다음 계단이 뭔지 볼 여유)")
    args = parser.parse_args()

    print(f"모델 로드: {args.model}")
    model = SAC.load(args.model)

    env = WhegEnv()
    viewer = None

    for ep in range(args.episodes):
        obs, info = env.reset()
        # reset()이 계단 높이를 랜덤화하면서 env.model/env.data를 완전히 새
        # 객체로 교체함 -> 뷰어를 예전 객체에 그대로 두면 화면이 멈춘 것처럼
        # 보임. 매 에피소드 새 model/data에 뷰어를 다시 연결해야 함.
        if viewer is not None:
            viewer.close()
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

        ep_reward = 0.0
        step_count = 0
        done = False
        print(f"\n=== 에피소드 {ep + 1}/{args.episodes} "
              f"(계단 높이: {info['step_h_true']*100:.2f}cm) ===")

        while not done and viewer.is_running():
            t0 = time.time()

            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step_count += 1
            done = terminated or truncated

            viewer.sync()

            leftover = (env.model.opt.timestep * 10 * args.slowdown) - (time.time() - t0)
            if leftover > 0:
                time.sleep(leftover)

        reason = info.get("termination_reason", "max_steps")
        print(f"에피소드 종료: reward={ep_reward:.2f}, reason={reason}, "
              f"steps={step_count}, mode={info.get('mode')}")
        time.sleep(args.pause_between)

    if viewer is not None:
        viewer.close()
    env.close()


if __name__ == "__main__":
    main()
