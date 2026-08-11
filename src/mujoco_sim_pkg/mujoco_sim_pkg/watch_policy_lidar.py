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
    args = parser.parse_args()

    print(f"모델 로드: {args.model}")
    model = SAC.load(args.model)

    env = WhegEnv()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(args.episodes):
            obs, info = env.reset()
            ep_reward = 0.0
            done = False
            print(f"\n=== 에피소드 {ep + 1}/{args.episodes} ===")

            while not done and viewer.is_running():
                t0 = time.time()

                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated

                viewer.sync()

                leftover = (env.model.opt.timestep * 10) - (time.time() - t0)
                if leftover > 0:
                    time.sleep(leftover)

            reason = info.get("termination_reason", "max_steps")
            print(f"에피소드 종료: reward={ep_reward:.2f}, reason={reason}")

    env.close()


if __name__ == "__main__":
    main()
