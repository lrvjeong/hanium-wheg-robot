# -*- coding: utf-8 -*-
"""
watch_policy_video.py — 학습된 WhegEnv 정책을 실행하고 MP4 영상으로 저장

watch_policy_lidar.py(라이브 뷰어 창)가 WSL 환경에서 계속 크래시가 나서,
그 대안으로 만든 스크립트입니다. 뷰어 창을 아예 안 띄우고, mujoco의
오프스크린 렌더러로 프레임을 그려서 바로 MP4 파일로 저장합니다.
GLFW 창 관리 자체가 없어서 지금까지 겪은 뷰어 문제(안 뜸, 버벅임,
longjmp 크래시)들을 원천적으로 피할 수 있습니다.

실행 예:
    python3 watch_policy_video.py --model ./checkpoints/sac_wheg_v1/best_model.zip
    (결과: policy_playback.mp4 파일 생성)
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")  # 오프스크린 렌더링 (창 불필요)

import argparse

import numpy as np
import mujoco
import imageio
from stable_baselines3 import SAC

from wheg_env import WhegEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--out", type=str, default="policy_playback.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    print(f"모델 로드: {args.model}")
    model = SAC.load(args.model)

    env = WhegEnv(randomize_terrain=True, domain_randomize=False)
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)

    # 카메라: 로봇을 옆에서 보는 각도로 고정 (필요하면 azimuth/distance 조정)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90, -20, 1.8
    cam.lookat[:] = [0.5, 0, 0.1]

    frames = []
    sim_dt = env.model.opt.timestep * 10  # 정책 스텝당 실제 시간
    frame_every = max(1, round(1.0 / (args.fps * sim_dt)))

    for ep in range(args.episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        step_count = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step_count += 1
            done = terminated or truncated
            if step_count % frame_every == 0 or done:
                renderer.update_scene(env.data, camera=cam)
                frames.append(renderer.render())
        reason = info.get("termination_reason", "max_steps")
        print(f"[{ep + 1}/{args.episodes}] 계단 {info['step_h_true']*100:.2f}cm, "
              f"reward={ep_reward:.2f}, reason={reason}, steps={step_count}")

    print(f"프레임 {len(frames)}개, 영상 저장 중: {args.out}")
    imageio.mimsave(args.out, frames, fps=args.fps)
    print("완료.")

    env.close()


if __name__ == "__main__":
    main()
