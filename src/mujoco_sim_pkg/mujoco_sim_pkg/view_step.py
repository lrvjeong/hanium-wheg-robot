# -*- coding: utf-8 -*-
"""
view_step.py — 특정 계단 높이로 고정한 씬을 MuJoCo 뷰어로 직접 확인

인터랙티브 모드(기본): 뷰어의 오른쪽 Control 패널에서 drive_l/drive_r/
deploy_l/deploy_r을 직접 움직여보면서 저단차에서 무슨 일이 일어나는지 확인 가능.

실행:
    python3 view_step.py --height 0.02          # 2cm 계단, 수동 조작
    python3 view_step.py --height 0.02 --auto    # 학습된 정책으로 자동 재생
"""
import os
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import argparse
import time

import mujoco
import mujoco.viewer

from climber_scene import build_xml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=float, default=0.02,
                         help="계단 높이 [m] (기본 0.02 = 2cm)")
    parser.add_argument("--auto", action="store_true",
                         help="지정하면 학습된 정책으로 자동 재생 (--model 필요)")
    parser.add_argument("--model", type=str, default=None,
                         help="--auto일 때 불러올 SAC 모델 경로")
    args = parser.parse_args()

    xml = build_xml(args.height)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    print(f"계단 높이 {args.height*100:.1f}cm 로 씬을 만들었어요.")

    if not args.auto:
        print("수동 조작 모드: 뷰어 오른쪽 Control 패널에서")
        print("  drive_l / drive_r : 바퀴 토크")
        print("  deploy_l / deploy_r : 스포크 전개각 (0=오므림)")
        print("을 직접 움직여보면서 확인하세요. Space=일시정지, Backspace=리셋")
        mujoco.viewer.launch(model, data)
        return

    # --auto: 학습된 정책으로 재생 (wheg_env.py의 상태기계 그대로 사용)
    if args.model is None:
        print("--auto를 쓰려면 --model 경로도 같이 주세요.")
        return

    from stable_baselines3 import SAC
    from wheg_env import WhegEnv

    sac_model = SAC.load(args.model)
    env = WhegEnv(randomize_terrain=False)
    env.model = model
    env.data = data
    env._step_h = args.height
    obs, info = env.reset()
    # reset()이 모델을 다시 랜덤화해버릴 수 있으니, 다시 강제 지정
    env.model, env.data = model, data
    env._step_h = args.height
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    mujoco.mj_forward(env.model, env.data)
    x, _, z = env.data.body("chassis").xpos
    env._prev_x, env._prev_z, env._start_z = float(x), float(z), float(z)
    env._mode, env._t_in_mode, env._clear_t = "DRIVE", 0.0, 0.0
    env._measure_max_h = 0.0
    obs, _ = env._get_obs()
    obs = None  # 아래에서 다시 구성
    lidar, rest = env._get_obs()
    import numpy as np
    obs = np.concatenate([lidar, rest])

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            action, _ = sac_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            if terminated or truncated:
                print(f"에피소드 종료: {info}")
                obs, info = env.reset()
            leftover = env.model.opt.timestep * 10 - (time.time() - t0)
            if leftover > 0:
                time.sleep(leftover)


if __name__ == "__main__":
    main()
