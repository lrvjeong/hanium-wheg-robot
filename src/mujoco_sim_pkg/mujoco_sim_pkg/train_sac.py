# -*- coding: utf-8 -*-
"""
train_sac.py — WhegEnv(계단 등반 wheg 로봇)를 SAC로 학습

실행:
    python3 train_sac.py

필요 패키지:
    pip install stable-baselines3 gymnasium mujoco --break-system-packages

같은 폴더에 climber_scene.py, wheg_env.py가 있어야 합니다.
"""
import os
import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from wheg_env import WhegEnv


def make_env():
    def _init():
        env = WhegEnv()
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=8,
                         help="병렬 환경 개수 (CPU 코어 수에 맞춰 조정, 기본 8)")
    parser.add_argument("--total-steps", type=int, default=500_000,
                         help="총 학습 스텝 수")
    parser.add_argument("--run-name", type=str, default="sac_wheg_v1")
    parser.add_argument("--resume", type=str, default=None,
                         help="이어서 학습할 체크포인트 .zip 경로")
    args = parser.parse_args()

    log_dir = f"./logs/{args.run_name}"
    ckpt_dir = f"./checkpoints/{args.run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ----- 학습용 병렬 환경 -----
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env()])

    # ----- 평가용 단일 환경 (학습 중 별도로 성능 체크) -----
    eval_env = DummyVecEnv([make_env()])

    if args.resume:
        print(f"체크포인트에서 이어서 학습: {args.resume}")
        model = SAC.load(args.resume, env=env, tensorboard_log=log_dir)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            verbose=1,
            tensorboard_log=log_dir,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path=ckpt_dir,
        name_prefix="sac_wheg",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=ckpt_dir,
        log_path=log_dir,
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(
        total_timesteps=args.total_steps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = os.path.join(ckpt_dir, f"{args.run_name}_final")
    model.save(final_path)
    print(f"학습 완료. 최종 모델 저장: {final_path}.zip")
    print(f"텐서보드로 확인: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
