# -*- coding: utf-8 -*-
"""
test_low_step.py — 1~6cm 사이 정수(cm) 높이 중 매번 무작위로 하나 뽑아
반복 테스트해서 PUSH/STOP 분기 판단이 제대로 되는지 검증

실행: python3 test_low_step.py
"""
import numpy as np
from wheg_env import WhegEnv, LOW_STEP_THRESH, HEIGHT_SAFETY_MARGIN

N_TRIALS = 20
CANDIDATE_CM = [1, 2, 3, 4, 5, 6]  # 이 정수(cm)들 중에서 매번 무작위로 하나 뽑음

env = WhegEnv(randomize_terrain=False)  # 우리가 직접 높이를 지정할 것

print(f"PUSH 분기 기준: 추정치 <= {LOW_STEP_THRESH - HEIGHT_SAFETY_MARGIN:.4f}m "
      f"(LOW_STEP_THRESH={LOW_STEP_THRESH} - SAFETY_MARGIN={HEIGHT_SAFETY_MARGIN})\n")

results = []
rng = np.random.default_rng()
for trial in range(N_TRIALS):
    true_h = rng.choice(CANDIDATE_CM) / 100.0  # cm -> m
    if True:  # (아래 for문 들여쓰기 유지용)
        obs, info = env.reset()
        env._step_h = true_h  # reset 이후 강제로 높이 지정
        # 모델은 randomize_terrain=False라 이미 기본 STEP_H로 빌드됨.
        # 실제로 높이를 바꾸려면 모델 자체를 재빌드해야 함:
        from climber_scene import build_xml
        import mujoco
        env.model = mujoco.MjModel.from_xml_string(build_xml(true_h))
        env.data = mujoco.MjData(env.model)
        mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
        mujoco.mj_forward(env.model, env.data)
        x, _, z = env.data.body("chassis").xpos
        env._prev_x, env._prev_z, env._start_z = float(x), float(z), float(z)
        env._mode, env._t_in_mode, env._clear_t = "DRIVE", 0.0, 0.0
        env._measure_max_h = 0.0

        prev_mode = "DRIVE"
        modes_seen_this_trial = set()
        resolved = False
        for step in range(400):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            modes_seen_this_trial.add(info["mode"])
            if true_h <= 0.03 and step < 20:
                x, _, z = env.data.body("chassis").xpos
                print(f"    [진단] step={step:3d} mode={env._mode:8s} "
                      f"t_in_mode={env._t_in_mode:.3f}s measure_max_h={env._measure_max_h:.4f} "
                      f"z={float(z):.4f} terminated={terminated} truncated={truncated} "
                      f"info={info}")
            if prev_mode == "MEASURE" and info["mode"] in ("PUSH", "STOP"):
                correct = (info["mode"] == "PUSH") == (true_h <= LOW_STEP_THRESH)
                results.append((true_h, info["step_h_estimated"], info["mode"], correct))
                mark = "OK" if correct else "!! 오분류 !!"
                print(f"  실제={true_h*100:.0f}cm  추정={info['step_h_estimated']*100:.2f}cm  "
                      f"분기={info['mode']:>5}  {mark}")
                resolved = True
                break
            prev_mode = info["mode"]
            if terminated or truncated:
                break
        if not resolved:
            print(f"  [스킵] 실제={true_h*100:.0f}cm  400스텝 동안 MEASURE 통과 못함  "
                  f"(이번 시도에서 본 모드들: {modes_seen_this_trial})")

if results:
    n_correct = sum(1 for r in results if r[3])
    print(f"\n총 {len(results)}회 중 {n_correct}회 올바른 분기 "
          f"({100*n_correct/len(results):.0f}%)")
else:
    print("MEASURE를 통과한 표본이 없었어요.")
