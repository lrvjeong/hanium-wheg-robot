# -*- coding: utf-8 -*-
"""
read_tb_log.py — 텐서보드 웹 접속 없이 로그(csv 대신 event 파일)에서 직접 값 확인

실행:
    python3 read_tb_log.py --logdir ./logs/sac_wheg_mesh_v2
"""
import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_files(logdir):
    pattern = os.path.join(logdir, "**", "events.out.tfevents.*")
    return sorted(glob.glob(pattern, recursive=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--n-points", type=int, default=10,
                         help="각 지표별로 앞/뒤 몇 개 값을 보여줄지")
    args = parser.parse_args()

    files = find_event_files(args.logdir)
    if not files:
        print(f"이벤트 파일을 못 찾았어요: {args.logdir}")
        return

    print(f"발견된 이벤트 파일 {len(files)}개\n")

    for fpath in files:
        print(f"=== {fpath} ===")
        ea = EventAccumulator(fpath)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if not tags:
            print("  (스칼라 데이터 없음)\n")
            continue

        for tag in sorted(tags):
            events = ea.Scalars(tag)
            values = [e.value for e in events]
            steps = [e.step for e in events]
            n = args.n_points
            print(f"  [{tag}]  총 {len(values)}개 포인트")
            print(f"    처음 값들: {list(zip(steps[:n], [round(v,3) for v in values[:n]]))}")
            print(f"    마지막 값들: {list(zip(steps[-n:], [round(v,3) for v in values[-n:]]))}")
            print(f"    최소={min(values):.3f}  최대={max(values):.3f}\n")


if __name__ == "__main__":
    main()
