"""시연/테스트용 가상 CGM 하루치 CSV 생성기. (앱 코드와 무관한 입력 파일 만들기용)

    python make_demo_cgm.py --date 2026-10-24 --meals 08:00:72,12:30:33,15:30:57,19:00:64 --out demo_cgm_1024.csv

--meals  "식사시각:상승폭(mg/dL)"을 쉼표로. 이 값은 이 파일의 곡선을 만들 때만 쓰이고,
         앱은 파일의 혈당 숫자만 읽어서 baseline/peak/rise를 새로 계산한다.
출력     실제 기기 내보내기처럼 현지 시간(시간대 표시 없음) + 컬럼명 "Timestamp", "Glucose (mg/dL)".
모든 값은 가상이며 실제 측정값이 아니다.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

PEAK_MINUTE = 45
BASE = 100.0


def parse_meals(spec: str):
    meals = []
    for item in spec.split(","):
        hh, mm, rise = item.strip().split(":")
        meals.append((int(hh) * 60 + int(mm), float(rise)))
    return meals


def build_day(date: str, meals, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(f"{date} 00:00", f"{date} 23:55", freq="5min")
    minutes = np.arange(len(times)) * 5.0

    glucose = BASE + 3.0 * np.sin(2 * np.pi * minutes / 720.0)  # 완만한 하루 변동
    for start, rise in meals:
        e = minutes - start
        up = (e >= 0) & (e <= PEAK_MINUTE)
        down = (e > PEAK_MINUTE) & (e <= 120)
        glucose[up] += rise * (0.5 - 0.5 * np.cos(np.pi * e[up] / PEAK_MINUTE))
        glucose[down] += rise * (0.5 + 0.5 * np.cos(np.pi * (e[down] - PEAK_MINUTE) / (120 - PEAK_MINUTE)))
    glucose += rng.normal(0, 0.8, size=len(glucose))
    glucose = np.clip(glucose, 60, 250)

    return pd.DataFrame(
        {"Timestamp": times.strftime("%Y-%m-%d %H:%M"), "Glucose (mg/dL)": np.round(glucose, 1)}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--meals", required=True, help="08:00:72,12:30:33 형식")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    df = build_day(args.date, parse_meals(args.meals), args.seed)
    df.to_csv(args.out, index=False, lineterminator="\n")
    print(f"Created {args.out}: {len(df)} rows (synthetic demo values, not real measurements)")


if __name__ == "__main__":
    main()