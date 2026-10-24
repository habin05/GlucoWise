"""Generate deterministic synthetic CGM and meal data for GlucoWise Step 1.

All glucose values and scenarios are invented for a hackathon demo.
They are not real measurements or medically validated predictions.

Run from the project root:
    python src/data_generator.py

Design notes
- Baseline and peak values come from the SCENARIOS table below and are
  reproduced exactly (baseline = mean of the 4 readings from t-15 to t;
  peak = the single maximum reading in (t, t+120 min]).
- Time-to-peak comes from PEAK_MINUTES, a schedule-position cycle that is
  identical for every user. It is a synthetic drawing choice, NOT a food
  effect, and must not be interpreted as one.
- All local times are built from wall-clock dates, so the generator stays
  correct if the date range crosses a daylight-saving change.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
START_DATE = "2026-10-01"
DAYS = 7
TIMEZONE = "America/Los_Angeles"  # keep identical to schema.TIMEZONE
CGM_INTERVAL = "5min"
RANDOM_SEED = 42
# Non-peak readings in a meal's 2-hour window are capped this far below the
# peak so the target peak is the one and only maximum (no ties).
PEAK_MARGIN = 0.2

# Each tuple is (time, primary food key, display description, primary portion g,
# pre-meal baseline mg/dL, post-meal peak mg/dL).
SCENARIOS = {
    "A": {
        "user_id": "A",
        "meals": [
            ("08:00", "white_rice", "White rice 150g; steamed egg 100g; spinach 70g; kimchi 30g", 150, 100, 172),
            ("12:30", "brown_rice", "Brown rice 170g; grilled chicken breast 100g; broccoli 80g; stir-fried mushrooms 60g", 170, 99, 138),
            ("18:30", "whole_oats", "Cooked whole oats 220g; grilled beef 100g; bok choy 80g; radish salad 50g", 220, 101, 130),
            ("08:00", "brown_rice", "Brown rice 170g; steamed egg 100g; spinach 70g; kimchi 30g", 170, 101, 144),
            ("12:30", "whole_oats", "Cooked whole oats 220g; grilled chicken breast 100g; broccoli 80g; stir-fried mushrooms 60g", 220, 97, 131),
            ("18:30", "white_rice", "White rice 150g; grilled beef 100g; bok choy 80g; radish salad 50g", 150, 102, 178),
            ("08:00", "whole_oats", "Cooked whole oats 220g; steamed egg 100g; spinach 70g; kimchi 30g", 220, 98, 129),
            ("12:30", "white_rice", "White rice 150g; grilled chicken breast 100g; broccoli 80g; stir-fried mushrooms 60g", 150, 103, 173),
            ("18:30", "brown_rice", "Brown rice 170g; grilled beef 100g; bok choy 80g; radish salad 50g", 170, 100, 137),
            ("08:00", "white_rice", "White rice 150g; pan-fried tofu 100g; cucumber salad 70g; rolled omelet 100g", 150, 99, 173),
            ("12:30", "brown_rice", "Brown rice 170g; grilled pork 100g; stir-fried cabbage 80g; seasoned bean sprouts 60g", 170, 102, 143),
            ("18:30", "whole_oats", "Cooked whole oats 220g; chicken and mushroom stir-fry 120g; bell pepper 60g; grilled eggplant 80g", 220, 96, 124),
            ("08:00", "brown_rice", "Brown rice 170g; pan-fried tofu 100g; cucumber salad 70g; rolled omelet 100g", 170, 100, 145),
            ("12:30", "whole_oats", "Cooked whole oats 220g; grilled pork 100g; stir-fried cabbage 80g; seasoned bean sprouts 60g", 220, 98, 133),
            ("18:30", "white_rice", "White rice 150g; chicken and mushroom stir-fry 120g; bell pepper 60g; grilled eggplant 80g", 150, 101, 180),
            ("08:00", "whole_oats", "Cooked whole oats 220g; pan-fried tofu 100g; cucumber salad 70g; rolled omelet 100g", 220, 99, 131),
            ("12:30", "white_rice", "White rice 150g; grilled pork 100g; stir-fried cabbage 80g; seasoned bean sprouts 60g", 150, 100, 171),
            ("18:30", "brown_rice", "Brown rice 170g; chicken and mushroom stir-fry 120g; bell pepper 60g; grilled eggplant 80g", 170, 103, 143),
            ("08:00", "white_rice", "White rice 150g; steamed egg 100g; spinach 70g; kimchi 30g", 150, 97, 172),
            ("12:30", "brown_rice", "Brown rice 170g; grilled chicken breast 100g; broccoli 80g; stir-fried mushrooms 60g", 170, 99, 141),
            ("18:30", "whole_oats", "Cooked whole oats 220g; grilled beef 100g; bok choy 80g; radish salad 50g", 220, 102, 132),
        ],
    },
    "B": {
        "user_id": "B",
        "meals": [
            ("08:00", "banana", "Banana 80g; boiled eggs 100g; Greek yogurt 150g", 80, 98, 162),
            ("12:30", "apple", "Apple 110g; grilled chicken breast 120g; salad 100g; whole-wheat bread 50g", 110, 101, 129),
            ("18:30", "mango", "Mango 100g; grilled salmon 120g; brown rice 100g; broccoli 100g", 100, 100, 169),
            ("08:00", "orange", "Orange 130g; boiled eggs 100g; Greek yogurt 150g", 130, 99, 129),
            ("12:30", "pineapple", "Pineapple 120g; grilled chicken breast 120g; salad 100g; whole-wheat bread 50g", 120, 102, 165),
            ("18:30", "strawberry", "Strawberries 180g; grilled salmon 120g; brown rice 100g; broccoli 100g", 180, 98, 122),
            ("08:00", "mango", "Mango 100g; boiled eggs 100g; Greek yogurt 150g", 100, 100, 166),
            ("12:30", "orange", "Orange 130g; grilled chicken breast 120g; salad 100g; whole-wheat bread 50g", 130, 101, 133),
            ("18:30", "banana", "Banana 80g; grilled salmon 120g; brown rice 100g; broccoli 100g", 80, 103, 171),
            ("08:00", "strawberry", "Strawberries 180g; omelet 120g; plain yogurt 150g", 180, 97, 120),
            ("12:30", "banana", "Banana 80g; grilled beef 120g; lettuce salad 100g; whole-wheat bread 50g", 80, 100, 162),
            ("18:30", "apple", "Apple 110g; grilled chicken 120g; brown rice 100g; spinach 80g", 110, 101, 130),
            ("08:00", "pineapple", "Pineapple 120g; omelet 120g; plain yogurt 150g", 120, 99, 161),
            ("12:30", "strawberry", "Strawberries 180g; grilled beef 120g; lettuce salad 100g; whole-wheat bread 50g", 180, 98, 125),
            ("18:30", "mango", "Mango 100g; grilled chicken 120g; brown rice 100g; spinach 80g", 100, 102, 173),
            ("08:00", "apple", "Apple 110g; omelet 120g; plain yogurt 150g", 110, 100, 127),
            ("12:30", "mango", "Mango 100g; grilled beef 120g; lettuce salad 100g; whole-wheat bread 50g", 100, 101, 165),
            ("18:30", "orange", "Orange 130g; grilled chicken 120g; brown rice 100g; spinach 80g", 130, 99, 130),
            ("08:00", "banana", "Banana 80g; boiled eggs 100g; Greek yogurt 150g", 80, 101, 166),
            ("12:30", "pineapple", "Pineapple 120g; grilled chicken breast 120g; salad 100g; whole-wheat bread 50g", 120, 100, 160),
            ("18:30", "strawberry", "Strawberries 180g; grilled salmon 120g; brown rice 100g; broccoli 100g", 180, 102, 128),
        ],
    },
    "C": {
        "user_id": "C",
        "meals": [
            ("08:00", "boiled_potato", "Boiled potato 160g; scrambled eggs 100g; cucumber 80g", 160, 98, 168),
            ("12:30", "baked_sweet_potato", "Baked sweet potato 140g; grilled chicken breast 120g; broccoli 100g", 140, 100, 159),
            ("18:30", "white_rice", "White rice 130g; grilled beef 120g; spinach 80g; stir-fried mushrooms 60g", 130, 101, 136),
            ("08:00", "baked_sweet_potato", "Baked sweet potato 140g; scrambled eggs 100g; cucumber 80g", 140, 97, 158),
            ("12:30", "white_rice", "White rice 130g; grilled chicken breast 120g; broccoli 100g", 130, 102, 140),
            ("18:30", "boiled_potato", "Boiled potato 160g; grilled beef 120g; spinach 80g; stir-fried mushrooms 60g", 160, 99, 167),
            ("08:00", "white_rice", "White rice 130g; scrambled eggs 100g; cucumber 80g", 130, 99, 132),
            ("12:30", "boiled_potato", "Boiled potato 160g; grilled chicken breast 120g; broccoli 100g", 160, 103, 176),
            ("18:30", "baked_sweet_potato", "Baked sweet potato 140g; grilled beef 120g; spinach 80g; stir-fried mushrooms 60g", 140, 100, 157),
            ("08:00", "boiled_potato", "Boiled potato 160g; boiled eggs 100g; tomato 100g", 160, 101, 167),
            ("12:30", "baked_sweet_potato", "Baked sweet potato 140g; grilled pork 120g; stir-fried cabbage 100g", 140, 98, 161),
            ("18:30", "white_rice", "White rice 130g; grilled fish 120g; bok choy 80g; grilled eggplant 80g", 130, 100, 137),
            ("08:00", "baked_sweet_potato", "Baked sweet potato 140g; boiled eggs 100g; tomato 100g", 140, 100, 160),
            ("12:30", "white_rice", "White rice 130g; grilled pork 120g; stir-fried cabbage 100g", 130, 101, 137),
            ("18:30", "boiled_potato", "Boiled potato 160g; grilled fish 120g; bok choy 80g; grilled eggplant 80g", 160, 102, 174),
            ("08:00", "white_rice", "White rice 130g; boiled eggs 100g; tomato 100g", 130, 96, 130),
            ("12:30", "boiled_potato", "Boiled potato 160g; grilled pork 120g; stir-fried cabbage 100g", 160, 100, 169),
            ("18:30", "baked_sweet_potato", "Baked sweet potato 140g; grilled fish 120g; bok choy 80g; grilled eggplant 80g", 140, 99, 157),
            ("08:00", "boiled_potato", "Boiled potato 160g; scrambled eggs 100g; cucumber 80g", 160, 99, 173),
            ("12:30", "baked_sweet_potato", "Baked sweet potato 140g; grilled chicken breast 120g; broccoli 100g", 140, 102, 164),
            ("18:30", "white_rice", "White rice 130g; grilled beef 120g; spinach 80g; stir-fried mushrooms 60g", 130, 98, 137),
        ],
    },
}

# Approximate, illustrative timing only; never a physiological claim.
PEAK_MINUTES = [40, 50, 45, 55, 45, 60, 50, 40, 55, 45, 60, 50, 40, 55, 45, 60, 50, 40, 55, 45, 50]

def iso_timestamp(ts: pd.Timestamp) -> str:
    """Return an ISO-8601 timestamp including UTC offset."""
    return ts.isoformat(timespec="seconds")


def _local(date_str: str, clock: str = "00:00") -> pd.Timestamp:
    """Wall-clock local time -> tz-aware Timestamp (DST-safe)."""
    return pd.Timestamp(f"{date_str} {clock}").tz_localize(TIMEZONE)


def _day_string(start_date: str, offset_days: int) -> str:
    """Calendar-date arithmetic on naive dates (no 24h-span assumption)."""
    return (pd.Timestamp(start_date) + pd.Timedelta(days=offset_days)).strftime("%Y-%m-%d")


def build_meals(start_date: str = START_DATE) -> pd.DataFrame:
    records = []
    for scenario_id, scenario in SCENARIOS.items():
        for index, (clock, food, description, portion, baseline, peak) in enumerate(scenario["meals"]):
            day = index // 3
            meal_time = _local(_day_string(start_date, day), clock)
            records.append({
                "meal_id": f"{scenario['user_id']}_D{day+1:02d}_{index%3+1}_{food}",
                "user_id": scenario["user_id"],
                "scenario_id": scenario_id,
                "timestamp": iso_timestamp(meal_time),
                "food": food,
                "description": description,
                # grams of the primary comparison food only (not the whole meal)
                "primary_portion_g": float(portion),
                # Intentionally blank: exact carbohydrate values were not provided.
                "carbs_g": np.nan,
                "meal_order": "unspecified",
                "_baseline_target": float(baseline),
                "_peak_target": float(peak),
                "_peak_minute": int(PEAK_MINUTES[index]),
            })
    return pd.DataFrame(records)


def build_cgm(meals: pd.DataFrame, start_date: str = START_DATE, days: int = DAYS) -> pd.DataFrame:
    """Create a continuous 5-minute synthetic trace with specified meal targets.

    Each meal's baseline target is held through the 15 minutes before meal time.
    A smooth response curve reaches the specified peak at the configured time.
    Between meals the trace gradually returns toward the user's baseline.
    Small deterministic noise is added everywhere, then the exact baseline
    readings and the single peak reading are re-asserted; every other reading
    in the 2-hour window is capped at (peak - PEAK_MARGIN) so the peak is unique.
    This is a constructed demo trace, not a physiological model.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    start = _local(start_date)
    end = _local(_day_string(start_date, days))
    # Absolute 5-minute grid: a DST day simply has 23 or 25 hours of readings.
    timestamps = pd.date_range(start=start, end=end, freq=CGM_INTERVAL, inclusive="left")
    ts_utc = timestamps.tz_convert("UTC")
    all_records = []

    for user_id, user_meals in meals.groupby("user_id", sort=True):
        user_meals = user_meals.copy()
        user_meals["timestamp_parsed"] = pd.to_datetime(user_meals["timestamp"], utc=True).dt.tz_convert(TIMEZONE)
        anchors = [(start, float(user_meals.iloc[0]["_baseline_target"]))]
        for _, meal in user_meals.iterrows():
            t = meal["timestamp_parsed"]
            anchors.append((t - pd.Timedelta(minutes=15), float(meal["_baseline_target"])))
            anchors.append((t, float(meal["_baseline_target"])))
        anchors.append((end, float(user_meals.iloc[-1]["_baseline_target"])))
        anchors = sorted(anchors, key=lambda item: item[0])
        anchor_x = np.array([a.timestamp() for a, _ in anchors], dtype=float)
        anchor_y = np.array([val for _, val in anchors], dtype=float)
        x = np.array([t.timestamp() for t in timestamps], dtype=float)
        glucose = np.interp(x, anchor_x, anchor_y)

        for _, meal in user_meals.iterrows():
            t0 = meal["timestamp_parsed"]
            peak_min = int(meal["_peak_minute"])
            baseline = float(meal["_baseline_target"])
            peak = float(meal["_peak_target"])
            elapsed = (x - t0.timestamp()) / 60
            rise = (elapsed >= 0) & (elapsed <= peak_min)
            fall = (elapsed > peak_min) & (elapsed <= 120)
            if rise.any():
                fraction = elapsed[rise] / peak_min
                glucose[rise] = baseline + (peak - baseline) * (0.5 - 0.5 * np.cos(np.pi * fraction))
            if fall.any():
                fraction = (elapsed[fall] - peak_min) / max(120 - peak_min, 1)
                glucose[fall] = peak - (peak - baseline) * (0.5 - 0.5 * np.cos(np.pi * fraction))

        glucose += rng.normal(0, 0.45, size=len(glucose))

        for _, meal in user_meals.iterrows():
            t0 = meal["timestamp_parsed"]
            baseline = float(meal["_baseline_target"])
            peak = float(meal["_peak_target"])
            peak_ts = t0 + pd.Timedelta(minutes=int(meal["_peak_minute"]))
            for offset in (-15, -10, -5, 0):
                idx = np.where(timestamps == t0 + pd.Timedelta(minutes=offset))[0]
                if len(idx):
                    glucose[idx[0]] = baseline
            window = (timestamps >= t0) & (timestamps <= t0 + pd.Timedelta(minutes=120))
            glucose[window] = np.minimum(glucose[window], peak - PEAK_MARGIN)
            idx = np.where(timestamps == peak_ts)[0]
            glucose[idx[0]] = peak  # the one and only maximum of the window

        glucose = np.clip(glucose, 60, 250)
        for ts, value in zip(timestamps, glucose):
            all_records.append({
                "user_id": user_id,
                "timestamp": iso_timestamp(ts),
                "glucose_mg_dl": round(float(value), 1),
            })

    cgm = pd.DataFrame(all_records)
    cgm["_utc"] = pd.to_datetime(cgm["timestamp"], utc=True)
    cgm = cgm.sort_values(["user_id", "_utc"]).drop(columns="_utc").reset_index(drop=True)
    return cgm


MEAL_COLUMNS = [
    "meal_id", "user_id", "scenario_id", "timestamp", "food",
    "description", "primary_portion_g", "carbs_g", "meal_order",
]


def generate(start_date: str = START_DATE, days: int = DAYS):
    """Return (cgm, meals) DataFrames without writing anything."""
    if days != 7:
        raise ValueError("The scenario tables define exactly 7 days (21 meals per user).")
    meals_internal = build_meals(start_date)
    cgm = build_cgm(meals_internal, start_date, days)
    meals = meals_internal[MEAL_COLUMNS].copy()
    meals["_utc"] = pd.to_datetime(meals["timestamp"], utc=True)
    meals = meals.sort_values(["user_id", "_utc"]).drop(columns="_utc").reset_index(drop=True)
    return cgm, meals


def main(data_dir: Path = DATA_DIR) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cgm, meals = generate()
    cgm.to_csv(data_dir / "cgm.csv", index=False, lineterminator="\n")
    meals.to_csv(data_dir / "meals.csv", index=False, lineterminator="\n")
    print(f"Created {data_dir / 'cgm.csv'}: {len(cgm):,} CGM rows")
    print(f"Created {data_dir / 'meals.csv'}: {len(meals):,} meal rows")
    print("Scenario counts:")
    print(meals.groupby(["user_id", "scenario_id"]).size().to_string())
    print("Reminder: these are synthetic demo values, not medical measurements.")


if __name__ == "__main__":
    main()
