
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

rng = np.random.default_rng(42)


def generate_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Generate 3 days of simulated meal records
    start = pd.Timestamp("2026-10-20 00:00:00")
    meal_schedule = [
        ("08:00", "Oatmeal", 25),
        ("13:00", "Pasta", 40),
        ("19:00", "Chicken Salad", 15),
        ("08:00", "Pasta", 40),
        ("13:00", "Chicken Salad", 15),
        ("19:00", "Oatmeal", 25),
        ("08:00", "Chicken Salad", 15),
        ("13:00", "Oatmeal", 25),
        ("19:00", "Pasta", 40),
    ]

    meals = []

    for i, (time_str, food, simulated_rise) in enumerate(
        meal_schedule, start=1
    ):
        day = (i - 1) // 3
        hour, minute = map(int, time_str.split(":"))

        meal_time = (
            start
            + pd.Timedelta(days=day, hours=hour, minutes=minute)
        )

        meals.append({
            "meal_id": i,
            "timestamp": meal_time,
            "food": food,
            "_simulated_rise": simulated_rise,
        })

    meals_df = pd.DataFrame(meals)

    # Generate CGM readings every 5 minutes for 3 days
    timestamps = pd.date_range(
        start=start,
        periods=3 * 24 * 12,
        freq="5min",
    )

    hours = (
        timestamps.hour.to_numpy()
        + timestamps.minute.to_numpy() / 60
    )

    glucose = (
        95
        + 5 * np.sin(2 * np.pi * (hours - 6) / 24)
        + rng.normal(0, 2, len(timestamps))
    )

    # Add a simulated response after each meal
    for _, meal in meals_df.iterrows():
        elapsed = (
            timestamps - meal["timestamp"]
        ).total_seconds() / 60

        elapsed = np.asarray(elapsed)
        active = (elapsed >= 0) & (elapsed <= 180)
        t = elapsed[active]

        rise = meal["_simulated_rise"]
        response = rise * (t / 45) * np.exp(1 - t / 45)
        glucose[active] += response

    glucose = np.clip(glucose, 65, 250)

    cgm_df = pd.DataFrame({
        "timestamp": timestamps,
        "glucose_mg_dl": np.round(glucose, 1),
    })

    # Keep only the agreed team columns
    meals_df = meals_df[["meal_id", "timestamp", "food"]]

    cgm_df.to_csv(DATA_DIR / "cgm.csv", index=False)
    meals_df.to_csv(DATA_DIR / "meals.csv", index=False)

    print("Simulated CSV files created successfully!")
    print(f"CGM rows: {len(cgm_df)}")
    print(f"Meal rows: {len(meals_df)}")
    print("Files: data/cgm.csv, data/meals.csv")
    print("All values are simulated, not real medical data.")


if __name__ == "__main__":
    generate_data()
