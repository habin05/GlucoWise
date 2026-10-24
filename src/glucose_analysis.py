"""GlucoWise Step 2: per-meal glucose metrics and past-meal comparison.

All data is SYNTHETIC demo data (not real measurements, not medical advice). The functions only
report numbers computed from the CSVs; they do not say that a food causes a glucose response.

Definitions (team-agreed defaults, change the constants below if the team decides otherwise):
  baseline = mean of CGM readings in the 15 minutes before the meal (meal time excluded)
  peak     = highest CGM reading from the meal time up to 2 hours after it
  rise     = peak - baseline

Every function takes optional ``cgm`` / ``meals`` DataFrames; if omitted, data/*.csv is loaded.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BASELINE_WINDOW = pd.Timedelta(minutes=15)
PEAK_WINDOW = pd.Timedelta(hours=2)
CURVE_BEFORE = pd.Timedelta(minutes=30)
UNIT = "mg/dL"
DATA_TYPE = "simulated"


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cgm, meals) with an extra UTC ``time`` column used for window arithmetic."""
    cgm = pd.read_csv(data_dir / "cgm.csv")
    meals = pd.read_csv(data_dir / "meals.csv")
    cgm["time"] = pd.to_datetime(cgm["timestamp"], utc=True)
    meals["time"] = pd.to_datetime(meals["timestamp"], utc=True)
    return cgm, meals


def _get(cgm, meals):
    if cgm is None or meals is None:
        loaded_cgm, loaded_meals = load_data()
        cgm = loaded_cgm if cgm is None else cgm
        meals = loaded_meals if meals is None else meals
    return cgm, meals


def _meal_row(meals, user_id, meal_id):
    rows = meals[(meals["user_id"] == user_id) & (meals["meal_id"] == meal_id)]
    if rows.empty:
        raise KeyError(f"No meal {meal_id!r} for user {user_id!r}")
    return rows.iloc[0]


def _metrics(user_cgm, meal_time):
    """(baseline, peak, rise) or None when the CGM has no data in either window."""
    before = user_cgm[(user_cgm["time"] >= meal_time - BASELINE_WINDOW) & (user_cgm["time"] < meal_time)]
    after = user_cgm[(user_cgm["time"] >= meal_time) & (user_cgm["time"] <= meal_time + PEAK_WINDOW)]
    if before.empty or after.empty:
        return None
    baseline = float(before["glucose_mg_dl"].mean())
    peak = float(after["glucose_mg_dl"].max())
    return baseline, peak, peak - baseline


def get_meal_glucose(user_id, meal_id, cgm=None, meals=None):
    """Baseline, peak and rise for one meal. Returns None if the CGM window is empty."""
    cgm, meals = _get(cgm, meals)
    meal = _meal_row(meals, user_id, meal_id)
    result = _metrics(cgm[cgm["user_id"] == user_id], meal["time"])
    if result is None:
        return None
    baseline, peak, rise = result
    return {
        "user_id": user_id,
        "meal_id": meal_id,
        "baseline": round(baseline, 1),
        "peak": round(peak, 1),
        "rise": round(rise, 1),
        "unit": UNIT,
        "data_type": DATA_TYPE,
    }


def get_glucose_curve(user_id, meal_id, cgm=None, meals=None):
    """CGM readings from 30 min before to 2 h after the meal (for charts)."""
    cgm, meals = _get(cgm, meals)
    meal = _meal_row(meals, user_id, meal_id)
    t = meal["time"]
    window = cgm[(cgm["user_id"] == user_id) & (cgm["time"] >= t - CURVE_BEFORE) & (cgm["time"] <= t + PEAK_WINDOW)]
    return [
        {
            "timestamp": row.timestamp,
            "minutes_from_meal": int((row.time - t).total_seconds() // 60),
            "glucose_mg_dl": float(row.glucose_mg_dl),
        }
        for row in window.itertuples()
    ]


def get_meal_history(user_id, food=None, cgm=None, meals=None):
    """Past meals of one user, optionally only those whose primary food is ``food``.

    ``food`` matches the ``food`` key in meals.csv (e.g. "white_rice"; spaces/case are tolerated).
    Side dishes are only visible in ``description``; they are not searchable keys.
    """
    cgm, meals = _get(cgm, meals)
    user_meals = meals[meals["user_id"] == user_id].sort_values("time")
    if food is not None:
        key = str(food).strip().lower().replace(" ", "_")
        user_meals = user_meals[user_meals["food"] == key]
    user_cgm = cgm[cgm["user_id"] == user_id]
    history = []
    for meal in user_meals.itertuples():
        result = _metrics(user_cgm, meal.time)
        history.append(
            {
                "meal_id": meal.meal_id,
                "timestamp": meal.timestamp,
                "food": meal.food,
                "description": meal.description,
                "primary_portion_g": float(meal.primary_portion_g),
                "baseline": None if result is None else round(result[0], 1),
                "peak": None if result is None else round(result[1], 1),
                "rise": None if result is None else round(result[2], 1),
                "unit": UNIT,
                "data_type": DATA_TYPE,
            }
        )
    return history


def compare_foods(user_id, cgm=None, meals=None):
    """Per-food summary of one user's past meals (count, average rise, average portion)."""
    history = [h for h in get_meal_history(user_id, cgm=cgm, meals=meals) if h["rise"] is not None]
    if not history:
        return []
    df = pd.DataFrame(history)
    summary = (
        df.groupby("food")
        .agg(
            meal_count=("rise", "size"),
            average_baseline=("baseline", "mean"),
            average_peak=("peak", "mean"),
            average_rise=("rise", "mean"),
            average_portion_g=("primary_portion_g", "mean"),
        )
        .reset_index()
        .sort_values("average_rise", ascending=False)
    )
    out = []
    for row in summary.itertuples():
        out.append(
            {
                "food": row.food,
                "meal_count": int(row.meal_count),
                "average_baseline": round(row.average_baseline, 1),
                "average_peak": round(row.average_peak, 1),
                "average_rise_mg_dl": round(row.average_rise, 1),
                "average_portion_g": round(row.average_portion_g, 1),
            }
        )
    return out


def analyze_user_patterns(user_id, cgm=None, meals=None):
    """Pre-computed summary to hand to the AI layer. Descriptive only: no causal or medical claims."""
    return {
        "user_id": user_id,
        "data_type": DATA_TYPE,
        "unit": UNIT,
        "definition": "baseline = mean of the 15 min before the meal; peak = max within 2 h after; rise = peak - baseline",
        "patterns": compare_foods(user_id, cgm=cgm, meals=meals),
    }


if __name__ == "__main__":
    import json

    for uid in ("A", "B", "C"):
        print(json.dumps(analyze_user_patterns(uid), indent=2))
    print("Reminder: synthetic demo values, not medical measurements.")
