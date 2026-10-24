"""GlucoWise P0 step 3-2: link meals (time + foods) to CGM readings.

Each meal gets exactly ONE glucose response (baseline, peak, rise). The rise is never split among
the foods of the meal, and nothing here says that a food causes a glucose response.

baseline / peak / rise are computed by glucose_analysis._metrics (not re-implemented here):
  baseline = mean of readings in [meal - 15 min, meal)      (meal time excluded)
  peak     = max of readings in [meal, meal + 2 h]          (both ends included)
  rise     = peak - baseline
This module only adds (a) the "enough readings" gate, (b) overlap cautions, (c) multi-food meals.

Demo thresholds (agreed for the demo, NOT medically validated):
  fewer than MIN_PRE_READINGS readings in the 15 min before the meal  -> insufficient_data
  fewer than MIN_POST_READINGS readings from the meal to +2 h         -> insufficient_data
No value is ever invented when the data is insufficient.

Inputs
  cgm   : DataFrame with ``time`` (UTC, tz-aware) and ``glucose_mg_dl`` (e.g. cgm_import's
          ``result.readings``), or the CgmImportResult itself. If a ``user_id`` column exists and
          ``user_id=`` is given, only that user's rows are used.
  meals : iterable of {"meal_time": str|datetime, "foods": [str, ...],
                        optional "meal_id", optional "data_type"}.
          A meal_time without a UTC offset is read as America/Los_Angeles.
"""
from __future__ import annotations

import pandas as pd

try:  # same import style as agent_context.py: package import first, then flat import
    from .glucose_analysis import BASELINE_WINDOW, PEAK_WINDOW, UNIT, _metrics
    from .cgm_import import DEFAULT_TZ, parse_timestamp
except ImportError:
    from glucose_analysis import BASELINE_WINDOW, PEAK_WINDOW, UNIT, _metrics
    from cgm_import import DEFAULT_TZ, parse_timestamp

MIN_PRE_READINGS = 2
MIN_POST_READINGS = 6

STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient_data"

INTERPRETATION_NOTE = (
    "Baseline, peak and rise describe the whole meal. They are not assigned to individual foods "
    "and do not show that any food caused the change."
)


def _fallback_normalize(name: str) -> str:
    """Same key rule glucose_analysis.get_meal_history already applies to ``food``."""
    return str(name).strip().lower().replace(" ", "_")


def _load_normalizer():
    """Use the team's normalization.normalize_food when it exists, else the fallback rule."""
    try:
        from .normalization import normalize_food  # type: ignore
    except ImportError:
        try:
            from normalization import normalize_food  # type: ignore
        except ImportError:
            return _fallback_normalize
    return normalize_food


def _clean_foods(foods, normalizer) -> list[str]:
    if isinstance(foods, str):
        foods = [foods]
    keys: list[str] = []
    for item in foods or []:
        if not str(item).strip():
            continue
        key = normalizer(str(item))
        if key and key not in keys:
            keys.append(key)
    if not keys:
        raise ValueError("a meal needs at least one food")
    return keys


def _prepare_cgm(cgm, user_id) -> pd.DataFrame:
    if hasattr(cgm, "readings"):
        cgm = cgm.readings
    if "time" not in cgm.columns or "glucose_mg_dl" not in cgm.columns:
        raise ValueError("cgm needs 'time' (UTC) and 'glucose_mg_dl' columns; use cgm_import.parse_cgm_csv")
    if user_id is not None and "user_id" in cgm.columns:
        cgm = cgm[cgm["user_id"] == user_id]
    if getattr(cgm["time"].dt, "tz", None) is None:
        raise ValueError("cgm['time'] must be timezone-aware (UTC)")
    return cgm


def _window_counts(cgm_time: pd.Series, t: pd.Timestamp) -> tuple[int, int]:
    """Reading counts with exactly the same boundaries _metrics uses."""
    pre = int(((cgm_time >= t - BASELINE_WINDOW) & (cgm_time < t)).sum())
    post = int(((cgm_time >= t) & (cgm_time <= t + PEAK_WINDOW)).sum())
    return pre, post


def _minutes(delta: pd.Timedelta) -> float:
    return round(delta.total_seconds() / 60.0, 1)


def analyze_meals(
    cgm,
    meals,
    *,
    user_id=None,
    default_tz: str = DEFAULT_TZ,
    data_type: str = "uploaded",
    normalizer=None,
) -> list[dict]:
    """One result per meal, in meal-time order. Raises ValueError for an unusable meal input.

    ``data_type`` is the default label ("uploaded", "simulated", ...); a meal may carry its own.
    """
    cgm = _prepare_cgm(cgm, user_id)
    normalizer = normalizer or _load_normalizer()

    parsed = []
    for index, meal in enumerate(meals):
        try:
            ts = parse_timestamp(meal["meal_time"], default_tz)
            foods = _clean_foods(meal.get("foods"), normalizer)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"meal #{index + 1}: {exc}") from exc
        parsed.append(
            {
                "index": index,
                "ts": ts,
                "utc": ts.tz_convert("UTC"),
                "foods": foods,
                "meal_id": meal.get("meal_id"),
                "data_type": meal.get("data_type", data_type),
            }
        )
    parsed.sort(key=lambda m: (m["utc"], m["index"]))

    results = []
    for pos, meal in enumerate(parsed):
        t = meal["utc"]
        pre, post = _window_counts(cgm["time"], t)

        reasons = []
        if pre < MIN_PRE_READINGS:
            reasons.append("too_few_readings_before_meal")
        if post < MIN_POST_READINGS:
            reasons.append("too_few_readings_after_meal")

        baseline = peak = rise = None
        message = None
        if not reasons:
            metrics = _metrics(cgm, t)  # the existing calculation
            if metrics is None:  # cannot happen once the gate passed; kept so we never invent a value
                reasons = ["no_readings_in_window"]
            else:
                baseline, peak, rise = (round(x, 1) for x in metrics)
        if reasons:
            message = (
                f"Not enough CGM data to compute a response: {pre} reading(s) in the 15 min before the meal "
                f"(need {MIN_PRE_READINGS}), {post} reading(s) from the meal to 2 h after (need {MIN_POST_READINGS})."
            )

        warnings = []
        if pos + 1 < len(parsed):
            gap = parsed[pos + 1]["utc"] - t
            if gap <= PEAK_WINDOW:
                warnings.append(
                    {
                        "code": "overlaps_next_meal",
                        "minutes": _minutes(gap),
                        "message": (
                            f"The next meal starts {_minutes(gap)} min later, inside this meal's 2-hour window. "
                            "The peak may not be explained by this meal alone."
                        ),
                    }
                )
        if pos > 0:
            gap = t - parsed[pos - 1]["utc"]
            if gap <= PEAK_WINDOW:
                warnings.append(
                    {
                        "code": "overlaps_previous_meal",
                        "minutes": _minutes(gap),
                        "message": (
                            f"The previous meal started {_minutes(gap)} min earlier, so its response may still be "
                            "affecting this meal's baseline and peak."
                        ),
                    }
                )

        results.append(
            {
                "meal_id": meal["meal_id"],
                "meal_time": meal["ts"].isoformat(),
                "meal_time_utc": t.isoformat(),
                "foods": meal["foods"],
                "status": STATUS_INSUFFICIENT if reasons else STATUS_OK,
                "reasons": reasons,
                "message": message,
                "baseline_mg_dl": baseline,
                "peak_mg_dl": peak,
                "rise_mg_dl": rise,
                "pre_meal_readings": pre,
                "post_meal_readings": post,
                "warnings": warnings,
                "unit": UNIT,
                "data_type": meal["data_type"],
                "interpretation_note": INTERPRETATION_NOTE,
            }
        )
    return results


def summarize_by_food(results: list[dict]) -> list[dict]:
    """Group meals by every food they contain (descriptive only).

    A meal with several foods appears under each of its foods, with its single whole-meal response.
    The averages are "meals that contained this food", NOT the effect of the food.
    """
    buckets: dict[str, list[dict]] = {}
    for result in results:
        for food in result["foods"]:
            buckets.setdefault(food, []).append(result)
    rows = []
    for food, meals in buckets.items():
        ok = [m for m in meals if m["status"] == STATUS_OK]
        rows.append(
            {
                "food": food,
                "meals_containing_food": len(meals),
                "meals_with_response": len(ok),
                "meals_insufficient_data": len(meals) - len(ok),
                "meals_with_overlap_caution": sum(1 for m in ok if m["warnings"]),
                "average_baseline_mg_dl": round(sum(m["baseline_mg_dl"] for m in ok) / len(ok), 1) if ok else None,
                "average_peak_mg_dl": round(sum(m["peak_mg_dl"] for m in ok) / len(ok), 1) if ok else None,
                "average_rise_mg_dl": round(sum(m["rise_mg_dl"] for m in ok) / len(ok), 1) if ok else None,
                "note": "Average over meals that contained this food; not the effect of the food itself.",
            }
        )
    rows.sort(key=lambda r: (r["average_rise_mg_dl"] is None, -(r["average_rise_mg_dl"] or 0)))
    return rows
