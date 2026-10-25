
"""Accumulate uploaded CGM and meal records in SQLite and convert them into DataFrames for analysis.

Workflow:
    Upload CGM CSV -> parse_cgm_csv() -> save_cgm()           (SQLite cgm_readings)
    Save meal      -> local_db.save_meal()                   (SQLite meals, existing table unchanged)
    Before analysis -> with_user_data(base_cgm, base_meals, user) -> (cgm, meals)
                       -> Pass cgm=..., meals=... to all glucose_analysis functions

Glucose values are taken directly from the uploaded file. This module does not generate any values.
Sample CSV data (base) and database records are combined based on user_id.
"""
from __future__ import annotations

from contextlib import closing

import pandas as pd

try:  # Support both `src.user_data` package imports and direct execution from `src/`
    from . import local_db
    from .normalization import normalize_food
except ImportError:  # pragma: no cover
    import local_db
    from normalization import normalize_food

TIMEZONE = "America/Los_Angeles"  # Must match data_generator.TIMEZONE
BASELINE_WINDOW = pd.Timedelta(minutes=15)
PEAK_WINDOW = pd.Timedelta(hours=2)

TIME_HINTS = ("timestamp", "datetime", "date_time", "time", "date", "시간", "일시", "날짜")
GLUCOSE_HINTS = ("glucose", "sgv", "mg/dl", "mg_dl", "혈당", "value")


# ----------------------------------------------------------------- tables
def init_user_tables() -> None:
    """Create the meals table (via local_db.init_db) and the CGM table."""
    local_db.init_db()
    with closing(local_db.get_connection()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cgm_readings (
                demo_user TEXT NOT NULL,
                time_utc TEXT NOT NULL,
                glucose_mg_dl REAL NOT NULL,
                PRIMARY KEY (demo_user, time_utc)
            )
            """
        )


def reset_user(demo_user: str) -> None:
    """Reset demo data: delete this user's uploaded CGM and meal records (sample CSV data remains)."""
    with closing(local_db.get_connection()) as conn, conn:
        conn.execute("DELETE FROM cgm_readings WHERE demo_user = ?", (demo_user,))
        conn.execute("DELETE FROM meals WHERE demo_user = ?", (demo_user,))


# ------------------------------------------------------------- CGM upload
def _find_col(columns, hints, exclude=None):
    for hint in hints:
        for col in columns:
            if col != exclude and hint in str(col).strip().lower():
                return col
    return None


def parse_cgm_csv(source, tz: str = TIMEZONE) -> pd.DataFrame:
    """Convert a CGM CSV (file path or uploaded file object) into DataFrame[time_utc, glucose_mg_dl].

    - Finds time and glucose columns using keywords in their names (timestamp, time, glucose, etc.).
    - Treats timestamps without timezone information as local times in ``tz``.
      (Meal times are entered using the same timezone.)
    - Raises ValueError with a user-facing English error message if a problem occurs.
    """
    try:
        raw = pd.read_csv(source)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read the CSV file: {exc}") from exc

    t_col = _find_col(raw.columns, TIME_HINTS)
    g_col = _find_col(raw.columns, GLUCOSE_HINTS, exclude=t_col)
    if t_col is None or g_col is None:
        raise ValueError(f"Could not find the time and glucose columns. File columns: {list(raw.columns)}")

    text = raw[t_col].astype(str).str.strip()
    aware = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    if aware.all():
        times = pd.to_datetime(text, errors="coerce", utc=True)
    elif aware.any():
        raise ValueError("Some timestamps include timezone information while others do not. Please use a consistent format.")
    else:
        times = pd.to_datetime(text, errors="coerce").dt.tz_localize(
            tz, ambiguous="NaT", nonexistent="shift_forward"
        )
        times = times.dt.tz_convert("UTC")

    out = pd.DataFrame(
        {"time_utc": times, "glucose_mg_dl": pd.to_numeric(raw[g_col], errors="coerce")}
    ).dropna()
    out = out.drop_duplicates("time_utc").sort_values("time_utc").reset_index(drop=True)

    if out.empty:
        raise ValueError("No valid rows containing both time and glucose values were found.")
    if out["glucose_mg_dl"].median() < 35:
        raise ValueError("The glucose values appear to be in mmol/L. Please upload a file with values in mg/dL.")
    if not out["glucose_mg_dl"].between(20, 600).all():
        raise ValueError("Some glucose values are outside the range of 20–600 mg/dL. Please check the file.")
    return out


def save_cgm(demo_user: str, readings: pd.DataFrame) -> int:
    """Save the result of parse_cgm_csv. Overwrite values with the same timestamp and return the number of rows saved."""
    rows = [
        (demo_user, t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S+00:00"), float(g))
        for t, g in zip(readings["time_utc"], readings["glucose_mg_dl"])
    ]
    with closing(local_db.get_connection()) as conn, conn:
        conn.executemany(
            "INSERT OR REPLACE INTO cgm_readings (demo_user, time_utc, glucose_mg_dl) VALUES (?, ?, ?)",
            rows,
        )
    return len(rows)


def cgm_summary(demo_user: str) -> dict:
    """Return a summary of saved CGM data for UI display: row count, first/last timestamps (local), and number of days."""
    df = load_cgm_df(demo_user)
    if df.empty:
        return {"rows": 0, "first": None, "last": None, "days": 0}
    local = df["time"].dt.tz_convert(TIMEZONE)
    return {
        "rows": int(len(df)),
        "first": local.min().isoformat(timespec="minutes"),
        "last": local.max().isoformat(timespec="minutes"),
        "days": int(local.dt.date.nunique()),
    }


# ------------------------------------------- DB -> analysis DataFrames
def load_cgm_df(demo_user: str) -> pd.DataFrame:
    """Convert DB CGM data to the same columns as cgm.csv (user_id, timestamp, glucose_mg_dl), plus UTC ``time``."""
    with closing(local_db.get_connection()) as conn:
        rows = conn.execute(
            "SELECT time_utc, glucose_mg_dl FROM cgm_readings WHERE demo_user = ? ORDER BY time_utc",
            (demo_user,),
        ).fetchall()
    time = pd.to_datetime([r["time_utc"] for r in rows], utc=True)
    return pd.DataFrame(
        {
            "user_id": demo_user,
            "timestamp": [t.tz_convert(TIMEZONE).isoformat(timespec="seconds") for t in time],
            "glucose_mg_dl": [float(r["glucose_mg_dl"]) for r in rows],
            "time": time,
        }
    )


def meals_df(demo_user: str) -> pd.DataFrame:
    """Convert meals from the local_db meals table to the same columns as meals.csv, plus UTC ``time``.

    meal_id is ``db_<id>``. If portion size is unknown (e.g., from a photo), primary_portion_g is NaN.
    """
    records = []
    for row in local_db.get_meals(demo_user=demo_user):
        local = pd.Timestamp(row["timestamp"]).tz_localize(
            TIMEZONE, ambiguous=True, nonexistent="shift_forward"
        )
        portion = row["portion_g"]
        carbs = row["carbs_g"]
        records.append(
            {
                "meal_id": f"db_{row['id']}",
                "user_id": demo_user,
                "scenario_id": "user_input",
                "timestamp": local.isoformat(timespec="seconds"),
                "food": normalize_food(row["food"]),
                "description": row["description"] or row["food"],
                "primary_portion_g": float("nan") if portion is None else float(portion),
                "carbs_g": float("nan") if carbs is None else float(carbs),
                "meal_order": "unspecified",
                "time": local.tz_convert("UTC"),
            }
        )
    columns = [
        "meal_id", "user_id", "scenario_id", "timestamp", "food", "description",
        "primary_portion_g", "carbs_g", "meal_order", "time",
    ]
    return pd.DataFrame(records, columns=columns)


def with_user_data(base_cgm: pd.DataFrame, base_meals: pd.DataFrame, demo_user: str):
    """Combine sample CSV data (base) with this user's DB records and return (cgm, meals).

    If DB CGM data exists for a date, the sample CSV CGM data for that date is excluded
    and the uploaded values are used instead.
    """
    db_cgm = load_cgm_df(demo_user)
    db_meals = meals_df(demo_user)

    cgm = base_cgm
    if not db_cgm.empty:
        uploaded_days = set(db_cgm["time"].dt.tz_convert(TIMEZONE).dt.date)
        base_days = base_cgm["time"].dt.tz_convert(TIMEZONE).dt.date
        drop = (base_cgm["user_id"] == demo_user) & base_days.isin(uploaded_days)
        cgm = pd.concat([base_cgm[~drop], db_cgm], ignore_index=True)

    meals = base_meals if db_meals.empty else pd.concat([base_meals, db_meals], ignore_index=True)
    return cgm, meals


MIN_PRE_READINGS = 2   # demo threshold, not a medical standard
MIN_POST_READINGS = 6  # demo threshold, not a medical standard


def meal_status(cgm: pd.DataFrame, meals: pd.DataFrame, user_id: str) -> dict:
    """{meal_id: {"insufficient": bool, "notes": [str]}} for meals that need a note."""
    user_cgm = cgm[cgm["user_id"] == user_id]
    cgm_time = user_cgm["time"]
    user_meals = meals[meals["user_id"] == user_id].sort_values("time")
    ids, times = list(user_meals["meal_id"]), list(user_meals["time"])

    out: dict[str, dict] = {}
    for i, (meal_id, t) in enumerate(zip(ids, times)):
        notes = []
        n_pre = int(((cgm_time >= t - BASELINE_WINDOW) & (cgm_time < t)).sum())
        n_post = int(((cgm_time >= t) & (cgm_time <= t + PEAK_WINDOW)).sum())
        insufficient = n_pre < MIN_PRE_READINGS or n_post < MIN_POST_READINGS
        if n_pre < MIN_PRE_READINGS:
            notes.append(
                f"Not calculated: only {n_pre} CGM reading(s) in the 15 minutes before the meal "
                f"(minimum {MIN_PRE_READINGS})."
            )
        if n_post < MIN_POST_READINGS:
            notes.append(
                f"Not calculated: only {n_post} CGM reading(s) within 2 hours after the meal "
                f"(minimum {MIN_POST_READINGS})."
            )
        if not insufficient and not cgm_time.empty and cgm_time.max() < t + PEAK_WINDOW:
            notes.append("Only part of the 2-hour period after the meal is recorded, so the result may be incomplete.")
        if i + 1 < len(times) and times[i + 1] - t <= PEAK_WINDOW:  # exactly 2 h counts as overlap
            gap = int((times[i + 1] - t).total_seconds() // 60)
            notes.append(f"The next meal is in {gap} minutes, so the analysis windows overlap.")
        if notes:
            out[meal_id] = {"insufficient": insufficient, "notes": notes}
    return out


def check_meals(cgm: pd.DataFrame, meals: pd.DataFrame, user_id: str) -> dict:
    """{meal_id: [notes]} (kept so existing callers still work)."""
    return {k: v["notes"] for k, v in meal_status(cgm, meals, user_id).items()}