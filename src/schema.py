"""Validate the GlucoWise Step 1 CSV files.

Run from the project root:
    python src/schema.py

Errors make validation fail. Warnings are printed but do not fail validation.
All CSV cells are read as text (no pandas NA guessing), so IDs such as "NA"
stay IDs and blank optional cells are detected explicitly.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TIMEZONE = "America/Los_Angeles"  # keep identical to data_generator.TIMEZONE
CGM_INTERVAL_SECONDS = 300
POST_MEAL_WINDOW_MIN = 120

CGM_COLUMNS = ["user_id", "timestamp", "glucose_mg_dl"]
MEAL_COLUMNS = [
    "meal_id", "user_id", "scenario_id", "timestamp", "food",
    "description", "primary_portion_g", "carbs_g", "meal_order",
]

VALID_SCENARIO_IDS = {"A", "B", "C"}          # Scenario D is not implemented yet
VALID_MEAL_ORDER = {"unspecified"}            # extend when Scenario D is defined

USER_ID_RE = re.compile(r"[A-Z][A-Z0-9]{0,15}")               # "A", "NA", "U001"
FOOD_RE = re.compile(r"[a-z][a-z0-9_]*")
MEAL_ID_RE = re.compile(r"(?P<user>[A-Z][A-Z0-9]{0,15})_D(?P<day>\d{2})_(?P<slot>[1-9])_(?P<food>[a-z][a-z0-9_]*)")
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}")


def read_csv_as_text(path: Path) -> pd.DataFrame:
    """Read every cell as text; never turn "NA"/"null"/"" into NaN."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _sample(index, limit: int = 3) -> str:
    items = [str(i) for i in list(index)[:limit]]
    more = "" if len(index) <= limit else f" (+{len(index) - limit} more)"
    return ", ".join(items) + more


def check_timestamps(series: pd.Series, label: str, errors: list) -> pd.Series | None:
    """Strict ISO-8601 with an offset that matches TIMEZONE at that instant.

    Returns UTC-parsed timestamps, or None if any value is unusable.
    """
    raw = series.astype(str)
    bad_format = ~raw.str.fullmatch(TIMESTAMP_RE.pattern)
    if bad_format.any():
        errors.append(
            f"{label}: {int(bad_format.sum())} timestamp(s) not in 'YYYY-MM-DDTHH:MM:SS±HH:MM' form "
            f"(rows {_sample(raw.index[bad_format])})."
        )
        return None
    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
    if parsed.isna().any():
        errors.append(f"{label}: {int(parsed.isna().sum())} timestamp(s) could not be parsed.")
        return None
    expected = parsed.dt.tz_convert(TIMEZONE).dt.strftime("%z")
    expected = expected.str[:3] + ":" + expected.str[3:]
    mismatch = raw.str[-6:] != expected
    if mismatch.any():
        errors.append(
            f"{label}: {int(mismatch.sum())} timestamp(s) have a UTC offset that is wrong for {TIMEZONE} "
            f"at that instant (rows {_sample(raw.index[mismatch])})."
        )
        return None
    return parsed


def _numeric(series: pd.Series) -> pd.Series:
    """Text -> float; blank or non-numeric cells become NaN (no .replace, so no pandas FutureWarning)."""
    return pd.to_numeric(series.str.strip(), errors="coerce")


def validate_frames(cgm: pd.DataFrame, meals: pd.DataFrame) -> tuple[list, list]:
    """Return (errors, warnings) for text-typed cgm and meals DataFrames."""
    errors: list[str] = []
    warnings: list[str] = []

    # ---- columns -------------------------------------------------------
    for name, frame, expected in (("cgm.csv", cgm, CGM_COLUMNS), ("meals.csv", meals, MEAL_COLUMNS)):
        missing = [c for c in expected if c not in frame.columns]
        extra = [c for c in frame.columns if c not in expected]
        if missing:
            errors.append(f"{name} missing columns: {missing}")
        if extra:
            errors.append(f"{name} has unexpected columns: {extra}")
        if len(frame) == 0:
            errors.append(f"{name} has no data rows.")
    if errors:
        return errors, warnings

    # ---- required text fields ------------------------------------------
    for name, frame, cols in (
        ("cgm.csv", cgm, ["user_id", "timestamp", "glucose_mg_dl"]),
        ("meals.csv", meals, ["meal_id", "user_id", "scenario_id", "timestamp", "food", "description", "meal_order"]),
    ):
        for col in cols:
            blank = frame[col].str.strip().eq("")
            if blank.any():
                errors.append(f"{name}: {int(blank.sum())} blank value(s) in required column '{col}'.")

    # ---- ID formats -----------------------------------------------------
    for name, frame in (("cgm.csv", cgm), ("meals.csv", meals)):
        bad = ~frame["user_id"].str.fullmatch(USER_ID_RE.pattern)
        if bad.any():
            errors.append(f"{name}: invalid user_id format {_sample(frame['user_id'][bad].unique())} "
                          "(expected capital letter then capitals/digits, e.g. A, NA, U001).")
    bad_food = ~meals["food"].str.fullmatch(FOOD_RE.pattern)
    if bad_food.any():
        errors.append(f"meals.csv: food keys must be lowercase letters, digits and underscores "
                      f"(got {_sample(meals['food'][bad_food].unique())}).")
    if meals["meal_id"].duplicated().any():
        errors.append("meals.csv has duplicate meal_id values.")
    parts = meals["meal_id"].str.extract(MEAL_ID_RE.pattern)
    bad_id = parts["user"].isna()
    if bad_id.any():
        errors.append(f"meals.csv: meal_id not in 'USER_Ddd_slot_food' form: {_sample(meals['meal_id'][bad_id])}.")
    ok = ~bad_id
    if (parts.loc[ok, "user"] != meals.loc[ok, "user_id"]).any():
        errors.append("meals.csv: meal_id prefix does not match user_id.")
    if (parts.loc[ok, "food"] != meals.loc[ok, "food"]).any():
        errors.append("meals.csv: meal_id suffix does not match food.")

    # ---- categorical columns -------------------------------------------
    bad_scn = ~meals["scenario_id"].isin(VALID_SCENARIO_IDS)
    if bad_scn.any():
        errors.append(f"meals.csv: invalid scenario_id {_sample(meals['scenario_id'][bad_scn].unique())}; "
                      f"allowed {sorted(VALID_SCENARIO_IDS)}.")
    scn_per_user = meals.groupby("user_id")["scenario_id"].nunique()
    if (scn_per_user > 1).any():
        errors.append(f"meals.csv: user(s) with more than one scenario_id: {_sample(scn_per_user[scn_per_user > 1].index)}.")
    bad_order = ~meals["meal_order"].isin(VALID_MEAL_ORDER)
    if bad_order.any():
        errors.append(f"meals.csv: invalid meal_order {_sample(meals['meal_order'][bad_order].unique())}; "
                      f"allowed {sorted(VALID_MEAL_ORDER)}.")

    # ---- numerics ----------------------------------------------------------
    glucose = _numeric(cgm["glucose_mg_dl"])
    if glucose.isna().any() or not np.isfinite(glucose.dropna()).all():
        errors.append("cgm.csv: glucose_mg_dl has blank/non-numeric/non-finite values.")
    else:
        out = ~glucose.between(40, 400)
        if out.any():
            warnings.append(f"{int(out.sum())} glucose value(s) outside 40-400 mg/dL.")
    for col, strictly_positive in (("primary_portion_g", True), ("carbs_g", False)):
        text = meals[col].str.strip()
        num = _numeric(meals[col])
        nonblank = text.ne("")
        if (nonblank & num.isna()).any():
            errors.append(f"meals.csv: {col} has a non-numeric value.")
        known = num.dropna()
        if (known < 0).any() or (strictly_positive and (known == 0).any()):
            errors.append(f"meals.csv: {col} must be {'> 0' if strictly_positive else '>= 0'} when present.")
    if (_numeric(meals["carbs_g"]) == 0).any():
        warnings.append("carbs_g contains 0: confirm it is truly zero carbohydrate, not unknown (unknown must be blank).")
    # NOTE: no carbs_g <= primary_portion_g rule. primary_portion_g covers only the
    # main food, while carbs_g (when known) describes the whole meal.

    # ---- timestamps -------------------------------------------------------
    cgm_t = check_timestamps(cgm["timestamp"], "cgm.csv", errors)
    meal_t = check_timestamps(meals["timestamp"], "meals.csv", errors)
    if cgm_t is None or meal_t is None:
        return errors, warnings

    # ---- CGM: uniqueness, order, 5-minute grid ---------------------------
    if cgm.duplicated(["user_id", "timestamp"]).any():
        errors.append("cgm.csv has duplicate (user_id, timestamp) rows.")
    cgm = cgm.assign(_t=cgm_t)
    order_ok = True
    for user, grp in cgm.groupby("user_id", sort=False):
        deltas = grp["_t"].diff().dropna().dt.total_seconds()
        if (deltas <= 0).any():
            order_ok = False
            errors.append(f"cgm.csv: user {user} rows are not in strictly increasing time order.")
            continue
        if not (deltas == CGM_INTERVAL_SECONDS).all():
            gaps = deltas[deltas != CGM_INTERVAL_SECONDS]
            errors.append(f"cgm.csv: user {user} has {len(gaps)} step(s) that are not exactly 5 minutes "
                          f"(e.g. {_sample(gaps.unique())} s).")
        local = grp["_t"].dt.tz_convert(TIMEZONE)
        if ((local.dt.minute % 5 != 0) | (local.dt.second != 0)).any():
            errors.append(f"cgm.csv: user {user} has readings not aligned to :00/:05/... minutes.")
    users_in_order = list(dict.fromkeys(cgm["user_id"]))
    if users_in_order != sorted(users_in_order) or cgm["user_id"].ne(cgm["user_id"].shift()).sum() != len(users_in_order):
        errors.append("cgm.csv: rows must be sorted by user_id, with each user's rows contiguous.")

    # ---- meals: order, uniqueness, linkage -------------------------------
    meals = meals.assign(_t=meal_t)
    meal_users = set(meals["user_id"])
    cgm_users = set(cgm["user_id"])
    if not meal_users <= cgm_users:
        errors.append(f"Meal users without CGM data: {sorted(meal_users - cgm_users)}")
    if meals["user_id"].tolist() != sorted(meals["user_id"]):
        errors.append("meals.csv: rows must be sorted by user_id.")
    for user, grp in meals.groupby("user_id", sort=False):
        if (grp["_t"].diff().dropna().dt.total_seconds() <= 0).any():
            errors.append(f"meals.csv: user {user} meals are not in strictly increasing time order.")
        close = grp["_t"].diff().dropna().dt.total_seconds() < POST_MEAL_WINDOW_MIN * 60
        if close.any():
            warnings.append(f"user {user}: {int(close.sum())} meal(s) start < {POST_MEAL_WINDOW_MIN} min after the previous one "
                            "(overlapping analysis windows).")

    if order_ok:
        bounds = cgm.groupby("user_id")["_t"].agg(["min", "max"])
        for _, meal in meals.iterrows():
            user = meal["user_id"]
            if user not in bounds.index:
                continue
            lo, hi = bounds.loc[user, "min"], bounds.loc[user, "max"]
            if not (lo <= meal["_t"] <= hi):
                errors.append(f"Meal {meal['meal_id']} is outside the CGM time range for user {user}.")
                continue
            if meal["_t"] + pd.Timedelta(minutes=POST_MEAL_WINDOW_MIN) > hi:
                warnings.append(f"Meal {meal['meal_id']} has an incomplete {POST_MEAL_WINDOW_MIN}-min post-meal CGM window.")
            if (meal["_t"] - lo).total_seconds() < 15 * 60:
                warnings.append(f"Meal {meal['meal_id']} has an incomplete 15-min pre-meal CGM window.")
            if meal["_t"].tz_convert(TIMEZONE).minute % 5 or meal["_t"].second:
                warnings.append(f"Meal {meal['meal_id']} is not on the 5-minute CGM grid.")
    return errors, warnings


def validate(data_dir: Path = DATA_DIR, verbose: bool = True) -> tuple[list, list]:
    """Validate data_dir/cgm.csv and data_dir/meals.csv; raise ValueError on errors."""
    data_dir = Path(data_dir)
    cgm_path, meals_path = data_dir / "cgm.csv", data_dir / "meals.csv"
    if not cgm_path.exists() or not meals_path.exists():
        raise FileNotFoundError(f"Missing {cgm_path} or {meals_path}. Run python src/data_generator.py first.")
    cgm, meals = read_csv_as_text(cgm_path), read_csv_as_text(meals_path)
    errors, warnings = validate_frames(cgm, meals)
    if verbose:
        for w in warnings:
            print(f"WARNING: {w}")
    if errors:
        raise ValueError("Validation failed:\n- " + "\n- ".join(errors))
    if verbose:
        print("Validation passed.")
        print(f"CGM rows: {len(cgm):,}; meal rows: {len(meals):,}")
        print(f"Users: {', '.join(sorted(cgm['user_id'].unique()))}")
        print("Timestamps: ISO-8601 with offsets that match "
              f"{TIMEZONE}. Data is synthetic.")
    return errors, warnings


if __name__ == "__main__":
    validate()
