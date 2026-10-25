"""GlucoWise P0 step 3-3: persistent storage for uploaded CGM data and meals (data/user_uploads.db).

Separate SQLite file; local_db.py, auth_db.py and their databases are not touched.

Tables
  uploads        one row per CSV upload (counts + exclusion report), history only
  cgm_readings   raw readings, PRIMARY KEY (user_id, ts_utc): re-uploading the same instant replaces the value
  meals          raw meals, UNIQUE (user_id, meal_time_utc): the same user + same instant is the same meal
  meal_foods     foods of a meal (several per meal, ordered)
  meal_analysis  CACHE of meal_analysis.analyze_meals; recomputed from the raw tables after every change,
                 so it can always be rebuilt (refresh_analysis)

Rules
  * Every query filters by user_id, so users never see each other's rows.
  * Connections are always closed (Windows keeps SQLite files locked otherwise) and every public
    function is one transaction: it either fully applies or not at all.
  * Raw data is never invented or altered; analysis results are derived and replaceable.
  * All demo data is synthetic, so every row has data_type "simulated". ``source`` ("uploaded" or
    "generated") only records where a row came from.

All public functions take an optional ``db_path`` (tests use a temp file).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

try:  # same import style as agent_context.py: package import first, then flat import
    from .cgm_import import DEFAULT_TZ, parse_timestamp
    from .meal_analysis import INTERPRETATION_NOTE, _clean_foods, _load_normalizer, analyze_meals
except ImportError:
    from cgm_import import DEFAULT_TZ, parse_timestamp
    from meal_analysis import INTERPRETATION_NOTE, _clean_foods, _load_normalizer, analyze_meals

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "user_uploads.db"
SIMULATED = "simulated"  # data_type of every row: all demo data is synthetic
UPLOADED = "uploaded"  # source: the row came from a user CSV upload
GENERATED = "generated"  # source: the row came from data_generator.py (data/*.csv)
_UTC_FMT = "%Y-%m-%dT%H:%M:%SZ"  # fixed width, so text order == time order
_MAX_REPORT_ROWS = 200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    filename TEXT,
    uploaded_at TEXT NOT NULL,
    rows_read INTEGER NOT NULL,
    rows_imported INTEGER NOT NULL,
    rows_excluded INTEGER NOT NULL,
    readings_new INTEGER NOT NULL,
    readings_updated INTEGER NOT NULL,
    readings_unchanged INTEGER NOT NULL,
    report TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cgm_readings (
    user_id TEXT NOT NULL,
    ts_utc TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    glucose_mg_dl REAL NOT NULL,
    upload_id INTEGER NOT NULL REFERENCES uploads(upload_id),
    PRIMARY KEY (user_id, ts_utc)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS meals (
    meal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    meal_time_utc TEXT NOT NULL,
    meal_time TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (user_id, meal_time_utc)
);
CREATE TABLE IF NOT EXISTS meal_foods (
    meal_id INTEGER NOT NULL REFERENCES meals(meal_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    food TEXT NOT NULL,
    PRIMARY KEY (meal_id, position)
);
CREATE TABLE IF NOT EXISTS meal_analysis (
    meal_id INTEGER PRIMARY KEY REFERENCES meals(meal_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    reasons TEXT NOT NULL,
    message TEXT,
    baseline_mg_dl REAL,
    peak_mg_dl REAL,
    rise_mg_dl REAL,
    pre_meal_readings INTEGER NOT NULL,
    post_meal_readings INTEGER NOT NULL,
    warnings TEXT NOT NULL,
    data_type TEXT NOT NULL,
    computed_at TEXT NOT NULL
);
"""


@contextmanager
def _connect(db_path=None):
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()  # sqlite3's own "with" does NOT close, which locks the file on Windows


def init_db(db_path=None) -> None:
    with _connect(db_path):
        pass


def _now() -> str:
    return datetime.now(timezone.utc).strftime(_UTC_FMT)


def _utc_text(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime(_UTC_FMT)


def _check_user(user_id) -> str:
    user_id = str(user_id).strip() if user_id is not None else ""
    if not user_id:
        raise ValueError("user_id is required")
    return user_id


def _local_bounds(start_date, end_date, tz=DEFAULT_TZ):
    """Inclusive local dates -> (start_utc_text or None, end_utc_text_exclusive or None)."""
    start = end = None
    if start_date is not None:
        start = _utc_text(pd.Timestamp(start_date).normalize().tz_localize(tz))
    if end_date is not None:
        end = _utc_text((pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)).tz_localize(tz))
    return start, end


def _empty_cgm() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series([], dtype="object"),
            "glucose_mg_dl": pd.Series([], dtype="float64"),
            "time": pd.to_datetime(pd.Series([], dtype="object"), utc=True),
        }
    )


def _read_cgm(conn, user_id, start_utc=None, end_utc=None) -> pd.DataFrame:
    sql, args = "SELECT ts_utc, timestamp, glucose_mg_dl FROM cgm_readings WHERE user_id = ?", [user_id]
    if start_utc:
        sql, args = sql + " AND ts_utc >= ?", args + [start_utc]
    if end_utc:
        sql, args = sql + " AND ts_utc < ?", args + [end_utc]
    rows = conn.execute(sql + " ORDER BY ts_utc", args).fetchall()
    if not rows:
        return _empty_cgm()
    return pd.DataFrame(
        {
            "timestamp": [r["timestamp"] for r in rows],
            "glucose_mg_dl": [float(r["glucose_mg_dl"]) for r in rows],
            "time": pd.to_datetime([r["ts_utc"] for r in rows], utc=True),
        }
    )


def _read_meals(conn, user_id, start_utc=None, end_utc=None) -> list[dict]:
    sql, args = "SELECT meal_id, meal_time, meal_time_utc FROM meals WHERE user_id = ?", [user_id]
    if start_utc:
        sql, args = sql + " AND meal_time_utc >= ?", args + [start_utc]
    if end_utc:
        sql, args = sql + " AND meal_time_utc < ?", args + [end_utc]
    meals = []
    for row in conn.execute(sql + " ORDER BY meal_time_utc, meal_id", args).fetchall():
        foods = [
            r["food"]
            for r in conn.execute("SELECT food FROM meal_foods WHERE meal_id = ? ORDER BY position", (row["meal_id"],))
        ]
        meals.append(
            {"meal_id": row["meal_id"], "meal_time": row["meal_time"], "meal_time_utc": row["meal_time_utc"], "foods": foods}
        )
    return meals


def _refresh(conn, user_id) -> None:
    """Recompute the cached analysis of ALL of this user's meals (neighbouring meals affect each other)."""
    conn.execute("DELETE FROM meal_analysis WHERE meal_id IN (SELECT meal_id FROM meals WHERE user_id = ?)", (user_id,))
    meals = _read_meals(conn, user_id)
    if not meals:
        return
    results = analyze_meals(
        _read_cgm(conn, user_id),
        [{"meal_time": m["meal_time"], "foods": m["foods"], "meal_id": m["meal_id"]} for m in meals],
        data_type=SIMULATED,
        normalizer=lambda name: name,  # foods were already normalized when saved
    )
    computed_at = _now()
    conn.executemany(
        "INSERT INTO meal_analysis (meal_id, status, reasons, message, baseline_mg_dl, peak_mg_dl, rise_mg_dl, "
        "pre_meal_readings, post_meal_readings, warnings, data_type, computed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                r["meal_id"], r["status"], json.dumps(r["reasons"]), r["message"], r["baseline_mg_dl"], r["peak_mg_dl"],
                r["rise_mg_dl"], r["pre_meal_readings"], r["post_meal_readings"], json.dumps(r["warnings"]),
                r["data_type"], computed_at,
            )
            for r in results
        ],
    )


def refresh_analysis(user_id, db_path=None) -> None:
    """Rebuild the analysis cache from the raw tables (e.g. after changing meal_analysis thresholds)."""
    user_id = _check_user(user_id)
    with _connect(db_path) as conn:
        _refresh(conn, user_id)


# ----------------------------------------------------------------------------- CGM uploads


def save_cgm_upload(user_id, result, filename=None, db_path=None) -> dict:
    """Store the readings of a cgm_import result (or its ``readings`` frame).

    The same instant for the same user is replaced by the newest upload. Re-uploading an identical file
    changes nothing except adding a history row. Raises ValueError when there is nothing usable.
    """
    user_id = _check_user(user_id)
    readings = result.readings if hasattr(result, "readings") else result
    if readings.empty:
        raise ValueError("no usable CGM rows to save")
    rows: dict[str, tuple[str, float]] = {}
    for ts_utc, text, value in zip(
        readings["time"].dt.strftime(_UTC_FMT), readings["timestamp"], readings["glucose_mg_dl"].astype(float)
    ):
        rows[ts_utc] = (text, value)  # last wins if two rows collapse to the same second

    with _connect(db_path) as conn:
        existing = {
            r["ts_utc"]: r["glucose_mg_dl"]
            for r in conn.execute(
                "SELECT ts_utc, glucose_mg_dl FROM cgm_readings WHERE user_id = ? AND ts_utc BETWEEN ? AND ?",
                (user_id, min(rows), max(rows)),
            )
        }
        new = updated = unchanged = 0
        to_write = []
        for ts_utc, (text, value) in rows.items():
            if ts_utc not in existing:
                new += 1
            elif abs(existing[ts_utc] - value) > 1e-9:
                updated += 1
            else:
                unchanged += 1
                continue
            to_write.append((ts_utc, text, value))

        summary = getattr(result, "summary", {}) or {}
        excluded = getattr(result, "excluded", []) or []
        report = {
            "warnings": list(getattr(result, "warnings", []) or []),
            "excluded": [e.__dict__ for e in excluded[:_MAX_REPORT_ROWS]],
            "excluded_total": len(excluded),
            "summary": summary,
        }
        upload_id = conn.execute(
            "INSERT INTO uploads (user_id, filename, uploaded_at, rows_read, rows_imported, rows_excluded, "
            "readings_new, readings_updated, readings_unchanged, report) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                user_id, filename, _now(), int(summary.get("rows_read", len(readings))), len(rows),
                int(summary.get("rows_excluded", len(excluded))), new, updated, unchanged, json.dumps(report, default=str),
            ),
        ).lastrowid
        conn.executemany(
            "INSERT INTO cgm_readings (user_id, ts_utc, timestamp, glucose_mg_dl, upload_id) VALUES (?,?,?,?,?) "
            "ON CONFLICT(user_id, ts_utc) DO UPDATE SET timestamp = excluded.timestamp, "
            "glucose_mg_dl = excluded.glucose_mg_dl, upload_id = excluded.upload_id",
            [(user_id, ts_utc, text, value, upload_id) for ts_utc, text, value in to_write],
        )
        _refresh(conn, user_id)
    return {
        "upload_id": upload_id,
        "readings_new": new,
        "readings_updated": updated,
        "readings_unchanged": unchanged,
        "rows_excluded": len(excluded),
    }


def get_upload_history(user_id, limit=50, db_path=None) -> list[dict]:
    user_id = _check_user(user_id)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM uploads WHERE user_id = ? ORDER BY upload_id DESC LIMIT ?", (user_id, int(limit))
        ).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        item["report"] = json.loads(item["report"])
        out.append(item)
    return out


def load_cgm(user_id, start_date=None, end_date=None, db_path=None) -> pd.DataFrame:
    """Stored readings (timestamp, glucose_mg_dl, time[UTC]) for one user; dates are inclusive local dates."""
    user_id = _check_user(user_id)
    start, end = _local_bounds(start_date, end_date)
    with _connect(db_path) as conn:
        return _read_cgm(conn, user_id, start, end)


# ----------------------------------------------------------------------------- meals


def save_meal(user_id, meal_time, foods, *, default_tz=DEFAULT_TZ, normalizer=None, db_path=None) -> dict:
    """Save one meal (time + one or more foods).

    The same user at the same instant is the same meal: saving it again replaces its food list instead
    of creating a duplicate. ``action`` is "created", "updated" or "unchanged".
    """
    user_id = _check_user(user_id)
    ts = parse_timestamp(meal_time, default_tz)  # TimestampError is a ValueError
    keys = _clean_foods(foods, normalizer or _load_normalizer())
    ts_utc = _utc_text(ts)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT meal_id FROM meals WHERE user_id = ? AND meal_time_utc = ?", (user_id, ts_utc)
        ).fetchone()
        now = _now()
        if row is None:
            meal_id = conn.execute(
                "INSERT INTO meals (user_id, meal_time_utc, meal_time, created_at, updated_at) VALUES (?,?,?,?,?)",
                (user_id, ts_utc, ts.isoformat(), now, now),
            ).lastrowid
            action = "created"
        else:
            meal_id = row["meal_id"]
            old = [r["food"] for r in conn.execute("SELECT food FROM meal_foods WHERE meal_id = ? ORDER BY position", (meal_id,))]
            if old == keys:
                return {"meal_id": meal_id, "action": "unchanged", "foods": keys, "meal_time": ts.isoformat(), "meal_time_utc": ts_utc}
            conn.execute("DELETE FROM meal_foods WHERE meal_id = ?", (meal_id,))
            conn.execute("UPDATE meals SET updated_at = ?, meal_time = ? WHERE meal_id = ?", (now, ts.isoformat(), meal_id))
            action = "updated"
        conn.executemany(
            "INSERT INTO meal_foods (meal_id, position, food) VALUES (?,?,?)", [(meal_id, i, f) for i, f in enumerate(keys)]
        )
        _refresh(conn, user_id)
    return {"meal_id": meal_id, "action": action, "foods": keys, "meal_time": ts.isoformat(), "meal_time_utc": ts_utc}


def delete_meal(user_id, meal_id, db_path=None) -> bool:
    """Delete one of this user's meals (False if it is not theirs / does not exist)."""
    user_id = _check_user(user_id)
    with _connect(db_path) as conn:
        deleted = conn.execute("DELETE FROM meals WHERE meal_id = ? AND user_id = ?", (meal_id, user_id)).rowcount > 0
        if deleted:
            _refresh(conn, user_id)
    return deleted


def list_meals(user_id, start_date=None, end_date=None, db_path=None) -> list[dict]:
    """Stored meals with their cached analysis, in time order (same keys as meal_analysis.analyze_meals)."""
    user_id = _check_user(user_id)
    start, end = _local_bounds(start_date, end_date)
    with _connect(db_path) as conn:
        meals = _read_meals(conn, user_id, start, end)
        analysis = {
            r["meal_id"]: r
            for r in conn.execute(
                "SELECT a.* FROM meal_analysis a JOIN meals m ON m.meal_id = a.meal_id WHERE m.user_id = ?", (user_id,)
            )
        }
    out = []
    for meal in meals:
        a = analysis.get(meal["meal_id"])
        out.append(
            {
                "meal_id": meal["meal_id"],
                "meal_time": meal["meal_time"],
                "meal_time_utc": meal["meal_time_utc"],
                "foods": meal["foods"],
                "status": a["status"] if a else None,
                "reasons": json.loads(a["reasons"]) if a else [],
                "message": a["message"] if a else None,
                "baseline_mg_dl": a["baseline_mg_dl"] if a else None,
                "peak_mg_dl": a["peak_mg_dl"] if a else None,
                "rise_mg_dl": a["rise_mg_dl"] if a else None,
                "pre_meal_readings": a["pre_meal_readings"] if a else None,
                "post_meal_readings": a["post_meal_readings"] if a else None,
                "warnings": json.loads(a["warnings"]) if a else [],
                "unit": "mg/dL",
                "data_type": a["data_type"] if a else SIMULATED,
                "source": UPLOADED,
                "interpretation_note": INTERPRETATION_NOTE,
            }
        )
    return out


def list_dates(user_id, db_path=None) -> list[dict]:
    """Local dates that have stored data: [{"date", "cgm_readings", "meals"}], oldest first."""
    user_id = _check_user(user_id)
    with _connect(db_path) as conn:
        cgm_ts = [r[0] for r in conn.execute("SELECT ts_utc FROM cgm_readings WHERE user_id = ?", (user_id,))]
        meal_ts = [r[0] for r in conn.execute("SELECT meal_time_utc FROM meals WHERE user_id = ?", (user_id,))]
    counts: dict[str, dict] = {}
    for key, values in (("cgm_readings", cgm_ts), ("meals", meal_ts)):
        if not values:
            continue
        for d in pd.to_datetime(values, utc=True).tz_convert(DEFAULT_TZ).date:
            counts.setdefault(d.isoformat(), {"date": d.isoformat(), "cgm_readings": 0, "meals": 0})[key] += 1
    return [counts[k] for k in sorted(counts)]


# ----------------------------------------------------------------------------- frames for glucose_analysis


def uploaded_frames(user_id, start_date=None, end_date=None, db_path=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(cgm, meals) in the layout glucose_analysis.load_data returns, for the uploaded data of one user.

    meals: food = first food key, description = all foods, primary_portion_g / carbs_g = NaN (not collected),
    meal_id = "<user>_UP<number>". Use with get_meal_history / get_glucose_curve / compare_foods via cgm=, meals=.
    """
    user_id = _check_user(user_id)
    start, end = _local_bounds(start_date, end_date)
    with _connect(db_path) as conn:
        cgm = _read_cgm(conn, user_id, start, end)
        meals = _read_meals(conn, user_id, start, end)
    cgm = cgm.assign(user_id=user_id, data_type=SIMULATED, source=UPLOADED)[
        ["user_id", "timestamp", "glucose_mg_dl", "time", "data_type", "source"]
    ]
    meals_df = pd.DataFrame(
        {
            "meal_id": [f"{user_id}_UP{m['meal_id']:05d}" for m in meals],
            "user_id": [user_id] * len(meals),
            "timestamp": [m["meal_time"] for m in meals],
            "food": [m["foods"][0] for m in meals],
            "description": [", ".join(m["foods"]) for m in meals],
            "primary_portion_g": [float("nan")] * len(meals),
            "carbs_g": [float("nan")] * len(meals),
            "time": pd.to_datetime([m["meal_time_utc"] for m in meals], utc=True)
            if meals
            else pd.to_datetime(pd.Series([], dtype="object"), utc=True),
            "data_type": [SIMULATED] * len(meals),
            "source": [UPLOADED] * len(meals),
        }
    )
    return cgm, meals_df


def combined_frames(user_id, base_cgm=None, base_meals=None, *, db_path=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulated CSV data + uploaded data for ONE user, as (cgm, meals) frames for glucose_analysis.

    Every row keeps data_type "simulated" (all demo data is synthetic); ``source`` is "generated" for the
    CSV rows and "uploaded" for stored uploads. Where both have the same instant (a CGM reading time or a meal
    time) the uploaded row wins and the simulated one is left out of the merged frame; the CSV files are not
    changed. If base_* are omitted, glucose_analysis.load_data() is used.
    """
    user_id = _check_user(user_id)
    if base_cgm is None or base_meals is None:
        try:
            from .glucose_analysis import load_data
        except ImportError:
            from glucose_analysis import load_data
        loaded_cgm, loaded_meals = load_data()
        base_cgm = loaded_cgm if base_cgm is None else base_cgm
        base_meals = loaded_meals if base_meals is None else base_meals

    up_cgm, up_meals = uploaded_frames(user_id, db_path=db_path)
    sim_cgm = base_cgm[base_cgm["user_id"] == user_id].copy()
    sim_meals = base_meals[base_meals["user_id"] == user_id].copy()
    if "data_type" not in sim_cgm.columns:
        sim_cgm["data_type"] = SIMULATED
    if "data_type" not in sim_meals.columns:
        sim_meals["data_type"] = SIMULATED
    if "source" not in sim_cgm.columns:
        sim_cgm["source"] = GENERATED
    if "source" not in sim_meals.columns:
        sim_meals["source"] = GENERATED
    sim_cgm = sim_cgm[~sim_cgm["time"].isin(up_cgm["time"])]
    sim_meals = sim_meals[~sim_meals["time"].isin(up_meals["time"])]

    cgm = sim_cgm.reset_index(drop=True) if up_cgm.empty else pd.concat([sim_cgm, up_cgm], ignore_index=True)
    meals = sim_meals.reset_index(drop=True) if up_meals.empty else pd.concat([sim_meals, up_meals], ignore_index=True)
    return cgm, meals
