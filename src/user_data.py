"""업로드한 CGM·식사 기록을 SQLite에 누적 저장하고, 분석 함수용 DataFrame으로 변환한다.

흐름
    CGM CSV 업로드 -> parse_cgm_csv() -> save_cgm()            (SQLite cgm_readings)
    식사 저장       -> local_db.save_meal()                    (SQLite meals, 기존 테이블 그대로)
    분석 직전       -> with_user_data(base_cgm, base_meals, user) -> (cgm, meals)
                      -> glucose_analysis의 모든 함수에 cgm=..., meals=... 로 전달

혈당 값은 업로드 파일의 숫자를 그대로 쓴다. 이 모듈은 어떤 값도 만들어 내지 않는다.
CSV 샘플(base)과 DB 데이터는 user_id 기준으로 합쳐진다.
"""
from __future__ import annotations

from contextlib import closing

import pandas as pd

try:  # `src.user_data`(패키지)와 `src/`에서 직접 실행 모두 지원
    from . import local_db
    from .normalization import normalize_food
except ImportError:  # pragma: no cover
    import local_db
    from normalization import normalize_food

TIMEZONE = "America/Los_Angeles"  # data_generator.TIMEZONE와 동일해야 함
BASELINE_WINDOW = pd.Timedelta(minutes=15)
PEAK_WINDOW = pd.Timedelta(hours=2)

TIME_HINTS = ("timestamp", "datetime", "date_time", "time", "date", "시간", "일시", "날짜")
GLUCOSE_HINTS = ("glucose", "sgv", "mg/dl", "mg_dl", "혈당", "value")


# ----------------------------------------------------------------- tables
def init_user_tables() -> None:
    """meals 테이블(local_db.init_db)과 CGM 테이블을 만든다."""
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
    """시연 초기화: 이 사용자의 업로드 CGM과 식사 기록을 모두 지운다 (CSV 샘플은 그대로)."""
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
    """CGM CSV(파일 경로 또는 업로드 파일 객체) -> DataFrame[time_utc, glucose_mg_dl].

    - 시간/혈당 컬럼은 이름에 포함된 단어(timestamp, time, glucose, 혈당 ...)로 찾는다.
    - 시간대 표시가 없는 시간은 ``tz``의 현지 시간으로 본다. (식사 시간 입력도 같은 기준)
    - 문제가 있으면 사용자에게 보여줄 한국어 메시지로 ValueError를 낸다.
    """
    try:
        raw = pd.read_csv(source)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"CSV를 읽지 못했어요: {exc}") from exc

    t_col = _find_col(raw.columns, TIME_HINTS)
    g_col = _find_col(raw.columns, GLUCOSE_HINTS, exclude=t_col)
    if t_col is None or g_col is None:
        raise ValueError(f"시간/혈당 컬럼을 찾지 못했어요. 파일의 컬럼: {list(raw.columns)}")

    text = raw[t_col].astype(str).str.strip()
    aware = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    if aware.all():
        times = pd.to_datetime(text, errors="coerce", utc=True)
    elif aware.any():
        raise ValueError("시간대 표시가 있는 행과 없는 행이 섞여 있어요. 형식을 통일해 주세요.")
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
        raise ValueError("읽을 수 있는 (시간, 혈당) 행이 없어요.")
    if out["glucose_mg_dl"].median() < 35:
        raise ValueError("혈당 값이 mmol/L로 보여요. mg/dL 단위 파일을 올려 주세요.")
    if not out["glucose_mg_dl"].between(20, 600).all():
        raise ValueError("혈당 값이 20~600 mg/dL 범위를 벗어난 행이 있어요. 파일을 확인해 주세요.")
    return out


def save_cgm(demo_user: str, readings: pd.DataFrame) -> int:
    """parse_cgm_csv 결과를 저장한다. 같은 시각의 값은 새 값으로 덮어쓴다. 저장한 행 수 반환."""
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
    """저장된 CGM 요약 (UI 표시용): 행 수, 첫/마지막 시각(현지), 일수."""
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
    """DB의 CGM을 cgm.csv와 같은 컬럼(user_id, timestamp, glucose_mg_dl) + UTC ``time``으로."""
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
    """meals 테이블(local_db)의 식사를 meals.csv와 같은 컬럼 + UTC ``time``으로.

    meal_id는 ``db_<id>``. 사진처럼 섭취량을 모르면 primary_portion_g는 NaN이다.
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
    """CSV 샘플(base)에 이 사용자의 DB 데이터를 합쳐 (cgm, meals)로 돌려준다.

    같은 날짜의 CGM이 DB에 있으면 그 날의 CSV 샘플 CGM은 제외하고 업로드 값을 쓴다.
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
            notes.append(f"식사 전 15분의 CGM 값이 {n_pre}개뿐이라 계산하지 않았어요. (최소 {MIN_PRE_READINGS}개)")
        if n_post < MIN_POST_READINGS:
            notes.append(f"식사 후 2시간 안의 CGM 값이 {n_post}개뿐이라 계산하지 않았어요. (최소 {MIN_POST_READINGS}개)")
        if not insufficient and not cgm_time.empty and cgm_time.max() < t + PEAK_WINDOW:
            notes.append("식사 후 2시간 중 일부만 기록되어 있어, 결과가 불완전할 수 있어요.")
        if i + 1 < len(times) and times[i + 1] - t <= PEAK_WINDOW:  # exactly 2 h counts as overlap
            gap = int((times[i + 1] - t).total_seconds() // 60)
            notes.append(f"다음 식사가 {gap}분 뒤라 분석 구간이 겹쳐요.")
        if notes:
            out[meal_id] = {"insufficient": insufficient, "notes": notes}
    return out


def check_meals(cgm: pd.DataFrame, meals: pd.DataFrame, user_id: str) -> dict:
    """{meal_id: [notes]} (kept so existing callers still work)."""
    return {k: v["notes"] for k, v in meal_status(cgm, meals, user_id).items()}