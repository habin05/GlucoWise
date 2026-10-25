"""GlucoWise P0 step 3-1: parse and validate a user-uploaded CGM CSV.

Required columns: ``timestamp`` and ``glucose_mg_dl`` (extra columns are ignored).

Rules
  * ``timestamp`` must be ISO 8601 with a time part (``2026-10-20T07:45:00-07:00``,
    ``2026-10-20 07:45``, ``...Z``). Anything else is excluded and reported.
  * A timestamp WITHOUT a UTC offset is read as America/Los_Angeles. A timestamp WITH an offset keeps
    that offset in the ``timestamp`` text; the ``time`` column is the same instant in UTC and is what
    every window calculation uses (same convention as glucose_analysis.load_data).
  * Naive times that are ambiguous or do not exist because of a daylight-saving change are excluded
    (never guessed).
  * ``glucose_mg_dl`` must be a finite number > 0. Readings are NOT required to be 5 minutes apart.
  * Rows that cannot be used are excluded and listed in ``result.excluded`` with the line number and
    reason. Nothing is invented or interpolated.
  * Duplicate instants inside one file: the LAST row in the file wins (same "latest upload wins"
    policy as the planned database).

The returned ``readings`` frame has the same columns glucose_analysis expects from cgm.csv
(``timestamp``, ``glucose_mg_dl``, ``time``), sorted by time.
"""
from __future__ import annotations

import csv
import io
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_TZ = "America/Los_Angeles"
REQUIRED_COLUMNS = ("timestamp", "glucose_mg_dl")

# Demo warning band only (NOT a medical range). Values outside it are kept but flagged.
WARN_LOW_MG_DL = 40.0
WARN_HIGH_MG_DL = 400.0

_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?\s*(Z|[+-]\d{2}(:?\d{2})?)?$"
)


class CgmImportError(ValueError):
    """The file as a whole cannot be used (empty, undecodable, missing columns)."""


class TimestampError(ValueError):
    """A single timestamp cannot be interpreted."""


@dataclass
class ExcludedRow:
    line: int
    reason: str
    detail: str
    raw_timestamp: str = ""
    raw_glucose: str = ""


@dataclass
class CgmImportResult:
    readings: pd.DataFrame
    excluded: list[ExcludedRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        return not self.readings.empty

    def excluded_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [e.__dict__ for e in self.excluded],
            columns=["line", "reason", "detail", "raw_timestamp", "raw_glucose"],
        )

    def report_lines(self, max_excluded: int = 20) -> list[str]:
        s = self.summary
        lines = [
            f"Rows read: {s.get('rows_read', 0)}; imported: {s.get('rows_imported', 0)}; "
            f"excluded: {s.get('rows_excluded', 0)}; duplicate timestamps replaced: "
            f"{s.get('duplicates_replaced', 0)}."
        ]
        lines += list(self.warnings)
        for e in self.excluded[:max_excluded]:
            lines.append(f"Line {e.line}: excluded ({e.reason}) - {e.detail}")
        if len(self.excluded) > max_excluded:
            lines.append(f"... and {len(self.excluded) - max_excluded} more excluded rows.")
        return lines


def parse_timestamp(raw, default_tz: str = DEFAULT_TZ) -> pd.Timestamp:
    """Return a tz-aware Timestamp. A naive value is read as ``default_tz``.

    Raises TimestampError when the value is not ISO 8601 date-time, or is naive and falls into a
    daylight-saving gap/overlap of ``default_tz``.
    """
    if isinstance(raw, datetime):
        ts = pd.Timestamp(raw)
        label = ts.isoformat()
    else:
        label = str(raw).strip()
        if not _ISO_RE.match(label):
            raise TimestampError(f"not an ISO 8601 date-time (expected e.g. 2026-10-20T07:45:00-07:00): {label!r}")
        try:
            ts = pd.Timestamp(label)
        except (ValueError, OverflowError) as exc:
            raise TimestampError(f"cannot read date-time {label!r}: {exc}") from exc
    if ts is pd.NaT or pd.isna(ts):
        raise TimestampError(f"cannot read date-time {label!r}")
    if ts.tzinfo is None:
        try:
            ts = ts.tz_localize(default_tz, ambiguous="raise", nonexistent="raise")
        except Exception as exc:  # pandas raises ValueError or a tz-library error
            raise TimestampError(
                f"{label!r} is ambiguous or does not exist in {default_tz} (daylight-saving change); "
                "add a UTC offset such as -07:00"
            ) from exc
    return ts


def _read_text(source) -> str:
    """Accept a path, raw bytes, CSV text (contains a newline) or a file-like/Streamlit upload."""
    if isinstance(source, (str, os.PathLike)) and not (isinstance(source, str) and "\n" in source):
        data = Path(source).read_bytes()
    elif isinstance(source, str):
        return source.lstrip("\ufeff")
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif hasattr(source, "getvalue"):
        data = source.getvalue()
    elif hasattr(source, "read"):
        data = source.read()
    else:
        raise CgmImportError(f"unsupported input type: {type(source).__name__}")
    if isinstance(data, str):
        return data.lstrip("\ufeff")
    for encoding in ("utf-8-sig", "cp949"):  # cp949: CSV saved by Korean-locale Excel
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise CgmImportError("the file is not UTF-8 or CP949 text")


def parse_cgm_csv(source, default_tz: str = DEFAULT_TZ) -> CgmImportResult:
    """Parse and validate a CGM CSV. See the module docstring for the rules."""
    text = _read_text(source)
    reader = csv.reader(io.StringIO(text))

    header = None
    for row in reader:
        if any(cell.strip() for cell in row):
            header = [cell.strip().lstrip("\ufeff").lower() for cell in row]
            break
    if header is None:
        raise CgmImportError("The file is empty.")
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise CgmImportError(
            f"Missing required column(s): {', '.join(missing)}. Found columns: {', '.join(header)}"
        )
    t_idx, g_idx = header.index("timestamp"), header.index("glucose_mg_dl")

    excluded: list[ExcludedRow] = []
    kept: list[tuple[pd.Timestamp, float]] = []
    rows_read = 0
    for row in reader:
        if not any(cell.strip() for cell in row):
            continue
        rows_read += 1
        line = reader.line_num
        if len(row) <= max(t_idx, g_idx):
            excluded.append(ExcludedRow(line, "missing_column", "row has fewer columns than the header"))
            continue
        raw_ts, raw_g = row[t_idx].strip(), row[g_idx].strip()
        try:
            ts = parse_timestamp(raw_ts, default_tz)
        except TimestampError as exc:
            excluded.append(ExcludedRow(line, "invalid_timestamp", str(exc), raw_ts, raw_g))
            continue
        try:
            value = float(raw_g)
        except ValueError:
            excluded.append(ExcludedRow(line, "invalid_glucose", f"not a number: {raw_g!r}", raw_ts, raw_g))
            continue
        if not math.isfinite(value) or value <= 0:
            excluded.append(ExcludedRow(line, "invalid_glucose", f"must be a finite number > 0: {raw_g!r}", raw_ts, raw_g))
            continue
        kept.append((ts, value))

    warnings: list[str] = []
    if kept:
        utc = pd.to_datetime([ts.tz_convert("UTC") for ts, _ in kept], utc=True)
        frame = pd.DataFrame(
            {
                "timestamp": [ts.isoformat() for ts, _ in kept],
                "glucose_mg_dl": [v for _, v in kept],
                "time": utc,
            }
        )
        frame = frame.sort_values("time", kind="stable")  # stable: file order kept among equal instants
        before = len(frame)
        frame = frame.drop_duplicates("time", keep="last")
        duplicates = before - len(frame)
        frame = frame.reset_index(drop=True)
        if duplicates:
            warnings.append(
                f"{duplicates} reading(s) shared a timestamp with a later row in this file; the later value was kept."
            )
        out_of_band = int(((frame["glucose_mg_dl"] < WARN_LOW_MG_DL) | (frame["glucose_mg_dl"] > WARN_HIGH_MG_DL)).sum())
        if out_of_band:
            warnings.append(
                f"{out_of_band} reading(s) are outside {WARN_LOW_MG_DL:.0f}-{WARN_HIGH_MG_DL:.0f} mg/dL; kept, but please "
                "check the file (this band is a demo check, not a medical range)."
            )
    else:
        duplicates = 0
        frame = pd.DataFrame(
            {
                "timestamp": pd.Series([], dtype="object"),
                "glucose_mg_dl": pd.Series([], dtype="float64"),
                "time": pd.to_datetime(pd.Series([], dtype="object"), utc=True),
            }
        )
        warnings.append("No usable rows were found in the file.")

    summary = {
        "rows_read": rows_read,
        "rows_imported": int(len(frame)),
        "rows_excluded": len(excluded),
        "duplicates_replaced": duplicates,
        "first_time_utc": None,
        "last_time_utc": None,
        "local_dates": [],
        "median_interval_minutes": None,
        "max_gap_minutes": None,
        "default_tz": default_tz,
    }
    if not frame.empty:
        summary["first_time_utc"] = frame["time"].iloc[0].isoformat()
        summary["last_time_utc"] = frame["time"].iloc[-1].isoformat()
        summary["local_dates"] = sorted({d.isoformat() for d in frame["time"].dt.tz_convert(default_tz).dt.date})
        if len(frame) > 1:
            gaps = frame["time"].diff().dropna().dt.total_seconds() / 60.0
            summary["median_interval_minutes"] = round(float(gaps.median()), 2)
            summary["max_gap_minutes"] = round(float(gaps.max()), 2)

    return CgmImportResult(readings=frame, excluded=excluded, warnings=warnings, summary=summary)
