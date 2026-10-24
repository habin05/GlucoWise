"""Read-only check of CSV files already in your repo against the GlucoWise v1.1 schema.

Usage (from anywhere):
    python scripts/inspect_existing_csvs.py path/to/existing/data
It only reads files; it never writes or changes anything.
"""
import sys
from pathlib import Path

import pandas as pd

NEW = {
    "cgm.csv": ["user_id", "timestamp", "glucose_mg_dl"],
    "meals.csv": ["meal_id", "user_id", "scenario_id", "timestamp", "food",
                  "description", "primary_portion_g", "carbs_g", "meal_order"],
}


def main(folder: str) -> None:
    folder = Path(folder)
    for name, new_cols in NEW.items():
        path = folder / name
        print(f"\n=== {path} ===")
        if not path.exists():
            print("  not found: nothing to overwrite.")
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        print(f"  rows: {len(df):,}   columns: {list(df.columns)}")
        print(f"  columns missing vs new schema: {[c for c in new_cols if c not in df.columns]}")
        print(f"  columns only in existing file (would be lost): {[c for c in df.columns if c not in new_cols]}")
        if "timestamp" in df.columns and len(df):
            sample = df["timestamp"].iloc[0]
            has_offset = df["timestamp"].str.contains(r"(?:Z|[+-]\d{2}:\d{2})$").all()
            print(f"  first timestamp: {sample!r}; every value has a UTC offset: {has_offset}")
            if not has_offset:
                print("  RISK: naive timestamps; new files use offset-aware ISO-8601 (mixing them raises errors in pandas).")
        if "user_id" not in df.columns:
            print("  RISK: no user_id column; code that assumes one user will now see 3 users mixed together.")
        if "portion_g" in df.columns:
            print("  RISK: code reading 'portion_g' must switch to 'primary_portion_g'.")
        if name == "cgm.csv" and "timestamp" in df.columns and len(df) > 1:
            t = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            step = t.diff().dropna().dt.total_seconds().mode()
            print(f"  most common step between readings: {step.tolist()} s (new files: 300 s)")
        if name == "meals.csv" and "food" in df.columns:
            print(f"  food values: {sorted(df['food'].unique())[:12]} (new keys are lowercase snake_case, e.g. white_rice)")
    print("\nAlso search your code for: 'portion_g', 'oat_rice', read_csv(...) without user_id filtering,")
    print("pd.to_datetime(...) without utc=True, and any hard-coded row counts (864 CGM rows / 9 meals).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
