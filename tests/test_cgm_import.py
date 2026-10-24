"""Tests for src/cgm_import.py (standard-library unittest, run from the project root:
python -m unittest discover -s tests -v)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cgm_import import CgmImportError, parse_cgm_csv, parse_timestamp  # noqa: E402

GOOD = (
    "timestamp,glucose_mg_dl\n"
    "2026-10-20T07:45:00-07:00,100\n"
    "2026-10-20T07:50:00-07:00,104\n"
    "2026-10-20T08:00:00-07:00,110\n"
    "2026-10-20T08:30:00-07:00,172\n"
    "2026-10-20T10:00:00-07:00,125\n"
)


class ParseTests(unittest.TestCase):
    def test_normal_csv(self):
        r = parse_cgm_csv(GOOD)
        self.assertTrue(r.is_usable)
        self.assertEqual(len(r.readings), 5)
        self.assertEqual(r.excluded, [])
        self.assertEqual(r.warnings, [])
        self.assertEqual(list(r.readings.columns), ["timestamp", "glucose_mg_dl", "time"])
        self.assertEqual(str(r.readings["time"].dt.tz), "UTC")
        self.assertEqual(r.summary["local_dates"], ["2026-10-20"])

    def test_irregular_spacing_is_accepted(self):
        r = parse_cgm_csv(GOOD)  # 5, 10, 30 and 90 minute gaps
        self.assertEqual(len(r.readings), 5)
        self.assertEqual(r.summary["max_gap_minutes"], 90.0)

    def test_unsorted_input_is_sorted(self):
        lines = GOOD.strip().split("\n")
        shuffled = "\n".join([lines[0]] + lines[:0:-1]) + "\n"
        r = parse_cgm_csv(shuffled)
        self.assertTrue(r.readings["time"].is_monotonic_increasing)
        self.assertEqual(r.readings["glucose_mg_dl"].tolist(), [100, 104, 110, 172, 125])

    def test_timezone_aware_offset_kept_and_converted_to_utc(self):
        r = parse_cgm_csv("timestamp,glucose_mg_dl\n2026-10-20T08:00:00-07:00,100\n2026-10-20T08:05:00+09:00,101\n")
        self.assertEqual(r.readings["timestamp"].tolist(), ["2026-10-20T08:05:00+09:00", "2026-10-20T08:00:00-07:00"])  # sorted by UTC instant
        self.assertEqual(r.readings["time"].iloc[0], pd.Timestamp("2026-10-19T23:05:00Z"))  # earlier instant first
        self.assertEqual(r.readings["time"].iloc[1], pd.Timestamp("2026-10-20T15:00:00Z"))

    def test_naive_timestamp_is_los_angeles(self):
        r = parse_cgm_csv("timestamp,glucose_mg_dl\n2026-10-20T08:00:00,100\n2026-12-01T08:00:00,100\n")
        self.assertEqual(r.readings["time"].iloc[0], pd.Timestamp("2026-10-20T15:00:00Z"))  # PDT -07
        self.assertEqual(r.readings["time"].iloc[1], pd.Timestamp("2026-12-01T16:00:00Z"))  # PST -08
        self.assertTrue(r.readings["timestamp"].iloc[0].endswith("-07:00"))
        self.assertTrue(r.readings["timestamp"].iloc[1].endswith("-08:00"))

    def test_naive_and_aware_same_instant_compare_equal(self):
        a = parse_timestamp("2026-10-20T08:00:00").tz_convert("UTC")
        b = parse_timestamp("2026-10-20T08:00:00-07:00").tz_convert("UTC")
        self.assertEqual(a, b)

    def test_daylight_saving_ambiguous_and_missing_naive_times_are_excluded(self):
        csv_text = (
            "timestamp,glucose_mg_dl\n"
            "2026-11-01T01:30:00,100\n"  # happens twice
            "2026-03-08T02:30:00,100\n"  # never happens
            "2026-11-01T01:30:00-07:00,100\n"  # explicit offset is fine
        )
        r = parse_cgm_csv(csv_text)
        self.assertEqual(len(r.readings), 1)
        self.assertEqual([e.reason for e in r.excluded], ["invalid_timestamp", "invalid_timestamp"])

    def test_bad_rows_are_excluded_with_line_numbers(self):
        csv_text = (
            "timestamp,glucose_mg_dl\n"
            "2026-10-20T08:00:00-07:00,100\n"  # line 2 ok
            "not a date,100\n"  # line 3
            "2026-10-20T08:05:00-07:00,abc\n"  # line 4
            "2026-10-20T08:10:00-07:00,\n"  # line 5
            "2026-10-20T08:15:00-07:00,-5\n"  # line 6
            "2026-10-20T08:20:00-07:00,nan\n"  # line 7
            "2026-10-20\n"  # line 8: missing column
            "2026-10-20T08:30:00-07:00,120\n"  # line 9 ok
        )
        r = parse_cgm_csv(csv_text)
        self.assertEqual(len(r.readings), 2)
        self.assertEqual([e.line for e in r.excluded], [3, 4, 5, 6, 7, 8])
        self.assertEqual(r.summary["rows_read"], 8)
        self.assertEqual(r.summary["rows_excluded"], 6)
        self.assertIn("Line 3", "\n".join(r.report_lines()))

    def test_date_only_and_non_iso_formats_are_excluded(self):
        r = parse_cgm_csv("timestamp,glucose_mg_dl\n2026-10-20,100\n10/20/2026 8:00 AM,100\n")
        self.assertFalse(r.is_usable)
        self.assertEqual(len(r.excluded), 2)

    def test_missing_required_column_raises(self):
        with self.assertRaises(CgmImportError):
            parse_cgm_csv("time,glucose\n2026-10-20T08:00:00,100\n")

    def test_empty_file_raises(self):
        with self.assertRaises(CgmImportError):
            parse_cgm_csv("   \n\n")
        with self.assertRaises(CgmImportError):
            parse_cgm_csv(b"")

    def test_header_only_is_not_usable(self):
        r = parse_cgm_csv("timestamp,glucose_mg_dl\n")
        self.assertFalse(r.is_usable)
        self.assertEqual(r.readings.shape[0], 0)
        self.assertEqual(list(r.readings.columns), ["timestamp", "glucose_mg_dl", "time"])

    def test_duplicate_timestamp_last_row_wins(self):
        r = parse_cgm_csv(
            "timestamp,glucose_mg_dl\n2026-10-20T08:00:00-07:00,100\n2026-10-20T08:00:00-07:00,130\n"
            "2026-10-20T15:00:00+00:00,150\n"  # same instant written with another offset
        )
        self.assertEqual(len(r.readings), 1)
        self.assertEqual(r.readings["glucose_mg_dl"].iloc[0], 150.0)
        self.assertEqual(r.summary["duplicates_replaced"], 2)

    def test_extra_columns_header_case_bom_and_whitespace(self):
        r = parse_cgm_csv("\ufeff Timestamp , Glucose_MG_DL ,user_id\n 2026-10-20T08:00:00-07:00 , 101.5 ,A\n")
        self.assertEqual(r.readings["glucose_mg_dl"].iloc[0], 101.5)

    def test_out_of_band_values_are_kept_with_warning(self):
        r = parse_cgm_csv("timestamp,glucose_mg_dl\n2026-10-20T08:00:00-07:00,30\n2026-10-20T08:05:00-07:00,450\n")
        self.assertEqual(len(r.readings), 2)
        self.assertEqual(len(r.warnings), 1)

    def test_reads_path_bytes_cp949_and_file_like(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cgm.csv"
            p.write_text(GOOD, encoding="utf-8")
            self.assertEqual(len(parse_cgm_csv(p).readings), 5)
            self.assertEqual(len(parse_cgm_csv(str(p)).readings), 5)
        self.assertEqual(len(parse_cgm_csv(GOOD.encode("utf-8")).readings), 5)
        korean = ("timestamp,glucose_mg_dl,메모\n2026-10-20T08:00:00-07:00,100,식전\n").encode("cp949")
        self.assertEqual(len(parse_cgm_csv(korean).readings), 1)

        class Upload:  # looks like a Streamlit UploadedFile
            def getvalue(self):
                return GOOD.encode("utf-8")

        self.assertEqual(len(parse_cgm_csv(Upload()).readings), 5)


if __name__ == "__main__":
    unittest.main()
