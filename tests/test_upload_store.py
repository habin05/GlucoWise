"""Tests for src/upload_store.py (P0 step 3-3). Run: python -m unittest discover -s tests

Every test uses its own temporary SQLite file, so data/user_uploads.db is never touched.
"""
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE):  # tests/ -> ../src ; flat layout -> same folder
    if (candidate / "upload_store.py").exists():
        sys.path.insert(0, str(candidate))

import upload_store as us  # noqa: E402
from cgm_import import parse_cgm_csv  # noqa: E402
from glucose_analysis import get_meal_history  # noqa: E402


def glucose_at(minute):
    """Flat 100 until minute 60, then +1 per minute up to 160 (so a meal at minute 60 has rise 60)."""
    return 100.0 + max(0, min(minute - 60, 60))


def make_csv(day="2026-10-20", start_hour=7, minutes=180, step=5, value=glucose_at, bad_rows=()):
    """CGM CSV text with naive local times (read as America/Los_Angeles, no DST change on these dates)."""
    base = pd.Timestamp(f"{day} {start_hour:02d}:00")
    lines = ["timestamp,glucose_mg_dl"]
    for m in range(0, minutes + 1, step):
        t = base + pd.Timedelta(minutes=m)
        lines.append(f"{t.strftime('%Y-%m-%dT%H:%M:%S')},{value(m)}")
    lines.extend(bad_rows)
    return "\n".join(lines) + "\n"


class UploadStoreTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # cleanup() would raise WinError 32 on Windows if any connection were left open
        self.addCleanup(self._tmp.cleanup)
        self.db = Path(self._tmp.name) / "uploads_test.db"
        us.init_db(self.db)  # create the tables up front so count() works even if a call fails before connecting

    def upload(self, user="A", csv_text=None, filename="cgm.csv", **csv_kwargs):
        result = parse_cgm_csv(csv_text if csv_text is not None else make_csv(**csv_kwargs))
        return us.save_cgm_upload(user, result, filename=filename, db_path=self.db)

    def count(self, table, user=None):
        with closing(sqlite3.connect(self.db)) as conn:
            if user is None:
                return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (user,)).fetchone()[0]


class CgmUploadTests(UploadStoreTestBase):
    def test_first_upload_stores_all_readings(self):
        info = self.upload()
        self.assertEqual(info["readings_new"], 37)
        self.assertEqual(info["readings_updated"], 0)
        self.assertEqual(info["readings_unchanged"], 0)
        cgm = us.load_cgm("A", db_path=self.db)
        self.assertEqual(len(cgm), 37)
        self.assertTrue(cgm["time"].is_monotonic_increasing)
        self.assertEqual(str(cgm["time"].dt.tz), "UTC")
        self.assertEqual(float(cgm["glucose_mg_dl"].iloc[0]), 100.0)

    def test_reupload_same_file_changes_nothing_but_history(self):
        self.upload()
        info = self.upload()
        self.assertEqual((info["readings_new"], info["readings_updated"], info["readings_unchanged"]), (0, 0, 37))
        self.assertEqual(self.count("cgm_readings"), 37)
        self.assertEqual(len(us.get_upload_history("A", db_path=self.db)), 2)

    def test_same_instant_is_overwritten_with_latest_value(self):
        self.upload()
        info = self.upload(value=lambda m: glucose_at(m) + 10)
        self.assertEqual((info["readings_new"], info["readings_updated"], info["readings_unchanged"]), (0, 37, 0))
        cgm = us.load_cgm("A", db_path=self.db)
        self.assertEqual(len(cgm), 37)
        self.assertEqual(float(cgm["glucose_mg_dl"].iloc[0]), 110.0)

    def test_partially_overlapping_files_merge_without_duplicates(self):
        self.upload(start_hour=7, minutes=120)  # 07:00-09:00 (25 readings)
        info = self.upload(start_hour=8, minutes=120)  # 08:00-10:00 (25 readings, 13 shared)
        self.assertEqual(info["readings_new"], 12)
        self.assertEqual(info["readings_unchanged"] + info["readings_updated"], 13)
        self.assertEqual(self.count("cgm_readings"), 37)

    def test_offset_notation_does_not_duplicate_an_instant(self):
        self.upload(csv_text="timestamp,glucose_mg_dl\n2026-10-20T07:00:00-07:00,100\n")
        self.upload(csv_text="timestamp,glucose_mg_dl\n2026-10-20T14:00:00Z,100\n")  # same instant
        self.assertEqual(self.count("cgm_readings"), 1)

    def test_excluded_rows_are_kept_in_upload_history(self):
        bad = ("not-a-date,120", "2026-10-20T11:00:00,abc")
        self.upload(bad_rows=bad)
        history = us.get_upload_history("A", db_path=self.db)
        self.assertEqual(len(history), 1)
        item = history[0]
        self.assertEqual(item["filename"], "cgm.csv")
        self.assertEqual(item["rows_excluded"], 2)
        self.assertEqual(item["report"]["excluded_total"], 2)
        self.assertEqual(len(item["report"]["excluded"]), 2)
        self.assertEqual(self.count("cgm_readings"), 37)  # bad rows were not stored

    def test_nothing_usable_raises_and_stores_nothing(self):
        result = parse_cgm_csv("timestamp,glucose_mg_dl\nbad,100\n2026-10-20T07:00:00,xyz\n")
        with self.assertRaises(ValueError):
            us.save_cgm_upload("A", result, db_path=self.db)
        self.assertEqual(self.count("uploads"), 0)

    def test_blank_user_id_is_rejected(self):
        result = parse_cgm_csv(make_csv(minutes=10))
        for bad in ("", "  ", None):
            with self.assertRaises(ValueError):
                us.save_cgm_upload(bad, result, db_path=self.db)


class MultiDateTests(UploadStoreTestBase):
    def setUp(self):
        super().setUp()
        self.upload(day="2026-10-20")
        self.upload(day="2026-10-21")
        us.save_meal("A", "2026-10-20T08:00:00", ["white rice", "egg"], db_path=self.db)
        us.save_meal("A", "2026-10-21T08:00:00", ["granola"], db_path=self.db)

    def test_dates_are_kept_apart(self):
        day1 = us.load_cgm("A", "2026-10-20", "2026-10-20", db_path=self.db)
        day2 = us.load_cgm("A", "2026-10-21", "2026-10-21", db_path=self.db)
        self.assertEqual((len(day1), len(day2)), (37, 37))
        self.assertTrue(set(day1["time"]).isdisjoint(set(day2["time"])))
        self.assertEqual(len(us.load_cgm("A", "2026-10-20", "2026-10-21", db_path=self.db)), 74)
        self.assertEqual(len(us.load_cgm("A", db_path=self.db)), 74)
        self.assertEqual(len(us.load_cgm("A", "2026-10-22", db_path=self.db)), 0)

    def test_meals_filtered_by_local_date(self):
        meals1 = us.list_meals("A", "2026-10-20", "2026-10-20", db_path=self.db)
        meals2 = us.list_meals("A", "2026-10-21", "2026-10-21", db_path=self.db)
        self.assertEqual([m["foods"] for m in meals1], [["white_rice", "egg"]])
        self.assertEqual([m["foods"] for m in meals2], [["granola"]])
        self.assertEqual(len(us.list_meals("A", db_path=self.db)), 2)

    def test_list_dates(self):
        dates = us.list_dates("A", db_path=self.db)
        self.assertEqual([d["date"] for d in dates], ["2026-10-20", "2026-10-21"])
        self.assertTrue(all(d["cgm_readings"] == 37 and d["meals"] == 1 for d in dates))

    def test_meals_on_different_days_do_not_overlap(self):
        for meal in us.list_meals("A", db_path=self.db):
            self.assertEqual(meal["warnings"], [])
            self.assertEqual(meal["status"], "ok")

    def test_late_evening_meal_belongs_to_its_local_date(self):
        # 22:30 PDT on the 20th is already the 21st in UTC; it must still be listed under the 20th.
        us.save_meal("A", "2026-10-20T22:30:00", ["pasta"], db_path=self.db)
        on_20 = [m["foods"] for m in us.list_meals("A", "2026-10-20", "2026-10-20", db_path=self.db)]
        on_21 = [m["foods"] for m in us.list_meals("A", "2026-10-21", "2026-10-21", db_path=self.db)]
        self.assertIn(["pasta"], on_20)
        self.assertNotIn(["pasta"], on_21)


class UserIsolationTests(UploadStoreTestBase):
    def setUp(self):
        super().setUp()
        self.upload("A", value=lambda m: 100.0)
        self.upload("B", value=lambda m: 200.0)  # same instants, different user
        self.meal_a = us.save_meal("A", "2026-10-20T08:00:00", ["rice"], db_path=self.db)
        self.meal_b = us.save_meal("B", "2026-10-20T08:00:00", ["pizza"], db_path=self.db)  # same instant, other user

    def test_same_instant_for_two_users_is_two_rows(self):
        self.assertEqual(self.count("cgm_readings", "A"), 37)
        self.assertEqual(self.count("cgm_readings", "B"), 37)
        self.assertEqual(self.count("meals"), 2)
        self.assertNotEqual(self.meal_a["meal_id"], self.meal_b["meal_id"])

    def test_reads_only_return_own_data(self):
        self.assertEqual(set(us.load_cgm("A", db_path=self.db)["glucose_mg_dl"]), {100.0})
        self.assertEqual(set(us.load_cgm("B", db_path=self.db)["glucose_mg_dl"]), {200.0})
        self.assertEqual([m["foods"] for m in us.list_meals("A", db_path=self.db)], [["rice"]])
        self.assertEqual([m["foods"] for m in us.list_meals("B", db_path=self.db)], [["pizza"]])
        self.assertEqual(len(us.get_upload_history("A", db_path=self.db)), 1)
        self.assertEqual(us.list_dates("C", db_path=self.db), [])
        self.assertEqual(len(us.load_cgm("C", db_path=self.db)), 0)

    def test_cannot_delete_another_users_meal(self):
        self.assertFalse(us.delete_meal("A", self.meal_b["meal_id"], db_path=self.db))
        self.assertEqual(len(us.list_meals("B", db_path=self.db)), 1)

    def test_one_users_upload_does_not_change_anothers_analysis(self):
        before = us.list_meals("B", db_path=self.db)[0]
        self.upload("A", value=lambda m: 300.0)  # A overwrites own values
        after = us.list_meals("B", db_path=self.db)[0]
        for key in ("baseline_mg_dl", "peak_mg_dl", "rise_mg_dl", "status"):
            self.assertEqual(before[key], after[key])
        self.assertEqual(after["baseline_mg_dl"], 200.0)

    def test_uploaded_frames_are_per_user(self):
        cgm, meals = us.uploaded_frames("A", db_path=self.db)
        self.assertEqual(set(cgm["user_id"]), {"A"})
        self.assertEqual(set(meals["user_id"]), {"A"})
        self.assertEqual(list(meals["food"]), ["rice"])


class MealTests(UploadStoreTestBase):
    def test_multiple_foods_saved_in_order_without_duplicates(self):
        info = us.save_meal("A", "2026-10-20T08:00:00", ["White Rice", "Egg", "spinach", "egg"], db_path=self.db)
        self.assertEqual(info["action"], "created")
        self.assertEqual(info["foods"], ["white_rice", "egg", "spinach"])
        meals = us.list_meals("A", db_path=self.db)
        self.assertEqual(len(meals), 1)
        self.assertEqual(meals[0]["foods"], ["white_rice", "egg", "spinach"])
        self.assertEqual(self.count("meal_foods"), 3)

    def test_blank_foods_are_ignored_and_empty_meal_is_rejected(self):
        info = us.save_meal("A", "2026-10-20T08:00:00", ["rice", "  ", ""], db_path=self.db)
        self.assertEqual(info["foods"], ["rice"])
        for bad in ([], ["", " "], None):
            with self.assertRaises(ValueError):
                us.save_meal("A", "2026-10-20T09:00:00", bad, db_path=self.db)
        self.assertEqual(self.count("meals"), 1)

    def test_bad_meal_time_is_rejected(self):
        for bad in ("lunch", "10/20/2026 8:00", ""):
            with self.assertRaises(ValueError):
                us.save_meal("A", bad, ["rice"], db_path=self.db)
        self.assertEqual(self.count("meals"), 0)

    def test_saving_same_meal_again_does_not_duplicate(self):
        first = us.save_meal("A", "2026-10-20T08:00:00", ["rice", "egg"], db_path=self.db)
        again = us.save_meal("A", "2026-10-20T08:00:00", ["rice", "egg"], db_path=self.db)
        self.assertEqual(again["action"], "unchanged")
        self.assertEqual(again["meal_id"], first["meal_id"])
        self.assertEqual(self.count("meals"), 1)
        self.assertEqual(self.count("meal_foods"), 2)

    def test_same_time_with_other_foods_replaces_the_food_list(self):
        first = us.save_meal("A", "2026-10-20T08:00:00", ["rice", "egg", "spinach"], db_path=self.db)
        again = us.save_meal("A", "2026-10-20T08:00:00", ["granola"], db_path=self.db)
        self.assertEqual(again["action"], "updated")
        self.assertEqual(again["meal_id"], first["meal_id"])
        self.assertEqual(us.list_meals("A", db_path=self.db)[0]["foods"], ["granola"])
        self.assertEqual(self.count("meals"), 1)
        self.assertEqual(self.count("meal_foods"), 1)

    def test_same_instant_in_another_offset_is_the_same_meal(self):
        first = us.save_meal("A", "2026-10-20T08:00:00-07:00", ["rice"], db_path=self.db)
        again = us.save_meal("A", "2026-10-20T15:00:00Z", ["rice", "egg"], db_path=self.db)
        self.assertEqual(again["meal_id"], first["meal_id"])
        self.assertEqual(self.count("meals"), 1)
        self.assertEqual(us.list_meals("A", db_path=self.db)[0]["foods"], ["rice", "egg"])

    def test_naive_meal_time_is_read_as_los_angeles(self):
        info = us.save_meal("A", "2026-10-20T08:00:00", ["rice"], db_path=self.db)
        self.assertEqual(info["meal_time_utc"], "2026-10-20T15:00:00Z")

    def test_delete_meal_removes_foods_and_analysis(self):
        info = us.save_meal("A", "2026-10-20T08:00:00", ["rice", "egg"], db_path=self.db)
        self.assertTrue(us.delete_meal("A", info["meal_id"], db_path=self.db))
        self.assertFalse(us.delete_meal("A", info["meal_id"], db_path=self.db))
        self.assertEqual(us.list_meals("A", db_path=self.db), [])
        for table in ("meals", "meal_foods", "meal_analysis"):
            self.assertEqual(self.count(table), 0, table)


class AnalysisCacheTests(UploadStoreTestBase):
    def test_meal_with_cgm_gets_one_response_for_the_whole_meal(self):
        self.upload()
        us.save_meal("A", "2026-10-20T08:00:00", ["white rice", "egg", "spinach"], db_path=self.db)
        meal = us.list_meals("A", db_path=self.db)[0]
        self.assertEqual(meal["status"], "ok")
        self.assertEqual((meal["baseline_mg_dl"], meal["peak_mg_dl"], meal["rise_mg_dl"]), (100.0, 160.0, 60.0))
        self.assertEqual(meal["data_type"], "simulated")  # all demo data is synthetic
        self.assertEqual(meal["source"], "uploaded")
        self.assertEqual(meal["foods"], ["white_rice", "egg", "spinach"])
        self.assertEqual(self.count("meal_analysis"), 1)  # one row per meal, not per food
        self.assertIn("not assigned to individual foods", meal["interpretation_note"])

    def test_meal_without_cgm_is_insufficient_not_invented(self):
        us.save_meal("A", "2026-10-20T08:00:00", ["rice"], db_path=self.db)
        meal = us.list_meals("A", db_path=self.db)[0]
        self.assertEqual(meal["status"], "insufficient_data")
        self.assertIsNone(meal["rise_mg_dl"])
        self.assertIsNone(meal["peak_mg_dl"])
        self.assertIn("too_few_readings_before_meal", meal["reasons"])
        self.assertIn("too_few_readings_after_meal", meal["reasons"])

    def test_cache_is_recomputed_when_cgm_arrives_later(self):
        us.save_meal("A", "2026-10-20T08:00:00", ["rice"], db_path=self.db)
        self.assertEqual(us.list_meals("A", db_path=self.db)[0]["status"], "insufficient_data")
        self.upload()
        meal = us.list_meals("A", db_path=self.db)[0]
        self.assertEqual(meal["status"], "ok")
        self.assertEqual(meal["rise_mg_dl"], 60.0)

    def test_cache_follows_overwritten_readings(self):
        self.upload()
        us.save_meal("A", "2026-10-20T08:00:00", ["rice"], db_path=self.db)
        self.upload(value=lambda m: glucose_at(m) + 20 if m >= 60 else glucose_at(m))  # peak 180
        meal = us.list_meals("A", db_path=self.db)[0]
        self.assertEqual((meal["peak_mg_dl"], meal["rise_mg_dl"]), (180.0, 80.0))

    def test_too_few_post_meal_readings(self):
        self.upload(minutes=90)  # CGM ends 08:30, only 7 readings after a 08:00 meal... use a later meal
        us.save_meal("A", "2026-10-20T08:25:00", ["rice"], db_path=self.db)  # readings 08:25, 08:30 only
        meal = us.list_meals("A", db_path=self.db)[0]
        self.assertEqual(meal["status"], "insufficient_data")
        self.assertEqual(meal["reasons"], ["too_few_readings_after_meal"])
        self.assertEqual(meal["post_meal_readings"], 2)

    def test_overlap_warnings_on_close_meals(self):
        self.upload()
        us.save_meal("A", "2026-10-20T08:00:00", ["rice"], db_path=self.db)
        us.save_meal("A", "2026-10-20T09:00:00", ["cake"], db_path=self.db)
        first, second = us.list_meals("A", db_path=self.db)
        self.assertEqual([w["code"] for w in first["warnings"]], ["overlaps_next_meal"])
        self.assertEqual([w["code"] for w in second["warnings"]], ["overlaps_previous_meal"])
        self.assertEqual(first["warnings"][0]["minutes"], 60.0)

    def test_refresh_analysis_rebuilds_a_wiped_cache(self):
        self.upload()
        us.save_meal("A", "2026-10-20T08:00:00", ["rice"], db_path=self.db)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("DELETE FROM meal_analysis")
        self.assertIsNone(us.list_meals("A", db_path=self.db)[0]["status"])
        us.refresh_analysis("A", db_path=self.db)
        meal = us.list_meals("A", db_path=self.db)[0]
        self.assertEqual((meal["status"], meal["rise_mg_dl"]), ("ok", 60.0))


class FramesForGlucoseAnalysisTests(UploadStoreTestBase):
    def setUp(self):
        super().setUp()
        self.upload()
        us.save_meal("A", "2026-10-20T08:00:00", ["white rice", "egg"], db_path=self.db)

    def test_uploaded_frames_layout(self):
        cgm, meals = us.uploaded_frames("A", db_path=self.db)
        self.assertEqual(list(meals.columns), [
            "meal_id", "user_id", "timestamp", "food", "description",
            "primary_portion_g", "carbs_g", "time", "data_type", "source",
        ])
        row = meals.iloc[0]
        self.assertEqual(row["food"], "white_rice")  # first food key
        self.assertEqual(row["description"], "white_rice, egg")  # all foods
        self.assertTrue(pd.isna(row["primary_portion_g"]))
        self.assertTrue(pd.isna(row["carbs_g"]))
        self.assertEqual(row["data_type"], "simulated")
        self.assertEqual(row["source"], "uploaded")
        self.assertTrue(row["meal_id"].startswith("A_UP"))
        self.assertEqual(set(cgm["data_type"]), {"simulated"})
        self.assertEqual(set(cgm["source"]), {"uploaded"})

    def test_existing_analysis_functions_match_the_cache(self):
        cgm, meals = us.uploaded_frames("A", db_path=self.db)
        history = get_meal_history("A", cgm=cgm, meals=meals)
        cached = us.list_meals("A", db_path=self.db)[0]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["baseline"], cached["baseline_mg_dl"])
        self.assertEqual(history[0]["peak"], cached["peak_mg_dl"])
        self.assertEqual(history[0]["rise"], cached["rise_mg_dl"])

    def test_empty_user_gives_empty_frames_with_columns(self):
        cgm, meals = us.uploaded_frames("C", db_path=self.db)
        self.assertTrue(cgm.empty and meals.empty)
        self.assertIn("time", cgm.columns)
        self.assertIn("food", meals.columns)

    def test_combined_frames_label_sources_and_prefer_uploads(self):
        stamps = pd.to_datetime(["2026-10-20T15:00:00Z", "2026-10-20T15:05:00Z", "2026-10-19T15:00:00Z"], utc=True)
        base_cgm = pd.DataFrame({
            "user_id": ["A", "A", "A"],
            "timestamp": [t.isoformat() for t in stamps],
            "glucose_mg_dl": [999.0, 999.0, 111.0],
            "time": stamps,
        })
        other_user = base_cgm.assign(user_id="B")
        base_cgm = pd.concat([base_cgm, other_user], ignore_index=True)
        base_meals = pd.DataFrame({
            "meal_id": ["A_SIM1", "A_SIM2", "B_SIM1"],
            "user_id": ["A", "A", "B"],
            "timestamp": ["2026-10-20T08:00:00-07:00", "2026-10-19T08:00:00-07:00", "2026-10-19T08:00:00-07:00"],
            "food": ["simulated_food_same_time", "other_day", "b_food"],
            "description": ["x", "y", "z"],
            "primary_portion_g": [100.0, 100.0, 100.0],
            "carbs_g": [float("nan")] * 3,
            "time": pd.to_datetime(
                ["2026-10-20T15:00:00Z", "2026-10-19T15:00:00Z", "2026-10-19T15:00:00Z"], utc=True
            ),
        })
        cgm, meals = us.combined_frames("A", base_cgm, base_meals, db_path=self.db)

        self.assertEqual(set(cgm["user_id"]), {"A"})  # user B's simulated rows are not included
        self.assertEqual(set(meals["user_id"]), {"A"})
        # the 15:00Z and 15:05Z instants exist in both sources: only the uploaded row remains
        shared = cgm[cgm["time"].isin(stamps[:2])]
        self.assertEqual(set(shared["source"]), {"uploaded"})
        self.assertNotIn(999.0, set(cgm["glucose_mg_dl"]))
        # the generated row on another day is kept and its source is "generated"
        sim = cgm[cgm["source"] == "generated"]
        self.assertEqual(list(sim["glucose_mg_dl"]), [111.0])
        # meal at the same instant: uploaded wins; the other-day generated meal stays
        self.assertEqual(sorted(meals["source"]), ["generated", "uploaded"])
        # every row is still "simulated": all demo data is synthetic, only the source differs
        self.assertEqual(set(cgm["data_type"]), {"simulated"})
        self.assertEqual(set(meals["data_type"]), {"simulated"})
        self.assertNotIn("simulated_food_same_time", set(meals["food"]))
        self.assertIn("other_day", set(meals["food"]))
        # the caller's frames are not modified
        self.assertNotIn("data_type", base_cgm.columns)
        self.assertNotIn("source", base_cgm.columns)
        self.assertEqual(len(base_cgm), 6)


class ConnectionHandlingTests(UploadStoreTestBase):
    def test_database_file_can_be_deleted_after_use(self):
        """On Windows an unclosed SQLite connection makes this raise PermissionError (WinError 32)."""
        self.upload()
        us.save_meal("A", "2026-10-20T08:00:00", ["rice"], db_path=self.db)
        us.list_meals("A", db_path=self.db)
        us.delete_meal("A", 1, db_path=self.db)
        us.get_upload_history("A", db_path=self.db)
        self.db.unlink()
        self.assertFalse(self.db.exists())

    def test_failed_call_leaves_database_unchanged(self):
        self.upload()
        before = self.count("meals")
        with self.assertRaises(ValueError):
            us.save_meal("A", "2026-10-20T08:00:00", [], db_path=self.db)
        self.assertEqual(self.count("meals"), before)


if __name__ == "__main__":
    unittest.main()
