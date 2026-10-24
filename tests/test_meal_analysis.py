"""Tests for src/meal_analysis.py (standard-library unittest, run from the project root:
python -m unittest discover -s tests -v)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

import glucose_analysis as ga  # noqa: E402
from cgm_import import parse_cgm_csv  # noqa: E402
from meal_analysis import (  # noqa: E402
    MIN_POST_READINGS,
    MIN_PRE_READINGS,
    STATUS_INSUFFICIENT,
    STATUS_OK,
    analyze_meals,
    summarize_by_food,
)

MEAL = "2026-10-20T08:00:00-07:00"


def cgm_from(points):
    """points: list of (clock 'HH:MM' on 2026-10-20 PDT, mg/dL)."""
    body = "".join(f"2026-10-20T{clock}:00-07:00,{value}\n" for clock, value in points)
    return parse_cgm_csv("timestamp,glucose_mg_dl\n" + body)


def grid(start_min, end_min, step, value_fn):
    """Readings every ``step`` minutes from minute offsets relative to 08:00."""
    pts = []
    for m in range(start_min, end_min + 1, step):
        total = 8 * 60 + m
        pts.append((f"{total // 60:02d}:{total % 60:02d}", value_fn(m)))
    return pts


def full_day(value_fn=lambda m: 100 + (72 if m == 30 else 0)):
    return cgm_from(grid(-60, 180, 5, value_fn))


class MetricsTests(unittest.TestCase):
    def test_baseline_peak_rise_and_data_type(self):
        r = analyze_meals(full_day(), [{"meal_time": MEAL, "foods": ["white_rice", "egg", "spinach"]}])[0]
        self.assertEqual(r["status"], STATUS_OK)
        self.assertEqual((r["baseline_mg_dl"], r["peak_mg_dl"], r["rise_mg_dl"]), (100.0, 172.0, 72.0))
        self.assertEqual(r["data_type"], "uploaded")  # real uploads are never labelled simulated
        self.assertEqual(r["unit"], "mg/dL")
        self.assertEqual(r["meal_time_utc"], "2026-10-20T15:00:00+00:00")

    def test_data_type_can_be_set_per_meal_or_default(self):
        meals = [{"meal_time": MEAL, "foods": ["a"], "data_type": "simulated"}]
        self.assertEqual(analyze_meals(full_day(), meals)[0]["data_type"], "simulated")
        meals = [{"meal_time": MEAL, "foods": ["a"]}]
        self.assertEqual(analyze_meals(full_day(), meals, data_type="simulated")[0]["data_type"], "simulated")

    def test_window_boundaries(self):
        # baseline: 07:45 (exactly -15 min) is IN, 08:00 (meal time) is OUT, 07:44 is OUT
        # peak: 08:00 is IN, 10:00 (exactly +2 h) is IN, 10:01 is OUT
        pts = [
            ("07:44", 500), ("07:45", 100), ("07:50", 100), ("07:55", 100),
            ("08:00", 90), ("08:30", 120), ("09:00", 130), ("09:30", 140), ("09:45", 150),
            ("10:00", 160), ("10:01", 900),
        ]
        r = analyze_meals(cgm_from(pts), [{"meal_time": MEAL, "foods": ["x"]}])[0]
        self.assertEqual(r["pre_meal_readings"], 3)
        self.assertEqual(r["post_meal_readings"], 6)  # 08:00, 08:30, 09:00, 09:30, 09:45, 10:00
        self.assertEqual(r["baseline_mg_dl"], 100.0)
        self.assertEqual(r["peak_mg_dl"], 160.0)

    def test_naive_meal_time_is_los_angeles(self):
        r = analyze_meals(full_day(), [{"meal_time": "2026-10-20T08:00:00", "foods": ["x"]}])[0]
        self.assertEqual(r["meal_time_utc"], "2026-10-20T15:00:00+00:00")
        self.assertEqual(r["rise_mg_dl"], 72.0)

    def test_irregular_spacing_still_works(self):
        pts = [("07:47", 100), ("07:58", 102), ("08:03", 105), ("08:20", 150), ("08:41", 140),
               ("09:10", 130), ("09:38", 125), ("09:59", 120)]
        r = analyze_meals(cgm_from(pts), [{"meal_time": MEAL, "foods": ["x"]}])[0]
        self.assertEqual(r["status"], STATUS_OK)
        self.assertEqual((r["baseline_mg_dl"], r["peak_mg_dl"]), (101.0, 150.0))


class InsufficientDataTests(unittest.TestCase):
    def test_too_few_readings_before_meal(self):
        pts = [("07:55", 100)] + grid(0, 120, 5, lambda m: 110)
        r = analyze_meals(cgm_from(pts), [{"meal_time": MEAL, "foods": ["x"]}])[0]
        self.assertEqual(r["status"], STATUS_INSUFFICIENT)
        self.assertEqual(r["reasons"], ["too_few_readings_before_meal"])
        self.assertIsNone(r["baseline_mg_dl"])
        self.assertIsNone(r["peak_mg_dl"])
        self.assertIsNone(r["rise_mg_dl"])
        self.assertIn("1 reading(s)", r["message"])

    def test_too_few_readings_after_meal(self):
        pts = grid(-15, -5, 5, lambda m: 100) + grid(0, 20, 5, lambda m: 120)  # 5 after
        r = analyze_meals(cgm_from(pts), [{"meal_time": MEAL, "foods": ["x"]}])[0]
        self.assertEqual(r["post_meal_readings"], 5)
        self.assertEqual(r["status"], STATUS_INSUFFICIENT)
        self.assertEqual(r["reasons"], ["too_few_readings_after_meal"])

    def test_both_missing_lists_both_reasons(self):
        r = analyze_meals(cgm_from([("12:00", 100)]), [{"meal_time": MEAL, "foods": ["x"]}])[0]
        self.assertEqual(r["status"], STATUS_INSUFFICIENT)
        self.assertEqual(len(r["reasons"]), 2)

    def test_exactly_at_thresholds_is_ok(self):
        pts = grid(-10, -5, 5, lambda m: 100) + grid(0, 25, 5, lambda m: 100 + m)  # 2 before, 6 after
        r = analyze_meals(cgm_from(pts), [{"meal_time": MEAL, "foods": ["x"]}])[0]
        self.assertEqual((r["pre_meal_readings"], r["post_meal_readings"]), (MIN_PRE_READINGS, MIN_POST_READINGS))
        self.assertEqual(r["status"], STATUS_OK)
        self.assertEqual(r["peak_mg_dl"], 125.0)

    def test_empty_cgm_never_invents_values(self):
        r = analyze_meals(parse_cgm_csv("timestamp,glucose_mg_dl\n"), [{"meal_time": MEAL, "foods": ["x"]}])[0]
        self.assertEqual(r["status"], STATUS_INSUFFICIENT)
        self.assertIsNone(r["rise_mg_dl"])


class OverlapTests(unittest.TestCase):
    def meals(self, second):
        return [{"meal_time": MEAL, "foods": ["a"]}, {"meal_time": second, "foods": ["b"]}]

    def test_next_meal_within_two_hours_flags_both_directions(self):
        first, second = analyze_meals(full_day(), self.meals("2026-10-20T09:30:00-07:00"))
        self.assertEqual([w["code"] for w in first["warnings"]], ["overlaps_next_meal"])
        self.assertEqual(first["warnings"][0]["minutes"], 90.0)
        self.assertEqual([w["code"] for w in second["warnings"]], ["overlaps_previous_meal"])
        self.assertEqual(first["status"], STATUS_OK)  # analysis still runs, with the caution attached

    def test_exactly_two_hours_is_flagged_and_beyond_is_not(self):
        first, _ = analyze_meals(full_day(), self.meals("2026-10-20T10:00:00-07:00"))
        self.assertEqual(len(first["warnings"]), 1)
        first, second = analyze_meals(full_day(), self.meals("2026-10-20T10:01:00-07:00"))
        self.assertEqual(first["warnings"], [])
        self.assertEqual(second["warnings"], [])

    def test_results_are_in_time_order_regardless_of_input_order(self):
        results = analyze_meals(full_day(), list(reversed(self.meals("2026-10-20T12:00:00-07:00"))))
        self.assertEqual([r["foods"] for r in results], [["a"], ["b"]])


class FoodsTests(unittest.TestCase):
    def test_multi_food_meal_has_one_response_not_split(self):
        r = analyze_meals(full_day(), [{"meal_time": MEAL, "foods": ["white_rice", "egg", "spinach"]}])[0]
        self.assertEqual(r["foods"], ["white_rice", "egg", "spinach"])
        self.assertEqual(r["rise_mg_dl"], 72.0)
        self.assertNotIn("per_food", r)
        self.assertIn("not assigned to individual foods", r["interpretation_note"])

    def test_foods_are_cleaned_and_deduplicated(self):
        r = analyze_meals(full_day(), [{"meal_time": MEAL, "foods": [" White Rice ", "white_rice", "", "Egg"]}])[0]
        self.assertEqual(r["foods"], ["white_rice", "egg"])

    def test_custom_normalizer_is_used(self):
        r = analyze_meals(full_day(), [{"meal_time": MEAL, "foods": ["흰쌀밥"]}],
                          normalizer=lambda s: {"흰쌀밥": "white_rice"}.get(s, s))[0]
        self.assertEqual(r["foods"], ["white_rice"])

    def test_invalid_meals_raise_value_error(self):
        for bad in ({"meal_time": MEAL, "foods": []}, {"meal_time": MEAL, "foods": [" "]},
                    {"meal_time": MEAL}, {"foods": ["x"]}, {"meal_time": "yesterday", "foods": ["x"]}):
            with self.assertRaises(ValueError, msg=str(bad)):
                analyze_meals(full_day(), [bad])

    def test_summary_groups_meals_containing_each_food_without_splitting_rise(self):
        cgm = cgm_from(grid(-60, 600, 5, lambda m: 100 + (50 if m in (30, 270) else 0) + (10 if m == 450 else 0)))
        meals = [
            {"meal_time": "2026-10-20T08:00:00-07:00", "foods": ["white_rice", "egg"]},
            {"meal_time": "2026-10-20T12:30:00-07:00", "foods": ["white_rice", "kimchi"]},
            {"meal_time": "2026-10-20T15:30:00-07:00", "foods": ["egg"]},
        ]
        results = analyze_meals(cgm, meals)
        self.assertEqual([r["rise_mg_dl"] for r in results], [50.0, 50.0, 10.0])
        by_food = {row["food"]: row for row in summarize_by_food(results)}
        self.assertEqual(by_food["white_rice"]["meals_containing_food"], 2)
        self.assertEqual(by_food["white_rice"]["average_rise_mg_dl"], 50.0)
        self.assertEqual(by_food["egg"]["meals_containing_food"], 2)
        self.assertEqual(by_food["egg"]["average_rise_mg_dl"], 30.0)
        self.assertIn("not the effect", by_food["egg"]["note"])


class MultiDayTests(unittest.TestCase):
    def test_meals_on_different_days_use_their_own_readings(self):
        day1 = "".join(f"2026-10-20T{h:02d}:{m:02d}:00-07:00,{100 if (h, m) != (8, 30) else 150}\n"
                       for h in range(6, 11) for m in range(0, 60, 5))
        day2 = "".join(f"2026-10-21T{h:02d}:{m:02d}:00-07:00,{110 if (h, m) != (8, 30) else 210}\n"
                       for h in range(6, 11) for m in range(0, 60, 5))
        cgm = parse_cgm_csv("timestamp,glucose_mg_dl\n" + day1 + day2)
        res = analyze_meals(cgm, [{"meal_time": "2026-10-21T08:00:00-07:00", "foods": ["b"]},
                                  {"meal_time": "2026-10-20T08:00:00-07:00", "foods": ["a"]}])
        self.assertEqual([r["rise_mg_dl"] for r in res], [50.0, 100.0])
        self.assertEqual(cgm.summary["local_dates"], ["2026-10-20", "2026-10-21"])
        self.assertTrue(all(not r["warnings"] for r in res))


class ConsistencyWithExistingAnalysisTests(unittest.TestCase):
    """New code must give the same numbers as the existing glucose_analysis functions."""

    @classmethod
    def setUpClass(cls):
        cls.cgm, cls.meals = ga.load_data()

    def test_every_meal_in_data_csvs_matches_get_meal_glucose(self):
        compared = 0
        for user in sorted(self.meals["user_id"].unique()):
            user_cgm = self.cgm[self.cgm["user_id"] == user]
            rows = self.meals[self.meals["user_id"] == user]
            meals = [{"meal_time": r.timestamp, "foods": [r.food], "meal_id": r.meal_id, "data_type": "simulated"}
                     for r in rows.itertuples()]
            ours = {r["meal_id"]: r for r in analyze_meals(user_cgm, meals)}
            for row in rows.itertuples():
                legacy = ga.get_meal_glucose(user, row.meal_id, self.cgm, self.meals)
                mine = ours[row.meal_id]
                self.assertEqual(mine["status"], STATUS_OK, row.meal_id)
                self.assertEqual(mine["baseline_mg_dl"], legacy["baseline"], row.meal_id)
                self.assertEqual(mine["peak_mg_dl"], legacy["peak"], row.meal_id)
                self.assertEqual(mine["rise_mg_dl"], legacy["rise"], row.meal_id)
                self.assertEqual(mine["data_type"], legacy["data_type"])
                self.assertEqual(mine["warnings"], [], row.meal_id)  # synthetic meals are >= 4 h apart
                compared += 1
        self.assertGreater(compared, 0)

    def test_gate_agrees_with_metrics_windows(self):
        # whenever the gate passes, the legacy _metrics must not return None
        user = self.meals["user_id"].iloc[0]
        user_cgm = self.cgm[self.cgm["user_id"] == user]
        for t in self.meals[self.meals["user_id"] == user]["time"]:
            self.assertIsNotNone(ga._metrics(user_cgm, t))

    def test_uploading_the_repo_csv_through_cgm_import_gives_same_numbers(self):
        user = "A"
        rows = self.cgm[self.cgm["user_id"] == user][["timestamp", "glucose_mg_dl"]]
        parsed = parse_cgm_csv(rows.to_csv(index=False))
        self.assertEqual(parsed.summary["rows_excluded"], 0)
        meal = self.meals[self.meals["user_id"] == user].iloc[0]
        mine = analyze_meals(parsed, [{"meal_time": meal["timestamp"], "foods": [meal["food"]]}])[0]
        legacy = ga.get_meal_glucose(user, meal["meal_id"], self.cgm, self.meals)
        self.assertEqual(mine["rise_mg_dl"], legacy["rise"])


if __name__ == "__main__":
    unittest.main()
