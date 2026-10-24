import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import glucose_analysis as ga  # noqa: E402


def _tiny():
    times = pd.date_range("2026-10-01 07:30", periods=30, freq="5min", tz="UTC")
    glucose = [100.0] * 6 + [100, 120, 150, 172, 160, 140] + [120] * 18
    cgm = pd.DataFrame({"user_id": "A", "timestamp": times.astype(str), "glucose_mg_dl": glucose[:30]})
    cgm["time"] = times
    t = pd.Timestamp("2026-10-01 08:00", tz="UTC")
    meals = pd.DataFrame(
        {
            "meal_id": ["M1"], "user_id": ["A"], "timestamp": [str(t)], "food": ["white_rice"],
            "description": ["White rice 150g"], "primary_portion_g": [150.0], "time": [t],
        }
    )
    return cgm, meals


class SyntheticMath(unittest.TestCase):
    def test_baseline_peak_rise(self):
        cgm, meals = _tiny()
        r = ga.get_meal_glucose("A", "M1", cgm, meals)
        self.assertEqual((r["baseline"], r["peak"], r["rise"]), (100.0, 172.0, 72.0))
        self.assertEqual(r["data_type"], "simulated")

    def test_empty_window_returns_none(self):
        cgm, meals = _tiny()
        meals.loc[0, "time"] = pd.Timestamp("2026-10-05 08:00", tz="UTC")
        self.assertIsNone(ga.get_meal_glucose("A", "M1", cgm, meals))

    def test_unknown_meal_raises(self):
        cgm, meals = _tiny()
        with self.assertRaises(KeyError):
            ga.get_meal_glucose("A", "nope", cgm, meals)


class RealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cgm, cls.meals = ga.load_data()

    def test_reference_meals(self):
        got = {}
        for food, expected in {"white_rice": (100, 172), "brown_rice": (99, 138), "whole_oats": (101, 130)}.items():
            mid = self.meals[(self.meals.user_id == "A") & (self.meals.food == food)].iloc[0].meal_id
            r = ga.get_meal_glucose("A", mid, self.cgm, self.meals)
            got[food] = (r["baseline"], r["peak"])
            self.assertEqual(got[food], expected)

    def test_user_a_average_rises_match_plan(self):
        rises = {p["food"]: (p["meal_count"], p["average_rise_mg_dl"]) for p in ga.compare_foods("A", self.cgm, self.meals)}
        self.assertEqual(rises, {"white_rice": (7, 73.9), "brown_rice": (7, 41.0), "whole_oats": (7, 31.3)})

    def test_all_63_meals_have_metrics(self):
        total = sum(len(ga.get_meal_history(u, cgm=self.cgm, meals=self.meals)) for u in "ABC")
        self.assertEqual(total, 63)
        for u in "ABC":
            self.assertTrue(all(h["rise"] is not None for h in ga.get_meal_history(u, cgm=self.cgm, meals=self.meals)))

    def test_history_filter_and_users_are_separate(self):
        a = ga.get_meal_history("A", "white rice", self.cgm, self.meals)
        c = ga.get_meal_history("C", "white_rice", self.cgm, self.meals)
        self.assertEqual((len(a), len(c)), (7, 7))
        self.assertTrue(all(h["food"] == "white_rice" for h in a))
        self.assertNotEqual(ga.compare_foods("A", self.cgm, self.meals)[0], ga.compare_foods("C", self.cgm, self.meals)[0])

    def test_curve_window(self):
        mid = self.meals.iloc[0].meal_id
        uid = self.meals.iloc[0].user_id
        curve = ga.get_glucose_curve(uid, mid, self.cgm, self.meals)
        self.assertEqual((curve[0]["minutes_from_meal"], curve[-1]["minutes_from_meal"]), (-30, 120))
        self.assertEqual(len(curve), 31)

    def test_pattern_output_shape(self):
        out = ga.analyze_user_patterns("B", self.cgm, self.meals)
        self.assertEqual(out["data_type"], "simulated")
        self.assertEqual(sum(p["meal_count"] for p in out["patterns"]), 21)


if __name__ == "__main__":
    unittest.main()
