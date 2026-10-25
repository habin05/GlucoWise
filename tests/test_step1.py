"""Tests for the GlucoWise Step 1 data pipeline (standard library unittest).

Run from the project root:
    python -m unittest discover -s tests -v
"""
import io
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data_generator as gen  # noqa: E402
import schema  # noqa: E402

DATA = ROOT / "data"


def as_text(df: pd.DataFrame) -> pd.DataFrame:
    """Round-trip through CSV text exactly like the validator reads files."""
    return schema.read_csv_as_text(io.StringIO(df.to_csv(index=False)))


def meal_metrics(cgm: pd.DataFrame, meals: pd.DataFrame) -> pd.DataFrame:
    """Independent re-implementation of the Step 2 metric definitions."""
    cgm = cgm.assign(t=pd.to_datetime(cgm["timestamp"], utc=True))
    rows = []
    for _, m in meals.iterrows():
        t0 = pd.Timestamp(m["timestamp"]).tz_convert("UTC")
        g = cgm[cgm["user_id"] == m["user_id"]].set_index("t")["glucose_mg_dl"].astype(float)
        pre = g[(g.index >= t0 - pd.Timedelta(minutes=15)) & (g.index < t0)]
        post = g[(g.index >= t0) & (g.index <= t0 + pd.Timedelta(minutes=120))]
        rows.append({
            "meal_id": m["meal_id"], "n_pre": len(pre), "n_post": len(post),
            "baseline": pre.mean(), "peak": post.max(), "n_at_peak": int((post == post.max()).sum()),
            "tpk": (post.idxmax() - t0).total_seconds() / 60,
        })
    return pd.DataFrame(rows)


def targets() -> pd.DataFrame:
    rows = []
    for sc in gen.SCENARIOS.values():
        for i, (_, food, _, _, base, peak) in enumerate(sc["meals"]):
            rows.append({"meal_id": f"{sc['user_id']}_D{i // 3 + 1:02d}_{i % 3 + 1}_{food}",
                         "t_base": float(base), "t_peak": float(peak), "t_tpk": gen.PEAK_MINUTES[i]})
    return pd.DataFrame(rows)


class ShippedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cgm = pd.read_csv(DATA / "cgm.csv", dtype={"user_id": str})
        cls.meals = pd.read_csv(DATA / "meals.csv", dtype={"user_id": str}, keep_default_na=False,
                                na_values=[""])

    def test_validator_passes_without_warnings(self):
        errors, warnings = schema.validate(DATA, verbose=False)
        self.assertEqual((errors, warnings), ([], []))

    def test_counts(self):
        self.assertEqual(len(self.cgm), 6048)
        self.assertEqual(len(self.meals), 63)
        self.assertEqual(self.meals.groupby("user_id").size().to_dict(), {"A": 21, "B": 21, "C": 21})
        counts = self.meals.groupby(["user_id", "food"]).size().to_dict()
        for food in ("white_rice", "brown_rice", "whole_oats"):
            self.assertEqual(counts[("A", food)], 7)
        for food in ("boiled_potato", "baked_sweet_potato", "white_rice"):
            self.assertEqual(counts[("C", food)], 7)
        self.assertEqual(sum(v for (u, _), v in counts.items() if u == "B"), 21)
        self.assertNotIn("oat_rice", set(self.meals["food"]))

    def test_targets_exact_and_peak_unique(self):
        r = meal_metrics(self.cgm, self.meals).merge(targets(), on="meal_id")
        self.assertEqual(len(r), 63)
        self.assertTrue((r["n_pre"] == 3).all() and (r["n_post"] == 25).all())
        self.assertEqual(float((r["baseline"] - r["t_base"]).abs().max()), 0.0)
        self.assertEqual(float((r["peak"] - r["t_peak"]).abs().max()), 0.0)
        self.assertTrue((r["n_at_peak"] == 1).all(), "every meal must have exactly one peak reading")
        self.assertTrue((r["tpk"] == r["t_tpk"]).all())

    def test_carbs_blank_not_zero(self):
        self.assertTrue(self.meals["carbs_g"].isna().all())
        raw = (DATA / "meals.csv").read_text().splitlines()[1:]
        self.assertTrue(all(",," in line for line in raw))  # blank carbs cell, no "0"

    def test_no_clipping_and_plausible_steps(self):
        self.assertTrue(self.cgm["glucose_mg_dl"].between(61, 249).all())
        steps = self.cgm.groupby("user_id")["glucose_mg_dl"].diff().abs()
        self.assertLess(steps.max(), 20)

    def test_files_are_reproducible(self):
        cgm, meals = gen.generate()
        self.assertEqual(cgm.to_csv(index=False, lineterminator="\n"), (DATA / "cgm.csv").read_text())
        self.assertEqual(meals.to_csv(index=False, lineterminator="\n"), (DATA / "meals.csv").read_text())

    def test_timezone_constants_agree(self):
        self.assertEqual(gen.TIMEZONE, schema.TIMEZONE)


class DaylightSaving(unittest.TestCase):
    """A range crossing the Nov 1 2026 DST end must keep local meal times."""

    @classmethod
    def setUpClass(cls):
        cls.cgm, cls.meals = gen.generate("2026-10-30")

    def test_meals_stay_on_wall_clock_times(self):
        local = pd.to_datetime(self.meals["timestamp"], utc=True).dt.tz_convert(gen.TIMEZONE)
        self.assertEqual(sorted(local.dt.strftime("%H:%M").unique()), ["08:00", "12:30", "18:30"])
        nov1 = self.meals[self.meals["timestamp"].str.startswith("2026-11-01T08:00")]
        self.assertTrue(nov1["timestamp"].str.endswith("-08:00").all())
        oct31 = self.meals[self.meals["timestamp"].str.startswith("2026-10-31T08:00")]
        self.assertTrue(oct31["timestamp"].str.endswith("-07:00").all())

    def test_cgm_has_one_extra_hour_and_validates(self):
        self.assertEqual(len(self.cgm), 3 * (7 * 288 + 12))  # Nov 1 has 25 hours
        errors, warnings = schema.validate_frames(as_text(self.cgm), as_text(self.meals))
        self.assertEqual((errors, warnings), ([], []))

    def test_targets_still_exact(self):
        r = meal_metrics(self.cgm, self.meals).merge(targets(), on="meal_id")
        self.assertEqual(float((r["baseline"] - r["t_base"]).abs().max()), 0.0)
        self.assertEqual(float((r["peak"] - r["t_peak"]).abs().max()), 0.0)
        self.assertTrue((r["n_at_peak"] == 1).all())


class ValidatorRejectsBadData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cgm0, cls.meals0 = as_text(gen.generate()[0]), as_text(gen.generate()[1])

    def run_case(self, mutate):
        cgm, meals = self.cgm0.copy(), self.meals0.copy()
        out = mutate(cgm, meals)
        if out is not None:
            cgm, meals = out
        return schema.validate_frames(cgm, meals)

    def assertRejected(self, mutate, needle):
        errors, _ = self.run_case(mutate)
        self.assertTrue(any(needle in e for e in errors), f"{needle!r} not in {errors}")

    def test_baseline_clean(self):
        self.assertEqual(self.run_case(lambda c, m: None), ([], []))

    def test_missing_column(self):
        self.assertRejected(lambda c, m: (c.drop(columns=["glucose_mg_dl"]), m), "missing columns")

    def test_old_portion_column_rejected(self):
        self.assertRejected(lambda c, m: (c, m.rename(columns={"primary_portion_g": "portion_g"})), "missing columns")

    def test_empty_files(self):
        self.assertRejected(lambda c, m: (c, m.iloc[0:0]), "no data rows")

    def test_naive_timestamp(self):
        self.assertRejected(lambda c, m: (c.assign(timestamp=c["timestamp"].str[:-6]), m), "not in 'YYYY-MM-DDTHH:MM:SS")

    def test_z_suffix_rejected(self):
        self.assertRejected(lambda c, m: (c.assign(timestamp=c["timestamp"].str[:-6] + "Z"), m), "not in 'YYYY-MM-DDTHH:MM:SS")

    def test_wrong_offset_on_one_row(self):
        def mut(c, m):
            c.loc[5, "timestamp"] = c.loc[5, "timestamp"].replace("-07:00", "-08:00")
            return c, m
        self.assertRejected(mut, "offset that is wrong")

    def test_duplicate_cgm_row(self):
        self.assertRejected(lambda c, m: (pd.concat([c, c.iloc[[10]]]).sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True), m),
                            "duplicate (user_id, timestamp)")

    def test_gap_in_cgm(self):
        self.assertRejected(lambda c, m: (c.drop(index=range(100, 130)).reset_index(drop=True), m), "not exactly 5 minutes")

    def test_off_grid_reading(self):
        def mut(c, m):
            ts = pd.Timestamp(c.loc[200, "timestamp"]) + pd.Timedelta(minutes=1)
            c.loc[200, "timestamp"] = ts.isoformat(timespec="seconds")
            return c, m
        self.assertRejected(mut, "not exactly 5 minutes")

    def test_unsorted_cgm(self):
        self.assertRejected(lambda c, m: (c.sample(frac=1, random_state=1).reset_index(drop=True), m), "increasing time order")

    def test_user_blocks_not_contiguous_or_sorted(self):
        def mut(c, m):
            return pd.concat([c[c["user_id"] == "B"], c[c["user_id"] != "B"]]).reset_index(drop=True), m
        self.assertRejected(mut, "sorted by user_id")

    def test_unsorted_meals(self):
        self.assertRejected(lambda c, m: (c, m.iloc[::-1].reset_index(drop=True)), "increasing time order")

    def test_duplicate_meal_id(self):
        self.assertRejected(lambda c, m: (c, m.assign(meal_id=m["meal_id"].where(m.index != 1, m["meal_id"][0]))),
                            "duplicate meal_id")

    def test_bad_meal_id_format(self):
        self.assertRejected(lambda c, m: (c, m.assign(meal_id=m["meal_id"].where(m.index != 0, "meal-1"))), "meal_id not in")

    def test_meal_id_user_mismatch(self):
        self.assertRejected(lambda c, m: (c, m.assign(meal_id=m["meal_id"].where(m.index != 0, "B_D01_1_white_rice"))), "prefix does not match")

    def test_bad_user_id_format(self):
        self.assertRejected(lambda c, m: (c, m.assign(user_id=m["user_id"].where(m.index != 0, "a b"))), "invalid user_id")

    def test_unknown_meal_user(self):
        def mut(c, m):
            m.loc[0, ["user_id", "meal_id"]] = ["Z", "Z_D01_1_white_rice"]
            return c, m
        self.assertRejected(mut, "Meal users without CGM data")

    def test_invalid_scenario_id(self):
        self.assertRejected(lambda c, m: (c, m.assign(scenario_id=m["scenario_id"].where(m.index != 0, "Q"))), "invalid scenario_id")

    def test_scenario_d_not_accepted_yet(self):
        self.assertRejected(lambda c, m: (c, m.assign(scenario_id=m["scenario_id"].where(m.index != 0, "D"))), "invalid scenario_id")

    def test_user_with_two_scenarios(self):
        self.assertRejected(lambda c, m: (c, m.assign(scenario_id=m["scenario_id"].where(m.index != 0, "B"))), "more than one scenario_id")

    def test_invalid_meal_order(self):
        self.assertRejected(lambda c, m: (c, m.assign(meal_order=m["meal_order"].where(m.index != 0, "rice_first"))), "invalid meal_order")

    def test_uppercase_food(self):
        self.assertRejected(lambda c, m: (c, m.assign(food=m["food"].where(m.index != 0, "White Rice"))), "food keys")

    def test_blank_and_bad_glucose(self):
        self.assertRejected(lambda c, m: (c.assign(glucose_mg_dl=c["glucose_mg_dl"].where(c.index != 3, "")), m), "glucose_mg_dl")
        self.assertRejected(lambda c, m: (c.assign(glucose_mg_dl=c["glucose_mg_dl"].where(c.index != 3, "abc")), m), "glucose_mg_dl")

    def test_negative_or_zero_portion(self):
        self.assertRejected(lambda c, m: (c, m.assign(primary_portion_g=m["primary_portion_g"].where(m.index != 0, "-5"))), "primary_portion_g must be > 0")
        self.assertRejected(lambda c, m: (c, m.assign(primary_portion_g=m["primary_portion_g"].where(m.index != 0, "0"))), "primary_portion_g must be > 0")

    def test_negative_carbs(self):
        self.assertRejected(lambda c, m: (c, m.assign(carbs_g=m["carbs_g"].where(m.index != 0, "-1"))), "carbs_g must be >= 0")

    def test_meal_outside_cgm_range(self):
        def mut(c, m):
            m.loc[0, "timestamp"] = "2026-11-20T08:00:00-08:00"
            return c, m
        self.assertRejected(mut, "outside the CGM time range")


class NumericParsingIsWarningFree(unittest.TestCase):
    """Regression: Series.replace("", NaN) raised a pandas 2.2 FutureWarning."""

    def test_numeric_helper_emits_no_warnings(self):
        text = pd.Series(["1.5", " 2 ", "", "abc", "-3", "1e2"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = schema._numeric(text)
        self.assertEqual(out[0], 1.5)
        self.assertEqual(out[1], 2.0)
        self.assertTrue(pd.isna(out[2]) and pd.isna(out[3]))  # blank and text -> NaN
        self.assertEqual(out[4], -3.0)
        self.assertEqual(out[5], 100.0)


class ValidatorAcceptsValidEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cgm, meals = gen.generate()
        cls.cgm0, cls.meals0 = as_text(cgm), as_text(meals)

    def test_user_id_NA_is_a_normal_id(self):
        cgm, meals = self.cgm0.copy(), self.meals0.copy()
        cgm["user_id"] = cgm["user_id"].str.replace(r"^A$", "NA", regex=True)
        meals["meal_id"] = meals["meal_id"].str.replace(r"^A_", "NA_", regex=True)
        meals["user_id"] = meals["user_id"].str.replace(r"^A$", "NA", regex=True)
        # files must stay sorted by user_id, and "NA" now sorts after "B" and "C"
        cgm = cgm.sort_values("user_id", kind="stable").reset_index(drop=True)
        meals = meals.sort_values("user_id", kind="stable").reset_index(drop=True)
        errors, warnings = schema.validate_frames(cgm, meals)
        self.assertEqual((errors, warnings), ([], []))

    def test_NA_survives_csv_round_trip(self):
        cgm, meals = self.cgm0.copy(), self.meals0.copy()
        cgm["user_id"] = cgm["user_id"].str.replace(r"^A$", "NA", regex=True)
        text = schema.read_csv_as_text(io.StringIO(cgm.to_csv(index=False)))
        self.assertIn("NA", set(text["user_id"]))

    def test_known_carbs_allowed_and_larger_than_primary_portion(self):
        meals = self.meals0.copy()
        meals.loc[0, "carbs_g"] = "90"  # whole-meal carbs may exceed the primary food's grams
        self.assertEqual(schema.validate_frames(self.cgm0, meals), ([], []))

    def test_zero_carbs_warns_but_passes(self):
        meals = self.meals0.copy()
        meals.loc[0, "carbs_g"] = "0"
        errors, warnings = schema.validate_frames(self.cgm0, meals)
        self.assertEqual(errors, [])
        self.assertTrue(any("carbs_g contains 0" in w for w in warnings))

    def test_incomplete_window_is_a_warning(self):
        cgm = self.cgm0.copy()
        keep = ~((cgm["user_id"] == "A") & (cgm.index >= cgm.index[cgm["user_id"] == "A"][-1] - 60))
        errors, warnings = schema.validate_frames(cgm[keep].reset_index(drop=True), self.meals0)
        self.assertEqual(errors, [])
        self.assertTrue(any("incomplete 120-min" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
