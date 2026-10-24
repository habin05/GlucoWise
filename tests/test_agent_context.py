import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import agent_context as ac  # noqa: E402
import glucose_analysis as ga  # noqa: E402


class AgentContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cgm, cls.meals = ga.load_data()

    def test_context_matches_analysis_and_is_json(self):
        ctx = ac.build_agent_context("A", self.cgm, self.meals)
        json.dumps(ctx)  # must be serialisable
        self.assertEqual(ctx["data_type"], "simulated")
        self.assertEqual(ctx["food_summary"], ga.compare_foods("A", self.cgm, self.meals))
        self.assertEqual(len(ctx["meals"]), 21)
        self.assertIn("SIMULATED", ctx["notice"])

    def test_context_only_contains_own_user(self):
        for u in "ABC":
            ctx = ac.build_agent_context(u, self.cgm, self.meals)
            self.assertTrue(all(m["meal_id"].startswith(u + "_") for m in ctx["meals"]))

    def test_prompt_contains_question_and_numbers(self):
        p = ac.build_agent_prompt("A", "Compare rice?", self.cgm, self.meals)
        self.assertIn("User question: Compare rice?", p)
        self.assertIn("73.9", p)


if __name__ == "__main__":
    unittest.main()
