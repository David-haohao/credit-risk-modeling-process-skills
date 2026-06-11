import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT_PATH = SCRIPTS / "06_scorecard.py"
SPEC = importlib.util.spec_from_file_location("stage6_scorecard", SCRIPT_PATH)
STAGE6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE6)


class Stage6ScoringTests(unittest.TestCase):
    def test_round_half_up(self):
        result = STAGE6.round_half_up(np.array([234.499, 234.500, 234.513, -1.5]))
        self.assertEqual(result.tolist(), [234, 235, 235, -2])

    def test_score_mapping_does_not_clip_by_default(self):
        raw, score = STAGE6.probability_to_score(
            np.array([1e-6, 0.999999]), base_score=600, base_odds=35, pdo=60, rate=2,
        )
        self.assertGreater(raw[0], 900)
        self.assertLess(raw[1], 300)
        self.assertEqual(len(score), 2)

    def test_train_grade_edges_are_reused(self):
        train_scores = pd.Series([900, 800, 700, 600, 500])
        edges = STAGE6.fit_grade_edges(train_scores, 5)
        train_grade = STAGE6.apply_grade_edges(train_scores, edges, ["E", "D", "C", "B", "A"])
        oot_grade = STAGE6.apply_grade_edges(pd.Series([850, 550]), edges, ["E", "D", "C", "B", "A"])
        self.assertEqual(len(train_grade), 5)
        self.assertEqual(len(oot_grade), 2)

    def test_platt_is_fit_once_and_applied_to_other_data(self):
        raw = np.array([0.1, 0.2, 0.8, 0.9])
        y = pd.Series([0, 0, 1, 1])
        calibrator = STAGE6.fit_platt_calibrator(raw, y)
        first = STAGE6.apply_calibrator(calibrator, raw)
        second = STAGE6.apply_calibrator(calibrator, np.array([0.3, 0.7]))
        self.assertEqual(len(first), 4)
        self.assertEqual(len(second), 2)
        self.assertGreater(calibrator.coef_[0][0], 0)

    def test_lr_bin_points_use_effective_coefficient(self):
        bins = pd.DataFrame([{"feature": "x", "woe": 0.5, "bin_label": "a"}])
        coefficients = pd.DataFrame([
            {"feature": "__INTERCEPT__", "coefficient": -1.0},
            {"feature": "x", "coefficient": 2.0},
        ])
        detail, base = STAGE6.build_lr_scorecard_tables(
            bins, coefficients, base_score=600, base_odds=35, pdo=60, rate=2,
        )
        self.assertIn("bin_points_raw", detail.columns)
        self.assertEqual(detail.loc[0, "bin_points"], STAGE6.round_half_up(detail.loc[0, "bin_points_raw"]))
        self.assertIn("base_points", base.columns)


if __name__ == "__main__":
    unittest.main()
