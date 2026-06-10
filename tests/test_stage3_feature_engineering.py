import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "03_feature_engineering.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("stage3_feature_engineering", SCRIPT_PATH)
STAGE3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE3)


class SmoothedWoeTests(unittest.TestCase):
    def setUp(self):
        self.binned = pd.DataFrame(
            {
                "feature": [0, 0, 1, 1],
                "target": [0, 0, 1, 1],
            }
        )

    def test_handles_zero_good_and_bad_bins(self):
        transformer, transformed = STAGE3.run_woe(self.binned, "target", smooth=0.5)

        expected_bin_0 = math.log((0 + 0.5) / (2 + 0.5 * 2) / ((2 + 0.5) / (2 + 0.5 * 2)))
        expected_bin_1 = -expected_bin_0
        self.assertAlmostEqual(transformed.loc[0, "feature"], expected_bin_0)
        self.assertAlmostEqual(transformed.loc[2, "feature"], expected_bin_1)
        self.assertTrue(transformed["feature"].map(math.isfinite).all())

        reapplied = transformer.transform(pd.DataFrame({"feature": [0, 1]}))
        self.assertEqual(reapplied["feature"].tolist(), [expected_bin_0, expected_bin_1])

    def test_iv_uses_same_smoothing_formula(self):
        transformer, _ = STAGE3.run_woe(self.binned, "target", smooth=0.5)
        iv = transformer.iv_table().set_index("column").loc["feature", "iv"]

        good_0, bad_0 = 2.5 / 3, 0.5 / 3
        good_1, bad_1 = 0.5 / 3, 2.5 / 3
        expected_iv = (
            (bad_0 - good_0) * math.log(bad_0 / good_0)
            + (bad_1 - good_1) * math.log(bad_1 / good_1)
        )
        self.assertAlmostEqual(iv, expected_iv)


class FeatureAdjustmentTests(unittest.TestCase):
    def test_binning_summary_marks_u_shape_candidate(self):
        binned = pd.DataFrame(
            {
                "x": [0] * 4 + [1] * 4 + [2] * 4,
                "target": [1, 1, 1, 0] + [0, 0, 0, 1] + [1, 1, 1, 0],
            }
        )
        transformer, _ = STAGE3.run_woe(binned, "target", smooth=0.5)

        _, summary = STAGE3.summarize_binning(
            binned, "target", transformer, "before_adjustment"
        )

        self.assertFalse(bool(summary.loc[0, "is_monotonic"]))
        self.assertTrue(bool(summary.loc[0, "u_shape_candidate"]))

    def test_u_shape_abs_replaces_source_feature_across_datasets(self):
        train = pd.DataFrame({"x": [1.0, 3.0, None], "target": [0, 1, 0]})
        oot = pd.DataFrame({"x": [2.0, 5.0], "target": [0, 1]})
        adjustments = pd.DataFrame(
            {
                "source_feature": ["x"],
                "action": ["u_shape_abs"],
                "output_feature": ["x__abs_mid"],
                "x_mid": [2.0],
                "manual_breaks": [""],
            }
        )

        adjusted, lineage = STAGE3.apply_feature_adjustments(
            {"train": train, "oot": oot}, adjustments
        )

        self.assertNotIn("x", adjusted["train"].columns)
        self.assertEqual(adjusted["train"]["x__abs_mid"].iloc[:2].tolist(), [1.0, 1.0])
        self.assertTrue(np.isnan(adjusted["train"]["x__abs_mid"].iloc[2]))
        self.assertEqual(adjusted["oot"]["x__abs_mid"].tolist(), [0.0, 3.0])
        self.assertEqual(lineage["x__abs_mid"]["source_feature"], "x")
        self.assertEqual(lineage["x__abs_mid"]["transformation"], "u_shape_abs")

    def test_manual_breaks_override_combiner_rule(self):
        data = pd.DataFrame(
            {
                "x": [-3.0, -1.0, 0.5, 1.5, 3.0, 4.0],
                "target": [0, 0, 0, 1, 1, 1],
            }
        )
        combiner, _ = STAGE3.run_binning(data, "target")

        STAGE3.apply_manual_breaks(combiner, {"x": [0.0, 2.0]})
        rule = combiner.export()["x"]

        self.assertEqual(rule, [0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
