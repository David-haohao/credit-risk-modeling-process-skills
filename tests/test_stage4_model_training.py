import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "04_model_training.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("stage4_model_training", SCRIPT_PATH)
STAGE4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE4)


class Stage4TrainingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        x1 = rng.normal(size=300)
        x2 = rng.normal(size=300)
        y = (x1 + 0.2 * x2 + rng.normal(size=300) > 0).astype(int)
        self.data = pd.DataFrame(
            {
                "sample_id": [f"id_{i}" for i in range(300)],
                "sample_weight": np.ones(300),
                "x1": x1,
                "x2": x2,
                "target": y,
            }
        )

    def test_explicit_feature_list_excludes_reserved_columns(self):
        features = STAGE4.validate_feature_contract(
            self.data, ["x1", "x2"], "target", "sample_id", "sample_weight"
        )
        self.assertEqual(features, ["x1", "x2"])

    def test_stepwise_returns_iteration_history(self):
        selected, history = STAGE4.auditable_stepwise(
            self.data, "target", ["x1", "x2"], direction="forward", criterion="aic"
        )
        self.assertGreaterEqual(len(selected), 1)
        self.assertGreaterEqual(len(history), 1)
        self.assertTrue(
            {
                "iteration", "action", "feature", "features_before",
                "features_after", "criterion_before", "criterion_after",
                "delta_criterion", "decision_reason",
            }.issubset(history.columns)
        )

    def test_soft_limit_creates_pending_decisions_without_auto_drop(self):
        decisions = STAGE4.build_soft_limit_decisions(
            ["x1", "x2"], max_features=1, diagnostic_table=pd.DataFrame()
        )
        self.assertEqual(set(decisions["decision"]), {"pending"})
        self.assertEqual(set(decisions["issue_type"]), {"soft_feature_limit"})

    def test_final_coefficients_include_intercept(self):
        model = LogisticRegression(solver="liblinear").fit(
            self.data[["x1", "x2"]], self.data["target"]
        )
        coefficients = STAGE4.build_final_lr_coefficients(model, ["x1", "x2"])
        self.assertIn("__INTERCEPT__", coefficients["feature"].tolist())
        self.assertIn("odds_ratio", coefficients.columns)

    def test_xgb_core_audit_contract_contains_search_space_and_ks(self):
        self.assertIn("learning_rate", STAGE4.XGB_SEARCH_SPACE)
        ks = STAGE4.calculate_ks(
            pd.Series([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
        )
        self.assertEqual(ks, 1.0)


if __name__ == "__main__":
    unittest.main()
