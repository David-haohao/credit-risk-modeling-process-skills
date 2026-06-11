import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE4 = load_script("04_model_training.py")
STAGE5 = load_script("05_model_evaluation.py")


class DummyModel:
    def predict_proba(self, data):
        score = np.clip(data.iloc[:, 0].to_numpy(dtype=float), 0.01, 0.99)
        return np.column_stack([1 - score, score])


class Stage4PredictionContractTests(unittest.TestCase):
    def test_prediction_frame_preserves_time_column(self):
        data = pd.DataFrame({
            "sample_id": ["a", "b"], "apply_date": ["2024-01-01", "2024-02-01"],
            "target": [0, 1], "x": [0.2, 0.8],
        })
        result = STAGE4.prediction_frame(
            DummyModel(), data, ["x"], "target", "sample_id", None, "apply_date",
        )
        self.assertEqual(
            result.columns.tolist(),
            ["sample_id", "apply_date", "y_true", "y_pred_raw"],
        )

    def test_lightgbm_dependency_error_is_explicit(self):
        if STAGE4.lgb is None:
            with self.assertRaisesRegex(ImportError, "LightGBM"):
                STAGE4.require_lightgbm()


class Stage5EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "sample_id": [f"t{i}" for i in range(10)],
            "time": pd.date_range("2024-01-01", periods=10, freq="D"),
            "y_true": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "y_pred_raw": np.linspace(0.05, 0.95, 10),
        })
        self.oot = pd.DataFrame({
            "sample_id": [f"o{i}" for i in range(8)],
            "time": pd.date_range("2024-03-01", periods=8, freq="15D"),
            "y_true": [0, 0, 0, 1, 0, 1, 1, 1],
            "y_pred_raw": [0.1, 0.2, 0.4, 0.6, 0.3, 0.7, 0.8, 0.9],
        })

    def test_fixed_edges_are_reused_for_oot(self):
        edges = STAGE5.fit_score_edges(self.train["y_pred_raw"], 4)
        fixed = STAGE5.build_bucket_table(self.oot, edges, "OOT", "fixed")
        self.assertEqual(fixed["bucket"].max(), len(edges) - 2)
        self.assertIn("cumulative_lift", fixed.columns)

    def test_period_psi_uses_train_edges(self):
        edges = STAGE5.fit_score_edges(self.train["y_pred_raw"], 4)
        detail, summary = STAGE5.period_psi(self.train, self.oot, edges, "time", "month")
        self.assertGreaterEqual(len(summary), 2)
        self.assertIn("psi", summary.columns)
        self.assertIn("psi_contribution", detail.columns)

    def test_calibration_metrics_include_required_fields(self):
        summary, detail = STAGE5.calibration_metrics(self.oot, "OOT", 4)
        self.assertTrue({"brier", "log_loss", "calibration_intercept", "calibration_slope"}.issubset(summary))
        self.assertIn("mean_pred", detail.columns)

    def test_four_core_pending_decisions_are_created(self):
        decisions = STAGE5.build_core_decisions()
        self.assertEqual(len(decisions), 4)
        self.assertEqual(set(decisions["decision"]), {"pending"})


if __name__ == "__main__":
    unittest.main()
