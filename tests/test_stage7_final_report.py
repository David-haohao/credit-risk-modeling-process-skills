import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "07_final_report.py"
SPEC = importlib.util.spec_from_file_location("stage7_final_report", SCRIPT_PATH)
STAGE7 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE7)


class Stage7FinalReportTests(unittest.TestCase):
    def test_confirmed_conclusions_render_before_stage_details(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            decisions = pd.DataFrame([{
                "decision_id": "D001",
                "assessment": "模型通过评估",
                "decision": "accept",
            }])
            decisions.to_csv(root / "05_evaluation_decisions.csv", index=False, encoding="utf-8-sig")

            confirmed, issues = STAGE7.read_confirmed_decisions(root, stage6_enabled=False)
            report_path = root / "report.html"
            STAGE7.render_html(
                STAGE7.build_inventory(root),
                pd.DataFrame(issues),
                root,
                False,
                confirmed,
                report_path,
            )

            report = report_path.read_text(encoding="utf-8")
            self.assertLess(report.index("最终结论与建议"), report.index("阶段 0"))
            self.assertIn("模型通过评估", report)
            self.assertEqual(issues, [])

    def test_pending_decision_blocks_final_delivery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pd.DataFrame([{
                "decision_id": "D001",
                "decision": "pending",
            }]).to_csv(root / "05_evaluation_decisions.csv", index=False, encoding="utf-8-sig")

            confirmed, issues = STAGE7.read_confirmed_decisions(root, stage6_enabled=False)

            self.assertEqual(confirmed, [])
            self.assertTrue(any(issue["issue_type"] == "unconfirmed_decision" for issue in issues))

    def test_decision_file_can_be_read_from_stage_subdirectory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage_dir = root / "stage05"
            stage_dir.mkdir()
            pd.DataFrame([{
                "decision_id": "D001",
                "decision": "accept",
            }]).to_csv(stage_dir / "05_evaluation_decisions.csv", index=False, encoding="utf-8-sig")

            confirmed, issues = STAGE7.read_confirmed_decisions(root, stage6_enabled=False)

            self.assertEqual(len(confirmed), 1)
            self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
