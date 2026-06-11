import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACTS = load_script("stage_contracts.py")
STAGE7 = load_script("07_final_report.py")


class PipelineRouteTests(unittest.TestCase):
    def make_config(self, root, stage6_enabled):
        config = root / "00_modeling_config.yaml"
        config.write_text(json.dumps({"enable_stage6_scoring": stage6_enabled}), encoding="utf-8")
        return config

    def test_stage5_can_route_directly_to_stage7(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self.make_config(root, False)
            decisions = root / "05_evaluation_decisions.csv"
            pd.DataFrame([{"decision": "accept"}]).to_csv(decisions, index=False, encoding="utf-8-sig")
            manifest_path = CONTRACTS.write_stage_manifest(
                root, 5, "completed", config, {}, {"05_evaluation_decisions": decisions}, [], 7,
            )
            manifest = CONTRACTS.load_previous_manifest(manifest_path, 5)
            inventory = STAGE7.build_inventory_from_manifest(manifest)
            self.assertEqual(manifest["next_stage"], 7)
            self.assertIn("05_evaluation_decisions.csv", inventory["artifact"].tolist())

    def test_stage6_route_preserves_stage5_decisions_for_stage7(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self.make_config(root, True)
            stage5_decisions = root / "05_evaluation_decisions.csv"
            stage6_decisions = root / "06_scoring_decisions.csv"
            pd.DataFrame([{"decision": "accept"}]).to_csv(stage5_decisions, index=False, encoding="utf-8-sig")
            pd.DataFrame([{"decision": "accept"}]).to_csv(stage6_decisions, index=False, encoding="utf-8-sig")
            stage5_manifest = CONTRACTS.write_stage_manifest(
                root, 5, "completed", config, {}, {"05_evaluation_decisions": stage5_decisions}, [], 6,
            )
            stage6_manifest = CONTRACTS.write_stage_manifest(
                root, 6, "completed", config, {"previous_manifest": stage5_manifest},
                {"06_scoring_decisions": stage6_decisions}, [], 7,
            )
            manifest = CONTRACTS.load_previous_manifest(stage6_manifest, 6)
            self.assertIn("05_evaluation_decisions", manifest["outputs"])
            self.assertIn("06_scoring_decisions", manifest["outputs"])


if __name__ == "__main__":
    unittest.main()
