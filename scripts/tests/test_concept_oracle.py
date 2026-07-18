from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
ORACLE = SCRIPT_DIR / "crosscheck_concept_graph.py"


class ConceptGraphIndependentOracleTests(unittest.TestCase):
    def test_randomized_black_box_oracle_and_negative_controls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ORACLE),
                    "--cases",
                    "3",
                    "--seed",
                    "20260718",
                    "--out",
                    str(report_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["method"]["production_modules_imported"], [])
            self.assertEqual(report["discrepancy_count"], 0)
            self.assertEqual(report["cli_failure_count"], 0)
            coverage = report["generated_coverage"]
            self.assertGreater(coverage["serial_case_count"], 0)
            self.assertGreater(coverage["branch_cases"], 0)
            self.assertGreater(coverage["negative_multiplier_mimic_joint_count"], 0)
            self.assertGreater(coverage["nested_mimic_joint_count"], 0)
            self.assertGreater(coverage["negative_controls_checked"], 0)


if __name__ == "__main__":
    unittest.main()
