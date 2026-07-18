from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
ORACLE = SCRIPT_DIR / "crosscheck_constraint_graph.py"


class ConstraintOracleTests(unittest.TestCase):
    def test_independent_loop_and_coordinate_oracle_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ORACLE),
                    "--fourbar-cases",
                    "2",
                    "--coordinate-cases",
                    "2",
                    "--out",
                    str(report_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["failure_count"], 0)
            self.assertEqual(report["coverage"]["fourbar_loop_case_count"], 2)
            self.assertEqual(report["coverage"]["coordinate_coupling_case_count"], 2)
            self.assertEqual(report["coverage"]["invalid_negative_control_count"], 4)
            self.assertEqual(report["coverage"]["local_solver_case_count"], 4)
            self.assertLessEqual(report["maximum_solved_fourbar_closure_error_m"], 2e-8)
            self.assertLessEqual(report["maximum_solved_coordinate_error"], 2e-8)


if __name__ == "__main__":
    unittest.main()
