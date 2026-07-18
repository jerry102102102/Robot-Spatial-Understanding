from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
ORACLE = SCRIPT_DIR / "crosscheck_cross_representation.py"


class CrossRepresentationOracleTests(unittest.TestCase):
    def test_independent_oracle_smoke_covers_common_and_joint_anchor_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "oracle.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ORACLE),
                    "--cases",
                    "2",
                    "--post-anchor-cases",
                    "1",
                    "--poses-per-case",
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
            self.assertEqual(report["coverage"]["urdf_sdf_mjcf_common_case_count"], 2)
            self.assertEqual(report["coverage"]["sdf_mjcf_non_identity_post_motion_case_count"], 1)
            self.assertGreater(report["coverage"]["independent_all_frame_evaluation_count"], 0)
            self.assertLessEqual(report["maximum_matrix_absolute_error"], 1e-11)


if __name__ == "__main__":
    unittest.main()
