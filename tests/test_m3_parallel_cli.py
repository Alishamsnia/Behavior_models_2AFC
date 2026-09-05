"""M3-only process-pool command-line controls."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.fit_m3 import _parse_m3_args  # noqa: E402


class M3ParallelCliTests(unittest.TestCase):
    def test_worker_controls_are_exposed_and_parsed(self):
        argv = [
            "fit_m3.py", "--raw-data-root", "/tmp/raw", "--output-dir", "/tmp/out",
            "--parallel-workers", "4", "--cpu-threads-per-worker", "2",
        ]
        with patch.object(sys, "argv", argv):
            args = _parse_m3_args()
        self.assertEqual(args.parallel_workers, 4)
        self.assertEqual(args.cpu_threads_per_worker, 2)

    def test_worker_controls_reject_nonpositive_values(self):
        argv = [
            "fit_m3.py", "--raw-data-root", "/tmp/raw", "--output-dir", "/tmp/out",
            "--parallel-workers", "0",
        ]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            _parse_m3_args()


if __name__ == "__main__":
    unittest.main()
