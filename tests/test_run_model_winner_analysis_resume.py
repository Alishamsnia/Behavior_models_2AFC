"""Tests for the notebook --resume staleness fix in
raw_behavior_by_winner/run_model_winner_analysis.py.

The real module imports nbformat/nbclient (heavy notebook-execution
dependencies not needed to test this control flow), so they are stubbed in
sys.modules before import -- this test only exercises run_model's
skip-vs-rerun decision, never actual notebook execution.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_BEHAVIOR_DIR = REPO_ROOT / "raw_behavior_by_winner"
for path in (str(RAW_BEHAVIOR_DIR),):
    if path not in sys.path:
        sys.path.insert(0, path)

for _name in ("nbformat", "nbclient"):
    if _name not in sys.modules:
        stub = types.ModuleType(_name)
        if _name == "nbclient":
            stub.NotebookClient = object  # never instantiated in these tests
        sys.modules[_name] = stub

import run_model_winner_analysis as runner  # noqa: E402


class RunModelResumeStalenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.results_dir = Path(self._tmp.name) / "M1_hmm_true_v1"
        self.results_dir.mkdir(parents=True)
        self.pooled_csv = Path(self._tmp.name) / "pooled.csv"
        self.pooled_csv.write_text("a,b\n1,2\n", encoding="utf-8")
        self.input_manifest = {
            "status": "ok",
            "pooled_csv": str(self.pooled_csv),
            "pooled_subjects": ["MC06"],
        }
        self._results_root_patch = patch.object(runner, "RESULTS_ROOT", Path(self._tmp.name))
        self._results_root_patch.start()
        self._kernel_patch = patch.object(runner, "_install_local_kernelspec", lambda _dir: None)
        self._kernel_patch.start()

    def tearDown(self) -> None:
        self._kernel_patch.stop()
        self._results_root_patch.stop()
        self._tmp.cleanup()

    def _fake_execute(self, calls: list) -> callable:
        def _execute(notebook_name, results_dir, pooled_path, subjects, timeout=1800):
            calls.append(notebook_name)
            return {"status": "completed", "error": None}
        return _execute

    def test_unchanged_input_is_skipped_on_second_run(self) -> None:
        calls: list = []
        with patch.object(runner, "execute_notebook", self._fake_execute(calls)):
            runner.run_model("M1_hmm_true_v1", self.input_manifest, ["psychometric_analysis.ipynb"], timeout=10)
            runner.run_model("M1_hmm_true_v1", self.input_manifest, ["psychometric_analysis.ipynb"], timeout=10)
        self.assertEqual(calls, ["psychometric_analysis.ipynb"])  # only the first run executed it

    def test_changed_pooled_csv_forces_rerun_despite_completed_status(self) -> None:
        calls: list = []
        with patch.object(runner, "execute_notebook", self._fake_execute(calls)):
            runner.run_model("M1_hmm_true_v1", self.input_manifest, ["psychometric_analysis.ipynb"], timeout=10)
            # Simulate winner assignments / upstream model results changing,
            # which rebuilds the pooled CSV with different content.
            self.pooled_csv.write_text("a,b\n1,3\n", encoding="utf-8")
            runner.run_model("M1_hmm_true_v1", self.input_manifest, ["psychometric_analysis.ipynb"], timeout=10)
        self.assertEqual(calls, ["psychometric_analysis.ipynb", "psychometric_analysis.ipynb"])

    def test_manifest_records_the_input_checksum(self) -> None:
        calls: list = []
        with patch.object(runner, "execute_notebook", self._fake_execute(calls)):
            manifest = runner.run_model(
                "M1_hmm_true_v1", self.input_manifest, ["psychometric_analysis.ipynb"], timeout=10
            )
        recorded = manifest["notebook_runs"]["psychometric_analysis.ipynb"]["input_pooled_csv_sha256"]
        from hashlib import sha256
        self.assertEqual(recorded, sha256(self.pooled_csv.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
