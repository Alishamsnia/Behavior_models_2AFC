"""Regression test: the L-BFGS post-step gradient fix (recomputing the
gradient with a fresh forward+backward pass after optimizer.step()) must
not leave `fresh.loss` un-detached when it is later converted to a plain
float for logging/checkpointing. Doing so triggers PyTorch's "Converting a
tensor with requires_grad=True to a scalar" UserWarning on every affected
fitter (observed in a real M1_dynamic_block_length_v1 rolling-window run).

PyTorch's underlying warning is emitted at most once per PROCESS (a
static C++-side flag, not reset by warnings.simplefilter or
catch_warnings), so testing multiple models' fit calls in one process
would let an earlier model's warning mask a later regression. Each model
is therefore run in its own subprocess with -W error::UserWarning: the
subprocess exits nonzero if the warning (now an error) is raised, and 0
otherwise.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"

_SESSION_HELPER = """
import torch
def make_session(cls, n=40, seed=0, **extra):
    g = torch.Generator().manual_seed(seed)
    duration = torch.where(
        torch.rand(n, generator=g) < 0.5,
        0.2 + 0.6 * torch.rand(n, generator=g),
        1.7 + 0.6 * torch.rand(n, generator=g),
    ).to(torch.float64)
    category = (torch.rand(n, generator=g) < 0.5).to(torch.float64)
    choice = (torch.rand(n, generator=g) < 0.5).to(torch.float64)
    feedback = torch.rand(n, generator=g) < 0.9
    return cls(mouse_id="mouse", session_id="sess", duration_s=duration,
        choice_long=choice, category_long=category, feedback_valid=feedback,
        split_role="train", **extra)
"""

_MODEL_SCRIPTS = {
    "M1_dynamic_block_length_v1": """
from m1_dynamic_block_length_v1.fit import fit_m1_dynamic_block_length_single_initialization as fit_fn
from m1_dynamic_block_length_v1.registry import load_m1_dynamic_block_length_model_registry as load_reg
from m1_dynamic_block_length_v1.session import M1DynamicBlockLengthSessionTrials as Session
""",
    "M2_dynamic_block_length_v1": """
from m2_dynamic_block_length_v1.fit import fit_m2_dynamic_block_length_single_initialization as fit_fn
from m2_dynamic_block_length_v1.registry import load_m2_dynamic_block_length_model_registry as load_reg
from m2_dynamic_block_length_v1.session import M2DynamicBlockLengthSessionTrials as Session
""",
    "M3_hmm_delta_v1": """
from m3_hmm_delta_v1.fit import fit_m3_single_initialization as fit_fn
from m3_hmm_delta_v1.registry import load_m3_model_registry as load_reg
from m3_hmm_delta_v1.session import M3SessionTrials as Session
""",
    "M4_hsmm_delta_v1": """
from m4_hsmm_delta_v1.fit import fit_m4_single_initialization as fit_fn
from m4_hsmm_delta_v1.registry import load_m4_model_registry as load_reg
from m4_hsmm_delta_v1.session import M4SessionTrials as Session
""",
    "M0_sensory_v1 (m1_hmm_true_v1, already fixed)": """
from m0_sensory_v1.model import fit_single_initialization_with_strategy as fit_fn
from m0_sensory_v1.m1_registry import load_m1_model_registry as load_reg
from m0_sensory_v1.model.m1_session import M1SessionTrials as Session
""",
}

_DRIVER_TEMPLATE = """
import sys
sys.path.insert(0, {source_root!r})
import warnings
warnings.filterwarnings(
    "error",
    message=r".*Converting a tensor with requires_grad=True to a scalar.*",
    category=UserWarning,
)
from dataclasses import replace
import torch
{session_helper}
{model_import}
from m0_sensory_v1.model.fit import LBFGSSettings

reg = load_reg()
train = [replace(make_session(Session, seed=10), split_role="train")]
val = [replace(make_session(Session, seed=11), split_role="validation")]
result = fit_fn(
    train, val, reg, reg.initializations[0],
    optimizer_strategy="lbfgs", allow_short_convergence_budget=True,
    lbfgs_settings=LBFGSSettings(maximum_steps=5),
)
# A warnings.filterwarnings("error", ...) match would raise UserWarning
# as an exception, which the fitter's own try/except would catch and
# record as exception_message -- so "the process exited 0" alone is not
# enough evidence the repaired line was actually reached; a fit that fails
# for an unrelated reason before ever getting there would exit 0 too.
assert result.exception_message is None, "fit recorded a failure: " + str(result.exception_message)
assert len(result.history) > 0, "fit recorded no steps at all"
print("NO_WARNING_RAISED")
"""


class LBFGSFreshLossDetachWarningTests(unittest.TestCase):
    def _run_subprocess(self, model_import: str) -> subprocess.CompletedProcess:
        script = _DRIVER_TEMPLATE.format(
            source_root=str(SOURCE_ROOT), session_helper=_SESSION_HELPER,
            model_import=model_import,
        )
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=90,
        )

    def test_m1_dynamic_block_length_lbfgs_step_does_not_warn(self) -> None:
        result = self._run_subprocess(_MODEL_SCRIPTS["M1_dynamic_block_length_v1"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NO_WARNING_RAISED", result.stdout)

    def test_m2_dynamic_block_length_lbfgs_step_does_not_warn(self) -> None:
        result = self._run_subprocess(_MODEL_SCRIPTS["M2_dynamic_block_length_v1"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NO_WARNING_RAISED", result.stdout)

    def test_m3_lbfgs_step_does_not_warn(self) -> None:
        result = self._run_subprocess(_MODEL_SCRIPTS["M3_hmm_delta_v1"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NO_WARNING_RAISED", result.stdout)

    def test_m4_lbfgs_step_does_not_warn(self) -> None:
        result = self._run_subprocess(_MODEL_SCRIPTS["M4_hsmm_delta_v1"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NO_WARNING_RAISED", result.stdout)

    def test_m0_lbfgs_step_does_not_warn(self) -> None:
        """M0 already used .detach().cpu() before this fix; guards against
        a future regression removing it."""
        result = self._run_subprocess(
            _MODEL_SCRIPTS["M0_sensory_v1 (m1_hmm_true_v1, already fixed)"]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NO_WARNING_RAISED", result.stdout)


if __name__ == "__main__":
    unittest.main()
