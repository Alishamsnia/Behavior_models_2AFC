"""Frozen model registry and analysis constants for the expanded comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from post_analysis.common import REPO_ROOT, RESULTS_SLIDING_ROOT

ANALYSIS_NAME = "expanded_model_comparison"
WORKFLOW_DIRECTORY = "full_session_matched_v1"
WORKFLOW_MANIFEST = "full_session_matched"
SPLIT_ROLES = ("train", "validation", "test")
EXPECTED_TRAIN_SESSIONS = 9
EXPECTED_VALIDATION_SESSIONS = 3
TIE_TOLERANCE_NLL = 1e-6
BOOTSTRAP_SEED = 20260914
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_BLOCK_LENGTH = 5
EQUIVALENCE_MARGIN_NLL = 0.005
# Every fitted prediction in a rolling window depends on the nine-session
# training prefix.  Use that full overlap span for the figure-00 paired
# uncertainty analysis, including validation and test comparisons.
EQUIVALENCE_BLOCK_LENGTH = EXPECTED_TRAIN_SESSIONS
EQUIVALENCE_ALPHA = 0.05
MODEL_FAMILY_ORDER = ("WSLS", "M0", "M1", "M2")
# Coarser than MODEL_FAMILY_ORDER: splits M1/M2 into the sub-clusters used to
# assign one shared color per group (models within a group told apart by
# linestyle instead) in the model-comparison figures, matching the visual
# groupings (with gaps) in the figure legends.
MODEL_GROUP_ORDER = ("wsls", "m0", "m1_history", "m1_switch", "m2_switch", "m2_bias")
# Seeds which windows, and which train/validation/test session within each,
# Figure 20 samples for its per-trial sensory/belief logit trajectories --
# fixed so the choice is stable across re-runs rather than changing every
# time collect() executes.
FIGURE20_WINDOW_SAMPLE_SEED = 20260917
FIGURE20_WINDOW_SAMPLE_COUNT = 5


@dataclass(frozen=True)
class ExpandedModelSpec:
    model_id: str
    directory: str
    label: str
    short_label: str
    family: str
    group: str
    parameter_names: tuple[str, ...]
    color: str
    oracle: bool = False

    def result_root(self, mouse_id: str) -> Path:
        return RESULTS_SLIDING_ROOT / self.directory / mouse_id / WORKFLOW_DIRECTORY


# One fixed, maximally-distinct color per MODEL_GROUP_ORDER cluster (Paul
# Tol's "Bright" qualitative palette, chosen for high pairwise contrast and
# colorblind-safety); every model in a group shares its group's color, and
# is told apart from its group-mates by group_style() (linestyle/marker)
# wherever a plot needs to distinguish them. Validated first on Figure 20;
# every figure reads this same MODEL_SPECS.color, so this one change is
# what makes the group palette apply everywhere.
GROUP_COLOR = {
    "wsls": "#AA3377",
    "m0": "#228833",
    "m1_history": "#4477AA",
    "m1_switch": "#CCBB44",
    "m2_switch": "#EE6677",
    "m2_bias": "#66CCEE",
}
GROUP_LINE_STYLES = ("-", "--", ":")
GROUP_MARKER_STYLES = ("o", "s", "^")

MODEL_SPECS = (
    ExpandedModelSpec("pure_wsls", "Pure_WSLS_v1", "wslsSingle_lapse", "wslsSingle_lapse", "WSLS", "wsls", ("ell_short", "ell_long"), GROUP_COLOR["wsls"]),
    ExpandedModelSpec("extended_wsls", "Extended_WSLS_v1", "wslsHistory_lapse", "wslsHistory_lapse", "WSLS", "wsls", ("ell_short", "ell_long", "a"), GROUP_COLOR["wsls"]),
    ExpandedModelSpec("m0_sensory", "M0_sensory_v1", "sensory_lapse", "sensory_lapse", "M0", "m0", ("w", "s0", "ell_short", "ell_long", "kappa"), GROUP_COLOR["m0"]),
    ExpandedModelSpec("m1_action_kernel", "M1_action_kernel_v1", "sensory_actionHistory_lapse", "sensory_actionHistory_lapse", "M1", "m1_history", ("w", "s0", "ell_short", "ell_long", "kappa", "a"), GROUP_COLOR["m1_history"]),
    ExpandedModelSpec("m1_stimulus_kernel", "M1_stimulus_kernel_v1", "sensory_wslsHistory_lapse", "sensory_wslsHistory_lapse", "M1", "m1_history", ("w", "s0", "ell_short", "ell_long", "kappa", "a"), GROUP_COLOR["m1_history"]),
    ExpandedModelSpec("m1_known_switch", "M1_known_switch_v1", "sensory_knownRatio_knownBlockSwitch_lapse", "sensory_knownRatio_knownBlockSwitch_lapse", "M1", "m1_switch", ("w", "s0", "ell_short", "ell_long", "kappa"), GROUP_COLOR["m1_switch"], True),
    ExpandedModelSpec("m1_hmm_true", "M1_hmm_true_v1", "sensory_knownRatio_learnStaticBlockSwitch_lapse", "sensory_knownRatio_learnStaticBlockSwitch_lapse", "M1", "m1_switch", ("w", "s0", "ell_short", "ell_long", "kappa", "ah"), GROUP_COLOR["m1_switch"]),
    ExpandedModelSpec("m1_dynamic", "M1_dynamic_block_length_v1", "sensory_knownRatio_learnDynamicBlockSwitch_lapse", "sensory_knownRatio_learnDynamicBlockSwitch_lapse", "M1", "m1_switch", ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "bh"), GROUP_COLOR["m1_switch"]),
    ExpandedModelSpec("m2_known_switch", "M2_known_switch_v1", "sensory_learnRatio_knownBlockSwitch_lapse", "sensory_learnRatio_knownBlockSwitch_lapse", "M2", "m2_switch", ("w", "s0", "ell_short", "ell_long", "kappa", "delta0"), GROUP_COLOR["m2_switch"], True),
    ExpandedModelSpec("m2_static", "M2_hmm_static_v1", "sensory_learnRatio_learnStaticBlockSwitch_lapse", "sensory_learnRatio_learnStaticBlockSwitch_lapse", "M2", "m2_switch", ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0"), GROUP_COLOR["m2_switch"]),
    ExpandedModelSpec("m2_dynamic", "M2_dynamic_block_length_v1", "sensory_learnRatio_learnDynamicBlockSwitch_lapse", "sensory_learnRatio_learnDynamicBlockSwitch_lapse", "M2", "m2_switch", ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh"), GROUP_COLOR["m2_switch"]),
    ExpandedModelSpec("m2_dynamic_wsls_bias", "M2_dynamic_previous_wsls_bias_v1", "sensory_learnRatio_learnDynamicBlockSwitch_wslsSingle_lapse", "sensory_learnRatio_learnDynamicBlockSwitch_wslsSingle_lapse", "M2", "m2_bias", ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh", "eta_logit"), GROUP_COLOR["m2_bias"]),
    ExpandedModelSpec("m2_dynamic_repetition_bias", "M2_dynamic_previous_action_bias_v1", "sensory_learnRatio_learnDynamicBlockSwitch_actionSingle_lapse", "sensory_learnRatio_learnDynamicBlockSwitch_actionSingle_lapse", "M2", "m2_bias", ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh", "eta_logit"), GROUP_COLOR["m2_bias"]),
)
MODEL_BY_ID = {spec.model_id: spec for spec in MODEL_SPECS}
MODEL_ORDER = tuple(spec.model_id for spec in MODEL_SPECS)

if {spec.family for spec in MODEL_SPECS} != set(MODEL_FAMILY_ORDER):
    raise RuntimeError("Expanded model family registry disagrees with MODEL_FAMILY_ORDER")

if {spec.group for spec in MODEL_SPECS} != set(MODEL_GROUP_ORDER):
    raise RuntimeError("Expanded model group registry disagrees with MODEL_GROUP_ORDER")


def group_style(model_id: str, kind: str = "line") -> str:
    """The within-group-relative linestyle ('line') or marker ('marker')
    for this model -- distinguishes group-mates that now share one color.
    """
    styles = GROUP_LINE_STYLES if kind == "line" else GROUP_MARKER_STYLES
    group = MODEL_BY_ID[model_id].group
    members = [m for m in MODEL_ORDER if MODEL_BY_ID[m].group == group]
    return styles[members.index(model_id) % len(styles)]


def validate_mouse_id(mouse_id: str) -> str:
    value = str(mouse_id)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Unsafe mouse ID: {value!r}")
    return value


def split_registry_path() -> Path:
    return REPO_ROOT / "config" / "splits" / "chronological_70_15_15_v1.csv"
