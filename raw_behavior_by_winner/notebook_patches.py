"""Generic, subject-count-agnostic source patches applied to in-memory
notebook cell copies before execution.

This generalizes the technique Behavior/New_version's own
``run_control_analyses.py`` already uses for its fixed MC16/MC17 cohort
(regex-replace a hardcoded subject-row list / color map / expected-subject
set with the actual cohort). Here the replacement is always driven by
whichever subjects are actually present in a given model-winner's pooled
CSV, so adding a new subject later (once it has model results) needs no
change to this file -- the next run just picks it up.

Nothing in this module touches the original notebooks on disk; it only
edits copies built by ``run_model_winner_analysis.py``.
"""

from __future__ import annotations

import re

# The original codebase's own contingency grouping (see
# EXPECTED_REVERSE_SUBJECTS / EXPECTED_NORMAL_SUBJECTS and the
# "MC06-08 Reverse" / "YH24LG/MC16/MC17 Normal" side-panel labels in
# licking_analysis.ipynb). Used only to intersect with whichever subjects
# are actually present -- never to add subjects that aren't there.
KNOWN_REVERSE_CONTINGENCY_SUBJECTS = {"MC06", "MC07", "MC08"}
KNOWN_NORMAL_CONTINGENCY_SUBJECTS = {"YH24LG", "MC16", "MC17"}

# Colorblind-safe qualitative palette (Okabe-Ito), cycled across however
# many subjects are actually present.
_SUBJECT_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
    "#56B4E9", "#F0E442", "#999999",
]

# Absolute-path literals baked into the original notebooks. Every occurrence
# is replaced with this model-folder's pooled CSV / results directory.
OLD_POOLED_PATH_SUFFIX = "New_version/all_subjects_final_data.csv"
OLD_OUTPUT_PREFIX = "/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Figures_behavior/Mega_pdf/"

# (notebook filename -> [(varname, has_leading_all_entry), ...]) for the
# "list of (label, subject_or_None) tuples" pattern used throughout.
TUPLE_ROW_VARS = {
    "performance_pooled_grand_average.ipynb": [("SUBJECT_ROWS", True)],
    "psychometric_analysis.ipynb": [
        ("SUBJECT_PAGE_ORDER", False),
        ("POST_OUTCOME_ROWS", True),
        ("PG_SUBJECT_ROWS", True),
        ("JB_SUBJECT_ROWS", True),
        ("SOM_SUBJECT_ROWS", True),
    ],
    "learning_curve.ipynb": [("WSLS_ROWS", True)],
}

# (notebook filename -> [varname, ...]) for a flat `VAR = ['MC06', ...]` list.
FLAT_LIST_VARS = {
    "performance_pooled_grand_average.ipynb": ["CONTEXT_META_LEARNING_SUBJECTS"],
}


def subject_colors(subjects: list[str]) -> dict[str, str]:
    return {s: _SUBJECT_PALETTE[i % len(_SUBJECT_PALETTE)] for i, s in enumerate(subjects)}


def _find_matching_close(source: str, open_index: int, opening: str, closing: str) -> int:
    """Index just past the bracket matching ``source[open_index]`` (which
    must be ``opening``), by simple depth counting. This codebase's own
    literal lists/dicts/sets never nest strings containing brackets, so
    plain depth counting is safe here.
    """
    depth = 0
    i = open_index
    while i < len(source):
        if source[i] == opening:
            depth += 1
        elif source[i] == closing:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError(f"Unbalanced {opening!r}/{closing!r} starting at index {open_index}")


def _replace_bracketed_assignment(source: str, varname: str, replacement: str) -> str:
    """Replace a `VARNAME = [...]` / `VARNAME = {...}` assignment (through
    its matching closing bracket, single- or multi-line) with
    ``replacement``. No-op if ``varname`` is not assigned in ``source``.
    """
    match = re.search(rf"(?m)^{re.escape(varname)}\s*=\s*([\[{{])", source)
    if match is None:
        return source
    opening = match.group(1)
    closing = "]" if opening == "[" else "}"
    open_index = match.end() - 1
    end = _find_matching_close(source, open_index, opening, closing)
    return source[: match.start()] + replacement + source[end:]


def _tuple_rows_literal(varname: str, subjects: list[str], include_all: bool) -> str:
    lines = [f"{varname} = ["]
    if include_all:
        lines.append('    ("All subjects", None),')
    for subject in subjects:
        lines.append(f"    ({subject!r}, {subject!r}),")
    lines.append("]")
    return "\n".join(lines)


def _flat_list_literal(varname: str, subjects: list[str]) -> str:
    return f"{varname} = {list(subjects)!r}"


def _set_literal(varname: str, subjects: set) -> str:
    if not subjects:
        return f"{varname} = set()"
    return f"{varname} = {{{', '.join(repr(s) for s in sorted(subjects))}}}"


def patch_paths(source: str, pooled_csv: str, results_dir: str) -> str:
    """Redirect every hardcoded pooled-CSV / output-directory literal to
    this model-winner's own pooled CSV and results directory.
    """
    source = re.sub(
        r"/home/ihsan/[^\"'\n]*?" + re.escape(OLD_POOLED_PATH_SUFFIX),
        pooled_csv,
        source,
    )
    source = source.replace(OLD_OUTPUT_PREFIX, f"{results_dir}/")
    return source


def patch_input_calls(source: str) -> str:
    """Route bare ``input(`` calls through the prelude's noninteractive
    stand-in (IPython otherwise routes bare ``input()`` through the kernel
    even when ``builtins.input`` is monkey-patched).
    """
    return re.sub(r"(?<![\w.])input\(", "_winner_input(", source)


def patch_subject_rows(notebook_name: str, source: str, subjects: list[str]) -> str:
    for varname, include_all in TUPLE_ROW_VARS.get(notebook_name, []):
        replacement = _tuple_rows_literal(varname, subjects, include_all)
        source = _replace_bracketed_assignment(source, varname, replacement)
    for varname in FLAT_LIST_VARS.get(notebook_name, []):
        replacement = _flat_list_literal(varname, subjects)
        source = _replace_bracketed_assignment(source, varname, replacement)
    return source


def patch_licking_cohorts(source: str, subjects: list[str]) -> str:
    """Restrict the reverse/normal contingency assertions in
    licking_analysis.ipynb to whichever of those subjects are actually
    present, instead of the original fixed 6-subject expectation. The
    isin()-based grouping lines already tolerate absent subjects and do not
    need patching.
    """
    subjects_set = set(subjects)
    reverse_present = KNOWN_REVERSE_CONTINGENCY_SUBJECTS & subjects_set
    normal_present = KNOWN_NORMAL_CONTINGENCY_SUBJECTS & subjects_set
    source = _replace_bracketed_assignment(
        source, "EXPECTED_REVERSE_SUBJECTS", _set_literal("EXPECTED_REVERSE_SUBJECTS", reverse_present),
    )
    source = _replace_bracketed_assignment(
        source, "EXPECTED_NORMAL_SUBJECTS", _set_literal("EXPECTED_NORMAL_SUBJECTS", normal_present),
    )
    return source


_META_TERTILE_SAFETY_PATCH = """
import pandas as _winner_meta_pd
def _winner_add_session_tertile(summary):
    # meta.add_session_tertile's own pd.qcut(..., 3, labels=[...]) raises
    # outright when a subject has fewer than 3 sessions (qcut then produces
    # fewer than 3 bins but is still handed 3 fixed labels). A model-winner
    # cohort can genuinely have 1-2 sessions for a subject, so degrade to
    # "no tertile" for that subject instead of crashing the whole notebook.
    summary = summary.copy()
    def _safe_tertile(s):
        if s.nunique() < 3:
            return _winner_meta_pd.Series([None] * len(s), index=s.index, dtype="object")
        return _winner_meta_pd.qcut(s, 3, labels=meta.SESSION_TERTILE_LABELS, duplicates="drop")
    summary["session_tertile"] = summary.groupby("subject")["subject_session_idx"].transform(_safe_tertile)
    return summary
meta.add_session_tertile = _winner_add_session_tertile
"""


def patch_meta_learning_import(source: str, subjects: list[str]) -> str:
    """After `import meta_learning_transition_analysis as meta`, override
    its module-level SUBJECT_ROWS/SUBJECT_COLORS to the actual cohort (same
    technique run_control_analyses.py uses, generalized) and monkeypatch
    add_session_tertile to tolerate <3 sessions for a subject instead of
    raising (see _META_TERTILE_SAFETY_PATCH).
    """
    marker = "import meta_learning_transition_analysis as meta"
    if marker not in source:
        return source
    rows = ["('All subjects', None)"] + [f"({s!r}, {s!r})" for s in subjects]
    colors = subject_colors(subjects)
    color_body = ", ".join(f"{s!r}: {c!r}" for s, c in colors.items())
    injection = (
        f"{marker}\n"
        f"meta.SUBJECT_ROWS = [{', '.join(rows)}]\n"
        f"meta.SUBJECT_COLORS = {{{color_body}}}\n"
        f"{_META_TERTILE_SAFETY_PATCH}"
    )
    return source.replace(marker, injection, 1)
