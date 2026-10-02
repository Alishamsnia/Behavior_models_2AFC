from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from post_analysis.expanded_model_comparison import collect, figures as comparison_figures, stats
from post_analysis.expanded_model_comparison.extended_collect import (
    EVENT_RANGES,
    FOCUSED_TRANSITION_RANGE,
    TABLE_COLUMNS,
    _event_rows,
    _validated_prior_rows,
)
from post_analysis.expanded_model_comparison.extended_figures import (
    FIGURE_IDS,
    _window_weighted,
    _winner_agreement_matrix,
)
from post_analysis.expanded_model_comparison.config import (
    MODEL_FAMILY_ORDER,
    MODEL_ORDER,
    MODEL_SPECS,
    validate_mouse_id,
)


class ExpandedModelComparisonTests(unittest.TestCase):
    def test_extended_registry_covers_figures_07_through_24(self):
        self.assertEqual(FIGURE_IDS, tuple(f"fig{number:02d}" for number in range(7, 25)))
        self.assertEqual(tuple(EVENT_RANGES["short_to_long"]), tuple(range(-15, 21)))
        self.assertEqual(tuple(FOCUSED_TRANSITION_RANGE), tuple(range(-5, 6)))
        self.assertEqual(tuple(EVENT_RANGES["rare_short"]), tuple(range(-2, 3)))

    def test_event_cohorts_respect_block_boundaries_and_have_distinct_spans(self):
        rows = []
        trial_index = 0
        for block_index, (block, length) in enumerate((("S", 20), ("L", 25), ("S", 20))):
            for run_length in range(1, length + 1):
                category_long = int((run_length % 5) != 0) if block == "L" else int((run_length % 5) == 0)
                rows.append({
                    "session_id": "s1", "trial_index": trial_index, "block_index": block_index,
                    "true_run_length": run_length, "true_block": block,
                    "category_long": category_long, "choice_long": category_long,
                    "model_probability_long": 0.8 if category_long else 0.2,
                    "per_trial_negative_log_likelihood": -np.log(0.8),
                    "model_evaluated": True, "split_role": "test",
                })
                trial_index += 1
        events = pd.DataFrame(_event_rows(pd.DataFrame(rows), "model", 1))
        transition = events[events.event.eq("short_to_long")]
        self.assertEqual(set(transition.event_span), {"focus", "wide"})
        focused = transition[transition.event_span.eq("focus")]
        self.assertEqual((focused.relative_position.min(), focused.relative_position.max()), (-5, 5))
        # A cohort is accepted only if every requested position exists and stays
        # inside the two blocks participating in the transition.
        self.assertFalse((events.event.eq("long_to_short") & events.event_span.eq("wide")).any())

    def test_event_grid_collapses_trial_types_with_event_count_weights(self):
        source = pd.DataFrame({
            "model_id": ["m", "m", "m"], "window_id": [1, 1, 2],
            "relative_position": [0, 0, 0], "mean_nll": [0.0, 1.0, 0.5],
            "n_events": [1, 3, 2],
        })
        result = _window_weighted(source, "mean_nll").sort_values("window_id")
        np.testing.assert_allclose(result.mean_nll, [0.75, 0.5])

    def test_prior_rows_exclude_unevaluated_and_validate_probability(self):
        source = pd.DataFrame({"model_evaluated": [True, False], "model_prior_long": [0.25, np.nan]})
        self.assertEqual(_validated_prior_rows(source).model_prior_long.tolist(), [0.25])
        source.loc[0, "model_prior_long"] = 1.1
        with self.assertRaises(ValueError):
            _validated_prior_rows(source)

    def test_tied_winner_agreement_assigns_one_total_window_credit(self):
        wins = pd.DataFrame({
            "window_id": [1, 1], "split_role": ["train", "test"],
            "winning_models": ["a|b", "b|c"],
        })
        matrix = _winner_agreement_matrix(wins, ("a", "b", "c"))
        self.assertAlmostEqual(matrix.sum(), 1.0)
        np.testing.assert_allclose(matrix[[0, 0, 1, 1], [1, 2, 1, 2]], 0.25)

    def test_empty_extended_tables_have_explicit_schemas(self):
        for columns in TABLE_COLUMNS.values():
            self.assertEqual(pd.DataFrame([], columns=columns).columns.tolist(), columns)

    def test_registry_has_requested_thirteen_models_including_stimulus_kernel(self):
        self.assertEqual(len(MODEL_SPECS), 13)
        self.assertEqual(len(set(MODEL_ORDER)), 13)
        self.assertIn("m1_stimulus_kernel", MODEL_ORDER)
        self.assertIn("M1_stimulus_kernel_v1", [spec.directory for spec in MODEL_SPECS])
        self.assertEqual(sum(spec.oracle for spec in MODEL_SPECS), 2)
        self.assertIn("M2_dynamic_previous_wsls_bias_v1", [spec.directory for spec in MODEL_SPECS])
        self.assertIn("M2_dynamic_previous_action_bias_v1", [spec.directory for spec in MODEL_SPECS])
        self.assertEqual({spec.family for spec in MODEL_SPECS}, set(MODEL_FAMILY_ORDER))

    def test_complete_matrix_rejects_missing_model_window(self):
        table = pd.DataFrame({
            "window_id": [1, 1, 2],
            "model_id": ["a", "b", "a"],
            "mean_nll": [0.4, 0.5, 0.3],
        })
        matrix = stats.complete_matrix(table, ("a", "b"))
        self.assertEqual(matrix.index.tolist(), [1])

    def test_ranking_is_separate_by_scope_and_lower_is_better(self):
        matrix = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.4, 0.5, 0.6]})
        ranked = stats.ranking_table(matrix, {"a": "A", "b": "B"}, {"a": False, "b": False}, "all")
        self.assertEqual(ranked.loc[0, "model_id"], "a")
        self.assertEqual(ranked["rank"].tolist(), [1, 2])

    def test_pairwise_marks_provisional_and_retains_ties(self):
        matrix = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.1, 0.25, 0.2]})
        table = stats.pairwise_table(matrix, "test", final_all_models=False)
        self.assertEqual(table.loc[0, "inferential_status"], "provisional_descriptive_do_not_interpret")
        self.assertEqual(table.loc[0, "ties"], 1)
        self.assertFalse(bool(table.loc[0, "units_independent"]))

    def test_moving_block_bootstrap_is_deterministic_and_in_bounds(self):
        first = stats.moving_block_indices(7, replicates=10, block_length=3)
        second = stats.moving_block_indices(7, replicates=10, block_length=3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (10, 7))
        self.assertTrue(((first >= 0) & (first < 7)).all())
        within_block_differences = np.diff(first, axis=1)[:, [0, 1, 3, 4]]
        self.assertTrue((within_block_differences == 1).all())

    def test_strict_complete_matrix_rejects_silently_dropped_window(self):
        table = pd.DataFrame({
            "window_id": [1, 1, 2],
            "model_id": ["a", "b", "a"],
            "mean_nll": [0.4, 0.5, 0.3],
        })
        with self.assertRaisesRegex(ValueError, "incomplete windows"):
            stats.strict_complete_matrix(table, ("a", "b"))
        complete_but_short = table[table["window_id"].eq(1)]
        with self.assertRaisesRegex(ValueError, "requires 2 complete windows"):
            stats.strict_complete_matrix(
                complete_but_short, ("a", "b"), expected_window_count=2,
            )

    def test_basic_bootstrap_ci_inverts_asymmetric_tost_tails(self):
        rng = np.random.default_rng(77)
        indices = stats.moving_block_indices(30, replicates=5000, block_length=9)
        for _ in range(133):
            difference = rng.lognormal(-5, 1.2, 30) - rng.uniform(0.003, 0.012)
        result = stats._boundary_bootstrap_tost(difference, indices, 0.005)
        self.assertEqual(
            result["tost_p"] < 0.05,
            result["ci90_strictly_inside_margin"],
        )

    def test_block_bootstrap_tost_and_family_summary_identify_equivalence_set(self):
        rows = []
        models = ("extended_wsls", "m0_sensory", "m1_hmm_true")
        for split_role in ("train", "validation", "test"):
            for window_id in range(1, 21):
                base = 0.45 + 0.01 * np.sin(window_id / 3)
                for model_id, offset in (
                    ("extended_wsls", 0.020),
                    ("m0_sensory", 0.001),
                    ("m1_hmm_true", 0.000),
                ):
                    rows.append({
                        "split_role": split_role,
                        "window_id": window_id,
                        "model_id": model_id,
                        "mean_nll": base + offset,
                    })
        table = stats.equivalence_table(
            pd.DataFrame(rows), models, replicates=499, block_length=3,
        )
        test = table[table["split_role"].eq("test")].set_index("model_id")
        self.assertEqual(test.loc["m1_hmm_true", "reference_model_id"], "m1_hmm_true")
        self.assertTrue(bool(test.loc["m0_sensory", "equivalent_holm"]))
        self.assertFalse(bool(test.loc["extended_wsls", "equivalent_holm"]))
        self.assertGreater(test.loc["extended_wsls", "tost_p_holm_within_split"], 0.05)
        family = stats.family_summary_table(table).set_index("family")
        self.assertEqual(family.loc["M1", "family_best_model_id"], "m1_hmm_true")
        self.assertEqual(family.loc["M2", "eligible_deployable_models"], 0)

    def test_combined_selection_gates_total_fit_and_uses_six_event_conditions(self):
        models = ("m1_hmm_true", "m1_dynamic", "m2_static")
        metric_rows = []
        event_rows = []
        total_offsets = {"m1_hmm_true": 0.0, "m1_dynamic": 0.020, "m2_static": 0.002}
        event_offsets = {"m1_hmm_true": 0.0, "m1_dynamic": 0.002, "m2_static": -0.002}
        for window_id in range(1, 21):
            for model in models:
                metric_rows.append({
                    "model_id": model, "window_id": window_id, "split_role": "test",
                    "mean_nll": 0.45 + total_offsets[model] + window_id / 10000,
                })
                rare_specs = (
                    ("rare_long", "long"), ("rare_short", "short"),
                )
                transition_specs = (
                    ("short_to_long", "short"), ("long_to_short", "short"),
                    ("short_to_long", "long"), ("long_to_short", "long"),
                )
                for event, trial_type in rare_specs:
                    event_rows.append({
                        "model_id": model, "window_id": window_id, "split_role": "test",
                        "event": event, "event_span": "focus", "relative_position": 0,
                        "trial_type": trial_type,
                        "mean_nll": 0.50 + event_offsets[model], "n_events": 4,
                    })
                if window_id < 20:  # One held-out session has no qualifying transitions.
                    for event, trial_type in transition_specs:
                        for relative_position in range(-5, 6):
                            weighting_probe = (
                                0.1
                                if model == "m1_hmm_true"
                                and event == "short_to_long"
                                and trial_type == "short"
                                and relative_position == 5
                                else 0.0
                            )
                            event_rows.append({
                                "model_id": model, "window_id": window_id, "split_role": "test",
                                "event": event, "event_span": "focus",
                                "relative_position": relative_position,
                                "trial_type": trial_type,
                                "mean_nll": (
                                    0.50 + event_offsets[model]
                                    + relative_position / 1000 + weighting_probe
                                ),
                                "n_events": abs(relative_position) + 1,
                            })
        summary, domains = stats.combined_selection_tables(
            pd.DataFrame(metric_rows), pd.DataFrame(event_rows), models,
            replicates=499, block_length=3, expected_total_windows=20,
            expected_event_windows=19,
            expected_event_window_ids=tuple(range(1, 20)),
        )
        indexed = summary.set_index("model_id")
        self.assertFalse(bool(indexed.loc["m1_dynamic", "passes_total_noninferiority"]))
        self.assertTrue(bool(indexed.loc["m1_hmm_true", "passes_total_noninferiority"]))
        self.assertTrue(bool(indexed.loc["m2_static", "passes_total_noninferiority"]))
        self.assertEqual(indexed.loc["m2_static", "event_n_windows"], 19)
        self.assertTrue(bool(indexed.loc["m2_static", "observed_combined_winner"]))
        self.assertTrue(bool(indexed.loc["m1_hmm_true", "event_equivalent_holm"]))
        self.assertTrue(bool(indexed.loc["m2_static", "prevalence_weighted_event_winner"]))
        self.assertEqual(domains["event_domain"].nunique(), 6)
        self.assertEqual(len(domains), len(models) * 6)
        weighted_probe = domains[
            domains["model_id"].eq("m1_hmm_true")
            & domains["event_domain"].eq("short_to_long_short_trials")
        ].iloc[0]
        self.assertAlmostEqual(weighted_probe["mean_nll"], 0.50 + 0.6 / 41)
        self.assertAlmostEqual(
            indexed.loc[list(models), "conditional_probability_event_best"].sum(), 1.0,
        )
        with self.assertRaisesRegex(ValueError, "requires 20 complete event windows"):
            stats.combined_selection_tables(
                pd.DataFrame(metric_rows), pd.DataFrame(event_rows), models,
                replicates=99, block_length=3, expected_total_windows=20,
                expected_event_windows=20,
            )
        wrong_window_rows = pd.DataFrame(event_rows)
        copied_last = wrong_window_rows[
            wrong_window_rows["window_id"].eq(19)
            & wrong_window_rows["event"].isin(("short_to_long", "long_to_short"))
        ].copy()
        copied_last["window_id"] = 20
        wrong_window_rows = wrong_window_rows[~(
            wrong_window_rows["window_id"].eq(1)
            & wrong_window_rows["event"].isin(("short_to_long", "long_to_short"))
        )]
        wrong_window_rows = pd.concat([wrong_window_rows, copied_last], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "prespecified set"):
            stats.combined_selection_tables(
                pd.DataFrame(metric_rows), wrong_window_rows, models,
                replicates=99, block_length=3, expected_total_windows=20,
                expected_event_windows=19,
                expected_event_window_ids=tuple(range(1, 20)),
            )
        incomplete_positions = pd.DataFrame(event_rows)
        incomplete_positions = incomplete_positions[~(
            incomplete_positions["window_id"].eq(1)
            & incomplete_positions["event"].eq("short_to_long")
            & incomplete_positions["relative_position"].eq(-5)
        )]
        with self.assertRaisesRegex(ValueError, "complete positions"):
            stats.combined_selection_tables(
                pd.DataFrame(metric_rows), incomplete_positions, models,
                replicates=99, block_length=3, expected_total_windows=20,
                expected_event_windows=19,
            )

    def test_combined_selection_reports_exact_event_ties_as_co_winners(self):
        models = ("m1_hmm_true", "m2_static")
        metrics = pd.DataFrame([
            {"model_id": model, "window_id": window, "split_role": "test", "mean_nll": 0.45}
            for window in range(1, 4) for model in models
        ])
        rows = []
        for window in range(1, 4):
            for model in models:
                for event, trial_type in (("rare_long", "long"), ("rare_short", "short")):
                    rows.append({
                        "model_id": model, "window_id": window, "split_role": "test",
                        "event": event, "event_span": "focus", "relative_position": 0,
                        "trial_type": trial_type, "mean_nll": 0.5, "n_events": 2,
                    })
                for event in ("short_to_long", "long_to_short"):
                    for trial_type in ("short", "long"):
                        for relative_position in range(-5, 6):
                            rows.append({
                                "model_id": model, "window_id": window, "split_role": "test",
                                "event": event, "event_span": "focus",
                                "relative_position": relative_position,
                                "trial_type": trial_type, "mean_nll": 0.5, "n_events": 1,
                            })
        summary, _ = stats.combined_selection_tables(
            metrics, pd.DataFrame(rows), models, replicates=99, block_length=2,
            expected_total_windows=3, expected_event_windows=3,
        )
        self.assertTrue(summary["observed_combined_winner"].all())
        np.testing.assert_allclose(summary["conditional_probability_event_best"], [0.5, 0.5])

    def test_figure00_pdf_saver_writes_four_pages_and_one_png(self):
        records = []

        class RecordingPdf:
            def __init__(self, path, metadata=None):
                self.path = Path(path)
                self.saved = []
                records.append(self)

            def __enter__(self):
                self.path.write_bytes(b"%PDF-fake")
                return self

            def savefig(self, figure, **kwargs):
                self.saved.append(figure)

            def __exit__(self, exc_type, exc, traceback):
                return False

        with tempfile.TemporaryDirectory() as directory:
            pages = [plt.figure() for _ in range(4)]
            with mock.patch.object(comparison_figures, "PdfPages", RecordingPdf):
                png, pdf = comparison_figures._save_figure00_pages(pages, Path(directory))
            self.assertTrue(png.is_file())
            self.assertTrue(pdf.is_file())
            self.assertEqual(len(records[0].saved), 4)
            self.assertFalse(any(Path(directory).glob("*.pending.*")))

    def test_figure00_cli_dispatch_does_not_call_full_renderer(self):
        arguments = ["figures", "--mouse", "MC08", "--figure", "fig00"]
        with mock.patch("sys.argv", arguments), \
             mock.patch.object(comparison_figures, "run_figure00") as figure00_run, \
             mock.patch.object(comparison_figures, "run") as full_run:
            comparison_figures.main()
        figure00_run.assert_called_once_with("MC08")
        full_run.assert_not_called()

    def test_bootstrap_shares_best_probability_across_exact_ties(self):
        matrix = pd.DataFrame({"a": [0.2, 0.3, 0.4], "b": [0.2, 0.3, 0.4]})
        table = stats.ranking_table(matrix, {"a": "A", "b": "B"}, {"a": False, "b": False}, "all")
        self.assertEqual(table["rank"].tolist(), [1, 1])
        np.testing.assert_allclose(table["bootstrap_probability_best"], [0.5, 0.5])
        np.testing.assert_allclose(table["bootstrap_mean_rank"], [1.5, 1.5])

    def test_bootstrap_ranks_apply_nll_tolerance(self):
        matrix = pd.DataFrame({"a": [0.2, 0.3, 0.4], "b": [0.2000005, 0.3000005, 0.4000005]})
        table = stats.ranking_table(matrix, {"a": "A", "b": "B"}, {"a": False, "b": False}, "all")
        self.assertEqual(table["rank"].tolist(), [1, 1])
        np.testing.assert_allclose(table["bootstrap_mean_rank"], [1.5, 1.5])

    def test_mouse_id_rejects_path_traversal(self):
        self.assertEqual(validate_mouse_id("MC08"), "MC08")
        for value in ("../MC08", "/tmp/x", "MC08/other", ""):
            with self.assertRaises(ValueError):
                validate_mouse_id(value)

    def test_trial_validator_allows_valid_choice_excluded_for_invalid_duration(self):
        base = {
            "source_trial_index": 0,
            "choice_long": 1.0, "category_long": 1, "duration_s": 1.7,
            "model_evaluated": True, "model_probability_long": 0.8,
            "per_trial_negative_log_likelihood": -np.log(0.8),
        }
        rows = [
            base | {"split_role": "train", "session_id": "train", "choice_valid": True, "feedback_valid": True, "source_sha256": "a"},
            base | {"split_role": "validation", "session_id": "validation", "choice_valid": True, "feedback_valid": True, "source_sha256": "b"},
            base | {"split_role": "test", "session_id": "test", "source_trial_index": 0,
                    "choice_valid": True, "feedback_valid": True, "source_sha256": "c"},
            base | {"split_role": "test", "session_id": "test", "source_trial_index": 1,
                    "choice_valid": True, "duration_s": np.nan,
                    "model_evaluated": False, "model_probability_long": np.nan,
                    "per_trial_negative_log_likelihood": np.nan, "feedback_valid": True, "source_sha256": "c"},
        ]
        schedule = {"training_session_ids": ["train"], "validation_session_ids": ["validation"], "test_session_id": "test"}
        counts = pd.DataFrame({"n_timeline_trials": [2, 1, 1], "n_valid_choices": [2, 1, 1]}, index=["test", "train", "validation"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            comparison, summaries = collect._validate_trial_table(path, schedule, None, counts)
            self.assertEqual(len(comparison), 3)
            self.assertEqual(summaries["test"]["valid_trial_count"], 1)

    def test_workflow_mismatch_has_dedicated_fatal_error(self):
        self.assertTrue(issubclass(collect.WorkflowMismatchError, RuntimeError))
        self.assertNotIn(collect.WorkflowMismatchError, (ValueError, OSError))


if __name__ == "__main__":
    unittest.main()
