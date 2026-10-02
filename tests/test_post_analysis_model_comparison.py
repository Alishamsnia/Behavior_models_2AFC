"""Focused regression tests for matched model-comparison post-analysis."""

from __future__ import annotations

from contextlib import ExitStack
import json
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from post_analysis.common import MODEL_ORDER, MODEL_SPECS
from post_analysis.model_comparison import figures as mc_figures
from post_analysis.model_comparison import stats as mc_stats
from post_analysis.model_comparison.collect import (
    collect_psychometric_comparison_trials,
    collect_single_session_information_criteria,
    collect_information_criteria,
    collect_sliding_window_global_pooled_metrics,
    collect_sliding_window_trial_level_tables,
    collect_sliding_window_trial_subset_pooled_diamonds,
    common_metric_window_ids,
    common_session_ids,
    common_trial_subset_window_ids,
    compute_information_criteria_paired_tests,
    compute_sliding_window_event_aligned_paired_tests,
    compute_sliding_window_event_aligned_trial0_paired_tests,
    event_aligned_unit_scalar,
    summarize_single_session_information_criteria,
    BLOCK_TRANSITION_PERFORMANCE_RELATIVE_RANGE,
    BLOCK_TRANSITION_RELATIVE_RANGE,
    RARE_TRIAL_RELATIVE_RANGE,
    _aggregate_event_aligned_nll,
    _block_transition_performance_rows_by_split,
    _detect_block_transition_events,
    _detect_rare_trial_events,
    _trial_subset_rows,
)
from post_analysis.model_comparison.report import _append_adjacent_results


def _small_trial_predictions_df() -> pd.DataFrame:
    """8-trial synthetic trial_predictions.csv-shaped frame: one neutral
    block (2 trials), one short_block (3 trials: common_short x2, rare_long
    x1), one long_block (3 trials: common_long x2, rare_short x1) -- each
    trial_subset resolves to a distinct, hand-computable mean_nll/n_trials.
    """
    return pd.DataFrame({
        "session_id": "s1",
        "session_date": "2026-01-01",
        "trial_index": range(1, 9),
        "block_index": [1, 1, 2, 2, 2, 3, 3, 3],
        "true_run_length": [1, 2, 1, 2, 3, 1, 2, 3],
        "split_role": "test",
        "per_trial_negative_log_likelihood": [float(i) for i in range(1, 9)],
        "model_evaluated": True,
        "category_long": [0, 1, 0, 0, 1, 1, 1, 0],
        "true_block": ["N", "N", "S", "S", "S", "L", "L", "L"],
        "duration_s": [0.5, 1.5, 0.6, 0.7, 1.6, 1.7, 1.8, 0.8],
        "choice_long": [0, 1, 0, 0, 1, 1, 1, 0],
        "model_probability_long": [0.1, 0.9, 0.2, 0.3, 0.8, 0.7, 0.85, 0.25],
        "mouse_correct": [1, 1, 1, 1, 0, 0, 1, 1],
        "model_probability_correct": [0.9, 0.9, 0.8, 0.7, 0.2, 0.3, 0.85, 0.75],
    })


def _metric_rows(unit_col: str, units: list, roles: list[str]) -> pd.DataFrame:
    rows = []
    for role in roles:
        for unit_index, unit in enumerate(units):
            for model_index, model_id in enumerate(MODEL_ORDER):
                rows.append({
                    unit_col: unit,
                    "model_id": model_id,
                    "analysis_role": role,
                    "split_role": role,
                    "session_id": "ALL_SESSIONS" if unit_col == "window_id" else unit,
                    "mean_nll": 0.5 + 0.01 * unit_index + 0.001 * model_index,
                })
    return pd.DataFrame(rows)


class ModelComparisonPostAnalysisTests(unittest.TestCase):
    def test_dynamic_block_length_models_are_registered_but_not_in_main_comparison(self) -> None:
        """The two dynamic-block-length variants should be first-class,
        look-up-able ModelSpec entries (so any code CAN reference them by
        model_id) without being folded into MODEL_ORDER's nested M0-M4
        comparison family -- that requires a modeling decision (how nesting
        and delta-IC work across two branches), not just registration."""
        for model_id, dir_name, extra_param_count in (
            ("M1_dynamic", "M1_dynamic_block_length_v1", 2),  # +ah, +bh
            ("M2_dynamic", "M2_dynamic_block_length_v1", 3),  # +ah, +delta0, +bh
        ):
            self.assertIn(model_id, MODEL_SPECS)
            spec = MODEL_SPECS[model_id]
            self.assertEqual(spec.dir_name, dir_name)
            self.assertEqual(len(spec.param_names), 5 + extra_param_count)
            self.assertNotIn(model_id, MODEL_ORDER)

    def test_all_figure_registry_entries_dispatch_and_read_declared_tables(self) -> None:
        renderer_names = (
            "fig_global_and_detailed_nll",
            "fig_window_nll_trajectory",
            "fig_single_session_heldout_nll",
            "fig_single_session_win_counts",
            "fig_shared_parameter_trajectories",
            "fig_sliding_window_win_counts",
            "fig_win_counts_combined",
            "fig_chronological_winner_timeline_test",
            "fig_sliding_window_model_specific_parameters",
            "fig_single_session_availability",
            "fig_single_session_shared_parameter_trajectories",
            "fig_single_session_model_specific_parameters",
            "fig_chronological_winner_timeline_train",
            "fig_train_vs_test_winner_agreement",
            "fig_information_criteria",
            "fig_sliding_window_event_aligned_nll",
            "fig_single_session_event_aligned_nll",
            "fig_psychometric_comparison",
            "fig_block_transition_performance_by_trial_type",
            "fig_prior_trajectory_samples",
            "fig_sliding_window_parameter_mean_sd",
            "fig_m4_hazard_curves_across_windows",
            "fig_rare_trial_performance_by_type",
        )
        expected_tables = {
            "sliding_window_trial_subset_metrics.csv",
            "sliding_window_trial_subset_pooled_diamonds.csv",
            "sliding_window_trial_subset_paired_pairwise.csv",
            "sliding_window_split_session_metrics.csv",
            "single_session_trial_subset_metrics.csv",
            "single_session_trial_subset_paired_pairwise.csv",
            "single_session_win_counts_summary.csv",
            "sliding_window_parameters.csv",
            "sliding_window_win_counts_summary.csv",
            "chronological_winner_timeline.csv",
            "single_session_availability.csv",
            "single_session_parameters.csv",
            "chronological_winner_timeline_train.csv",
            "chronological_train_vs_test_winner_agreement_summary.csv",
            "information_criteria_sliding_window.csv",
            "information_criteria_single_session.csv",
            "information_criteria_paired_pairwise.csv",
            "sliding_window_event_aligned_nll.csv",
            "sliding_window_event_aligned_paired_pairwise.csv",
            "sliding_window_event_aligned_trial0_paired_pairwise.csv",
            "single_session_event_aligned_nll.csv",
            "single_session_event_aligned_paired_pairwise.csv",
            "psychometric_comparison_summary.csv",
            "sliding_window_block_transition_performance.csv",
            "sliding_window_prior_trajectory_samples.csv",
            "sliding_window_rare_trial_performance.csv",
        }
        expected_calls = {
            "fig_global_and_detailed_nll": (
                "sliding_window_trial_subset_metrics.csv",
                "sliding_window_trial_subset_pooled_diamonds.csv",
                "sliding_window_trial_subset_paired_pairwise.csv",
                "mouse", "output",
            ),
            "fig_window_nll_trajectory": (
                "sliding_window_split_session_metrics.csv",
                "sliding_window_trial_subset_metrics.csv", "output",
            ),
            "fig_single_session_heldout_nll": (
                "single_session_trial_subset_metrics.csv",
                "single_session_trial_subset_paired_pairwise.csv",
                "mouse", "output",
            ),
            "fig_single_session_win_counts": (
                "single_session_win_counts_summary.csv", "output",
            ),
            "fig_shared_parameter_trajectories": (
                "sliding_window_parameters.csv", "output",
            ),
            "fig_sliding_window_win_counts": (
                "sliding_window_win_counts_summary.csv", "output",
            ),
            "fig_win_counts_combined": (
                "single_session_win_counts_summary.csv",
                "sliding_window_win_counts_summary.csv", "output",
            ),
            "fig_chronological_winner_timeline_test": (
                "chronological_winner_timeline.csv", "output",
            ),
            "fig_sliding_window_model_specific_parameters": (
                "sliding_window_parameters.csv", "mouse", "output",
            ),
            "fig_single_session_availability": (
                "single_session_availability.csv", "output",
            ),
            "fig_single_session_shared_parameter_trajectories": (
                "single_session_parameters.csv", "output",
            ),
            "fig_single_session_model_specific_parameters": (
                "single_session_parameters.csv", "output",
            ),
            "fig_chronological_winner_timeline_train": (
                "chronological_winner_timeline_train.csv", "output",
            ),
            "fig_train_vs_test_winner_agreement": (
                "chronological_train_vs_test_winner_agreement_summary.csv", "output",
            ),
            "fig_information_criteria": (
                "information_criteria_sliding_window.csv",
                "information_criteria_single_session.csv",
                "information_criteria_paired_pairwise.csv",
                "mouse", "output",
            ),
            "fig_sliding_window_event_aligned_nll": (
                "sliding_window_event_aligned_nll.csv",
                "sliding_window_event_aligned_paired_pairwise.csv",
                "sliding_window_event_aligned_trial0_paired_pairwise.csv",
                "mouse", "output",
            ),
            "fig_single_session_event_aligned_nll": (
                "single_session_event_aligned_nll.csv",
                "single_session_event_aligned_paired_pairwise.csv",
                "mouse", "output",
            ),
            "fig_psychometric_comparison": (
                "psychometric_comparison_summary.csv", "mouse", "output",
            ),
            "fig_block_transition_performance_by_trial_type": (
                "sliding_window_block_transition_performance.csv", "mouse", "output",
            ),
            "fig_prior_trajectory_samples": (
                "sliding_window_prior_trajectory_samples.csv", "mouse", "output",
            ),
            "fig_sliding_window_parameter_mean_sd": (
                "sliding_window_parameters.csv", "mouse", "output",
            ),
            "fig_m4_hazard_curves_across_windows": (
                "sliding_window_parameters.csv", "mouse", "output",
            ),
            "fig_rare_trial_performance_by_type": (
                "sliding_window_rare_trial_performance.csv", "mouse", "output",
            ),
        }
        frames = {
            filename: pd.DataFrame({"source": [filename]})
            for filename in expected_tables
        }
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            tables_dir, figures_dir = root / "tables", root / "figures"
            stack.enter_context(patch.object(
                mc_figures.common, "ensure_output_dirs",
                return_value=(tables_dir, figures_dir),
            ))
            read_csv = stack.enter_context(
                patch.object(
                    mc_figures.pd, "read_csv",
                    side_effect=lambda path: frames[path.name],
                ),
            )
            renderers = {
                name: stack.enter_context(patch.object(mc_figures, name))
                for name in renderer_names
            }
            selected = mc_figures.run("synthetic")

        self.assertEqual(selected, tuple(f"fig{number:02d}" for number in range(1, 24)))
        self.assertEqual({call.args[0].name for call in read_csv.call_args_list}, expected_tables)
        self.assertEqual(read_csv.call_count, len(expected_tables))
        for renderer_name, expected_args in expected_calls.items():
            resolved_args = tuple(
                frames[value] if value in frames else
                "synthetic" if value == "mouse" else figures_dir
                for value in expected_args
            )
            renderers[renderer_name].assert_called_once_with(*resolved_args)

    def test_figure_selectors_normalize_deduplicate_and_sort(self) -> None:
        expected_all = tuple(f"fig{number:02d}" for number in range(1, 24))
        self.assertEqual(mc_figures.normalize_figure_selectors(None), expected_all)
        self.assertEqual(
            mc_figures.normalize_figure_selectors(["17", "fig03", 3, "03", "fig1"]),
            ("fig01", "fig03", "fig17"),
        )
        self.assertEqual(mc_figures.normalize_figure_selectors("all"), expected_all)
        self.assertEqual(len(mc_figures.list_figures()), 23)

    def test_invalid_figure_selection_fails_before_io(self) -> None:
        for selectors in ([], ["fig00"], ["fig24"], ["bad"], [True]):
            with self.subTest(selectors=selectors):
                with (
                    patch.object(mc_figures.common, "ensure_output_dirs") as ensure_dirs,
                    patch.object(mc_figures.pd, "read_csv") as read_csv,
                ):
                    with self.assertRaises(ValueError):
                        mc_figures.run("synthetic", selectors)
                ensure_dirs.assert_not_called()
                read_csv.assert_not_called()

    def test_selective_figure_run_reads_only_its_required_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables_dir, figures_dir = root / "tables", root / "figures"
            frame = pd.DataFrame({"value": [1.0]})
            with (
                patch.object(
                    mc_figures.common, "ensure_output_dirs",
                    return_value=(tables_dir, figures_dir),
                ),
                patch.object(mc_figures.pd, "read_csv", return_value=frame) as read_csv,
                patch.object(mc_figures, "fig_window_nll_trajectory") as renderer,
            ):
                selected = mc_figures.run("synthetic", ["fig02", "2"])
        self.assertEqual(selected, ("fig02",))
        self.assertEqual(
            [call.args[0] for call in read_csv.call_args_list],
            [
                tables_dir / "sliding_window_split_session_metrics.csv",
                tables_dir / "sliding_window_trial_subset_metrics.csv",
            ],
        )
        renderer.assert_called_once_with(frame, frame, figures_dir)

    def test_window_nll_trajectory_keeps_centers_and_joins_trial_sem(self) -> None:
        centers = pd.DataFrame({
            "model_id": ["M0", "M0"],
            "window_id": [11, 10],
            "split_role": ["test", "test"],
            "session_id": ["ALL_SESSIONS", "ALL_SESSIONS"],
            "test_rank": [2, 1],
            "mean_nll": [0.42, 0.37],
        })
        uncertainty = pd.DataFrame({
            "model_id": ["M0", "M0"],
            "window_id": [10, 11],
            "split_role": ["test", "test"],
            "trial_subset": ["all", "all"],
            "mean_nll": [0.37, 0.42],
            "nll_sem": [0.01, 0.02],
            "n_trials": [100, 120],
        })
        result = mc_figures._window_nll_trajectory_data(
            centers, uncertainty, "test", "M0",
        )
        self.assertEqual(result["test_rank"].tolist(), [1, 2])
        self.assertEqual(result["mean_nll"].tolist(), [0.37, 0.42])
        self.assertEqual(result["nll_sem"].tolist(), [0.01, 0.02])

    def test_window_nll_trajectory_test_title_defines_sem_unit(self) -> None:
        centers = pd.DataFrame({
            "model_id": ["M0"] * 3,
            "window_id": [1] * 3,
            "split_role": ["train", "validation", "test"],
            "session_id": ["ALL_SESSIONS"] * 3,
            "test_rank": [1] * 3,
            "mean_nll": [0.4, 0.5, 0.6],
        })
        uncertainty = pd.DataFrame({
            "model_id": ["M0"] * 3,
            "window_id": [1] * 3,
            "split_role": ["train", "validation", "test"],
            "trial_subset": ["all"] * 3,
            "mean_nll": [0.4, 0.5, 0.6],
            "nll_sem": [0.01, 0.02, 0.03],
            "n_trials": [100, 50, 25],
        })
        with patch.object(mc_figures, "_save") as save:
            mc_figures.fig_window_nll_trajectory(centers, uncertainty, Path("figures"))
        figure = save.call_args.args[0]
        try:
            self.assertIn("SEM across individual-trial NLL values", figure.axes[2].get_title())
            self.assertIn("held-out session", figure.axes[2].get_title())
            for axis in figure.axes:
                self.assertTrue(any(container.has_yerr for container in axis.containers))
        finally:
            mc_figures.plt.close(figure)

    def test_window_nll_trajectory_rejects_missing_or_mismatched_sem_rows(self) -> None:
        centers = pd.DataFrame({
            "model_id": ["M0"], "window_id": [1], "split_role": ["test"],
            "session_id": ["ALL_SESSIONS"], "test_rank": [1], "mean_nll": [0.4],
        })
        complete = pd.DataFrame({
            "model_id": ["M0"], "window_id": [1], "split_role": ["test"],
            "trial_subset": ["all"], "mean_nll": [0.4], "nll_sem": [0.02],
            "n_trials": [25],
        })
        with self.assertRaisesRegex(ValueError, "Missing trial-level SEM rows"):
            mc_figures._window_nll_trajectory_data(
                centers, complete.iloc[0:0], "test", "M0",
            )
        with self.assertRaisesRegex(ValueError, "mean NLL disagree"):
            mc_figures._window_nll_trajectory_data(
                centers, complete.assign(mean_nll=0.5), "test", "M0",
            )
        with self.assertRaisesRegex(ValueError, "uncertainty column"):
            mc_figures._window_nll_trajectory_data(
                centers, complete.drop(columns="nll_sem"), "test", "M0",
            )
        for malformed in (
            complete.assign(n_trials=np.inf),
            complete.assign(n_trials=2.5),
            complete.assign(nll_sem=-0.01),
        ):
            with self.subTest(values=malformed[["n_trials", "nll_sem"]].iloc[0].to_dict()):
                with self.assertRaisesRegex(ValueError, "Invalid trial-level SEM"):
                    mc_figures._window_nll_trajectory_data(
                        centers, malformed, "test", "M0",
                    )
        singleton = mc_figures._window_nll_trajectory_data(
            centers, complete.assign(n_trials=1, nll_sem=np.nan), "test", "M0",
        )
        self.assertTrue(np.isnan(singleton.loc[0, "nll_sem"]))

    def test_finite_nll_filter_is_opt_in_for_sliding_window_sem(self) -> None:
        trials = _small_trial_predictions_df().iloc[:3].copy()
        trials["per_trial_negative_log_likelihood"] = [1.0, 3.0, np.inf]
        finite = _trial_subset_rows(
            trials, ("test",), include_finite_nll_sem=True,
        )[0]
        legacy = _trial_subset_rows(trials, ("test",))[0]
        self.assertEqual(finite["n_trials"], 2)
        self.assertEqual(finite["mean_nll"], 2.0)
        self.assertEqual(finite["nll_sem"], 1.0)
        self.assertEqual(legacy["n_trials"], 3)
        self.assertTrue(np.isinf(legacy["mean_nll"]))
        self.assertNotIn("nll_sem", legacy)

    def test_multi_figure_run_caches_shared_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables_dir, figures_dir = root / "tables", root / "figures"
            frame = pd.DataFrame({"value": [1.0]})
            with (
                patch.object(
                    mc_figures.common, "ensure_output_dirs",
                    return_value=(tables_dir, figures_dir),
                ),
                patch.object(mc_figures.pd, "read_csv", return_value=frame) as read_csv,
                patch.object(mc_figures, "fig_shared_parameter_trajectories") as shared_renderer,
                patch.object(
                    mc_figures, "fig_sliding_window_model_specific_parameters",
                ) as specific_renderer,
                patch.object(
                    mc_figures, "fig_sliding_window_parameter_mean_sd",
                ) as summary_renderer,
            ):
                selected = mc_figures.run("synthetic", ["fig21", "fig09", "fig05"])
        self.assertEqual(selected, ("fig05", "fig09", "fig21"))
        read_csv.assert_called_once_with(tables_dir / "sliding_window_parameters.csv")
        shared_renderer.assert_called_once_with(frame, figures_dir)
        specific_renderer.assert_called_once_with(frame, "synthetic", figures_dir)
        summary_renderer.assert_called_once_with(frame, "synthetic", figures_dir)

    def test_sliding_window_parameter_summary_has_expected_means_sd_and_groups(self) -> None:
        rows = []
        for param_name in mc_figures.PARAMETER_SUMMARY_ORDER:
            model_ids = mc_figures._parameter_summary_models(param_name)
            for model_index, model_id in enumerate(model_ids):
                for window_id, offset in ((1, 1.0), (2, 3.0)):
                    rows.append({
                        "model_id": model_id,
                        "window_id": window_id,
                        "param_name": param_name,
                        "param_value": model_index + offset,
                    })
        parameters = pd.DataFrame(rows)
        summary = mc_figures.summarize_sliding_window_parameters(parameters)

        self.assertEqual(len(summary), 35)  # 5*5 shared + 4+3+2+1 specific
        w_m0 = summary[
            (summary["param_name"] == "w") & (summary["model_id"] == "M0")
        ].iloc[0]
        self.assertEqual(w_m0["mean"], 2.0)
        self.assertAlmostEqual(w_m0["sd"], np.sqrt(2.0))
        self.assertEqual(w_m0["n_windows"], 2)

        without_bh = mc_figures.summarize_sliding_window_parameters(
            parameters[parameters["param_name"] != "bh"],
        )
        bh_m4 = without_bh[
            (without_bh["param_name"] == "bh") & (without_bh["model_id"] == "M4")
        ].iloc[0]
        self.assertEqual(bh_m4["n_windows"], 0)
        self.assertTrue(np.isnan(bh_m4["mean"]))
        self.assertTrue(np.isnan(bh_m4["sd"]))

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            mc_figures.summarize_sliding_window_parameters(
                pd.concat([parameters, parameters.iloc[[0]]], ignore_index=True),
            )

        invalid_cases = (
            (parameters.drop(columns="param_value"), "missing column"),
            (parameters.assign(model_id="MX"), "parameter models"),
            (parameters.assign(param_name="unknown"), "parameters"),
            (
                parameters.iloc[[0]].assign(model_id="M0", param_name="ah"),
                "does not fit",
            ),
            (parameters.assign(window_id=np.nan), "window_id"),
            (parameters.assign(param_value="bad"), "numeric"),
            (
                pd.DataFrame({
                    "model_id": ["M4"], "window_id": [1],
                    "param_name": [np.nan], "param_value": [1.0],
                }),
                "placeholder",
            ),
        )
        for invalid, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    mc_figures.summarize_sliding_window_parameters(invalid)

        finite_only = mc_figures.summarize_sliding_window_parameters(pd.DataFrame({
            "model_id": ["M0", "M0"], "window_id": [1, 2],
            "param_name": ["w", "w"], "param_value": [2.0, np.inf],
        }))
        w_m0 = finite_only[
            (finite_only["param_name"] == "w") & (finite_only["model_id"] == "M0")
        ].iloc[0]
        self.assertEqual(w_m0["n_windows"], 1)
        self.assertEqual(w_m0["mean"], 2.0)
        self.assertTrue(np.isnan(w_m0["sd"]))

    def test_sliding_window_parameter_mean_sd_uses_nine_panel_model_pattern(self) -> None:
        rows = []
        for param_name in mc_figures.PARAMETER_SUMMARY_ORDER:
            for model_index, model_id in enumerate(
                mc_figures._parameter_summary_models(param_name)
            ):
                windows = ((1, 1.0), (2, 3.0))
                if param_name == "bh":
                    windows = ((1, 1.0),)  # n=1: mean dot without an SD whisker
                if param_name == "alpha_theta" and model_id == "M4":
                    windows = ()  # n=0: retain the model tick without a dot
                for window_id, offset in windows:
                    rows.append({
                        "model_id": model_id,
                        "window_id": window_id,
                        "param_name": param_name,
                        "param_value": model_index + offset,
                    })
        with patch.object(mc_figures, "_save") as save:
            mc_figures.fig_sliding_window_parameter_mean_sd(
                pd.DataFrame(rows), "MCXX", Path("figures"),
            )
        figure, _, stem = save.call_args.args
        try:
            self.assertEqual(stem, "fig21_sliding_window_parameter_mean_sd")
            self.assertEqual(len(figure.axes), 9)
            self.assertEqual(
                [len(axis.get_xticks()) for axis in figure.axes],
                [5, 5, 5, 5, 5, 4, 3, 2, 1],
            )
            self.assertEqual(
                [axis.get_title() for axis in figure.axes],
                list(mc_figures.PARAMETER_SUMMARY_ORDER),
            )
            labels = [
                label.get_text() for axis in figure.axes for label in axis.get_xticklabels()
            ]
            self.assertIn("M4\n(n=0)", labels)
            self.assertIn("M4\n(n=1)", labels)
            bh_markers = [line for line in figure.axes[8].lines if line.get_marker() == "o"]
            self.assertEqual(len(bh_markers), 1)
        finally:
            mc_figures.plt.close(figure)

    def test_figure_cli_lists_and_normalizes_repeated_selectors(self) -> None:
        with (
            patch("sys.argv", ["figures", "--list-figures"]),
            patch("builtins.print") as print_output,
            patch.object(mc_figures, "run") as run,
        ):
            mc_figures.main()
        run.assert_not_called()
        self.assertEqual(print_output.call_count, 23)

        with (
            patch(
                "sys.argv",
                ["figures", "--mouse", "MC07", "--figure", "11", "--figure", "fig03"],
            ),
            patch.object(mc_figures, "run") as run,
        ):
            mc_figures.main()
        run.assert_called_once_with("MC07", ("fig03", "fig11"))

    def test_figure_cli_rejects_selector_errors_but_preserves_renderer_errors(self) -> None:
        with (
            patch("sys.argv", ["figures", "--figure", "fig99"]),
            patch.object(mc_figures, "run") as run,
            self.assertRaises(SystemExit) as caught,
        ):
            mc_figures.main()
        self.assertEqual(caught.exception.code, 2)
        run.assert_not_called()

        with (
            patch("sys.argv", ["figures", "--figure", "fig02"]),
            patch.object(mc_figures, "run", side_effect=ValueError("renderer failed")),
            self.assertRaisesRegex(ValueError, "renderer failed"),
        ):
            mc_figures.main()

    def test_save_closes_figure_when_export_fails(self) -> None:
        figure = Mock()
        figure.savefig.side_effect = [None, OSError("PDF export failed")]
        with tempfile.TemporaryDirectory() as directory:
            figures_dir = Path(directory)
            png_path = figures_dir / "figure.png"
            pdf_path = figures_dir / "figure.pdf"
            png_path.write_bytes(b"old png")
            pdf_path.write_bytes(b"old pdf")
            with (
                patch.object(mc_figures.plt, "close") as close,
                self.assertRaisesRegex(OSError, "PDF export failed"),
            ):
                mc_figures._save(figure, figures_dir, "figure")
            self.assertEqual(png_path.read_bytes(), b"old png")
            self.assertEqual(pdf_path.read_bytes(), b"old pdf")
            self.assertEqual(
                sorted(path.name for path in figures_dir.iterdir()),
                ["figure.pdf", "figure.png"],
            )
        close.assert_called_once_with(figure)

    def test_run_closes_new_figure_when_renderer_fails_before_save(self) -> None:
        open_before = set(mc_figures.plt.get_fignums())

        def failing_renderer(*_args) -> None:
            mc_figures.plt.figure()
            raise ValueError("bad plotting input")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(
                    mc_figures.common, "ensure_output_dirs",
                    return_value=(root / "tables", root / "figures"),
                ),
                patch.object(
                    mc_figures, "fig_window_nll_trajectory",
                    side_effect=failing_renderer,
                ),
                patch.object(mc_figures.pd, "read_csv", return_value=pd.DataFrame()),
                self.assertRaisesRegex(ValueError, "bad plotting input"),
            ):
                mc_figures.run("synthetic", "fig02")
        self.assertEqual(set(mc_figures.plt.get_fignums()), open_before)

    def test_adjacent_tests_are_exactly_the_four_nested_pairs(self) -> None:
        matrix = pd.DataFrame({
            model_id: np.arange(1.0, 9.0) + model_index / 10
            for model_index, model_id in enumerate(MODEL_ORDER)
        })
        result = mc_stats.adjacent_pairwise_tests(matrix)
        observed = list(zip(result["model_a"], result["model_b"]))
        self.assertEqual(observed, list(mc_stats.ADJACENT_MODEL_PAIRS))
        self.assertEqual(result["n_pairs"].tolist(), [8] * 4)
        expected_holm = mc_stats.holm_adjust(result["wilcoxon_p"].tolist())
        np.testing.assert_allclose(result["wilcoxon_p_holm"], expected_holm)

    def test_adjacent_tests_require_every_registered_model(self) -> None:
        with self.assertRaises(ValueError):
            mc_stats.adjacent_pairwise_tests(pd.DataFrame({"M0": [1.0], "M1": [2.0]}))

    def test_empty_adjacent_tests_are_unavailable_not_nonsignificant(self) -> None:
        result = mc_stats.adjacent_pairwise_tests(pd.DataFrame(columns=MODEL_ORDER))
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), mc_stats.PAIRWISE_RESULT_COLUMNS)

    def test_common_session_population_requires_both_roles(self) -> None:
        data = _metric_rows(
            "session_id", ["session_a", "session_b"],
            ["fine_tune_prefix", "heldout"],
        )
        missing = data[
            ~(
                data["session_id"].eq("session_b")
                & data["analysis_role"].eq("heldout")
                & data["model_id"].eq("M4")
            )
        ]
        self.assertEqual(common_session_ids(missing), {"session_a"})

    def test_common_window_population_is_complete_in_every_panel(self) -> None:
        data = _metric_rows("window_id", [1, 2], ["train", "validation", "test"])
        missing = data[
            ~(
                data["window_id"].eq(2)
                & data["split_role"].eq("validation")
                & data["model_id"].eq("M3")
            )
        ]
        with patch(
            "post_analysis.model_comparison.collect.common_window_ids",
            return_value={1, 2},
        ):
            self.assertEqual(common_metric_window_ids(missing, "synthetic"), {1})

    def test_duplicate_matched_rows_raise_instead_of_being_averaged(self) -> None:
        data = _metric_rows(
            "session_id", ["session_a"], ["fine_tune_prefix", "heldout"],
        )
        duplicate = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            common_session_ids(duplicate)

    def test_single_session_ic_requires_joint_success_and_same_trial_n(self) -> None:
        rows = []
        for session_id, rank in [("eligible", 1), ("failed", 2), ("unequal_n", 3)]:
            for model_index, model_id in enumerate(MODEL_ORDER):
                rows.append({
                    "model_id": model_id,
                    "session_id": session_id,
                    "chronological_rank": rank,
                    "analysis_role": "fine_tune_prefix",
                    "elpd": -100.0 - model_index,
                    "valid_trial_count": 200 + (1 if session_id == "unequal_n" and model_id == "M2" else 0),
                    "numerically_sound": not (session_id == "failed" and model_id == "M3"),
                })
        table = collect_single_session_information_criteria(pd.DataFrame(rows))
        self.assertEqual(set(table["session_id"]), {"eligible"})
        self.assertEqual(len(table), len(MODEL_ORDER))
        m0 = table[table["model_id"].eq("M0")].iloc[0]
        k = len(MODEL_SPECS["M0"].param_names)
        self.assertAlmostEqual(m0["aic"], 2 * k + 200.0)
        self.assertAlmostEqual(m0["bic"], k * np.log(200) + 200.0)

    def test_empty_single_session_ic_still_has_a_readable_csv_schema(self) -> None:
        columns = [
            "analysis_role", "numerically_sound", "model_id",
            "session_id", "elpd", "valid_trial_count", "chronological_rank",
        ]
        table = collect_single_session_information_criteria(pd.DataFrame(columns=columns))
        self.assertTrue(table.empty)
        self.assertIn("delta_bic", table.columns)
        summary = summarize_single_session_information_criteria(table)
        paired = compute_information_criteria_paired_tests(
            table, "session_id", "single_session",
        )
        self.assertIn("n_sessions_total", summary.columns)
        self.assertIn("analysis_level", paired.columns)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty_ic.csv"
            table.to_csv(path, index=False)
            loaded = pd.read_csv(path)
        self.assertEqual(list(loaded.columns), list(table.columns))

    def test_rolling_ic_drops_units_with_model_specific_trial_counts(self) -> None:
        rows = []
        for window_id in (1, 2):
            for model_id in MODEL_ORDER:
                rows.append({
                    "model_id": model_id,
                    "window_id": window_id,
                    "test_rank": window_id + 10,
                    "split_role": "train",
                    "session_id": "ALL_SESSIONS",
                    "converged": True,
                    "valid_trial_count": 200 + (
                        1 if window_id == 2 and model_id == "M4" else 0
                    ),
                    "elpd": -100.0,
                })
        with patch(
            "post_analysis.model_comparison.collect.common_metric_window_ids",
            return_value={1, 2},
        ):
            table = collect_information_criteria("synthetic", pd.DataFrame(rows))
        self.assertEqual(set(table["window_id"]), {1})
        self.assertEqual(len(table), len(MODEL_ORDER))

    def test_pooled_diamonds_use_the_exact_metric_complete_window_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model_id in MODEL_ORDER:
                global_dir = root / model_id / "global_figures"
                global_dir.mkdir(parents=True)
                rows = []
                for window_id, nll in ((1, 0.2), (2, 0.8)):
                    for split_role in ("train", "validation", "test"):
                        rows.append({
                            "source_run_id": f"window_{window_id:05d}_synthetic",
                            "model_evaluated": True,
                            "split_role": split_role,
                            "choice_long": 1,
                            "model_probability_long": 0.8,
                            "per_trial_negative_log_likelihood": nll,
                            "mouse_correct": 1,
                            "model_probability_correct": 0.8,
                        })
                pd.DataFrame(rows).to_csv(
                    global_dir / "pooled_trial_predictions.csv", index=False,
                )
            with (
                patch(
                    "post_analysis.model_comparison.collect.common.sliding_window_dir",
                    side_effect=lambda model_id, mouse_id: root / model_id,
                ),
                patch(
                    "post_analysis.model_comparison.collect._per_model_window_ids",
                    return_value={model_id: {1, 2} for model_id in MODEL_ORDER},
                ),
                patch(
                    "post_analysis.model_comparison.collect.common_metric_window_ids",
                    return_value={1},
                ),
            ):
                pooled = collect_sliding_window_global_pooled_metrics(
                    "synthetic", pd.DataFrame(),
                )
                m4_path = root / "M4" / "global_figures" / "pooled_trial_predictions.csv"
                m4 = pd.read_csv(m4_path)
                m4 = m4[~m4["source_run_id"].str.startswith("window_00001_")]
                m4.to_csv(m4_path, index=False)
                with self.assertRaises(RuntimeError):
                    collect_sliding_window_global_pooled_metrics(
                        "synthetic", pd.DataFrame(),
                    )
        self.assertEqual(set(pooled["prediction_unit_count"]), {1})
        np.testing.assert_allclose(pooled["mean_nll"], 0.2)

    def test_single_session_ic_requires_explicit_numerically_sound_metadata(self) -> None:
        rows = []
        for model_id in MODEL_ORDER:
            rows.append({
                "model_id": model_id,
                "session_id": "session_a",
                "chronological_rank": 1,
                "analysis_role": "fine_tune_prefix",
                "elpd": -10.0,
                "valid_trial_count": 20,
                "converged": True,
            })
        with self.assertRaises(ValueError):
            collect_single_session_information_criteria(pd.DataFrame(rows))

    def test_report_marks_failed_pair_test_unavailable(self) -> None:
        table = pd.DataFrame([{
            "analysis_role": "heldout", "model_a": "M0", "model_b": "M1",
            "n_pairs": 5, "median_diff_a_minus_b": 0.0,
            "wilcoxon_p_holm": np.nan,
        }])
        lines = []
        _append_adjacent_results(
            lines, table, "analysis_role", [("heldout", "Held-out")],
        )
        self.assertIn("unavailable", "\n".join(lines))

    def test_report_uses_paired_difference_direction_for_significant_result(self) -> None:
        table = pd.DataFrame([{
            "analysis_role": "heldout", "model_a": "M3", "model_b": "M4",
            "n_pairs": 12, "median_diff_a_minus_b": 0.02,
            "wilcoxon_p_holm": 0.01,
        }])
        lines = []
        _append_adjacent_results(
            lines, table, "analysis_role", [("heldout", "Held-out")],
        )
        self.assertIn("M4 lower NLL", "\n".join(lines))

    def test_information_criteria_tests_keep_levels_and_adjacent_pairs(self) -> None:
        rows = []
        for unit in range(7):
            for model_index, model_id in enumerate(MODEL_ORDER):
                rows.append({
                    "window_id": unit,
                    "model_id": model_id,
                    "aic": 10 + unit + model_index * (1 + unit / 20),
                    "bic": 20 + unit + model_index * (2 + unit / 15),
                })
        result = compute_information_criteria_paired_tests(
            pd.DataFrame(rows), "window_id", "sliding_window",
        )
        self.assertEqual(len(result), 8)
        self.assertEqual(set(result["metric"]), {"aic", "bic"})
        self.assertEqual(set(result["analysis_level"]), {"sliding_window"})
        self.assertFalse(result["units_independent"].any())
        for _, group in result.groupby("metric"):
            self.assertEqual(
                list(zip(group["model_a"], group["model_b"])),
                list(mc_stats.ADJACENT_MODEL_PAIRS),
            )


    def test_trial_subset_metrics_read_per_window_trial_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model_id in MODEL_ORDER:
                window_dir = root / model_id / "window_00001_synthetic"
                window_dir.mkdir(parents=True)
                (window_dir / "window_manifest.json").write_text(json.dumps({"window_id": 1}))
                _small_trial_predictions_df().to_csv(window_dir / "trial_predictions.csv", index=False)
            with patch(
                "post_analysis.model_comparison.collect.common.sliding_window_dir",
                side_effect=lambda model_id, mouse_id: root / model_id,
            ):
                subset_table, event_table, block_transition_table, rare_trial_table = collect_sliding_window_trial_level_tables("synthetic")
        m0 = subset_table[
            (subset_table["model_id"] == "M0") & (subset_table["split_role"] == "test")
        ].set_index("trial_subset")
        self.assertAlmostEqual(m0.loc["all", "mean_nll"], 4.5)
        self.assertAlmostEqual(m0.loc["all", "nll_sem"], np.std(range(1, 9), ddof=1) / np.sqrt(8))
        self.assertEqual(m0.loc["all", "n_trials"], 8)
        self.assertAlmostEqual(m0.loc["short_block", "mean_nll"], 4.0)
        self.assertEqual(m0.loc["short_block", "n_trials"], 3)
        self.assertAlmostEqual(m0.loc["long_block", "mean_nll"], 7.0)
        self.assertAlmostEqual(m0.loc["common_short", "mean_nll"], 3.5)
        self.assertAlmostEqual(m0.loc["common_long", "mean_nll"], 6.5)
        self.assertAlmostEqual(m0.loc["rare_short", "mean_nll"], 8.0)
        self.assertTrue(np.isnan(m0.loc["rare_short", "nll_sem"]))
        self.assertEqual(m0.loc["rare_short", "n_trials"], 1)
        self.assertAlmostEqual(m0.loc["rare_long", "mean_nll"], 5.0)
        self.assertEqual(m0.loc["rare_long", "n_trials"], 1)

    def test_trial_subset_metrics_skip_missing_trial_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model_id in MODEL_ORDER:
                window_dir = root / model_id / "window_00001_synthetic"
                window_dir.mkdir(parents=True)
                (window_dir / "window_manifest.json").write_text(json.dumps({"window_id": 1}))
                if model_id != "M4":
                    _small_trial_predictions_df().to_csv(window_dir / "trial_predictions.csv", index=False)
            with patch(
                "post_analysis.model_comparison.collect.common.sliding_window_dir",
                side_effect=lambda model_id, mouse_id: root / model_id,
            ):
                subset_table, event_table, block_transition_table, rare_trial_table = collect_sliding_window_trial_level_tables("synthetic")
        self.assertNotIn("M4", set(subset_table["model_id"]))
        self.assertNotIn("M4", set(event_table["model_id"]))
        self.assertIn("M0", set(subset_table["model_id"]))

    def test_common_trial_subset_ids_require_all_five_models(self) -> None:
        rows = []
        for window_id in (1, 2):
            for model_id in MODEL_ORDER:
                for trial_subset in ("all", "rare_short"):
                    if trial_subset == "rare_short" and window_id == 2 and model_id == "M3":
                        continue  # M3 has zero rare_short trials in window 2.
                    rows.append({
                        "model_id": model_id, "window_id": window_id, "split_role": "test",
                        "trial_subset": trial_subset, "mean_nll": 0.5, "n_trials": 10,
                    })
        table = pd.DataFrame(rows)
        with patch(
            "post_analysis.model_comparison.collect.common_window_ids", return_value={1, 2},
        ):
            all_ids = common_trial_subset_window_ids(table, "synthetic", "test", "all")
            rare_ids = common_trial_subset_window_ids(table, "synthetic", "test", "rare_short")
        self.assertEqual(all_ids, {1, 2})
        self.assertEqual(rare_ids, {1})

    def test_psychometric_comparison_enforces_identical_population_across_sources(self) -> None:
        """collect_psychometric_comparison_trials claims (in its own
        docstring) an IDENTICAL trial population across sliding-window and
        single-session sources. Before the fix, the two sources' trial_index
        intersections were computed independently and never checked against
        each other, so a trial one model's single-session fine-tune happened
        to drop (but its own sliding-window evaluation did not) could leave
        the two sources' populations silently mismatched for that session."""
        trial_columns = [
            "session_id", "session_date", "trial_index", "block_index", "true_run_length",
            "split_role", "per_trial_negative_log_likelihood", "model_evaluated",
            "category_long", "true_block", "duration_s", "choice_long",
            "model_probability_long", "mouse_correct", "model_probability_correct",
        ]

        def _row(trial_index: int, block_index: int, split_role: str) -> dict:
            return {
                "session_id": "s1", "session_date": "2026-01-01",
                "trial_index": trial_index, "block_index": block_index,
                "true_run_length": 1, "split_role": split_role,
                "per_trial_negative_log_likelihood": 1.0, "model_evaluated": True,
                "category_long": 0, "true_block": "S", "duration_s": 0.6,
                "choice_long": 0, "model_probability_long": 0.2,
                "mouse_correct": 1, "model_probability_correct": 0.8,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model_id in MODEL_ORDER:
                # Sliding-window "test" side: every model sees block 1
                # (trials 1-2) and block 2 (trial 3, trimmed away below).
                window_dir = root / "sliding" / model_id / "window_00001_synthetic"
                window_dir.mkdir(parents=True)
                sw_rows = [_row(1, 1, "test"), _row(2, 1, "test"), _row(3, 2, "test")]
                pd.DataFrame(sw_rows, columns=trial_columns).to_csv(
                    window_dir / "trial_predictions.csv", index=False
                )

                # Single-session "train" prefix (block 1 only, matching the
                # sliding-window trim boundary) -- M4 alone is missing
                # trial_index 1, simulating one model's fine-tune dropping a
                # trial its own sliding-window evaluation kept.
                session_dir = root / "single" / model_id / "session_001_synthetic"
                session_dir.mkdir(parents=True)
                ss_rows = [_row(2, 1, "train")] if model_id == "M4" else [
                    _row(1, 1, "train"), _row(2, 1, "train"),
                ]
                pd.DataFrame(ss_rows, columns=trial_columns).to_csv(
                    session_dir / "trial_predictions.csv", index=False
                )
                (session_dir / "fine_tune_fit_result.json").write_text(json.dumps({
                    "convergence_checks": [
                        ["finite_objective", True], ["finite_parameters", True],
                        ["finite_gradients", True], ["small_gradient_norm", True],
                        ["stable_fresh_training_nll", True],
                        ["no_parameter_transform_errors", True],
                        ["no_probability_normalization_errors", True],
                    ],
                }))

            with (
                patch(
                    "post_analysis.model_comparison.collect.common.sliding_window_dir",
                    side_effect=lambda model_id, mouse_id: root / "sliding" / model_id,
                ),
                patch(
                    "post_analysis.model_comparison.collect.common.single_session_dir",
                    side_effect=lambda model_id, mouse_id: root / "single" / model_id,
                ),
            ):
                table = collect_psychometric_comparison_trials("synthetic")

        sw_index = set(table.loc[table["source"] == "sliding_window", "trial_index"])
        ss_index = set(table.loc[table["source"] == "single_session", "trial_index"])
        self.assertEqual(
            sw_index, ss_index,
            "sliding-window and single-session sources must share an identical "
            "trial population, per this function's own docstring",
        )
        # trial_index 1 is exactly the trial M4's single-session fine-tune
        # dropped -- it must not survive in either source once the two
        # sources' populations are correctly intersected with each other.
        self.assertNotIn(1, sw_index)
        self.assertNotIn(1, ss_index)
        self.assertEqual(sw_index, {2})

    def test_pooled_diamond_equals_raw_trial_pool(self) -> None:
        rows = []
        for window_id, mean_nll, n_trials in [(1, 0.4, 100), (2, 0.6, 300)]:
            for model_id in MODEL_ORDER:
                rows.append({
                    "model_id": model_id, "window_id": window_id, "split_role": "test",
                    "trial_subset": "all", "mean_nll": mean_nll, "n_trials": n_trials,
                })
        table = pd.DataFrame(rows)
        with patch(
            "post_analysis.model_comparison.collect.common_window_ids", return_value={1, 2},
        ):
            diamonds = collect_sliding_window_trial_subset_pooled_diamonds(table, "synthetic")
        row = diamonds[
            (diamonds["split_role"] == "test") & (diamonds["trial_subset"] == "all")
            & (diamonds["model_id"] == "M0")
        ].iloc[0]
        self.assertAlmostEqual(row["mean_nll"], (0.4 * 100 + 0.6 * 300) / 400)
        self.assertEqual(row["n_trials"], 400)

    def test_block_transition_events_exclude_post_neutral_and_short_flanks(self) -> None:
        # Blocks: neutral(2) -> short(3, too short for the +/-5 window) ->
        # long(6) -> short(6). neutral->short is post-neutral (excluded);
        # short(3)->long(6) has an insufficient flank (excluded); long(6)->
        # short(6) is a genuine, correctly-anchored long_to_short event.
        block_lengths = [2, 3, 6, 6]
        block_types = ["N", "S", "L", "S"]
        block_index, true_block, true_run_length = [], [], []
        for idx, (length, btype) in enumerate(zip(block_lengths, block_types), start=1):
            block_index += [idx] * length
            true_block += [btype] * length
            true_run_length += list(range(1, length + 1))
        n = sum(block_lengths)
        trials = pd.DataFrame({
            "session_id": "s1", "trial_index": range(1, n + 1), "block_index": block_index,
            "true_run_length": true_run_length, "split_role": "test",
            "per_trial_negative_log_likelihood": [float(i) for i in range(1, n + 1)],
            "model_evaluated": True, "category_long": [0] * n, "true_block": true_block,
        })
        events = _detect_block_transition_events(trials, BLOCK_TRANSITION_RELATIVE_RANGE)
        self.assertEqual(len(events), 1)
        row = events.iloc[0]
        self.assertEqual(row["event_category"], "block_transition_long_to_short")
        self.assertEqual(row["anchor_trial_index"], sum(block_lengths[:3]) + 1)

    def test_figure19_uses_15_trials_before_and_20_after_switch(self) -> None:
        self.assertEqual(
            tuple(BLOCK_TRANSITION_PERFORMANCE_RELATIVE_RANGE),
            tuple(range(-15, 21)),
        )
        self.assertEqual(len(BLOCK_TRANSITION_PERFORMANCE_RELATIVE_RANGE), 36)

        def transition_trials(previous_length: int, new_length: int) -> pd.DataFrame:
            n = previous_length + new_length
            return pd.DataFrame({
                "session_id": ["s1"] * n,
                "trial_index": range(1, n + 1),
                "block_index": [1] * previous_length + [2] * new_length,
                "true_run_length": (
                    list(range(1, previous_length + 1))
                    + list(range(1, new_length + 1))
                ),
                "split_role": ["test"] * n,
                "category_long": [index % 2 for index in range(n)],
                "true_block": ["S"] * previous_length + ["L"] * new_length,
                "mouse_correct": [1.0] * n,
                "model_probability_correct": [0.8] * n,
            })

        exact = pd.DataFrame(
            _block_transition_performance_rows_by_split(transition_trials(15, 21)),
        )
        self.assertEqual(set(exact["split_role"]), {"all", "test"})
        self.assertEqual(exact["relative_position"].min(), -15)
        self.assertEqual(exact["relative_position"].max(), 20)
        self.assertTrue(exact["support_signature"].str.len().eq(64).all())
        self.assertFalse(_block_transition_performance_rows_by_split(transition_trials(14, 21)))
        self.assertFalse(_block_transition_performance_rows_by_split(transition_trials(15, 20)))

    def test_figure19_overlay_matches_models_pointwise_and_deduplicates_mouse(self) -> None:
        rows = []
        for relative_position in (-15, 20):
            for window_id in (1, 2):
                for model_index, model_id in enumerate(MODEL_ORDER):
                    if relative_position == 20 and window_id == 2 and model_id == "M4":
                        continue
                    rows.append({
                        "model_id": model_id,
                        "window_id": window_id,
                        "relative_position": relative_position,
                        "mean_mouse_correct": 0.5 + 0.1 * window_id,
                        "mean_model_probability_correct": 0.4 + 0.01 * model_index,
                        "n_trials": 4,
                        "support_signature": f"support-{window_id}-{relative_position}",
                    })
        panel = pd.DataFrame(rows)
        summary = mc_figures._matched_block_transition_overlay_summary(panel)

        self.assertEqual(set(summary["series_id"]), {"Mouse", *MODEL_ORDER})
        self.assertEqual(len(summary), 2 * (len(MODEL_ORDER) + 1))
        mouse = summary.loc[summary["series_id"] == "Mouse"].set_index("relative_position")
        self.assertEqual(mouse.loc[-15, "n_windows"], 2)
        self.assertEqual(mouse.loc[20, "n_windows"], 1)
        self.assertAlmostEqual(mouse.loc[-15, "mean"], 0.65)
        self.assertAlmostEqual(mouse.loc[20, "mean"], 0.6)
        self.assertTrue(np.isnan(mouse.loc[20, "sem"]))
        self.assertEqual(
            mc_figures._window_count_range_label(
                mouse["n_windows"], "matched",
            ),
            "N=0–2 matched windows/position",
        )

        mismatched = panel.copy()
        mismatched.loc[
            (mismatched["model_id"] == "M4")
            & (mismatched["window_id"] == 1)
            & (mismatched["relative_position"] == -15),
            "mean_mouse_correct",
        ] = 0.99
        with self.assertRaisesRegex(ValueError, "Mouse correctness differs across models"):
            mc_figures._matched_block_transition_overlay_summary(mismatched)

        mismatched_support = panel.copy()
        mismatched_support.loc[
            (mismatched_support["model_id"] == "M4")
            & (mismatched_support["window_id"] == 1)
            & (mismatched_support["relative_position"] == -15),
            "support_signature",
        ] = "different-support"
        with self.assertRaisesRegex(ValueError, "Trial support differs across models"):
            mc_figures._matched_block_transition_overlay_summary(mismatched_support)

        missing_support = panel.copy()
        missing_support["support_signature"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing trial support signature"):
            mc_figures._matched_block_transition_overlay_summary(missing_support)

    def test_figure19_appends_all_model_overlay_as_sixth_page(self) -> None:
        columns = [
            "model_id", "window_id", "split_role", "event_category", "trial_type",
            "relative_position", "mean_mouse_correct",
            "mean_model_probability_correct", "n_trials", "support_signature",
        ]
        empty = pd.DataFrame(columns=columns)
        real_subplots = mc_figures.plt.subplots

        def small_subplots(*args, **kwargs):
            kwargs["figsize"] = (8.0, 6.0)
            return real_subplots(*args, **kwargs)

        with (
            patch.object(mc_figures.plt, "subplots", side_effect=small_subplots),
            patch.object(mc_figures, "_save_pdf_pages") as save_pages,
        ):
            mc_figures.fig_block_transition_performance_by_trial_type(
                empty, "synthetic", Path("unused"),
            )
        pages = save_pages.call_args.args[0]
        self.assertEqual(len(pages), len(MODEL_ORDER) + 1)
        for figure in pages:
            mc_figures.plt.close(figure)

    def test_rare_trial_events_require_same_block_window(self) -> None:
        # One 5-trial long_block: rare_short at position 3 has a full +/-2
        # margin inside the block (kept); rare_short at position 5 (the
        # block's last trial) has no margin after it (excluded).
        n = 5
        trials = pd.DataFrame({
            "session_id": "s1", "trial_index": range(1, n + 1), "block_index": [1] * n,
            "true_run_length": range(1, n + 1), "split_role": "test",
            "per_trial_negative_log_likelihood": [float(i) for i in range(1, n + 1)],
            "model_evaluated": True, "category_long": [1, 1, 0, 1, 0], "true_block": ["L"] * n,
        })
        events = _detect_rare_trial_events(trials, RARE_TRIAL_RELATIVE_RANGE)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["anchor_trial_index"], 3)
        self.assertEqual(events.iloc[0]["event_category"], "rare_short")

    def test_event_detection_rejects_a_gap_inside_the_displayed_window(self) -> None:
        trials = pd.DataFrame({
            "session_id": ["s1"] * 13,
            "trial_index": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14],
            "block_index": [1] * 6 + [2] * 7,
            "true_run_length": [1, 2, 3, 4, 5, 6, 1, 2, 3, 5, 6, 7, 8],
            "true_block": ["S"] * 6 + ["L"] * 7,
            "category_long": [0] * 13,
        })
        # Trial 10 is absent from the evaluated sequence. Although the block
        # bounds contain the requested -5..+5 span around anchor 7, this
        # event must be excluded instead of using different events by position.
        events = _detect_block_transition_events(trials, range(-5, 6))
        self.assertTrue(events.empty)

    def test_event_aligned_aggregation_happens_before_pooling(self) -> None:
        trials = pd.DataFrame({
            "session_id": ["s1"] * 6, "trial_index": [1, 2, 3, 4, 5, 6],
            "per_trial_negative_log_likelihood": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        })
        events = pd.DataFrame([
            {"session_id": "s1", "event_category": "rare_short", "anchor_trial_index": 2},
            {"session_id": "s1", "event_category": "rare_short", "anchor_trial_index": 5},
        ])
        agg = _aggregate_event_aligned_nll(trials, events, range(-1, 2)).set_index("relative_position")
        # One row PER relative position (aggregated across the 2 events), not
        # one row per event -- the pseudoreplication guard.
        self.assertEqual(len(agg), 3)
        self.assertAlmostEqual(agg.loc[-1, "mean_nll"], (1.0 + 10.0) / 2)
        self.assertAlmostEqual(agg.loc[0, "mean_nll"], (2.0 + 20.0) / 2)
        self.assertAlmostEqual(agg.loc[1, "mean_nll"], (3.0 + 30.0) / 2)
        self.assertEqual(agg.loc[0, "n_events"], 2)

    def test_event_aligned_paired_tests_use_one_scalar_per_window(self) -> None:
        rows = []
        for window_id in range(1, 6):
            for model_index, model_id in enumerate(MODEL_ORDER):
                for relative_position in range(-5, 6):
                    rows.append({
                        "model_id": model_id, "window_id": window_id, "split_role": "all",
                        "event_category": "block_transition_short_to_long",
                        "relative_position": relative_position,
                        "mean_nll": 0.5 + 0.01 * model_index + 0.001 * relative_position,
                        "n_events_in_unit": 3, "n_events_at_position": 3,
                    })
        table = pd.DataFrame(rows)
        with patch(
            "post_analysis.model_comparison.collect.common_window_ids",
            return_value=set(range(1, 6)),
        ):
            result = compute_sliding_window_event_aligned_paired_tests(table, "synthetic")
        # One row per adjacent pair (4 total) -- not one per relative
        # position -- confirming the windowed-average-first design.
        self.assertEqual(len(result), 4)
        self.assertEqual(
            list(zip(result["model_a"], result["model_b"])), list(mc_stats.ADJACENT_MODEL_PAIRS),
        )

    def test_event_aligned_trial0_tests_use_only_anchor_and_both_directions(self) -> None:
        rows = []
        categories = (
            "block_transition_short_to_long",
            "block_transition_long_to_short",
        )
        for category_index, event_category in enumerate(categories):
            for window_id in range(1, 6):
                for model_index, model_id in enumerate(MODEL_ORDER):
                    for relative_position in range(-5, 6):
                        rows.append({
                            "model_id": model_id, "window_id": window_id, "split_role": "all",
                            "event_category": event_category,
                            "relative_position": relative_position,
                            "mean_nll": (
                                0.5 + 0.01 * category_index +
                                (0.02 * model_index if relative_position == 0 else 0.0)
                            ),
                            "n_events_in_unit": 3, "n_events_at_position": 3,
                        })
        table = pd.DataFrame(rows)
        with patch(
            "post_analysis.model_comparison.collect.common_window_ids",
            return_value=set(range(1, 6)),
        ):
            result = compute_sliding_window_event_aligned_trial0_paired_tests(
                table, "synthetic",
            )
        self.assertEqual(len(result), 8)
        self.assertEqual(set(result["event_category"]), set(categories))
        self.assertEqual(set(result["n_pairs"]), {5})
        np.testing.assert_allclose(result["median_diff_a_minus_b"], -0.02)

    def test_event_aligned_scalars_reject_incomplete_positions_and_mismatched_counts(self) -> None:
        rows = []
        for model_id in MODEL_ORDER:
            for relative_position in range(-5, 6):
                rows.append({
                    "model_id": model_id, "window_id": 1, "split_role": "all",
                    "event_category": "block_transition_short_to_long",
                    "relative_position": relative_position,
                    "mean_nll": 0.5, "n_events_in_unit": 3,
                    "n_events_at_position": 3,
                })
        complete = pd.DataFrame(rows)
        incomplete = complete[
            ~(
                (complete["model_id"] == "M4")
                & (complete["relative_position"] == 5)
            )
        ]
        with self.assertRaisesRegex(ValueError, "Incomplete event-aligned"):
            event_aligned_unit_scalar(
                incomplete, "window_id", "block_transition_short_to_long",
            )

        nonfinite = complete.copy()
        nonfinite.loc[
            (nonfinite["model_id"] == "M4")
            & (nonfinite["relative_position"] == 5),
            "mean_nll",
        ] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite event-aligned NLL"):
            event_aligned_unit_scalar(
                nonfinite, "window_id", "block_transition_short_to_long",
            )

        duplicate = pd.concat([complete, complete.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate event-aligned"):
            event_aligned_unit_scalar(
                duplicate, "window_id", "block_transition_short_to_long",
            )

        reduced_position_count = complete.copy()
        reduced_position_count.loc[
            (reduced_position_count["model_id"] == "M4")
            & (reduced_position_count["relative_position"] == 5),
            "n_events_at_position",
        ] = 2
        with (
            patch(
                "post_analysis.model_comparison.collect.common_window_ids",
                return_value={1},
            ),
            self.assertRaisesRegex(ValueError, "Not every event contributes"),
        ):
            compute_sliding_window_event_aligned_paired_tests(
                reduced_position_count, "synthetic",
            )

        mismatched = complete.copy()
        mismatched.loc[mismatched["model_id"] == "M4", "n_events_in_unit"] = 2
        mismatched.loc[mismatched["model_id"] == "M4", "n_events_at_position"] = 2
        with (
            patch(
                "post_analysis.model_comparison.collect.common_window_ids",
                return_value={1},
            ),
            self.assertRaisesRegex(ValueError, "Event counts are not identical"),
        ):
            compute_sliding_window_event_aligned_paired_tests(
                mismatched, "synthetic",
            )

    def test_m4_expected_block_length_matches_brute_force_duration_mean(self) -> None:
        """The closed-form Figure 9 mean (ah, bh together) must match a
        brute-force truncated-sum expectation of M4's own hazard law, and
        must differ substantially from the M1-M3 formula 1/sigmoid(ah) --
        which is only the reciprocal hazard at run length 23, not the mean
        of M4's actual (run-length-varying) duration distribution."""
        ah, bh = -2.6326975441877343, 0.10493482454586804  # a real fitted MC06 pair

        def brute_force_mean(ah: float, bh: float, center=23, cap=65, horizon=20000) -> float:
            survival = 1.0
            total = 0.0
            for r in range(1, horizon + 1):
                run_length = min(r, cap)
                hazard = 1.0 / (1.0 + np.exp(-(ah + bh * (run_length - center))))
                pmf = survival * hazard
                total += r * pmf
                survival *= 1.0 - hazard
                if survival < 1e-15:
                    break
            return total + r * survival  # fold negligible remaining mass in at r

        closed_form = mc_figures._m4_expected_block_length(ah, bh)
        self.assertAlmostEqual(closed_form, brute_force_mean(ah, bh), places=3)

        wrong_constant_hazard_formula = 1.0 / (1.0 / (1.0 + np.exp(-ah)))
        self.assertGreater(closed_form / wrong_constant_hazard_formula, 1.3)

    def test_figure9_ah_panel_uses_full_duration_law_for_m4_only(self) -> None:
        sw_params = pd.DataFrame([
            {"test_rank": 1, "model_id": "M1", "param_name": "ah", "param_value": -2.0},
            {"test_rank": 1, "model_id": "M4", "param_name": "ah", "param_value": -2.6326975441877343},
            {"test_rank": 1, "model_id": "M4", "param_name": "bh", "param_value": 0.10493482454586804},
        ])
        curves = mc_figures._derived_quantity_curves("ah", sw_params)
        self.assertEqual(len(curves), 1)
        _label, _ylabel, derived = curves[0]
        m1_value = derived["M1"].loc[1]
        self.assertAlmostEqual(m1_value, 1.0 / (1.0 / (1.0 + np.exp(2.0))), places=9)
        m4_value = derived["M4"].loc[1]
        self.assertAlmostEqual(
            m4_value,
            mc_figures._m4_expected_block_length(-2.6326975441877343, 0.10493482454586804),
            places=9,
        )
        # The old bug used 1/sigmoid(ah) for every model including M4; confirm
        # the M4 curve is no longer that.
        old_buggy_value = 1.0 / (1.0 / (1.0 + np.exp(2.6326975441877343)))
        self.assertGreater(abs(m4_value - old_buggy_value), 5.0)


if __name__ == "__main__":
    unittest.main()
