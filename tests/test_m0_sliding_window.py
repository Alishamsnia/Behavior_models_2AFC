"""Contracts for the isolated M0 sliding-window experiment."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from m0_sensory_v1.registry import load_model_registry  # noqa: E402
from m0_sliding_window_v1 import (  # noqa: E402
    adaptive_psychometric_summary,
    aggregate_session_initializations,
    build_sliding_windows,
    clamp_fine_tune_initialization,
    match_window_winners_to_test_sessions,
    partition_final_two_blocks,
    partition_session_blocks,
    pool_prediction_tables,
)


def _assignments(count: int = 16) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mouse_id": ["MCXX"] * count,
            "session_id": [f"session_{rank:02d}" for rank in range(1, count + 1)],
            "chronological_rank": range(1, count + 1),
        }
    )


def _session_frame() -> pd.DataFrame:
    blocks = np.repeat(np.arange(1, 7), [3, 2, 4, 3, 2, 5])
    labels = {1: "N", 2: "S", 3: "L", 4: "S", 5: "L", 6: "S"}
    return pd.DataFrame(
        {
            "session_id": ["session"] * len(blocks),
            "trial_index": np.arange(1, len(blocks) + 1),
            "block_index": blocks,
            "true_block": [labels[int(value)] for value in blocks],
            "choice_long": np.resize([0.0, 1.0], len(blocks)),
        }
    )


class SlidingWindowContracts(unittest.TestCase):
    def test_boundary_rounded_winner_is_moved_inside_fine_tune_domain(self) -> None:
        names = ("w", "s0", "ell_short", "ell_long", "kappa")
        source = [0.2, 1e-10, 0.0, 1.0, 0.0]
        adjusted, changed = clamp_fine_tune_initialization(
            source,
            names,
            parameter_epsilon=1e-10,
            lapse_max=1.0,
        )
        self.assertEqual(source, [0.2, 1e-10, 0.0, 1.0, 0.0])
        self.assertEqual(changed, ("s0", "ell_short", "ell_long", "kappa"))
        self.assertGreater(adjusted[1], 1e-10)
        self.assertGreater(adjusted[2], 0.0)
        self.assertLess(adjusted[3], 1.0)
        self.assertGreater(adjusted[4], 0.0)

    def test_m2_delta_boundary_is_moved_inside_fine_tune_domain(self) -> None:
        names = ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0")
        adjusted, changed = clamp_fine_tune_initialization(
            [0.2, 0.1, 0.2, 0.3, 1.0, -3.0, 1e-10],
            names,
            parameter_epsilon=1e-10,
            lapse_max=1.0,
        )
        self.assertEqual(changed, ("delta0",))
        self.assertGreater(adjusted[6], 1e-10)

    def test_valid_near_boundary_start_is_not_changed(self) -> None:
        names = ("w", "s0", "ell_short", "ell_long", "kappa")
        source = [2e-10, 3e-10, 2e-12, 1.0 - 2e-12, 2e-12]
        adjusted, changed = clamp_fine_tune_initialization(
            source,
            names,
            parameter_epsilon=1e-10,
            lapse_max=1.0,
        )
        self.assertEqual(adjusted, source)
        self.assertEqual(changed, ())

    def test_sparse_psychometrics_reduce_only_the_required_bin_count(self) -> None:
        trial_types = ["short"] * 3 + ["long"] * 8
        frame = pd.DataFrame(
            {
                "model_evaluated": [True] * len(trial_types),
                "trial_type": trial_types,
                "duration_s": [0.2, 0.35, 0.5] + list(np.linspace(1.7, 2.3, 8)),
                "session_date": ["2026-01-01"] * len(trial_types),
                "session_id": ["session"] * len(trial_types),
                "trial_index": range(1, len(trial_types) + 1),
                "choice_long": np.resize([0.0, 1.0], len(trial_types)),
                "model_probability_long": np.linspace(0.1, 0.9, len(trial_types)),
            }
        )
        summary = adaptive_psychometric_summary(frame)
        bins = summary.groupby("trial_type")["n_bins_for_trial_type"].first().to_dict()
        self.assertEqual(bins, {"long": 4, "short": 3})
        self.assertEqual(summary.loc[summary["trial_type"].eq("short")].shape[0], 3)
        self.assertEqual(summary.loc[summary["trial_type"].eq("long")].shape[0], 4)

    def test_pooling_keeps_repeated_window_sessions_as_distinct_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = pd.DataFrame(
                {
                    "session_id": ["same_session"],
                    "split_role": ["train"],
                    "trial_index": [1],
                }
            )
            second = first.assign(split_role="validation")
            first.to_csv(root / "first.csv", index=False)
            second.to_csv(root / "second.csv", index=False)
            pooled = pool_prediction_tables(
                (("window_001", root / "first.csv"), ("window_002", root / "second.csv"))
            )
        self.assertEqual(pooled["original_session_id"].nunique(), 1)
        self.assertEqual(pooled["analysis_session_id"].nunique(), 2)
        self.assertEqual(
            set(pooled["analysis_session_id"]),
            {"window_001::same_session", "window_002::same_session"},
        )

    def test_registered_m0_retains_exactly_eight_initializations(self) -> None:
        registry = load_model_registry(ROOT / "config" / "m0_sensory_v1_mc06.json")
        self.assertEqual(len(registry.initializations), 8)
        self.assertEqual(
            [item.start_id for item in registry.initializations], list(range(1, 9))
        )

    def test_sliding_windows_are_exact_9_3_1_with_step_one(self) -> None:
        windows = build_sliding_windows(_assignments())
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[0].training_ranks, tuple(range(1, 10)))
        self.assertEqual(windows[0].validation_ranks, (10, 11, 12))
        self.assertEqual(windows[0].test_rank, 13)
        self.assertEqual(windows[-1].training_ranks, tuple(range(4, 13)))
        self.assertEqual(windows[-1].validation_ranks, (13, 14, 15))
        self.assertEqual(windows[-1].test_rank, 16)
        for first, second in zip(windows, windows[1:]):
            self.assertEqual(second.training_ranks[0] - first.training_ranks[0], 1)
            self.assertEqual(len(first.training_session_ids), 9)
            self.assertEqual(len(first.validation_session_ids), 3)

    def test_terminal_block_is_ignored_and_previous_two_are_held_out(self) -> None:
        partitioned, partition, audit = partition_session_blocks(_session_frame())
        self.assertEqual(partition.prefix_block_indices, (1, 2, 3))
        self.assertEqual(partition.heldout_block_indices, (4, 5))
        self.assertEqual(partition.ignored_terminal_block_index, 6)
        observed = partitioned.groupby("block_index")["experiment_partition"].first().to_dict()
        self.assertEqual(
            observed,
            {
                1: "prefix",
                2: "prefix",
                3: "prefix",
                4: "heldout",
                5: "heldout",
                6: "ignored",
            },
        )
        self.assertEqual(
            audit.loc[audit["block_index"].eq(6), "observed_complete"].item(), 0
        )
        self.assertTrue(
            audit.loc[
                audit["experiment_partition"].eq("heldout"), "observed_complete"
            ].eq(1).all()
        )

    def test_matched_workflow_holds_out_literal_final_two_blocks(self) -> None:
        partitioned, partition, audit = partition_final_two_blocks(_session_frame())
        self.assertEqual(partition.prefix_block_indices, (1, 2, 3, 4))
        self.assertEqual(partition.heldout_block_indices, (5, 6))
        self.assertIsNone(partition.ignored_terminal_block_index)
        observed = partitioned.groupby("block_index")[
            "experiment_partition"
        ].first().to_dict()
        self.assertEqual(
            observed,
            {1: "prefix", 2: "prefix", 3: "prefix", 4: "prefix", 5: "heldout", 6: "heldout"},
        )
        terminal = audit.loc[audit["block_index"].eq(6)].iloc[0]
        self.assertEqual(terminal["experiment_partition"], "heldout")
        self.assertEqual(int(terminal["right_censored_terminal"]), 1)

    def test_window_winner_maps_directly_to_its_test_session(self) -> None:
        assignments = _assignments()
        windows = build_sliding_windows(assignments)
        winners = {
            window.window_id: [float(window.window_id), 0.2, 0.1, 0.5, 1.0]
            for window in windows
        }
        matched = match_window_winners_to_test_sessions(
            assignments,
            windows,
            winners,
            ("w", "s0", "ell_short", "ell_long", "kappa"),
        )
        self.assertEqual(matched["chronological_rank"].tolist(), [13, 14, 15, 16])
        self.assertEqual(matched["source_window_id"].tolist(), [1, 2, 3, 4])
        self.assertEqual(matched["initial_w"].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertTrue(matched["n_contributing_windows"].eq(1).all())

    def test_session_initializations_average_training_windows_and_cover_tail(self) -> None:
        assignments = _assignments()
        windows = build_sliding_windows(assignments)
        winners = {
            window.window_id: [float(window.window_id), 0.2, 0.1, 0.5, 1.0]
            for window in windows
        }
        result = aggregate_session_initializations(assignments, windows, winners).set_index(
            "chronological_rank"
        )
        self.assertEqual(result.loc[1, "initial_w"], 1.0)
        self.assertEqual(result.loc[4, "initial_w"], 2.5)
        self.assertEqual(result.loc[12, "initial_w"], 4.0)
        self.assertEqual(
            result.loc[13, "initialization_policy"],
            "final_window_winner_fallback_no_training_membership",
        )
        self.assertEqual(result.loc[16, "initial_w"], 4.0)
        self.assertEqual(result.loc[16, "n_contributing_windows"], 1)


if __name__ == "__main__":
    unittest.main()
