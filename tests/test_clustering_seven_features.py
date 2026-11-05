"""Scientific contracts for signed gains, train-only baseline and equal geometry."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from post_analysis.session_clustering import pattern_rbf_seven_features as module
from post_analysis.session_clustering.pattern_rbf_seven_features import (
    FEATURES, DEFINITIONS, constant_choice_nll, gain_features, standardized, audit_predictions,
)


class SevenFeatureTests(unittest.TestCase):
    def test_signed_contrasts_do_not_use_parameters_or_trial_totals(self):
        raw = pd.DataFrame({
            'gain_bias__m0_sensory': [.12, -.03], 'nll__m0_sensory': [.6, .7],
            'nll__m1_previous_action_bias': [.5, .75], 'nll__m1_action_kernel': [.53, .65],
            'nll__m1_previous_wsls_bias': [.58, .68], 'nll__m1_stimulus_kernel': [.55, .72],
            'nll__m2_static': [.49, .62], 'nll__m2_dynamic': [.50, .60],
            'param__unused': [1e20, -1e20], 'n__m0_sensory': [100, 1000],
        })
        actual, manifest = gain_features(raw)
        expected = [[.12, .10, -.03, .02, .03, .11, -.01],
                    [-.03, -.05, .10, .02, -.04, .08, .02]]
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        self.assertEqual(list(actual.columns), FEATURES)
        self.assertTrue(manifest.weight.eq(1).all())
        changed = raw.copy(); changed['n__m0_sensory'] *= 10
        np.testing.assert_array_equal(gain_features(changed)[0], actual)

    def test_baseline_uses_training_choices_and_mean_test_loss(self):
        q, loss = constant_choice_nll([0, 0, 0, 1], [1, 1])
        self.assertEqual(q, .25)
        self.assertAlmostEqual(loss, -np.log(.25))
        self.assertEqual(constant_choice_nll([0, 0, 0, 1], [1]*20), (q, loss))
        q2, loss2 = constant_choice_nll([0, 0, 0, 1], [0, 0])
        self.assertEqual(q2, q)
        self.assertAlmostEqual(loss2, -np.log(.75))
        for choices in ([0]*5, [1]*5):
            self.assertTrue(np.isfinite(constant_choice_nll(choices, [0, 1])[1]))
        with self.assertRaises(ValueError): constant_choice_nll([], [0])
        with self.assertRaises(ValueError): constant_choice_nll([0, 1], [2])

    def test_scaling_is_equal_weight_and_within_mouse(self):
        base = np.array([-2., -1., 0., 1., 2.])
        rows = np.column_stack([base*(i+1)*.03+i for i in range(7)])
        values = pd.DataFrame(np.vstack([rows, rows*20+100]), columns=FEATURES)
        z, audit = standardized(values, ['A']*5+['B']*5)
        expected = np.tile((base/base.std(ddof=1))[:, None], (1, 7))
        np.testing.assert_allclose(z[:5], expected, atol=1e-12)
        np.testing.assert_allclose(z[5:], expected, atol=1e-12)
        self.assertTrue(audit.weight.eq(1).all())
        values.loc[:, 'sensory'] = 1.
        z, audit = standardized(values, ['A']*5+['B']*5)
        np.testing.assert_array_equal(z[:, 0], np.zeros(10))
        self.assertTrue(audit[audit.feature.eq('sensory')].denominator.eq(1).all())

    def test_clipping_and_degenerate_profiles(self):
        values = pd.DataFrame(np.zeros((30, 7)), columns=FEATURES)
        for i, feature in enumerate(FEATURES): values.loc[i, feature] = 100.
        z, audit = standardized(values, ['A']*30)
        self.assertEqual(z.max(), 3.)
        self.assertEqual(int(audit.clipped_count.sum()), 7)
        with self.assertRaises(ValueError):
            standardized(pd.DataFrame(np.ones((30, 7)), columns=FEATURES), ['A']*30)


    def test_k_selection_recovers_separated_groups_and_numbers_by_size(self):
        rng = np.random.default_rng(0)
        centers = np.eye(7)[:3]*6
        z = np.vstack([c+rng.normal(0, .3, (n, 7)) for c, n in zip(centers, (30, 20, 10))])
        affinity, _ = module.pe.gaussian_affinity(z)
        subjects = np.tile(['A', 'B', 'C'], 20)
        with patch.object(module, 'N_STABILITY', 5):
            table, eigenvalues, candidates, optimal = module.k_selection(z, affinity, subjects)
        self.assertEqual(optimal, 3)
        self.assertEqual(int(table.K[table.best_by_eigengap].iloc[0]), 3)
        self.assertTrue(table.all_mice_in_every_cluster[table.K.eq(3)].iloc[0])
        self.assertGreater(table.stability_ari_median[table.K.eq(3)].iloc[0], .9)
        self.assertEqual(list(table.K), list(module.K_RANGE))
        self.assertEqual(table.optimal.sum(), 1)
        self.assertAlmostEqual(eigenvalues[0], 0, places=8)
        lab, mapping = module.size_order(candidates[3])
        np.testing.assert_array_equal(lab, np.repeat([1, 2, 3], [30, 20, 10]))
        self.assertEqual(sorted(mapping.values()), [1, 2, 3])

    def test_published_layout_places_every_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = [n for _, _, group in module.LAYOUT for n in group] + list(module.TOP)
            names += ['K2_report.pdf', 'K3_1_fingerprint.png', 'K4_5_reaction_time.png']
            for name in names: (root/name).write_text('x')
            (root/'sequence_performance').mkdir(); (root/'sequence_performance'/'methods.json').write_text('{}')
            moved = module.organize(root)
            self.assertEqual(sorted(p.name for p in root.iterdir() if p.is_file()), sorted(module.TOP))
            self.assertEqual(moved['K2_report.pdf'], '04_clusters_K2/K2_report.pdf')
            self.assertEqual(moved['K3_1_fingerprint.png'], '04_clusters_K3/K3_1_fingerprint.png')
            self.assertEqual(moved['K4_5_reaction_time.png'], '04_clusters_K4/K4_5_reaction_time.png')
            self.assertTrue((root/'04_clusters_K3'/'sequence_performance'/'methods.json').exists())
            for path in moved.values(): self.assertTrue((root/path).exists(), path)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'unexpected.csv').write_text('x')
            with self.assertRaisesRegex(RuntimeError, 'unexpected.csv'): module.organize(Path(tmp))

    def test_trial_audit_rejects_equal_count_mismatches_and_stale_losses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = list(dict.fromkeys(m for d in DEFINITIONS for m in d[3:5] if m != 'constant_choice'))
            data = pd.DataFrame(dict(session_id=['train']*2+['validation']*2+['test']*2,
                                     trial_index=[1, 2]*3, source_trial_index=[0, 1]*3,
                                     split_role=['train']*2+['validation']*2+['test']*2,
                                     model_evaluated=[True]*6, choice_long=[0, 1]*3,
                                     duration_s=[.8, 1.7]*3, category_long=[0, 1]*3,
                                     model_probability_long=[.2, .6, .4, .7, .1, .7]))
            y = data.choice_long.to_numpy(); prob = data.model_probability_long.to_numpy()
            data['per_trial_negative_log_likelihood'] = -y*np.log(prob)-(1-y)*np.log1p(-prob)
            raw = dict(subject='A', window_id=1, session_id='test')
            registry, paths = {}, {}
            for model in models:
                model_root = root/model
                window = model_root/'window_001'; window.mkdir(parents=True)
                paths[model] = window/'trial_predictions.csv'
                data.to_csv(paths[model], index=False)
                registry[model] = SimpleNamespace(result_root=lambda subject, p=model_root: p)
                for split, prefix in [('train', 'train_nll__'), ('validation', 'validation_nll__'), ('test', 'nll__')]:
                    raw[prefix+model] = data.loc[data.split_role.eq(split), 'per_trial_negative_log_likelihood'].mean()
                raw['n__'+model] = 2
            raw['gain_bias__m0_sensory'] = np.log(2)-raw['nll__m0_sensory']
            raw = pd.DataFrame([raw])
            with patch('post_analysis.session_clustering.pattern_rbf_seven_features.MODEL_BY_ID', registry):
                sources = audit_predictions(raw, root)
                self.assertEqual(len(sources), 7)
                # Same number of trials is not sufficient: identities and flags must match.
                for field, value in [('trial_index', 3), ('choice_long', 1), ('model_evaluated', False),
                                     ('per_trial_negative_log_likelihood', 10.)]:
                    bad = data.copy(); bad.loc[4, field] = value
                    bad.to_csv(paths[models[1]], index=False)
                    with self.subTest(field=field), self.assertRaises((AssertionError, ValueError)):
                        audit_predictions(raw, root)
                    data.to_csv(paths[models[1]], index=False)
                stale = raw.copy(); stale['nll__m0_sensory'] += .1
                with self.assertRaises(AssertionError): audit_predictions(stale, root)
                duplicate = pd.concat([data, data[data.split_role.eq('train')]], ignore_index=True)
                duplicate.to_csv(paths['m0_sensory'], index=False)
                with self.assertRaisesRegex(ValueError, 'Duplicate trial identity'): audit_predictions(raw, root)
                data.to_csv(paths['m0_sensory'], index=False)
                leak = data.copy(); leak.loc[leak.split_role.eq('train'), 'session_id'] = 'test'
                leak.to_csv(paths['m0_sensory'], index=False)
                with self.assertRaisesRegex(ValueError, 'leakage|Duplicate trial identity'): audit_predictions(raw, root)



    def test_failed_run_never_replaces_existing_results(self):
        from pypdf import PdfWriter
        for mode in ('analysis', 'invalid_pdf', 'preservation', 'publish'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); destination = root/'result'; destination.mkdir()
                sentinel = destination/'old.txt'; sentinel.write_text('original result')
                def generate(out):
                    if mode == 'analysis':
                        (out/'partial.txt').write_text('partial')
                        raise RuntimeError('analysis failed')
                    if mode == 'invalid_pdf':
                        (out/'K2_report.pdf').write_text('not a PDF')
                        return {'K2_report.pdf': 5}
                    if mode == 'preservation': return
                    reports = {'K2_report.pdf': 5, 'K3_report.pdf': 5, 'K4_report.pdf': 5, 'k_selection.pdf': 1}
                    expected = {**reports, 'sequence_performance/rbf_sequence_overview.pdf': 3,
                                'sequence_performance/rbf_sequence_by_cluster.pdf': 9,
                                **{f'sequence_performance/rbf_cluster_{c}_sequence_performance.pdf': 3 for c in (1, 2, 3)}}
                    for name, count in expected.items():
                        path = out/name; path.parent.mkdir(exist_ok=True)
                        w = PdfWriter()
                        for _ in range(count): w.add_blank_page(width=72, height=72)
                        with path.open('wb') as handle: w.write(handle)
                    return reports
                original_rename = Path.rename
                def rename(path, target):
                    if mode == 'publish' and '.staging_' in path.name:
                        raise OSError('simulated publication failure')
                    return original_rename(path, target)
                snapshots = [{'old': 'same'}, {'old': 'changed' if mode == 'preservation' else 'same'}]
                with patch.object(module, 'ROOT', root), patch.object(module, 'OUT', destination), \
                     patch.object(module, 'protected_hashes', side_effect=snapshots), \
                     patch.object(module, 'run_analysis', side_effect=generate), patch.object(Path, 'rename', rename):
                    with self.assertRaises(Exception): module.run()
                self.assertEqual(sentinel.read_text(), 'original result')
                self.assertEqual(list(root.iterdir()), [destination])


if __name__ == '__main__': unittest.main()
