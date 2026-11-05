"""Requested K=2 and K=3 descriptive sliding-window K-means figures.

Run: python -m post_analysis.session_clustering
Primary clustering uses training-fit parameters and preceding validation performance. Test-session plots are descriptive checks.
"""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .features import (OUT, LEGACY_OUT, collect, matrix, normalize, compact_features,
                       dynamic_annotations, select_features)

SEED = 20261025
BLOCK = 13  # full train9/validation3/test1 span; resampling is still descriptive
COLORS = ['#4477AA', '#EE6677', '#228833', '#AA3377']
# Fixed gates, reported in summary.json. No tuning to get a desired number of clusters.
GATES = dict(min_per_mouse=3, max_mouse_fraction_deviation=.20, max_mouse_nmi=.10,
             min_bootstrap_ari=.60, min_lomo_ari=.40, min_silhouette=.10)


def partition(x, k):
    if k == 1:
        return np.ones(len(x), dtype=int)
    return KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(x) + 1


def mouse_checks(labels, subjects):
    table = pd.crosstab(pd.Series(subjects, name='mouse'), pd.Series(labels, name='cluster'))
    fractions = table.div(table.sum(axis=1), axis=0)
    deviation = float(np.max(np.abs(fractions.to_numpy() - pd.Series(labels).value_counts(normalize=True)
                                    .reindex(table.columns).to_numpy())))
    nmi = float(normalized_mutual_info_score(subjects, labels))
    passed = (table.to_numpy().min() >= GATES['min_per_mouse']
              and deviation <= GATES['max_mouse_fraction_deviation'] and nmi <= GATES['max_mouse_nmi'])
    return dict(mouse_gate=bool(passed), mouse_nmi=nmi, max_mouse_fraction_deviation=deviation,
                min_per_mouse=int(table.to_numpy().min()))


def block_sample(subjects, rng):
    """Moving blocks within each mouse; never wrap the end of its chronology to its start."""
    sampled = []
    for subject in np.unique(subjects):
        ix = np.flatnonzero(subjects == subject)
        block = min(BLOCK, len(ix))
        starts = rng.integers(0, len(ix) - block + 1, size=int(np.ceil(len(ix)/block)))
        sampled.extend(np.concatenate([ix[s:s+block] for s in starts])[:len(ix)])
    return np.asarray(sampled)


def stability(values, meta, subjects, labels, k, repeats, rng):
    """Refit preprocessing and clustering in each resample; compare sampled unique rows only.

    LOMO predicts the excluded mouse from nearest training-cluster centroids. Its own
    feature mean/SD are used (transductive descriptive normalization, not live prediction).
    """
    aris = []
    for _ in range(repeats):
        ix = block_sample(subjects, rng)
        xb, _ = matrix(values.iloc[ix].reset_index(drop=True), meta, subjects[ix])
        lb = partition(xb, k)
        _, first = np.unique(ix, return_index=True)
        if len(np.unique(labels[ix[first]])) > 1:
            aris.append(adjusted_rand_score(labels[ix[first]], lb[first]))
    transfers = []
    for subject in np.unique(subjects):
        keep = subjects != subject
        # Select features on donors only; excluded mouse cannot change donor geometry.
        xd, used = matrix(values.loc[keep].reset_index(drop=True), meta, subjects[keep])
        xt = normalize(values.loc[~keep, used.feature], subjects[~keep]) * used.weight.to_numpy()
        donor = partition(xd, k)
        centroids = np.vstack([xd[donor == c].mean(axis=0) for c in range(1, k+1)])
        predicted = cdist(xt, centroids).argmin(axis=1) + 1
        transfers.append(dict(subject=str(subject), ari=float(adjusted_rand_score(labels[~keep], predicted)),
                              predicted_groups=int(np.unique(predicted).size)))
    return dict(bootstrap_ari_median=float(np.median(aris)) if aris else 0.,
                bootstrap_ari_p10=float(np.quantile(aris, .1)) if aris else 0.,
                lomo_ari_min=min(t['ari'] for t in transfers), lomo=transfers)


def driver_manifest(manifest):
    """Keep nuisance parameters in the heatmap, outside the clustering distance."""
    return manifest[~manifest.block.eq('bias') &
                    ~manifest.description.isin(['ell_short', 'ell_long'])].copy()


def fit_solutions(values, manifest, subjects, repeats=200):
    """Render both requested solutions regardless of diagnostic thresholds; never substitute K=1."""
    subjects = np.asarray(subjects)
    meta = driver_manifest(manifest)
    x, used = matrix(values, meta, subjects)
    checks, solutions = [], {}
    for k in (2, 3):
        labels = partition(x, k)
        check = mouse_checks(labels, subjects)
        check.update(k=k, silhouette=float(silhouette_score(x, labels)))
        # Paired resample sequences across K; fixed preprocessing is refitted on each draw.
        check.update(stability(values, meta, subjects, labels, k, repeats, np.random.default_rng(SEED)))
        check['passes_diagnostics'] = bool(check['mouse_gate'] and check['silhouette'] >= GATES['min_silhouette']
                                          and check['bootstrap_ari_median'] >= GATES['min_bootstrap_ari']
                                          and check['lomo_ari_min'] >= GATES['min_lomo_ari'])
        checks.append(check)
        solutions[k] = labels
        print(f"K={k}: mouse NMI={check['mouse_nmi']:.3f}, bootstrap ARI={check['bootstrap_ari_median']:.3f}", flush=True)
    return x, used, solutions, checks


def raster_order(x):
    """Rastermap orders rows only. K-means is always performed on feature vectors."""
    from rastermap import Rastermap
    rm = Rastermap(n_clusters=min(20, max(5, len(x)//4)), n_PCs=min(20, x.shape[1], len(x)-1),
                   locality=.75, time_lag_window=0, random_state=SEED, verbose=False, normalize=False, mean_time=False)
    rm.fit(data=x)
    return np.asarray(rm.isort, dtype=int)


def cluster_figure(raw, values, manifest, labels, order, check, display_z=None):
    z = normalize(values[manifest.feature], raw.subject.to_numpy()) if display_z is None else np.asarray(display_z)
    k = len(np.unique(labels))
    # Keep cluster groups contiguous, preserving Rastermap order within each group.
    order = order[np.argsort(labels[order], kind='stable')]
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(1, 5, width_ratios=[.35, .35, 9, .20, 4],
                          left=.04, right=.97, bottom=.32, top=.90, wspace=.55)
    subjects = list(raw.subject.unique())
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.array([subjects.index(s) for s in raw.subject.iloc[order]])[:, None],
              aspect='auto', cmap=ListedColormap(['#1b9e77', '#d95f02', '#7570b3']), vmin=0, vmax=2)
    ax.set_title('Mouse', fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(labels[order, None], aspect='auto', cmap=ListedColormap(COLORS[:k]), vmin=1, vmax=max(2,k))
    ax.set_title('Group', fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(z[order], aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3, interpolation='nearest')
    for boundary in np.flatnonzero(np.diff(labels[order])) + .5:
        ax.axhline(boundary, color='black', lw=2)
    for boundary in np.flatnonzero(manifest.model.to_numpy()[1:] != manifest.model.to_numpy()[:-1]) + .5:
        ax.axvline(boundary, color='black', lw=.6)
    short = dict(pure_wsls='WSLS', extended_wsls='WSLS-hist', m0_sensory='M0',
                 m1_action_kernel='Action-hist', m1_stimulus_kernel='Stim-hist',
                 m1_known_switch='M1-oracle', m1_hmm_true='M1-static', m1_dynamic='M1-dyn',
                 m2_known_switch='M2-oracle', m2_static='M2-static', m2_dynamic='M2-dyn',
                 m2_dynamic_wsls_bias='M2-dyn+WSLS', m2_dynamic_repetition_bias='M2-dyn+repeat')
    compact_names = {'noise_short': 'M0: log SD at 0.8s', 'noise_long': 'M0: log SD at 1.7s',
                     'history_alpha': 'M1 action: update rate', 'prior_contrast': 'M2 static: block contrast',
                     'prior_hazard': 'M2 static: switch hazard',
                     'performance__m0_sensory': 'Validation: M0 vs coin',
                     'performance__m1_action_kernel': 'Validation: M1 vs M0',
                     'performance__m2_static': 'Validation: M2 vs M0'}
    names = [compact_names.get(r.feature, f'{short[r.model]}: {r.description}') for r in manifest.itertuples()]
    ax.set_xticks(np.arange(len(names)), names, rotation=60, ha='right', fontsize=9)
    ax.set_ylabel('Window / held-out session (cluster, then Rastermap order)')
    ax.set_title('Compact panel: training parameters + validation performance', fontsize=12)
    cax = fig.add_subplot(gs[0, 3])
    fig.colorbar(im, cax=cax)
    cax.tick_params(labelsize=8)
    cax.set_title('z', fontsize=9)
    table = pd.crosstab(raw.subject, labels)
    ax = fig.add_subplot(gs[0, 4]); ax.axis('off')
    tab = ax.table(cellText=table.to_numpy(), rowLabels=table.index,
                   colLabels=[f'C{c}' for c in table.columns], loc='upper center', cellLoc='center')
    tab.auto_set_font_size(False); tab.set_fontsize(11); tab.scale(1, 1.6)
    for i, subject in enumerate(subjects):
        ax.text(0, .79-i*.035, f'■ {subject}', color=['#1b9e77', '#d95f02', '#7570b3'][i],
                transform=ax.transAxes, fontsize=10)
    lines = ['Exploratory K-means solution', '',
             f"K={k}; mouse NMI={check['mouse_nmi']:.3f}",
             f"Minimum sessions/mouse/cluster: {check['min_per_mouse']}",
             f"Resampling ARI: {check['bootstrap_ari_median']:.3f}",
             f"Minimum LOMO ARI: {check['lomo_ari_min']:.3f}",
             f"Passes diagnostic thresholds: {check['passes_diagnostics']}",
             '', 'Drivers: M0, M1 action, M2 static', 'Dynamic M2: annotation only',
             'No target-test scores in distance', '', 'Feature blocks have equal weight.']
    ax.text(0, .66, '\n'.join(lines), transform=ax.transAxes, va='top', fontsize=10)
    status = f'Requested K={k} partition; exploratory. Cluster numbers are local to each K.'
    if not check['mouse_gate']:
        status += ' CAUTION: mouse-representation diagnostic failed.'
    fig.text(.11, .11, status + '\n' + '; '.join(f'{s}: {len(raw[raw.subject.eq(s)])} sessions' for s in subjects)
            + '\nTarget-test behavior is excluded from features. Overlapping windows and retrospective scaling remain.'
            + '\nParameters were fitted on nine training sessions, not independently on the displayed test session.',
            fontsize=10, va='top')
    fig.suptitle(f'Sliding-window session clustering — K={k}, K-means driver features', fontsize=18)
    return fig, order


def gain_transfer(raw):
    """Descriptive transfer from past validation to next test, never used to select features."""
    from scipy.stats import spearmanr
    rows = []
    for model, reference in [('m1_action_kernel', 'm0_sensory'), ('m2_static', 'm0_sensory'), ('m2_dynamic', 'm2_static')]:
        validation = raw['validation_nll__'+reference] - raw['validation_nll__'+model]
        test = raw['nll__'+reference] - raw['nll__'+model]
        for subject in ['ALL', *raw.subject.unique()]:
            ix = np.ones(len(raw), bool) if subject == 'ALL' else raw.subject.eq(subject).to_numpy()
            rows.append(dict(subject=subject, model=model, reference=reference, n_windows=int(ix.sum()),
                             validation_gain_mean=float(validation[ix].mean()), test_gain_mean=float(test[ix].mean()),
                             validation_test_spearman=float(spearmanr(validation[ix], test[ix]).statistic),
                             test_gain_when_validation_positive=float(test[ix & (validation.to_numpy()>0)].mean()),
                             test_fraction_positive=float((test[ix]>0).mean())))
    return pd.DataFrame(rows)


def compare_panels(raw, old_values, old_meta, values, meta, solutions, checks, repeats, out):
    """Fixed comparisons, identical resampling settings; no search for a winning panel."""
    rows, labels = [], []
    subjects = raw.subject.to_numpy()
    panels = [('legacy_all_models_test', old_values, old_meta),
              ('compact_test_sensitivity', *compact_features(raw, 'test')),
              ('compact_validation_primary', values, meta)]
    old_saved = LEGACY_OUT / 'session_clusters.csv'
    previous = pd.read_csv(old_saved) if old_saved.exists() else None
    for name, vv, mm in panels:
        if name == 'compact_validation_primary':
            sol, diagnostics = solutions, checks
        else:
            _, _, sol, diagnostics = fit_solutions(vv, mm, subjects, repeats)
        for diagnostic in diagnostics:
            k = diagnostic['k']
            row = {key:value for key,value in diagnostic.items() if key != 'lomo'}
            row.update(panel=name, n_features=len(matrix(vv, driver_manifest(mm), subjects)[1]),
                       ari_to_primary=float(adjusted_rand_score(solutions[k], sol[k])))
            if previous is not None:
                reference = raw[['subject','session_id']].merge(previous[previous.K.eq(k)], on=['subject','session_id'], validate='one_to_one')
                row['ari_to_saved_previous'] = float(adjusted_rand_score(reference.cluster_label, sol[k]))
            rows.append(row)
            frame = raw[['subject','window_id','session_id']].copy()
            frame['panel'], frame['K'], frame['cluster_label'] = name, k, sol[k]
            labels.append(frame)
    pd.DataFrame(rows).to_csv(out / 'panel_comparison.csv', index=False)
    pd.concat(labels).to_csv(out / 'panel_comparison_labels.csv', index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--output', type=str, default=str(OUT))
    args = parser.parse_args()
    if args.bootstrap < 20:
        parser.error('--bootstrap must be at least 20')
    from pathlib import Path
    out = Path(args.output).resolve()
    if not out.is_relative_to(OUT.parents[3].resolve()):
        parser.error('Output must be inside results_post_analysis')
    out.mkdir(parents=True, exist_ok=True)
    raw, legacy_values, legacy_manifest, _ = collect()
    values, manifest = compact_features(raw)
    print(f'Loaded {len(raw)} matched sliding windows', flush=True)
    x, used, solutions, checks = fit_solutions(values, manifest, raw.subject.to_numpy(), args.bootstrap)
    selection = select_features(values, driver_manifest(manifest), raw.subject.to_numpy())
    selection.to_csv(out / 'feature_selection.csv', index=False)
    # The heatmap contains only actual drivers, not excluded/duplicate inputs.
    manifest = used.drop(columns='weight')
    comparisons = compare_panels(raw, legacy_values, legacy_manifest, values, selection, solutions, checks, args.bootstrap, out)
    order = raster_order(x)
    label_frames, count_frames = [], []
    with PdfPages(out / 'clusters.pdf') as pdf:
        for check in checks:
            k = check['k']
            labels = solutions[k]
            fig, display_order = cluster_figure(raw, values, manifest, labels, order, check)
            pdf.savefig(fig)
            if k == 2:
                fig.savefig(out / 'clusters.png', dpi=160)
            plt.close(fig)
            lab = raw[['subject', 'window_id', 'test_rank', 'session_id']].copy()
            lab['K'], lab['cluster_label'] = k, labels
            lab['display_rank'] = np.argsort(display_order) + 1
            label_frames.append(lab)
            counts = pd.crosstab(raw.subject, labels).rename_axis(columns='cluster_label').stack().rename('n_sessions').reset_index()
            counts.insert(0, 'K', k)
            count_frames.append(counts)
    all_labels = pd.concat(label_frames, ignore_index=True)
    all_labels.to_csv(out / 'session_clusters.csv', index=False)
    pd.concat(count_frames, ignore_index=True).to_csv(out / 'cluster_by_mouse.csv', index=False)
    raw.to_csv(out / 'window_metrics_parameters.csv', index=False)
    values.to_csv(out / 'feature_values.csv', index=False)
    annotations = dynamic_annotations(raw)
    annotations.to_csv(out / 'dynamic_m2_annotations.csv', index=False)
    gain_transfer(raw).to_csv(out / 'validation_test_transfer.csv', index=False)
    dynamic_profiles = all_labels.merge(annotations, on=['subject', 'window_id', 'test_rank', 'session_id'], validate='many_to_one')
    dynamic_profiles.groupby(['K','cluster_label','subject']).mean(numeric_only=True).to_csv(out / 'dynamic_by_cluster_mouse.csv')
    manifest.to_csv(out / 'display_features.csv', index=False)
    used.to_csv(out / 'clustering_features.csv', index=False)
    pd.DataFrame(x, columns=used.feature).to_csv(out / 'clustering_matrix.csv', index=False)
    from .diagnostics import evaluate
    prediction_checks = evaluate(raw, all_labels, out, block=BLOCK)
    summary = dict(prediction_checks=prediction_checks, n_windows=len(raw), rendered_K=[2, 3], algorithm='KMeans', n_init=20,
                   seed=SEED, gates=GATES, bootstrap_repeats=args.bootstrap, bootstrap_block=BLOCK, solutions=checks,
                   panel_comparison=comparisons, target_test_used_in_features=False,
                   features='M0 effective noise at .8/1.7s; M1 action alpha; M2 static contrast/hazard; validation NLL gains',
                   feature_screen='10–90% physical spread in >=2 mice; within-block |r|>=.95 redundancy; refitted in resampling',
                   forced_fit_handling='same weights and inclusion as all other fits',
                   interpretation='Experimental compact panel; no automatic replacement of previous results. Both K values shown even when diagnostics fail.',
                   excluded_from_clustering=['other models including dynamic M2', 'raw w/s0', 'kappa', 'ell_short', 'ell_long', 'test performance'],
                   mouse_normalization='within-mouse mean and SD; LOMO uses excluded mouse feature statistics')
    from .behavior import figures, VISUALIZATION, WEIGHTING
    page_manifest = figures(all_labels, out)
    from .latency import METHOD
    summary.update(behavior_pages=page_manifest, behavior_visualization=VISUALIZATION, behavior_weighting=WEIGHTING,
                   reaction_time=METHOD)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    # Remove only obsolete outputs owned by the previous single-group pipeline.
    for name in ['psychometrics', 'block_transitions', 'rare_trials']:
        for extension in ['png', 'pdf']:
            (out / f'{name}.{extension}').unlink(missing_ok=True)
    for name in ['candidate_cluster_labels.csv', 'psychometric_session_bins.csv',
                 'transition_events.csv', 'rare_events.csv']:
        (out / name).unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))
    print(f'Wrote clusters.pdf (2 pages) and cluster_behavior_pooled.pdf (4 pages): {out}')


if __name__ == '__main__':
    main()
