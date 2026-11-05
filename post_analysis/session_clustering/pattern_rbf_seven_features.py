"""Seven signed predictive-gain coordinates; isolated first-round RBF analysis.

Run: python -m post_analysis.session_clustering.pattern_rbf_seven_features
Previous analysis code/output is read-only. No decision models are refitted.
"""
from __future__ import annotations

from contextlib import ExitStack
from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import scipy
from scipy.sparse.csgraph import laplacian
import sklearn
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)
from threadpoolctl import threadpool_limits

from . import features, pattern_exploration as pe, pattern_report as pr
from . import pattern_visuals as pv, behavior, latency, sequence_performance as seq
from .cluster import SEED, COLORS, driver_manifest, raster_order, mouse_checks
from .pattern_rbf_13_models import MODELS, EXPECTED
from post_analysis.expanded_model_comparison.config import MODEL_SPECS, MODEL_BY_ID

ROOT = features.BASE / 'pooled' / 'clustering'
OLD = ROOT / 'version1' / 'sliding_window_final'
PREVIOUS = ROOT / 'sliding_window_pattern_exploration_rbf_13_models'
OUT = ROOT / 'sliding_window_pattern_exploration_rbf_seven_features'
VARIANT = 'rbf_seven_features'
TITLE = 'RBF: seven signed predictive gains'
# Names are explicit, never inferred from mutable registry ordering.
DEFINITIONS = (
    ('sensory', 'Sensory-model\nbenefit', 'sensory', 'constant_choice', 'm0_sensory', 'baseline - 1'),
    ('action_single', 'Single-action\nbenefit', 'action', 'm0_sensory', 'm1_previous_action_bias', '1 - 3'),
    ('action_history_architecture', 'Action-history\narchitecture advantage', 'action', 'm1_previous_action_bias', 'm1_action_kernel', '3 - 2'),
    ('wsls_single', 'Single-WSLS\nbenefit', 'wsls', 'm0_sensory', 'm1_previous_wsls_bias', '1 - 7'),
    ('wsls_history_architecture', 'WSLS-history\narchitecture advantage', 'wsls', 'm1_previous_wsls_bias', 'm1_stimulus_kernel', '7 - 6'),
    ('static_context', 'Static-context\nbenefit', 'context', 'm0_sensory', 'm2_static', '1 - 10'),
    ('dynamic_switching', 'Dynamic-switching\nbenefit', 'context', 'm2_static', 'm2_dynamic', '10 - 11'),
)
FEATURES = [d[0] for d in DEFINITIONS]
KEYS = ['subject', 'session_id']
N_INIT = 50
# Always reported; they also have historical labels to align cluster numbers to.
REFERENCE_K = (2, 3)
# Additional reports do not change K selection or require historical labels.
EXTRA_REPORT_K = ()
K_RANGE = range(2, 9)
N_STABILITY, STABILITY_FRACTION = 100, .8
# Project colors first so K=2/3 figures are unchanged; extra slots only for larger K.
PALETTE = list(COLORS) + ['#997700', '#009988', '#CC3311', '#7755CC']
# Heatmaps: warm = high/positive, cold = low/negative, neutral gray at zero.
# Editable text in Illustrator: TrueType (Type 42) instead of Type 3; Liberation Sans has Arial metrics.
EDITABLE_PDF = {'pdf.fonttype': 42, 'ps.fonttype': 42}
ILLUSTRATOR = {**EDITABLE_PDF, 'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'DejaVu Sans']}
WARM_COLD = LinearSegmentedColormap.from_list('cold_warm', ['#2a5d9f', '#8fb3d9', '#f0efec', '#e8907e', '#b2302a'])


def spectral_labels(affinity, k):
    return SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', n_init=N_INIT,
                              eigen_solver='arpack', random_state=SEED).fit_predict(affinity)+1


def stability(affinity, labels, k):
    """ARI between full-data labels and reclustering of random session subsets (same spectral settings)."""
    rng = np.random.default_rng(SEED)
    scores = []
    for b in range(N_STABILITY):
        ix = np.sort(rng.choice(len(affinity), int(STABILITY_FRACTION*len(affinity)), replace=False))
        sub = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', n_init=N_INIT,
                                 eigen_solver='arpack', random_state=b).fit_predict(affinity[np.ix_(ix, ix)])
        scores.append(adjusted_rand_score(labels[ix], sub))
    return np.array(scores)


def k_selection(z, affinity, subjects):
    """Metrics over K_RANGE; optimal K = largest eigengap among K whose every cluster has all mice.

    The eigengap is the spectral-clustering criterion. Silhouette, Calinski-Harabasz,
    Davies-Bouldin and subsample stability are reported alongside, never used for the choice.
    The spectral embedding has k dimensions, so k-means inertia and embedding silhouettes
    are not comparable across k; the other scores use the z space the RBF kernel is built from.
    """
    eigenvalues = np.linalg.eigvalsh(laplacian(affinity, normed=True))
    rows, candidates = [], {}
    for k in K_RANGE:
        lab = candidates[k] = spectral_labels(affinity, k)
        # Same embedding SpectralClustering passes to its final k-means.
        embedding = spectral_embedding(affinity, n_components=k, eigen_solver='arpack',
                                       random_state=SEED, drop_first=False)
        ari = stability(affinity, lab, k)
        per_mouse = pd.crosstab(pd.Series(subjects, name='mouse'), pd.Series(lab, name='cluster'))
        rows.append(dict(K=k, eigenvalue_k=eigenvalues[k-1], eigenvalue_k_plus_1=eigenvalues[k],
                         eigengap=eigenvalues[k]-eigenvalues[k-1], silhouette_z=silhouette_score(z, lab),
                         silhouette_embedding=silhouette_score(embedding, lab),
                         calinski_harabasz=calinski_harabasz_score(z, lab), davies_bouldin=davies_bouldin_score(z, lab),
                         stability_ari_median=np.median(ari), stability_ari_q10=np.quantile(ari, .1),
                         stability_ari_q90=np.quantile(ari, .9), min_sessions_per_mouse_cluster=int(per_mouse.to_numpy().min()),
                         all_mice_in_every_cluster=bool(per_mouse.to_numpy().min() >= 1),
                         sizes='/'.join(map(str, sorted(np.bincount(lab)[1:], reverse=True)))))
    table = pd.DataFrame(rows)
    eligible = table[table.all_mice_in_every_cluster]
    if eligible.empty:
        raise ValueError('No K has sessions from every mouse in every cluster')
    # idxmax keeps the smaller K on ties.
    optimal = int(eligible.K[eligible.eigengap.idxmax()])
    table['optimal'] = table.K.eq(optimal)
    for name, column, best in [('eigengap', 'eigengap', 'max'), ('silhouette', 'silhouette_z', 'max'),
                               ('calinski_harabasz', 'calinski_harabasz', 'max'), ('davies_bouldin', 'davies_bouldin', 'min'),
                               ('stability', 'stability_ari_median', 'max')]:
        table[f'best_by_{name}'] = table.K.eq(table.K[getattr(table[column], 'idx'+best)()])
    return table, eigenvalues, candidates, optimal


def size_order(labels):
    """Number clusters by decreasing size when no historical labels exist at this K."""
    sizes = np.bincount(labels)[1:]
    mapping = {int(old): new for new, old in enumerate(np.argsort(-sizes, kind='stable')+1, 1)}
    return np.array([mapping[int(x)] for x in labels]), mapping


def constant_choice_nll(training_choices, test_choices):
    """Training-only Bernoulli MLE; same numerical clipping as legacy collector."""
    train, test = np.asarray(training_choices, float), np.asarray(test_choices, float)
    if not len(train) or not len(test) or not np.isin(train, [0, 1]).all() or not np.isin(test, [0, 1]).all():
        raise ValueError('Baseline requires nonempty binary evaluated choices')
    q = float(np.clip(train.mean(), 1e-12, 1-1e-12))
    return q, float(np.mean(-test*np.log(q)-(1-test)*np.log1p(-q)))


def gain_features(raw):
    values = pd.DataFrame(index=raw.index)
    rows = []
    for feature, label, family, reduced, full, formula in DEFINITIONS:
        values[feature] = (raw['gain_bias__m0_sensory'] if reduced == 'constant_choice'
                           else raw['nll__'+reduced] - raw['nll__'+full])
        rows.append(dict(variant=VARIANT, feature=feature, label=label.replace('\n', ' '),
                         family=family, reduced_model=reduced, full_model=full,
                         nll_subtraction=formula, units='nats/trial', weight=1.,
                         selected=True, interpretation='architecture comparison, not isolated memory'
                         if 'architecture' in feature else 'held-out predictive gain'))
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('Nonfinite signed gain')
    return values, pd.DataFrame(rows)


def standardized(values, subjects):
    """Equal coordinate weights; deliberately do NOT use legacy matrix()."""
    if list(values.columns) != FEATURES or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f'Expected {len(FEATURES)} finite ordered features')
    subjects = np.asarray(subjects)
    z = features.normalize(values, subjects)
    stats = []
    for subject in np.unique(subjects):
        sub = values.loc[subjects == subject]
        if len(sub) < 2:
            raise ValueError('Within-mouse scaling requires at least two sessions')
        for col in FEATURES:
            sd = float(sub[col].std(ddof=1))
            stats.append(dict(subject=subject, feature=col, mean=float(sub[col].mean()),
                              sd=sd, denominator=sd if sd > 1e-8 else 1., weight=1.,
                              clipped_count=int(np.sum(np.abs((sub[col]-sub[col].mean())/(sd if sd > 1e-8 else 1.)) > 3))))
    if np.unique(z, axis=0).shape[0] < 3:
        raise ValueError('Insufficient distinct profiles for K=3')
    return z, pd.DataFrame(stats)


def audit_predictions(raw, out):
    """Reconstruct all driver losses and baseline; verify identical test populations."""
    models = list(dict.fromkeys(m for d in DEFINITIONS for m in d[3:5] if m != 'constant_choice'))
    columns = ['session_id', 'trial_index', 'source_trial_index', 'split_role', 'model_evaluated',
               'choice_long', 'duration_s', 'category_long', 'per_trial_negative_log_likelihood',
               'model_probability_long']
    identity = [c for c in columns if c not in ['per_trial_negative_log_likelihood', 'model_probability_long']]
    files, audits, baselines = [], [], []
    for subject, windows in raw.groupby('subject', sort=False):
        paths = {}
        for model in models:
            root = MODEL_BY_ID[model].result_root(subject)
            if not root.exists():
                # Some old fits were moved inside model/mouse/model/mouse.
                # Accept only one unambiguous archived root and verify every
                # prediction against the existing aggregate tables below.
                candidates = list(root.parent.rglob('full_session_matched_v1'))
                if len(candidates) != 1:
                    raise ValueError(f'No unique prediction root: {subject}/{model}: {candidates}')
                root = candidates[0]
            predictions = sorted(root.glob('window_*/trial_predictions.csv'))
            paths[model] = {int(p.parent.name.split('_')[1]): p for p in predictions}
            if len(paths[model]) != len(predictions):
                raise ValueError(f'Duplicate window IDs: {root}')
            if set(paths[model]) != set(windows.window_id):
                raise ValueError(f'Canonical windows mismatch: {subject}/{model}')
        for row in windows.itertuples():
            reference = None
            for model in models:
                path = paths[model][row.window_id]
                payload = path.read_bytes()
                files.append(dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest()))
                t = pd.read_csv(BytesIO(payload), usecols=columns)
                flags = t.model_evaluated.astype(str).str.lower()
                if not flags.isin(['true', 'false']).all():
                    raise ValueError(f'Invalid evaluation mask: {path}')
                t['model_evaluated'] = flags.eq('true')
                if t.duplicated(['session_id', 'trial_index']).any():
                    raise ValueError(f'Duplicate trial identity in training/validation/test: {path}')
                test = t[t.split_role.eq('test')].sort_values(['session_id', 'trial_index']).reset_index(drop=True)
                if test.session_id.nunique() != 1 or test.session_id.iloc[0] != row.session_id or test.trial_index.duplicated().any():
                    raise ValueError(f'Invalid test session: {path}')
                population = t[identity].sort_values(['split_role', 'session_id', 'trial_index']).reset_index(drop=True)
                if reference is None:
                    reference = population
                else:
                    pd.testing.assert_frame_equal(reference, population, check_dtype=False, atol=1e-12, rtol=0)
                ev = t[t.model_evaluated]
                p = ev.model_probability_long.to_numpy(float)
                y = ev.choice_long.to_numpy(float)
                if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                    raise ValueError(f'Invalid evaluated probabilities/choices: {path}')
                p = np.clip(p, 1e-15, 1-1e-15)
                np.testing.assert_allclose(ev.per_trial_negative_log_likelihood, -y*np.log(p)-(1-y)*np.log1p(-p), atol=1e-10, rtol=1e-8)
                for split, prefix in [('train', 'train_nll__'), ('validation', 'validation_nll__'), ('test', 'nll__')]:
                    split_trials = ev[ev.split_role.eq(split)]
                    actual = float(split_trials.per_trial_negative_log_likelihood.mean())
                    np.testing.assert_allclose(actual, getattr(row, prefix+model), atol=1e-10, rtol=1e-9)
                    audits.append(dict(subject=subject, window_id=row.window_id, model=model, split=split,
                                       n_trials=len(split_trials), reconstructed_mean_nll=actual, matched=True))
                test_ev = ev[ev.split_role.eq('test')]
                if len(test_ev) != getattr(row, 'n__'+model):
                    raise ValueError(f'Test count differs from table: {path}')
                if model == 'm0_sensory':
                    train = ev[ev.split_role.eq('train')]
                    if set(train.session_id) & set(test_ev.session_id):
                        raise ValueError('Training/test leakage in baseline')
                    q, loss = constant_choice_nll(train.choice_long, test_ev.choice_long)
                    gain = loss - float(test_ev.per_trial_negative_log_likelihood.mean())
                    np.testing.assert_allclose(gain, row.gain_bias__m0_sensory, atol=1e-10, rtol=1e-9)
                    baselines.append(dict(subject=subject, window_id=row.window_id, session_id=row.session_id,
                                          train_n=len(train), train_p_long=q, test_n=len(test_ev),
                                          test_p_long=float(test_ev.choice_long.mean()), baseline_test_mean_nll=loss,
                                          sensory_test_mean_nll=row.nll__m0_sensory, sensory_gain=gain))
        print('Audited canonical trials:', subject, len(windows), 'windows', flush=True)
    pd.DataFrame(audits).to_csv(out/'prediction_input_audit.csv', index=False)
    pd.DataFrame(baselines).to_csv(out/'sensory_baseline_audit.csv', index=False)
    return files


# profile_figure, gain_heatmap, clean_annotation_text, readouts and render_reports keep the
# original report for pattern_rbf_single_session_train; this analysis uses render_cluster_reports.
def profile_figure(raw, gains, ll, k):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.subplots_adjust(left=.06, right=.98, bottom=.24, top=.79, hspace=.7, wspace=.34)
    means, by = pv.balanced(pd.concat([raw[KEYS], gains], axis=1), ll, FEATURES, k)
    formulas = ['L(constant choice) - L(S)', 'L(S) - L(S+A1)', 'L(S+A1) - L(S+AH)',
                'L(S) - L(S+W1)', 'L(S+W1) - L(S+WH)', 'L(S) - L(S+Bstatic)',
                'L(S+Bstatic) - L(S+Bdynamic)']
    for j, (feature, label, *_rest) in enumerate(DEFINITIONS):
        ax = axes.flat[j]
        ax.axvline(0, color='.5', lw=.9, linestyle='--')
        for c in range(1, k+1):
            for i, subject in enumerate(sorted(raw.subject.unique())):
                ax.plot(by.loc[(subject, c), feature], c+(i-1)*.12, marker=['o', '^', 's'][i],
                        color='.5', ms=5, linestyle='none')
            ax.plot(means[c-1, j], c, 'o', color=PALETTE[c-1], ms=9)
        ax.set_title(label.replace('benefit', 'gain').replace('advantage', 'contrast')+'\n'+formulas[j], fontsize=10)
        ax.set_yticks(range(1, k+1), [f'C{c}' for c in range(1, k+1)])
        ax.set_ylim(k+.5, .5)
        ax.set_xlabel('Signed mean-NLL gain (nats/trial)', fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.2)
    ax = axes.flat[-1]; ax.axis('off')
    counts = pd.crosstab(ll.cluster_label, ll.subject).reindex(index=range(1, k+1), columns=sorted(raw.subject.unique()), fill_value=0)
    table = ax.table(cellText=counts.to_numpy(), rowLabels=[f'C{c}' for c in counts.index],
                     colLabels=counts.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.6)
    ax.set_title('Sessions per mouse', fontsize=12)
    fig.suptitle(f'Seven signed predictive gains | K={k} | {len(raw)} sessions', fontsize=20)
    fig.text(.06, .90, '3 mice; descriptive means, no uncertainty intervals. Compare clusters within panels; x-axis scales differ across features.', fontsize=11)
    fig.text(.06, .065, 'Colored dots: strict equal-mouse means. Gray: MC06 circles, MC07 triangles, MC08 squares. Missing mice remain missing.\n'
             'L = mean test NLL. Positive gain favors the model on the right of each subtraction; negative gains are retained.\n'
             'S = sensory; A1/AH = single/history action; W1/WH = single/history WSLS; Bstatic/Bdynamic = static/dynamic block inference.\n'
             'Sensory baseline: constant choice probability fitted on training trials, evaluated on test trials.\n'
             'History-architecture contrasts change both memory span and integration rule; they are not isolated memory effects.', fontsize=10)
    return fig, means


def gain_heatmap(raw, z, ll, order, k):
    order = order[np.argsort(ll.cluster_label.to_numpy()[order], kind='stable')]
    fig, (mouse_ax, ax) = plt.subplots(1, 2, figsize=(16, 11), gridspec_kw={'width_ratios': [.25, 10]})
    fig.subplots_adjust(left=.08, right=.92, top=.85, bottom=.15, wspace=.1)
    subjects = sorted(raw.subject.unique())
    mouse_ax.imshow(np.array([subjects.index(s) for s in raw.subject.iloc[order]])[:, None], aspect='auto',
                    cmap=ListedColormap([pv.MOUSE_COLORS[s] for s in subjects]), vmin=0, vmax=2)
    mouse_ax.set_xticks([]); mouse_ax.set_yticks([]); mouse_ax.set_title('Mouse', fontsize=9)
    im = ax.imshow(z[order], cmap=pv.VALUE_CMAP, vmin=-3, vmax=3, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(FEATURES)), [d[1].replace('benefit', 'gain').replace('advantage', 'contrast') for d in DEFINITIONS], fontsize=10)
    ax.xaxis.tick_top(); ax.tick_params(axis='x', length=0, pad=12)
    labs = ll.cluster_label.to_numpy()[order]
    mids = [np.flatnonzero(labs == c).mean() for c in range(1, k+1)]
    ax.set_yticks(mids, [f'C{c} ({np.sum(labs == c)})' for c in range(1, k+1)])
    for border in np.flatnonzero(np.diff(labs))+.5:
        ax.axhline(border, color='.15', lw=2)
    fig.colorbar(im, ax=ax, fraction=.025, pad=.02,
                 label='Within-mouse z score: 0 = mouse average\nClipped at +/-3 before clustering')
    fig.suptitle(f'Actual seven clustering coordinates | K={k} | {len(raw)} sessions, 3 mice', fontsize=18)
    fig.text(.08, .91, 'Descriptive profiles of clusters defined by these features. One row per session.\n'
             'Equal weight per standardized coordinate; no fitted parameters enter the distance.', fontsize=11)
    fig.text(.08, .075, 'Blue = above that mouse\'s average; orange = below. A negative z score is NOT necessarily a negative NLL gain.\n'
             'Within-cluster rows use Rastermap similarity ordering, not chronological order.', fontsize=11)
    for i, subject in enumerate(subjects):
        fig.text(.08+i*.17, .035, '\u25a0 '+subject, color=pv.MOUSE_COLORS[subject], fontsize=11)
    return fig


def clean_annotation_text(fig):
    for text in fig.findobj(matplotlib.text.Text):
        value = text.get_text()
        value = value.replace('Existing cluster labels unchanged', 'Seven-feature labels; descriptive annotations')
        value = value.replace('These 12 columns', 'These 17 annotation columns')
        value = value.replace('Rows keep the previous within-cluster session order.', 'Rows follow the new within-cluster order.')
        value = value.replace('Common display: original 48 model-performance/parameter drivers',
                              'Explanatory appendix: all 50 model-performance/parameter columns (not clustering inputs)')
        if value.startswith('Shared heatmap basis'):
            value = ('Labels were estimated from seven signed test-NLL gains only, with equal coordinate weights.\n'
                     'This full-model appendix is explanatory; none of its fitted parameters enter the new clustering.\n'
                     'Within-mouse scaling and median-distance RBF settings are retained. Same-session behavior is descriptive.')
        text.set_text(value)


def readouts(raw):
    d, meta = pv.readouts(raw)
    meta.loc[meta.feature.eq('gain__m1_action_kernel'), 'label'] = 'Action hist.\nbenefit'
    meta.loc[meta.feature.eq('gain__m1_stimulus_kernel'), 'label'] = 'WSLS hist.\nbenefit'
    meta = pd.concat([meta[meta.family.eq(f)] for f in ['Sensory', 'History', 'Block context', 'Combined models']], ignore_index=True)
    return d, meta


def behavior_trials():
    trials = pd.read_csv(OLD/'reaction_time_trials.csv').drop(columns='cluster_label', errors='ignore')
    if trials.duplicated(KEYS+['trial_index']).any():
        raise ValueError('Duplicate behavior trial identities')
    return trials, trials[trials.latency_eligible.eq(True)]


def behavior_figure(kind, ll, trials, eligible, k):
    tagged = behavior.attach_labels(trials if kind == 'choice_behavior' else eligible, ll)
    stats = None
    if kind == 'choice_behavior':
        fig, _, _ = behavior.make_page(tagged, k)
    else:
        fig, stats, _ = latency.make_page(tagged, k)
        stats['variant'], stats['K'] = VARIANT, k
    fig._suptitle.set_text(f'{kind.replace("_", " ").title()} | Seven-feature RBF | K={k}')
    fig._suptitle.set_fontsize(15); behavior.annotate_mouse_support(fig, ll)
    return fig, stats


def k_selection_figure(table, eigenvalues, z, labels, optimal):
    ks = sorted(labels.K.unique())
    fig = plt.figure(figsize=(19, 17))
    gs = fig.add_gridspec(3, 6, left=.06, right=.98, top=.89, bottom=.08, hspace=.55, wspace=1.)
    excluded = table.K[~table.all_mice_in_every_cluster]
    best = table[table.optimal].iloc[0]

    def metric(ax, column, title, ylabel, better, band=None):
        for k in excluded:
            ax.axvspan(k-.4, k+.4, color='.93', lw=0, zorder=0)
        if band is not None:
            ax.fill_between(table.K, table[band[0]], table[band[1]], color=PALETTE[0], alpha=.15, lw=0,
                            label='10-90% of subsamples')
        ax.plot(table.K, table[column], '-o', color=PALETTE[0], lw=2, ms=8)
        ax.plot(optimal, best[column], '*', color=PALETTE[1], ms=20, zorder=5)
        ax.set_xticks(table.K); ax.set_xlabel('Number of clusters K'); ax.set_ylabel(ylabel)
        ax.set_title(f'{title}\n({better} is better)', loc='left', fontsize=12.5, fontweight='bold')
        if band is not None:
            ax.legend(frameon=False, fontsize=9, loc='lower left')

    ax = fig.add_subplot(gs[0, :2])
    n = min(12, len(eigenvalues)); idx = np.arange(1, n+1)
    ax.plot(idx, eigenvalues[:n], '-o', color='.25', lw=2, ms=8)
    ax.annotate('', xy=(optimal+.5, eigenvalues[optimal]), xytext=(optimal+.5, eigenvalues[optimal-1]),
                arrowprops=dict(arrowstyle='<->', color=PALETTE[1], lw=2))
    ax.text(optimal+.8, eigenvalues[optimal-1]-.1, f'Largest gap after\neigenvalue {optimal} -> K={optimal}', va='top', fontsize=10.5)
    ax.set_xticks(idx); ax.set_xlabel('Eigenvalue index'); ax.set_ylabel('Normalized-Laplacian eigenvalue')
    ax.set_title('A  Eigengap: the selection rule\n(gap after eigenvalue K)', loc='left', fontsize=12.5, fontweight='bold')
    metric(fig.add_subplot(gs[0, 2:4]), 'eigengap', 'B  Eigengap by K', 'Eigenvalue K+1 - eigenvalue K', 'higher')
    ax = fig.add_subplot(gs[0, 4:])
    ax.bar(table.K, table.min_sessions_per_mouse_cluster, color=[PALETTE[0] if ok else '.75' for ok in table.all_mice_in_every_cluster],
           width=.6)
    ax.set_xticks(table.K); ax.set_xlabel('Number of clusters K'); ax.set_ylabel('Fewest sessions of one mouse in a cluster')
    ax.set_title('C  Every mouse in every cluster?\n(0 = a cluster lacks a mouse; K excluded)', loc='left', fontsize=12.5, fontweight='bold')
    metric(fig.add_subplot(gs[1, :2]), 'stability_ari_median', 'D  Stability', f'Median ARI, {N_STABILITY} subsamples of {STABILITY_FRACTION:.0%}',
           'higher', band=('stability_ari_q10', 'stability_ari_q90'))
    metric(fig.add_subplot(gs[1, 2:4]), 'silhouette_z', 'E  Silhouette (z space)', 'Mean silhouette', 'higher')
    ax = fig.add_subplot(gs[1, 4:])
    metric(ax, 'davies_bouldin', 'F  Davies-Bouldin (z space)', 'Davies-Bouldin index', 'lower')
    for a in fig.axes:
        a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', alpha=.2)
    width = 6 // len(ks)
    for j, k in enumerate(ks):
        ax = fig.add_subplot(gs[2, j*width:(j+1)*width])
        lab = labels[labels.K.eq(k)].cluster_label.to_numpy()
        s = silhouette_samples(z, lab); y, ticks = 0, []
        for c in range(1, k+1):
            v = np.sort(s[lab == c])
            ax.barh(np.arange(y, y+len(v)), v, height=1., color=PALETTE[c-1], edgecolor='none')
            ticks.append((y+len(v)/2, f'C{c} ({len(v)})')); y += len(v)+3
        ax.axvline(0, color='.6', lw=.8); ax.axvline(s.mean(), color='.2', ls='--', lw=1.2)
        ax.set_yticks(*zip(*ticks)); ax.set_ylim(y, -3); ax.set_xlabel('Silhouette per session (z space)')
        ax.set_title(f'{"GHI"[j]}  K={k}{" (optimal)" if k == optimal else ""} | mean silhouette {s.mean():.3f}',
                     loc='left', fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'How many clusters? Optimal K={optimal}', fontsize=22, fontweight='bold', x=.06, ha='left', y=.975)
    fig.text(.06, .935, f'Rule: the largest eigengap among the K whose every cluster contains sessions from all mice (gray K are excluded). '
             f'Red star = K={optimal}.\nEvery K uses the same RBF affinity, normalized Laplacian and final k-means '
             f'({N_INIT} initializations, seed {SEED}); {len(z)} sessions.', fontsize=11.5, color='.25')
    fig.text(.06, .025, 'D-F are reported alongside the rule, not used for the choice. Silhouette and Davies-Bouldin use Euclidean distance on the clipped '
             'within-mouse z scores, the distance the RBF kernel is built from.\nCalinski-Harabasz is in k_selection.csv. Silhouette cannot test K=1. '
             'Overlapping windows are dependent; descriptive only.', fontsize=10, color='.3')
    return fig


# Column groups for the report heatmaps; family keys are DEFINITIONS[2].
FAMILIES = {'sensory': ('Sensory evidence', '#8a56c0'), 'action': ('Choice history', '#188f6e'),
            'wsls': ('Outcome feedback (WSLS)', '#c9652a'), 'context': ('Block context', '#3a72b8')}
COLUMN_LABELS = {'sensory': 'Stimulus readout\nvs constant choice', 'action_single': 'Last choice\n(1-back)',
                 'action_history_architecture': 'Longer choice\nhistory', 'wsls_single': 'Last rewarded side\n(1-back)',
                 'wsls_history_architecture': 'Longer outcome\nhistory', 'static_context': 'Static block\nprior',
                 'dynamic_switching': 'Duration-aware\nblock switching'}


def short_label(definition):
    return COLUMN_LABELS[definition[0]]


def family_headers(fig, ax, offset=40, height=18):
    """Colored family bands (offset/height in points) above a 7-column heatmap whose column labels sit on top."""
    from matplotlib.patches import Rectangle
    box, inches = ax.get_position(), fig.get_size_inches()[1]
    y0, h = box.y1 + offset/72/inches, height/72/inches
    families = [d[2] for d in DEFINITIONS]
    start = 0
    for j in range(1, len(families)+1):
        if j == len(families) or families[j] != families[start]:
            name, color = FAMILIES[families[start]]
            x0, width = box.x0 + box.width*start/len(families), box.width*(j-start)/len(families)
            fig.patches.append(Rectangle((x0+.002, y0), width-.004, h, transform=fig.transFigure, facecolor=color))
            fig.text(x0+width/2, y0+h/2, name, ha='center', va='center', color='white', fontsize=10.5, fontweight='bold')
            if j < len(families):
                ax.axvline(j-.5, color='white', lw=5)
            start = j


def mouse_legend(fig, x, y):
    for i, subject in enumerate(sorted(pv.MOUSE_COLORS)):
        fig.text(x+i*.07, y, '■ '+subject, color=pv.MOUSE_COLORS[subject], fontsize=11)


def cluster_means(data, ll, columns, k):
    """Sessions averaged within each mouse, then the mice present averaged equally; also returns mice per cluster."""
    t = data.merge(ll[['subject', 'session_id', 'cluster_label']], on=KEYS, validate='one_to_one')
    by = t.groupby(['cluster_label', 'subject'])[columns].mean()
    means = by.groupby(level='cluster_label').mean().reindex(range(1, k+1))
    return means.to_numpy(), by.groupby(level='cluster_label').size().reindex(range(1, k+1), fill_value=0).to_numpy()


def value_matrix(ax, values, columns, rows, limit, fmt, subtext=None):
    """Cluster x column heatmap on the warm/cold scale; cell shows fmt(value) and optional smaller subtext."""
    values = np.asarray(values, float)
    im = ax.imshow(np.ma.masked_invalid(values), cmap=WARM_COLD, vmin=-limit, vmax=limit, aspect='auto')
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            color = 'white' if np.isfinite(v) and abs(v) > .6*limit else '#25313b'
            ax.text(j, i-(.12 if subtext is not None else 0), fmt(v) if np.isfinite(v) else 'n/a', ha='center', va='center',
                    fontsize=13, fontweight='bold', color=color)
            if subtext is not None:
                ax.text(j, i+.24, subtext[i][j], ha='center', va='center', fontsize=9, color=color)
    ax.set_xticks(range(len(columns)), columns, fontsize=10); ax.xaxis.tick_top(); ax.tick_params(length=0, pad=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=11)
    ax.set_xticks(np.arange(-.5, len(columns)), minor=True); ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which='minor', color='white', lw=3); ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def fingerprint_page(raw, gains, z, ll, observed, k):
    """What defines each cluster (its seven coordinates) and what the mice did in those sessions."""
    counts = ll.cluster_label.value_counts().reindex(range(1, k+1)).to_numpy()
    zmean, n_mice = cluster_means(pd.concat([raw[KEYS], pd.DataFrame(z, columns=FEATURES)], axis=1), ll, FEATURES, k)
    gmean, _ = cluster_means(pd.concat([raw[KEYS], gains], axis=1), ll, FEATURES, k)
    all_mice = raw.subject.nunique()
    rows = [f'C{c}{"*" if m < all_mice else ""}  ({n} sessions)' for c, n, m in zip(range(1, k+1), counts, n_mice)]
    fig = plt.figure(figsize=(18, 11.5))
    fig.suptitle(f'K={k}: what defines each cluster?', fontsize=22, fontweight='bold', x=.05, ha='left', y=.97)
    fig.text(.05, .925, f'{len(raw)} held-out sessions, 3 mice. Clusters were formed from the seven signed gains below only '
             '(within-mouse z scores, RBF spectral clustering).', fontsize=12, color='.3')
    fig.text(.05, .865, '1  Cluster fingerprint: the seven clustering coordinates', fontsize=15, fontweight='bold')
    fig.text(.05, .842, 'Bold number and color = mean within-mouse z score, the value the clustering used (red = above the mouse\'s '
             'average, blue = below). Small number = mean signed gain in 10⁻³ nats/trial (positive favors the richer model).',
             fontsize=10, color='.3')
    ax = fig.add_axes([.17, .53, .62, .22])
    limit = max(1., float(np.ceil(np.nanmax(np.abs(zmean))*2)/2))
    im = value_matrix(ax, zmean, [short_label(d) for d in DEFINITIONS], rows, limit, lambda v: f'{v:+.2f}',
                      subtext=[[f'gain {v*1e3:+.1f}' for v in row] for row in gmean])
    family_headers(fig, ax)
    cb = fig.colorbar(im, cax=fig.add_axes([.80, .53, .01, .22])); cb.set_label('Mean z score', fontsize=10)
    side = fig.add_axes([.86, .53, .12, .22]); side.axis('off')
    table = pd.crosstab(ll.cluster_label, ll.subject).reindex(index=range(1, k+1), columns=sorted(pv.MOUSE_COLORS), fill_value=0)
    acc, _ = cluster_means(observed, ll, ['balanced_accuracy'], k)
    body = [[str(v) for v in table.loc[c]] + [f'{acc[c-1, 0]*100:.1f}' if np.isfinite(acc[c-1, 0]) else 'n/a'] for c in range(1, k+1)]
    t = side.table(cellText=body, colLabels=[s[2:] for s in table.columns]+['Acc.%'],
                   rowLabels=[f'C{c}' for c in range(1, k+1)], cellLoc='center', bbox=[0, 0, 1, .85])
    t.auto_set_font_size(False); t.set_fontsize(10)
    side.text(.5, .93, 'Sessions per mouse\n& balanced accuracy', ha='center', fontsize=10, fontweight='bold')
    fig.text(.05, .42, '2  What did the mice do in these sessions?', fontsize=15, fontweight='bold')
    fig.text(.05, .397, 'Equal-mouse means in percentage points; red = larger effect, blue = smaller or reversed. '
             'Descriptive: behavior did not enter the clustering.', fontsize=10, color='.3')
    ax = fig.add_axes([.17, .13, .62, .2])
    bmean, _ = cluster_means(observed, ll, [m for m, _ in pv.BEHAVIOR], k)
    im = value_matrix(ax, bmean*100, [n for _, n in pv.BEHAVIOR], rows, 30, lambda v: f'{v:.1f}')
    cb = fig.colorbar(im, cax=fig.add_axes([.80, .13, .01, .2]), extend='both'); cb.set_label('Effect (pp)', fontsize=10)
    missing = [f'C{c} (no {", ".join(table.columns[table.loc[c] == 0])} sessions)' for c in range(1, k+1) if n_mice[c-1] < all_mice]
    if missing:
        fig.text(.05, .075, '* Mean over the mice present only: ' + '; '.join(missing) + '. Do not read it as a strategy shared by all mice.',
                 fontsize=10, color='#a32626')
    fig.text(.05, .045, 'Means: sessions averaged within each mouse, then mice averaged equally. '
             'Context shift compares matched stimulus bins; recovery compares newly common trials 3-4 with 0-1 after adjusted transitions.',
             fontsize=9.5, color='.3')
    return fig, gmean, n_mice


def gain_detail_page(raw, gains, ll, k):
    """The seven gains in their own units, with each mouse shown separately."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.subplots_adjust(left=.06, right=.98, bottom=.17, top=.8, hspace=.75, wspace=.34)
    _, by = pv.balanced(pd.concat([raw[KEYS], gains], axis=1), ll, FEATURES, k)
    means, _ = cluster_means(pd.concat([raw[KEYS], gains], axis=1), ll, FEATURES, k)
    model_names = {'constant_choice': 'constant choice', 'm0_sensory': 'S',
                   'm1_previous_action_bias': 'S+A1', 'm1_action_kernel': 'S+AH',
                   'm1_previous_wsls_bias': 'S+W1', 'm1_stimulus_kernel': 'S+WH',
                   'm2_static': 'S+Bstatic', 'm2_dynamic': 'S+Bdynamic'}
    formulas = [f'L({model_names[d[3]]}) - L({model_names[d[4]]})' for d in DEFINITIONS]
    subjects = sorted(raw.subject.unique())
    for j, definition in enumerate(DEFINITIONS):
        ax = axes.flat[j]
        ax.axvline(0, color='.5', lw=.9, linestyle='--')
        for c in range(1, k+1):
            for i, subject in enumerate(subjects):
                ax.plot(by.loc[(subject, c), definition[0]], c+(i-1)*.14, marker='o', ms=5, linestyle='none',
                        color=pv.MOUSE_COLORS[subject], alpha=.75)
            ax.plot(means[c-1, j], c, 'D', color='.1', ms=8)
        ax.set_title(short_label(definition).replace('\n', ' ')+'\n'+formulas[j], fontsize=11)
        ax.set_yticks(range(1, k+1), [f'C{c}' for c in range(1, k+1)]); ax.set_ylim(k+.5, .5)
        ax.set_xlabel('Signed gain (nats/trial)', fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', alpha=.2)
    for unused in list(axes.flat)[len(DEFINITIONS):-1]:
        unused.axis('off')
    ax = axes.flat[-1]; ax.axis('off')
    ax.text(0, .95, 'How to read', fontsize=12, fontweight='bold', va='top', transform=ax.transAxes)
    ax.text(0, .8, 'Black diamond: equal-mouse mean.\nColored dots: one mouse each.\nRight of the dashed line:\n'
            'the richer model predicts better.\nX scales differ between panels.', fontsize=10.5, va='top', transform=ax.transAxes)
    for subject in subjects:
        ax.plot([], [], 'o', color=pv.MOUSE_COLORS[subject], label=subject)
    ax.plot([], [], 'D', color='.1', label='Equal-mouse mean')
    ax.legend(loc='lower left', frameon=False, fontsize=10.5)
    fig.suptitle(f'K={k}: the seven signed gains by cluster (raw units)', fontsize=20, fontweight='bold', x=.06, ha='left')
    fig.text(.06, .895, 'L = mean held-out NLL. S = sensory; A1/AH = single/history action; W1/WH = single/history WSLS; '
             'Bstatic/Bdynamic = static/dynamic block inference.', fontsize=11, color='.3')
    fig.text(.06, .05, 'Sensory gain is against a constant choice probability fitted on training trials. History contrasts change '
             'both memory span and integration rule; they are not isolated memory effects.', fontsize=10, color='.3')
    return fig


def pooled_z(gains):
    """Each gain standardized across all sessions and mice (sample SD), clipped at +/-3; display only."""
    return np.clip(((gains-gains.mean())/gains.std(ddof=1)).to_numpy(), -3, 3)


def session_heatmap_page(raw, gains, ll, order, k):
    """Every session's seven gains on one scale shared by all mice, grouped by cluster.

    The clustering used within-mouse z scores (a mouse's own average = 0). Here each gain is
    standardized across all sessions instead, so a red column means high relative to every session.
    """
    z = pooled_z(gains)
    order = order[np.argsort(ll.cluster_label.to_numpy()[order], kind='stable')]
    labs = ll.cluster_label.to_numpy()[order]
    fig, (mouse_ax, ax) = plt.subplots(1, 2, figsize=(16, 11), gridspec_kw={'width_ratios': [.25, 10]})
    fig.subplots_adjust(left=.1, right=.9, top=.8, bottom=.1, wspace=.08)
    subjects = sorted(raw.subject.unique())
    # pcolormesh (one vector cell per value) instead of imshow, so the PDF has no raster images;
    # edges at -.5 keep imshow's cell-centre coordinates for ticks and cluster borders.
    rows = np.arange(len(order)+1)-.5
    mouse_ax.pcolormesh([-.5, .5], rows, np.array([subjects.index(s) for s in raw.subject.iloc[order]])[:, None],
                        cmap=ListedColormap([pv.MOUSE_COLORS[s] for s in subjects]), vmin=0, vmax=len(subjects)-1,
                        edgecolors='face', linewidth=0)
    mouse_ax.set_ylim(rows[-1], rows[0])
    mouse_ax.set_xticks([]); mouse_ax.set_title('Mouse', fontsize=10)
    mouse_ax.set_yticks([np.flatnonzero(labs == c).mean() for c in range(1, k+1)],
                        [f'C{c}\n{np.sum(labs == c)} sessions' for c in range(1, k+1)], fontsize=11)
    im = ax.pcolormesh(np.arange(len(FEATURES)+1)-.5, rows, z[order], cmap=WARM_COLD, vmin=-3, vmax=3, edgecolors='face', linewidth=0)
    ax.set_ylim(rows[-1], rows[0]); ax.set_xlim(-.5, len(FEATURES)-.5)
    ax.set_xticks(range(len(FEATURES)), [short_label(d) for d in DEFINITIONS], fontsize=10)
    ax.xaxis.tick_top(); ax.tick_params(axis='x', length=0, pad=10)
    ax.set_yticks([])
    for border in np.flatnonzero(np.diff(labs))+.5:
        ax.axhline(border, color='.1', lw=2.5)
    cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.02, ticks=[-3, 0, 3])
    cb.solids.set_rasterized(False); cb.solids.set_edgecolor('face')  # matplotlib rasterizes >=50-step colorbars
    cb.ax.set_yticklabels(['-3\nlower', '0\naverage\nsession', '+3\nhigher']); cb.set_label('z score across all sessions and mice (clipped)', fontsize=10)
    family_headers(fig, ax)
    fig.suptitle(f'K={k}: every session\'s seven gains, one scale for all mice', fontsize=20, fontweight='bold', x=.1, ha='left')
    fig.text(.1, .91, 'One row per session, grouped by cluster; within a cluster rows are ordered by similarity (Rastermap), not by date.',
             fontsize=11, color='.3')
    fig.text(.1, .05, 'Each column is standardized across all 92 sessions of all mice: red = higher gain than the average session, '
             'blue = lower, regardless of mouse. The clustering itself used\nwithin-mouse z scores (page 1). Blue is not necessarily '
             'a negative gain: see the raw-units page.', fontsize=11, color='.3')
    mouse_legend(fig, .1, .02)
    return fig, order


def methods_page():
    """How each model's history term is computed and how the seven coordinates are derived from the fits."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('How the seven clustering coordinates are calculated', fontsize=22, fontweight='bold', x=.05, ha='left', y=.975)
    fig.text(.05, .935, 'Each coordinate is the held-out predictive gain of one model over a simpler one. All models share one Bayesian '
             'choice model; they differ only in the history or block term.', fontsize=12, color='.3')

    def section(y, title):
        fig.text(.05, y, title, fontsize=15, fontweight='bold')

    def line(y, x, text, size=13, **kw):
        fig.text(x, y, text, fontsize=size, va='center', **kw)

    section(.885, '1  Shared choice model')
    line(.85, .07, r'Noisy duration measurement:   $X_t \sim \mathcal{N}(s_t,\ w\,s_t^2 + s_0^2)$')
    line(.815, .07, r'Decision variable:   $\Lambda(X_t) = \log\frac{f_{long}(X_t)}{f_{short}(X_t)} + \mathrm{logit}(\pi_t) - \log\kappa$'
         r',   $q_t = P(\Lambda(X_t) > 0\ |\ s_t)$   (choose long when $\Lambda > 0$)')
    line(.78, .07, r'Lapse:   $p_t = (1-\epsilon_t)\,m_t + \epsilon_t/2$,   with $m_t = q_t$ unless a 1-back term is mixed in;   '
         r'$\epsilon_t$ = short- or long-trial lapse rate')

    section(.725, '2  What each model adds (prior $\\pi_t$ or mixed term)')
    rows = [
        ('sensory', 'S', 'Sensory only', r'$\pi_t = 0.5$', 'no history'),
        ('action', 'S+A1', 'Last choice', r'$m_t = (1-\eta)\,q_t + \eta\,A_{t-1}$',
         '$A$ = the mouse\'s last valid choice (1 = long)'),
        ('action', 'S+AH', 'Choice history kernel', r'$\pi_{t+1} = \pi_t + \alpha\,(A_t - \pi_t)$   $\Rightarrow$   '
         r'$\pi_t = (1-\alpha)^n\,0.5 + \alpha\sum_{i=1}^{n}(1-\alpha)^{n-i}A_{k_i}$', 'exponential kernel over past choices'),
        ('wsls', 'S+W1', 'Last rewarded side', r'$m_t = (1-\eta)\,q_t + \eta\,C_{t-1}$',
         '$C$ = last rewarded side: stay after a win, shift after a loss'),
        ('wsls', 'S+WH', 'Outcome history kernel', r'$\pi_{t+1} = \pi_t + \alpha\,(C_t - \pi_t)$',
         'exponential kernel over past rewarded sides'),
        ('context', 'S+Bstatic', 'Static block prior', r'$\pi_t = \sum_z P(z_t\ |\ C_{1:t-1})\,\theta_z$,   '
         r'$\theta = (0.5,\ \sigma(-\delta_0),\ \sigma(\delta_0))$,   $h_B = \sigma(a_h)$, $h_N = h_B/2$',
         'hidden block z in {neutral, short, long}'),
        ('context', 'S+Bdynamic', 'Duration-aware switching', r'as S+Bstatic with   $h_B(r) = \sigma(a_h + b_h\,(r - 23))$',
         'r = trials since the last block switch'),
    ]
    for i, (family, short, name, math, note) in enumerate(rows):
        y = .685 - i*.052
        line(y, .07, short, color=FAMILIES[family][1], fontweight='bold')
        line(y, .16, name, size=12)
        line(y, .33, math)
        line(y-.019, .33, note, size=10, color='.4')
    fig.text(.07, .31, 'Kernel models put the history into the prior; 1-back models mix it into the output, so the sensory readout '
             'stays intact. Each history state resets to 0.5 at the start of a session.', fontsize=10.5, color='.3')

    section(.26, '3  From model fits to clustering coordinates')
    line(.225, .07, r'Held-out loss:   $L(M) = -\frac{1}{n}\sum_t\,[\,y_t \log p_t + (1-y_t)\log(1-p_t)\,]$   '
         'on the test session, after fitting the preceding window')
    line(.19, .07, r'Coordinate = $L(\mathrm{simpler}) - L(\mathrm{richer})$:   '
         r'$L(\mathrm{const}) - L(S)$,  $L(S) - L(S{+}A1)$,  $L(S{+}A1) - L(S{+}AH)$,  $L(S) - L(S{+}W1)$,  '
         r'$L(S{+}W1) - L(S{+}WH)$,  $L(S) - L(S{+}B_{static})$,  $L(S{+}B_{static}) - L(S{+}B_{dynamic})$', size=12)
    line(.155, .07, r'Scaling:   $z = \mathrm{clip}\left((g - \bar g_{mouse})\,/\,\mathrm{sd}_{mouse},\ -3,\ 3\right)$   '
         r'Affinity:   $A_{ij} = \exp\left(-\Vert z_i - z_j\Vert^2\,/\,(2\,\mathrm{median}\,\Vert z_i - z_j\Vert^2)\right)$')
    line(.12, .07, f'Spectral clustering on A: normalized Laplacian, then k-means ({N_INIT} initializations) on its first K eigenvectors.')
    fig.text(.05, .05, 'Positive coordinate = the richer model predicts the held-out choices better (nats/trial). const = constant choice '
             'probability fitted on training trials. History contrasts (AH vs A1, WH vs W1)\nchange both memory span and integration rule, so '
             'they are architecture comparisons, not isolated memory effects. Parameters are fitted per window; only the losses enter the clustering.',
             fontsize=10.5, color='.3')
    return fig


# Original latency.make_page conditions, in its order (keeps reaction_time_binned.csv identical), with readable titles.
RT_ROWS = [
    ('Block', [('All trials', 'All trials'), ('Neutral block', 'Neutral block'), ('Short block', 'Short block'),
               ('Long block', 'Long block')]),
    ('Stimulus expectation and outcome', [('Common trials', 'Common stimulus'), ('Rare trials', 'Rare stimulus'),
                                          ('Reward · non-neutral', 'Reward (biased blocks)'),
                                          ('Punish · non-neutral', 'Punish (biased blocks)')]),
    ('Block x outcome', [('Short · reward', 'Short block · reward'), ('Short · punish', 'Short block · punish'),
                            ('Long · reward', 'Long block · reward'), ('Long · punish', 'Long block · punish')]),
    ('Expectation x outcome', [('Common · reward', 'Common · reward'), ('Common · punish', 'Common · punish'),
                               ('Rare · reward', 'Rare · reward'), ('Rare · punish', 'Rare · punish')]),
]


def rt_masks(tagged):
    biased = tagged.true_block.isin(['S', 'L'])
    reward, punish = tagged.outcome_strict.eq('Reward'), tagged.outcome_strict.eq('Punish')
    common, rare = tagged.is_common.eq(1), tagged.is_common.eq(0)
    return {'All trials': pd.Series(True, index=tagged.index), 'Neutral block': tagged.true_block.eq('N'),
            'Short block': tagged.true_block.eq('S'), 'Long block': tagged.true_block.eq('L'),
            'Common trials': common, 'Rare trials': rare, 'Reward · non-neutral': biased & reward,
            'Punish · non-neutral': biased & punish, 'Common · reward': common & reward, 'Common · punish': common & punish,
            'Rare · reward': rare & reward, 'Rare · punish': rare & punish,
            'Short · reward': tagged.true_block.eq('S') & reward, 'Short · punish': tagged.true_block.eq('S') & punish,
            'Long · reward': tagged.true_block.eq('L') & reward, 'Long · punish': tagged.true_block.eq('L') & punish}


def reaction_time_page(ll, eligible, k):
    """Reaction time by stimulus duration, one line per cluster across short and long stimuli; own y range per panel."""
    tagged = behavior.attach_labels(eligible, ll)
    masks = rt_masks(tagged)
    fig, axes = plt.subplots(len(RT_ROWS), 4, figsize=(20, 5*len(RT_ROWS)))
    fig.subplots_adjust(left=.07, right=.98, top=.86, bottom=.08, hspace=.55, wspace=.3)
    records = []
    offsets = {c: (c-(k+1)/2)*.03 for c in range(1, k+1)}
    for r, (row_title, conditions) in enumerate(RT_ROWS):
        fig.text(.07, axes[r, 0].get_position().y1+.035, row_title, fontsize=14, fontweight='bold')
        for ax, (condition, title) in zip(axes[r], conditions):
            for c in range(1, k+1):
                data = tagged[masks[condition] & tagged.cluster_label.eq(c)]
                summary = latency.binned_summary(data)
                if summary.empty:
                    continue
                records.append(summary.assign(K=k, cluster=c, condition=condition))
                seg = summary.sort_values('isi_mean')
                x = seg.isi_mean+offsets[c]
                ax.fill_between(x, seg.latency_mean-seg.latency_sem, seg.latency_mean+seg.latency_sem,
                                color=PALETTE[c-1], alpha=.15, lw=0)
                ax.plot(x, seg.latency_mean, '-o', color=PALETTE[c-1], lw=2, ms=4.5)
            ax.axvline(1.25, color='.6', ls=':', lw=1)
            ax.set_xticks([.5, 1, 1.5, 2]); ax.set_xlim(.1, 2.4)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Stimulus duration (s)', fontsize=10)
            ax.spines[['top', 'right']].set_visible(False); ax.grid(False); ax.tick_params(labelsize=9)
            ax.set_ylabel('Reaction time (s)', fontsize=10)
    sessions = ll.cluster_label.value_counts()
    handles = [plt.Line2D([], [], color=PALETTE[c-1], lw=2.5, marker='o', label=f'C{c} ({sessions.get(c, 0)} sessions)')
               for c in range(1, k+1)]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.98, .955), ncol=k, frameon=False, fontsize=12)
    fig.suptitle(f'K={k}: reaction time by stimulus duration', fontsize=20, fontweight='bold', x=.07, ha='left', y=.975)
    fig.text(.07, .925, 'Mean ± SEM across session means, four duration bins per stimulus category (short left, long right); '
             'dotted line = category boundary (1.25 s). Each panel has its own y range.', fontsize=11, color='.3')
    fig.text(.07, .025, 'First lick at or after choice-window onset. Warm-up, invalid latencies and pre-choice-lick trials excluded; '
             'no latency trimming. Reward/punish = outcome of the current trial. Held-out test sessions only.', fontsize=10, color='.3')
    behavior.annotate_mouse_support(fig, ll)
    stats = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    stats['variant'], stats['K'] = VARIANT, k
    return fig, stats


def equal_row_heights(fig, rows=(0, 0, 0, 1, 1, 2, 2, 3, 3), top=1.6, gap=1.1, bottom=1.2):
    """Give every row of the choice-behavior page the psychometric panels' height.

    rows[i] is the row of fig.axes[i]; by default behavior.make_page's 3 psychometric, 4 transition
    (2 rows) and 2 rare-trial panels. Widths and horizontal positions are kept.
    """
    axes = fig.axes
    if len(axes) != len(rows):
        raise ValueError('Unexpected choice-behavior layout')
    width, height = fig.get_size_inches()
    panel = axes[0].get_position().height*height
    boxes = [a.get_position() for a in axes]
    n = max(rows)+1
    total = top + n*panel + (n-1)*gap + bottom
    fig.set_size_inches(width, total)
    for a, box, row in zip(axes, boxes, rows):
        y0 = total - top - panel - row*(panel+gap)
        a.set_position([box.x0, y0/total, box.width, panel/total])
    return fig


def choice_behavior_page(ll, trials, eligible, k):
    """behavior.make_page plus a 0.5 reference line and one short- vs long-block psychometric panel per cluster."""
    fig, _ = behavior_figure('choice_behavior', ll, trials, eligible, k)
    tagged = behavior.attach_labels(trials, ll)
    valid = tagged[tagged.model_evaluated.astype(bool) & tagged.choice_long.isin([0, 1]) & tagged.warm_up.eq(0)]
    psych = fig.axes[:3]
    for ax in psych:
        ax.axhline(.5, color='.6', lw=.8, ls='--', zorder=0)
    xs = np.linspace(.15, 2.35, 200)
    for c in range(1, k+1):
        if k <= len(psych):
            box = psych[c-1].get_position()
        else:
            # The source page has three pooled psychometric panels regardless of K.
            # Lay out the per-cluster block panels across their full horizontal span.
            from matplotlib.transforms import Bbox
            left, right = psych[0].get_position(), psych[-1].get_position()
            gap = .025
            width = (right.x1-left.x0-gap*(k-1))/k
            box = Bbox.from_bounds(left.x0+(c-1)*(width+gap), left.y0, width, left.height)
        ax = fig.add_axes(box)
        g = valid[valid.cluster_label.eq(c)]
        for block, name, line, face in [('S', 'Short block', '--', 'white'), ('L', 'Long block', '-', PALETTE[c-1])]:
            b_trials = g[g.true_block.eq(block)].copy()
            if b_trials.empty:
                continue
            b_trials['_bin'] = behavior.assign_bins(b_trials)
            b = behavior.bin_stats(b_trials.assign(v=b_trials.choice_long.astype(float)), 'v')
            ax.errorbar(b[:, 0], b[:, 1], yerr=b[:, 2], fmt='o', color=PALETTE[c-1], mfc=face, ms=5, capsize=2, lw=1.1,
                        label=f'{name} ({len(b_trials)} trials)')
            popt = behavior.fit_data(b_trials.duration_s.to_numpy(float), b_trials.choice_long.to_numpy(float))
            if np.isfinite(popt).all():
                ax.plot(xs, behavior.logist(xs, *popt), color=PALETTE[c-1], lw=1.8, ls=line)
        ax.axhline(.5, color='.6', lw=.8, ls='--', zorder=0); ax.axvline(1.25, color='.6', lw=.8, ls=':')
        ax.set_ylim(-.03, 1.03); ax.set_box_aspect(1)
        ax.set_title(f'Cluster {c}: short vs long block', fontsize=12)
        ax.set_xlabel('Stimulus duration (s)'); ax.set_ylabel('P(choose long)')
        ax.legend(frameon=False, fontsize=8, loc='upper left'); behavior.style(ax)
    return equal_row_heights(fig, rows=(0, 0, 0, 2, 2, 3, 3, 4, 4) + (1,)*k)


def render_cluster_reports(raw, gains, z, labels, observed, table, eigenvalues, optimal, out):
    """k-selection page plus one six-page report per K, all specific to the seven-gain clustering."""
    with matplotlib.rc_context(EDITABLE_PDF):
        return _render_cluster_reports(raw, gains, z, labels, observed, table, eigenvalues, optimal, out)


def _render_cluster_reports(raw, gains, z, labels, observed, table, eigenvalues, optimal, out):
    ks = sorted(labels.K.unique())
    fig = k_selection_figure(table, eigenvalues, z, labels, optimal)
    fig.savefig(out/'k_selection.pdf'); fig.savefig(out/'k_selection.png', dpi=120); plt.close(fig)
    order = raster_order(z)
    trials, eligible = behavior_trials()
    pages, gain_profiles, orders, bins = [], [], [], []
    for k in ks:
        ll = labels[labels.K.eq(k)].copy()
        with PdfPages(out/f'K{k}_report.pdf') as pdf:
            fig, means, n_mice = fingerprint_page(raw, gains, z, ll, observed, k)
            gain_profiles += [dict(K=k, cluster_label=c, n_mice=n_mice[c-1], **dict(zip(FEATURES, means[c-1])))
                              for c in range(1, k+1)]
            figures = [('1_fingerprint', fig), ('2_methods', methods_page()),
                       ('3_gains_raw_units', gain_detail_page(raw, gains, ll, k))]
            fig, session_order = session_heatmap_page(raw, gains, ll, order, k)
            figures.append(('4_session_heatmap', fig))
            # Fonts are fixed when text is created, so the Illustrator copy is drawn inside its own rc context.
            with matplotlib.rc_context(ILLUSTRATOR):
                vector, _ = session_heatmap_page(raw, gains, ll, order, k)
                vector.savefig(out/f'K{k}_4_session_heatmap_illustrator.pdf'); plt.close(vector)
            rows = raw[KEYS].copy(); rows['K'], rows['display_rank'] = k, np.argsort(session_order)+1; orders.append(rows)
            figures.append(('5_choice_behavior', choice_behavior_page(ll, trials, eligible, k)))
            fig, stats = reaction_time_page(ll, eligible, k)
            bins.append(stats); figures.append(('6_reaction_time', fig))
            for name, fig in figures:
                pdf.savefig(fig); fig.savefig(out/f'K{k}_{name}.png', dpi=100); plt.close(fig)
                pages.append(dict(K=k, page=len([p for p in pages if p['K'] == k])+1, content=name.split('_', 1)[1]))
    pd.DataFrame(pages).to_csv(out/'report_pages.csv', index=False)
    pd.DataFrame(gain_profiles).to_csv(out/'gain_profiles_equal_mouse.csv', index=False)
    pd.concat(orders).to_csv(out/'session_heatmap_order.csv', index=False)
    pd.concat(bins).to_csv(out/'reaction_time_binned.csv', index=False)
    (out/'reaction_time_input_audit.csv').write_bytes((OLD/'reaction_time_input_audit.csv').read_bytes())
    return {**{f'K{k}_report.pdf': 6 for k in ks}, 'k_selection.pdf': 1}


def render_reports(raw, values, manifest, gains, z, labels, profiles, observed, out):
    order = raster_order(z)
    d, meta = readouts(raw)
    d.to_csv(out/'visual_readouts.csv', index=False); meta.to_csv(out/'visual_readout_manifest.csv', index=False)
    pages, orders, gain_profiles = [], [], []
    with PdfPages(out/'clusters.pdf') as pdf:
        for k in (3, 2):
            ll = labels[labels.K.eq(k)].copy()
            fig, means = profile_figure(raw, gains, ll, k)
            pdf.savefig(fig); fig.savefig(out/f'gain_profiles_K{k}.png', dpi=120); plt.close(fig)
            pages.append(dict(page=len(pages)+1, K=k, analysis='signed_gain_profiles'))
            for c in range(1, k+1):
                gain_profiles.append(dict(K=k, cluster_label=c, **dict(zip(FEATURES, means[c-1]))))
            fig = gain_heatmap(raw, z, ll, order, k)
            pdf.savefig(fig); fig.savefig(out/f'gain_heatmap_K{k}.png', dpi=120); plt.close(fig)
            pages.append(dict(page=len(pages)+1, K=k, analysis='seven_feature_heatmap'))
            fig = pv.summary_page(raw, d, meta, ll, observed, VARIANT, k)
            fig.set_size_inches(25, 13); clean_annotation_text(fig)
            pdf.savefig(fig); fig.savefig(out/f'clusters_K{k}.png', dpi=100); plt.close(fig)
            pages.append(dict(page=len(pages)+1, K=k, analysis='readable_summary'))
            fig, oo = pv.session_page(d, meta, ll, order, VARIANT, k)
            fig.set_size_inches(25, 12); clean_annotation_text(fig)
            pdf.savefig(fig); plt.close(fig)
            pages.append(dict(page=len(pages)+1, K=k, analysis='readable_sessions'))
            rows = raw[KEYS].copy(); rows['variant'], rows['K'], rows['display_rank'] = VARIANT, k, np.argsort(oo)+1
            orders.append(rows)
        display = driver_manifest(manifest)
        display.to_csv(out/'annotation_feature_manifest.csv', index=False)
        fig = pr.cluster_page(raw, values, display, labels[labels.K.eq(3)].cluster_label.to_numpy(), order, VARIANT, 3, profiles)
        fig.set_size_inches(28, 14); fig.axes[2].images[0].set_cmap(pv.VALUE_CMAP)
        clean_annotation_text(fig); pdf.savefig(fig); plt.close(fig)
        pages.append(dict(page=len(pages)+1, K=3, analysis='all_model_annotation_appendix'))
    pd.DataFrame(pages).to_csv(out/'cluster_pages.csv', index=False)
    pd.concat(orders).to_csv(out/'visual_session_order.csv', index=False)
    pd.DataFrame(gain_profiles).to_csv(out/'gain_profiles_equal_mouse.csv', index=False)
    trials, eligible = behavior_trials()
    pages, bins = [], []
    with PdfPages(out/'cluster_behavior_pooled.pdf') as pdf:
        for kind in ('choice_behavior', 'reaction_time'):
            for k in (3, 2):
                ll = labels[labels.K.eq(k)]
                fig, stats = behavior_figure(kind, ll, trials, eligible, k)
                if stats is not None:
                    bins.append(stats)
                pdf.savefig(fig); fig.savefig(out/f'{kind}_K{k}.png', dpi=100); plt.close(fig)
                pages.append(dict(page=len(pages)+1, K=k, analysis=kind, n_sessions=len(raw)))
    pd.DataFrame(pages).to_csv(out/'behavior_pages.csv', index=False)
    pd.concat(bins).to_csv(out/'reaction_time_binned.csv', index=False)
    (out/'reaction_time_input_audit.csv').write_bytes((OLD/'reaction_time_input_audit.csv').read_bytes())


def render_sequences(labels, out):
    """Original E/F collectors and K=3 figures; no historical population assertions."""
    dest = out/'sequence_performance'; dest.mkdir(exist_ok=True)
    ll = labels[labels.K.eq(3)].copy()
    with patch.object(seq, 'LEGACY_OUT', OLD):
        data, audit = seq.load_trials(ll)
    source = seq.source_module()
    methods = json.loads((ROOT/'version1/sliding_window_pattern_exploration/sequence_performance/methods.json').read_text())
    actual_settings = dict(window_size=source.WINDOW_SIZE, window_step=source.WINDOW_STEP,
                           max_window_center=source.MAX_WINDOW_CENTER)
    if any(methods[key] != value for key, value in actual_settings.items()):
        raise ValueError('Sequence window settings differ from the archived RBF analysis')
    e = seq.collect_strict_common_events(data, source)
    f = source.collect_sequence_events(data, seq.adjusted_block_start)
    ew = seq.tag_events(source.collect_complete_train_sliding_events(e), data)
    fw = seq.tag_events(source.collect_complete_train_sliding_events(f), data)
    summary = pd.concat([seq.summarize(ew, 'E'), seq.summarize(fw, 'F')], ignore_index=True)
    for name, frame in [('session_input_audit', audit), ('common_outcome_window_events', ew),
                        ('rare_sequence_window_events', fw), ('window_summary', summary), ('session_clusters_used', ll)]:
        frame.to_csv(dest/(name+'.csv'), index=False)
    # Categorical value_counts includes zero for absent mice: legacy renderer then
    # reports zero, rather than falling back to the pooled count for that mouse.
    display_labels = ll.copy()
    display_labels['subject'] = pd.Categorical(display_labels.subject, categories=seq.SUBJECTS)
    seq.make_figures(summary, display_labels, dest)
    methods.update(actual_settings)
    methods.update(variant=VARIANT, K=3, session_counts=ll.cluster_label.value_counts().sort_index().to_dict(),
                   selected_label_file_sha256=hashlib.sha256((out/'selected_session_clusters.csv').read_bytes()).hexdigest(),
                   raw_source=str(seq.RAW_ROOT/'block_sequence_epoch_analysis.py'),
                   source_sha256=hashlib.sha256((seq.RAW_ROOT/'block_sequence_epoch_analysis.py').read_bytes()).hexdigest())
    (dest/'methods.json').write_text(json.dumps(methods, indent=2)+'\n')
    (dest/'README.md').write_text('Same E/F sequence definitions and plotting procedure as archived RBF analysis; newly assigned seven-feature K=3 labels.\n'
                                  'Overview: 3 pages. Details: 9 pages, also split into 3 per-cluster PDFs. See methods.json for event pooling and exclusions.\n')


def run_analysis(out):
    OUT = out
    if {s.label for s in MODEL_SPECS} != EXPECTED:
        raise ValueError('The model registry changed; review the frozen 13-model annotation roster')
    OUT.mkdir(parents=True, exist_ok=True)
    raw, values, manifest, _ = features.collect()
    if len(raw) != 92:
        raise ValueError('Expected the same 92 held-out sessions')
    print('Collected 92 sessions; checking paired source predictions', flush=True)
    sources = audit_predictions(raw, OUT)
    gains, used = gain_features(raw)
    z, scaling = standardized(gains, raw.subject.to_numpy())
    affinity, bandwidth = pe.gaussian_affinity(z)
    np.save(OUT/'clustering_matrix.npy', z); np.save(OUT/'affinity_rbf_seven_features.npy', affinity)
    raw.to_csv(OUT/'window_metrics_parameters.csv', index=False)
    pd.concat([raw[features.IDS], gains], axis=1).to_csv(OUT/'signed_feature_values.csv', index=False)
    pd.concat([raw[features.IDS], pd.DataFrame(z, columns=FEATURES)], axis=1).to_csv(OUT/'standardized_feature_values.csv', index=False)
    used.to_csv(OUT/'feature_manifest.csv', index=False); scaling.to_csv(OUT/'feature_scaling.csv', index=False)
    gains.corr().to_csv(OUT/'raw_feature_correlations.csv')
    ref = pd.read_csv(OLD/'session_clusters.csv')
    previous = pd.read_csv(PREVIOUS/'selected_session_clusters.csv')
    with patch.object(pe, 'LEGACY_OUT', OLD):
        observed = pe.session_metrics(raw)
    observed.to_csv(OUT/'observed_session_metrics.csv', index=False)
    selection, eigenvalues, candidates, optimal = k_selection(z, affinity, raw.subject.to_numpy())
    selection.to_csv(OUT/'k_selection.csv', index=False)
    print('K selection: optimal', optimal, 'by eigengap among K with every mouse in every cluster', flush=True)
    labels, summaries, profiles, contrasts, parameter_profiles = [], [], [], [], []
    keys = pd.MultiIndex.from_frame(raw[KEYS])
    for k in sorted({*REFERENCE_K, *EXTRA_REPORT_K, optimal}):
        lab = candidates[k]
        if k in REFERENCE_K:
            reference = ref[ref.K.eq(k)].set_index(KEYS).reindex(keys).cluster_label.to_numpy()
            prev = previous[previous.K.eq(k)].set_index(KEYS).reindex(keys).cluster_label.to_numpy()
            if not np.isfinite(reference).all() or not np.isfinite(prev).all():
                raise ValueError('Missing reference session labels')
            lab, mapping = pe.align_labels(lab, reference, k)
            ari = dict(ari_to_reference=float(adjusted_rand_score(reference, lab)),
                       ari_to_13_model_rbf=float(adjusted_rand_score(prev, lab)))
        else:
            lab, mapping = size_order(lab)
            ari = dict(ari_to_reference=np.nan, ari_to_13_model_rbf=np.nan)
        ll = raw[features.IDS].copy(); ll['variant'], ll['K'], ll['cluster_label'] = VARIANT, k, lab; labels.append(ll)
        tagged = observed.copy(); tagged['cluster_label'] = lab
        p, c = pe.behavior_profiles(tagged, k)
        p['variant'], p['K'] = VARIANT, k; c['variant'], c['K'] = VARIANT, k
        profiles.append(p); contrasts.append(c)
        sizes = pd.Series(lab).value_counts().reindex(range(1, k+1)).astype(int).tolist()
        summaries.append(dict(variant=VARIANT, K=k, optimal=k == optimal, method='rbf', n_features=len(FEATURES), n_sessions=len(raw),
                              sizes='/'.join(map(str, sizes)), bandwidth=bandwidth,
                              silhouette_z=float(silhouette_score(z, lab)), **ari,
                              label_mapping=json.dumps(mapping), **mouse_checks(lab, raw.subject.to_numpy())))
        group = raw.copy(); group['cluster_label'] = lab
        for cluster, g in group.groupby('cluster_label'):
            for col in raw.columns:
                if col.startswith(('nll__', 'param__')):
                    parameter_profiles.append(dict(variant=VARIANT, K=k, cluster_label=cluster, feature=col,
                                                   median=g[col].median(), q25=g[col].quantile(.25), q75=g[col].quantile(.75)))
        print('Seven-feature RBF:', k, sizes, flush=True)
    labels = pd.concat(labels, ignore_index=True); profiles = pd.concat(profiles, ignore_index=True)
    for name in ['all_candidate_labels', 'selected_session_clusters']:
        labels.to_csv(OUT/(name+'.csv'), index=False)
    pd.DataFrame(summaries).to_csv(OUT/'candidate_summary.csv', index=False)
    profiles.to_csv(OUT/'behavior_profiles_equal_mouse.csv', index=False)
    pd.concat(contrasts).to_csv(OUT/'behavior_contrasts_by_mouse.csv', index=False)
    pd.DataFrame(parameter_profiles).to_csv(OUT/'model_parameter_profiles.csv', index=False)
    protocol = dict(variant=VARIANT, seed=SEED, K=sorted({*REFERENCE_K, *EXTRA_REPORT_K, optimal}),
                    k_selection=dict(K_range=list(K_RANGE), optimal_K=optimal,
                                     rule='Largest eigengap (normalized-Laplacian eigenvalue k+1 minus eigenvalue k) among K whose every '
                                          'cluster has at least one session from every mouse; smaller K on ties',
                                     reported_only='Silhouette, Calinski-Harabasz, Davies-Bouldin (z space) and stability: median ARI of '
                                                   f'{N_STABILITY} reclusterings of random {STABILITY_FRACTION:.0%} session subsets',
                                     embedding_silhouette='Silhouette in each K-dimensional spectral embedding; reported only, dimension changes with K',
                                     label_numbering='K=2/K=3 aligned to historical labels; other K numbered by decreasing cluster size'),
                    n_sessions=92, units='signed mean NLL gain, nats/trial',
                    features=used.to_dict('records'), scaling='Within mouse mean/sample SD (ddof=1); clip +/-3; equal weight=1 per coordinate',
                    constant_sd_policy='If within-mouse SD <=1e-8, denominator=1; no feature dropped',
                    baseline='Training evaluated choice_long mean, clipped [1e-12,1-1e-12]; cross entropy on evaluated test choices',
                    kernel='exp(-squared_distance / (2 * median_positive_squared_distance))', bandwidth=bandwidth,
                    spectral=dict(assign_labels='kmeans', n_init=N_INIT, eigen_solver='arpack'),
                    label_alignment_reference=str(OLD/'session_clusters.csv'),
                    exclusions='No calendar or fit-status filtering/weighting. Original model evaluation masks retained, including evaluated warm-up trials.',
                    annotations='Parameters, percent loss reductions and behavior are explanatory only; do not enter clustering.',
                    interpretation='Architecture comparisons are not isolated memory; same-session behavioral checks are descriptive.',
                    versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for subject in raw.subject.unique():
        for name in ['sliding_window_metrics.csv', 'extended_parameters.csv']:
            path = features.BASE/subject/'tables'/name
            sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for path in [OLD/'session_clusters.csv', OLD/'interpretation_K3/session_metrics.csv', OLD/'reaction_time_trials.csv',
                 PREVIOUS/'selected_session_clusters.csv', Path(__file__)]:
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/'source_manifest.json').write_text(json.dumps(sources, indent=2)+'\n')
    # Scope all compatibility overrides; no entrypoint with historical output paths is called.
    with ExitStack() as context:
        for obj, key, val in [(pe, 'OUT', OUT), (pv, 'MODELS', MODELS),
                              (pv, 'NAMES', {**pv.NAMES, VARIANT: TITLE}), (pr, 'NAMES', {**pr.NAMES, VARIANT: TITLE}),
                              (pr, 'SHORT', {s.model_id: s.model_id for s in MODEL_SPECS}),
                              (pv, 'names_for', lambda variant, k: {c: f'Group {c}' for c in range(1, k+1)}),
                              (pv, 'COLORS', PALETTE), (behavior, 'COLORS', PALETTE), (latency, 'COLORS', PALETTE)]:
            context.enter_context(patch.object(obj, key, val))
        pe.export_functional_readouts(raw, labels)
        pdf_pages = render_cluster_reports(raw, gains, z, labels, observed, selection, eigenvalues, optimal, OUT)
    render_sequences(labels, OUT)
    ks = sorted(labels.K.unique())
    guide = ''.join(f'- `{folder}/`: {text}\n' for folder, text, _ in LAYOUT[:3])
    guide += ''.join(f'- `04_clusters_K{k}/`: K{k}_report.pdf{" (optimal K)" if k == optimal else ""}'
                     f'{" and sequence_performance/" if k == 3 else ""}\n' for k in ks)
    guide += ''.join(f'- `{folder}/`: {text}\n' for folder, text, _ in LAYOUT[3:])
    (OUT/'README.md').write_text(
        '# Seven-feature RBF spectral clustering\n\n'
        'Run `python -m post_analysis.session_clustering.pattern_rbf_seven_features`.\n\n'
        '## Where to find things\n\n'
        '- `selected_session_clusters.csv`: session labels for every K (main result)\n'
        '- `completion.json`: page counts of every PDF and where each one is\n'
        + guide + '\n'
        '## Reports\n\n'
        f'`03_number_of_clusters/k_selection.pdf`: choice of K over K={K_RANGE.start}-{K_RANGE.stop-1}. Optimal K={optimal} = largest '
        'eigengap of the normalized Laplacian among the K whose every cluster contains sessions from all mice. Stability '
        f'({N_STABILITY} reclusterings of {STABILITY_FRACTION:.0%} subsets), silhouette, Calinski-Harabasz and Davies-Bouldin are reported '
        'alongside in k_selection.csv, not used for the choice.\n\n'
        f'Each `K<k>_report.pdf` (K={", ".join(map(str, ks))}) has six pages: (1) cluster fingerprint, the mean of each of the seven '
        'clustering coordinates plus the observed behavior of the same sessions; (2) how the coordinates are calculated, with the '
        'model equations; (3) the seven signed gains in raw units, per mouse; (4) every session\'s seven gains, z-scored across all sessions so '
        'colors compare across mice; (5) choice behavior, with short- vs long-block psychometrics per cluster; (6) reaction time. Columns are grouped as sensory evidence, choice history, outcome feedback (win-stay / '
        'lose-shift) and block context. Heatmaps: red = high/positive, blue = low/negative. K=3 also has the E/F '
        'sequence-performance plots.\n\n'
        '## Analysis\n\n'
        f'All 92 sessions, RBF spectral clustering ({N_INIT} k-means initializations). Seven signed mean-NLL gains, '
        'within-mouse z scores clipped at +/-3, equal coordinate weights. No parameter features or extra contrasts. '
        'Sensory gain is against a training-fitted constant-choice baseline. Negative gains remain negative in signed_feature_values.csv; '
        'negative standardized values mean below-mouse-average, not necessarily a negative gain. '
        'K=2/K=3 cluster numbers are aligned to the historical labels; other K are numbered by cluster size.\n\n'
        'feature_manifest.csv and protocol.json specify the exact comparisons. sensory_baseline_audit.csv and '
        'prediction_input_audit.csv verify the baseline, paired test identities and source per-trial NLLs. '
        'feature_scaling.csv preserves scaling constants; source_manifest.json hashes all contributing prediction files.\n\n'
        'Clusters describe held-out test-session predictive profiles under preceding-window fits. '
        'Same-session behavior is descriptive; overlapping windows are dependent. No independent validation or biological strategy labels are claimed. '
        'Earlier source and result files are not modified.\n')
    print('DONE', OUT, flush=True)
    return pdf_pages


# Published layout. The analysis writes flat; organize() sorts only after all checks pass.
# selected_session_clusters.csv and completion.json stay on top: the single-session variant reads them.
TOP = ('README.md', 'selected_session_clusters.csv', 'completion.json')
LAYOUT = (
    ('01_clustering_input', 'Seven signed gains, within-mouse z scores, RBF affinity and feature definitions',
     ('signed_feature_values.csv', 'standardized_feature_values.csv', 'clustering_matrix.npy',
      'affinity_rbf_seven_features.npy', 'feature_manifest.csv', 'feature_scaling.csv',
      'raw_feature_correlations.csv', 'window_metrics_parameters.csv')),
    ('02_cluster_labels', 'Labels for every K, per-K summary (sizes, silhouette, ARI, mouse checks) and protocol',
     ('all_candidate_labels.csv', 'candidate_summary.csv', 'protocol.json')),
    ('03_number_of_clusters', 'How many clusters: eigengap and silhouette over K',
     ('k_selection.csv', 'k_selection.png', 'k_selection.pdf')),
    ('05_cluster_tables', 'Per-cluster numbers behind the reports, plus explanatory fitted-model readouts',
     ('gain_profiles_equal_mouse.csv', 'behavior_profiles_equal_mouse.csv', 'behavior_contrasts_by_mouse.csv',
      'observed_session_metrics.csv', 'reaction_time_binned.csv', 'reaction_time_input_audit.csv',
      'session_heatmap_order.csv', 'report_pages.csv', 'model_parameter_profiles.csv',
      'functional_model_readouts.csv', 'selected_functional_parameter_profiles.csv')),
    ('06_audit', 'Input audits, source hashes and preservation check',
     ('prediction_input_audit.csv', 'sensory_baseline_audit.csv', 'source_manifest.json', 'preservation_audit.json')),
)


def destination(name):
    """Published path of one flat staging entry (None = stays on top); K<k>_ files go to their K folder."""
    if name in TOP:
        return None
    if name == 'sequence_performance':
        return '04_clusters_K3/sequence_performance'
    if name.startswith('K') and name.split('_')[0][1:].isdigit():
        return f'04_clusters_{name.split("_")[0]}/{name}'
    for folder, _, names in LAYOUT:
        if name in names:
            return f'{folder}/{name}'
    raise RuntimeError(f'No published location for {name}; add it to LAYOUT')


def organize(staging):
    """Move flat staging entries into LAYOUT; returns {flat name: published path}."""
    moved = {}
    for path in sorted(staging.iterdir()):
        target = destination(path.name)
        if target is None:
            continue
        (staging/target).parent.mkdir(parents=True, exist_ok=True)
        path.rename(staging/target); moved[path.name] = target
    return moved


def protected_hashes():
    paths = list((ROOT/'version1').rglob('*')) + list(PREVIOUS.rglob('*'))
    paths += [p for p in Path(__file__).parent.glob('*.py') if p.resolve() != Path(__file__).resolve()]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def run():
    """Publish only a complete run; verify historical files before replacement."""
    before = protected_hashes()
    staging = Path(tempfile.mkdtemp(prefix=OUT.name+'.staging_', dir=ROOT))
    try:
        expected = run_analysis(staging)
        after = protected_hashes()
        if before != after:
            raise RuntimeError('Historical analysis files changed during execution')
        (staging/'preservation_audit.json').write_text(json.dumps(dict(
            unchanged=True, n_protected_files=len(before), sha256=before), indent=2)+'\n')
        from pypdf import PdfReader
        expected = {**expected, 'sequence_performance/rbf_sequence_overview.pdf': 3,
                    'sequence_performance/rbf_sequence_by_cluster.pdf': 9,
                    **{f'sequence_performance/rbf_cluster_{c}_sequence_performance.pdf': 3 for c in (1, 2, 3)}}
        for name, pages in expected.items():
            if len(PdfReader(staging/name).pages) != pages:
                raise RuntimeError(f'Incomplete report: {name}')
        moved = organize(staging)
        located = {}
        for name, pages in expected.items():
            head, _, rest = name.partition('/')
            located[moved[head] + ('/'+rest if rest else '')] = pages
        (staging/'completion.json').write_text(json.dumps(dict(complete=True, pdf_pages=located), indent=2)+'\n')
        # An existing result of THIS new analysis is retained as a backup until
        # the staged directory has been published successfully.
        backup = None
        if OUT.exists():
            backup = Path(tempfile.mkdtemp(prefix=OUT.name+'.previous_', dir=ROOT))
            backup.rmdir()
            OUT.rename(backup)
        try:
            staging.rename(OUT)
        except BaseException:
            if backup is not None:
                backup.rename(OUT)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        print('PUBLISHED', OUT, flush=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
