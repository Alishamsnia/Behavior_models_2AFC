"""Scaling sensitivity for the frozen seven-feature RBF clustering (PI's scaling concern).

Run: python -m post_analysis.session_clustering.scaling_sensitivity_seven_features
Reads the frozen signed gains and K=2/3 labels; reruns only the clustering step with the
frozen RBF median bandwidth, spectral settings and seed. No decision models are refitted
and nothing in the frozen seven-feature folder is written.

Worry (a): per-mouse centering removes between-mouse differences -> pooled z sensitivity.
Worry (b): per-feature SD division removes magnitude differences between features ->
raw-nats clustering, its distance budget, and the frozen clusters shown in raw nats.
"""
from __future__ import annotations

import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from . import features, pattern_exploration as pe, pattern_visuals as pv
from . import pattern_rbf_seven_features as frozen
from .cluster import SEED, COLORS

SOURCE = frozen.OUT
# Lives inside the frozen folder; rerunning the frozen pipeline republishes that folder and drops it, so rerun this too.
OUT = frozen.OUT / '08_scaling_sensitivity'
WORRY_A = OUT / 'worry_a_between_mouse_offsets'
WORRY_B = OUT / 'worry_b_feature_magnitudes'
FROZEN_SHA = 'ba822972205b21a1e7f3c7fadc7ccdf58d07743491cfad0e695ea8b652e6a774'
FEATURES, KS = frozen.FEATURES, frozen.REFERENCE_K
# Definition labels, not the report's "Longer ... history" column labels: those contrasts change memory span AND integration rule.
LABELS = {d[0]: d[1].replace('\n', ' ').replace('benefit', 'gain').replace('advantage', 'contrast') for d in frozen.DEFINITIONS}
FAMILY = {d[0]: d[2] for d in frozen.DEFINITIONS}
SCALINGS = {
    'within_mouse_z': 'Within-mouse z, clip ±3 (frozen)',
    'pooled_z': 'Pooled z across all sessions, clip ±3',
    'raw_nats': 'Raw nats/trial, no scaling',
}


def load():
    labels_path = SOURCE / 'selected_session_clusters.csv'
    if hashlib.sha256(labels_path.read_bytes()).hexdigest() != FROZEN_SHA:
        raise ValueError('Frozen K=2/3 labels changed; refusing to compare against them')
    gains = pd.read_csv(SOURCE / '01_clustering_input' / 'signed_feature_values.csv')
    labels = pd.read_csv(labels_path)
    keys = pd.MultiIndex.from_frame(gains[frozen.KEYS])
    ref = {k: labels[labels.K.eq(k)].set_index(frozen.KEYS).reindex(keys).cluster_label.to_numpy() for k in KS}
    if len(gains) != 92 or any(np.isnan(r).any() for r in ref.values()):
        raise ValueError('Expected the frozen 92 sessions with K=2/3 labels')
    return gains, {k: r.astype(int) for k, r in ref.items()}


def coordinates(gains):
    x, subjects = gains[FEATURES], gains.subject.to_numpy()
    return {'within_mouse_z': features.normalize(x, subjects),
            'pooled_z': frozen.pooled_z(x),
            'raw_nats': x.to_numpy(float)}


def cramers_v(labels, subjects):
    table = pd.crosstab(labels, subjects).to_numpy()
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.sum() * (min(table.shape)-1))))


def cluster_all(coords, ref, subjects):
    labels, rows = {}, []
    with threadpool_limits(1):
        for name, z in coords.items():
            affinity, bandwidth = pe.gaussian_affinity(z)
            for k in KS:
                lab, _ = pe.align_labels(frozen.spectral_labels(affinity, k), ref[k], k)
                labels[name, k] = lab
                mix = pd.crosstab(lab, subjects)
                rows.append(dict(scaling=name, description=SCALINGS[name], K=k, bandwidth=bandwidth,
                                 sizes='/'.join(map(str, np.bincount(lab)[1:])),
                                 ari_to_frozen=adjusted_rand_score(ref[k], lab),
                                 cramers_v_with_mouse=cramers_v(lab, subjects),
                                 every_mouse_in_every_cluster=bool((mix.to_numpy() > 0).all() and mix.shape[0] == k)))
    summary = pd.DataFrame(rows)
    if not np.allclose(summary.query("scaling == 'within_mouse_z'").ari_to_frozen, 1):
        raise ValueError('Frozen procedure did not reproduce the frozen labels')
    return labels, summary


def eta_squared(values, lab):
    """Share of a raw gain's across-session variance explained by cluster membership."""
    v = pd.Series(values)
    return float(v.groupby(lab).transform('mean').sub(v.mean()).pow(2).sum() / v.sub(v.mean()).pow(2).sum())


def separation(gains, labels):
    return pd.DataFrame([dict(scaling=name, K=k, feature=f, eta_squared=eta_squared(gains[f].to_numpy(), lab))
                         for (name, k), lab in labels.items() for f in FEATURES])


def distance_share(coords):
    """Share of the mean pairwise squared Euclidean distance owed to each feature (= its variance share)."""
    rows = []
    for name in ('raw_nats', 'within_mouse_z', 'pooled_z'):
        var = coords[name].var(axis=0)
        rows += [dict(scaling=name, feature=f, share=v / var.sum()) for f, v in zip(FEATURES, var)]
    return pd.DataFrame(rows)


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors='.3', labelsize=9)


def save(fig, path):
    fig.savefig(path.with_suffix('.pdf')); fig.savefig(path.with_suffix('.png'), dpi=200); plt.close(fig)


def share_figure(share, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    fig.subplots_adjust(left=.2, right=.97, top=.78, bottom=.16, wspace=.12)
    y = np.arange(len(FEATURES))[::-1]
    for ax, name in zip(axes, ('raw_nats', 'within_mouse_z')):
        s = share[share.scaling.eq(name)].set_index('feature').share.reindex(FEATURES)
        ax.barh(y, s*100, height=.62, color=[frozen.FAMILIES[FAMILY[f]][1] for f in FEATURES])
        for yi, v in zip(y, s):
            ax.text(v*100+1, yi, f'{v:.0%}', va='center', fontsize=9, color='.25')
        ax.set_xlim(0, 100); ax.set_xlabel('% of mean pairwise squared distance', fontsize=10)
        ax.set_title(SCALINGS[name], fontsize=12, loc='left'); style(ax)
    axes[0].set_yticks(y, [LABELS[f] for f in FEATURES])
    fig.suptitle('Which gains decide who is "close" to whom', fontsize=16, fontweight='bold', x=.02, ha='left')
    fig.text(.02, .86, 'Squared Euclidean distance is a sum over features, so each bar is that feature\'s variance share. '
             'Without scaling, the sensory gain owns most of the distance\nand the RBF affinity (hence the clustering) '
             'is essentially a one-feature comparison. Scaling gives each coordinate an equal budget by construction.',
             fontsize=10, color='.3')
    fig.legend(handles=[Patch(color=c, label=n) for n, c in frozen.FAMILIES.values()], loc='lower center',
               ncol=4, frameon=False, fontsize=9)
    save(fig, path)


def strips(fig, gs_row, gains, lab, k, mean_z=None):
    """One panel per feature: sessions' raw gain by cluster; own y-axis per panel."""
    rng = np.random.default_rng(SEED)
    for j, f in enumerate(FEATURES):
        ax = fig.add_subplot(gs_row[j])
        for c in range(1, k+1):
            v = gains[f].to_numpy()[lab == c]
            ax.scatter(c + rng.uniform(-.18, .18, len(v)), v, s=11, color=COLORS[c-1], alpha=.75, linewidths=0)
            ax.hlines(np.median(v), c-.3, c+.3, color='.1', lw=2)
            if mean_z is not None:
                ax.annotate(f'{mean_z.loc[c, f]:+.1f}', (c, 0), xycoords=('data', 'axes fraction'),
                            xytext=(0, -24), textcoords='offset points', ha='center', fontsize=8, color='.3')
        if mean_z is not None and j == 0:
            ax.annotate('mean z', (0, 0), xycoords='axes fraction', xytext=(-8, -24), textcoords='offset points',
                        ha='right', fontsize=8, color='.3')
        ax.axhline(0, color='.55', lw=.8, ls='--')
        ax.set_xticks(range(1, k+1), [f'C{c}' for c in range(1, k+1)])
        ax.set_xlim(.4, k+.6); ax.set_title(LABELS[f], pad=14, fontsize=9, color=frozen.FAMILIES[FAMILY[f]][1])
        ax.text(.5, 1.0, f'η² = {eta_squared(gains[f].to_numpy(), lab):.2f}', transform=ax.transAxes, ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color='.2')
        style(ax)
        if j == 0:
            ax.set_ylabel(f'K={k}\nheld-out gain (nats/trial)', fontsize=10)


def raw_cluster_figure(gains, labels, summary, path):
    fig = plt.figure(figsize=(18, 9.5))
    gs = fig.add_gridspec(2, len(FEATURES), left=.06, right=.99, top=.74, bottom=.07, hspace=.5, wspace=.45)
    for i, k in enumerate(KS):
        strips(fig, [gs[i, j] for j in range(len(FEATURES))], gains, labels['raw_nats', k], k)
    r = summary.set_index(['scaling', 'K'])
    fig.suptitle('What the clustering finds without scaling (raw nats/trial)', fontsize=16, fontweight='bold', x=.02, ha='left')
    fig.text(.02, .88, 'Clusters here are a rerun of the frozen RBF/spectral procedure on raw gains (no model refits). '
             'Dots = sessions, bar = cluster median, dashed = zero gain. Each panel has its own y-axis.\n'
             'η² = share of that gain\'s across-session variance explained by the clusters (which gains the clusters differ on).',
             fontsize=10, color='.3')
    fig.text(.02, .82, '   '.join(f'K={k}: sizes {r.loc[("raw_nats", k), "sizes"]}, ARI to frozen={r.loc[("raw_nats", k), "ari_to_frozen"]:.2f}.' for k in KS)
             + '\nUnscaled, the distance is ~80% sensory gain: the method isolates the sessions where the stimulus model predicts worse than a constant,'
             '\nthen splits the rest by sensory level. WSLS and block-context gains barely differ between clusters (η² ≈ 0).',
             fontsize=10.5)
    save(fig, path)


def frozen_raw_figure(gains, coords, ref, path):
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, len(FEATURES), left=.06, right=.99, top=.79, bottom=.08, hspace=.6, wspace=.45)
    z = pd.DataFrame(coords['within_mouse_z'], columns=FEATURES)
    for i, k in enumerate(KS):
        strips(fig, [gs[i, j] for j in range(len(FEATURES))], gains, ref[k], k, z.groupby(ref[k]).mean())
    fig.suptitle('Frozen clusters shown in raw units: nothing is hidden by scaling', fontsize=16, fontweight='bold', x=.02, ha='left')
    fig.text(.02, .875, 'Same frozen K=2/3 labels as the report. Dots = sessions\' held-out gain in nats/trial; bar = cluster median; '
             'dashed = zero gain. Under each cluster: its mean within-mouse z (the value the clustering saw).\n'
             'η² = share of that gain\'s across-session variance explained by the clusters: every family contributes, unlike the raw-nats clustering.\n'
             'Read magnitude from the y-axis, not from heatmap colour: a high z can be a small (or negative) raw gain, and y-axes differ by panel.',
             fontsize=10, color='.3')
    save(fig, path)


def composition_figure(labels, ref, subjects, summary, path):
    mice = sorted(pv.MOUSE_COLORS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    fig.subplots_adjust(left=.09, right=.97, top=.82, bottom=.12, hspace=.55, wspace=.25)
    r = summary.set_index(['scaling', 'K'])
    for i, k in enumerate(KS):
        for j, (name, lab) in enumerate((('within_mouse_z', ref[k]), ('pooled_z', labels['pooled_z', k]))):
            ax = axes[i, j]
            mix = pd.crosstab(lab, subjects).reindex(index=range(1, k+1), columns=mice, fill_value=0)
            frac, left = mix.div(mix.sum(axis=1), axis=0), np.zeros(k)
            for m in mice:
                ax.barh(range(k), frac[m], left=left, height=.6, color=pv.MOUSE_COLORS[m], edgecolor='white', linewidth=1.5)
                left += frac[m].to_numpy()
            ax.set_yticks(range(k), [f'C{c} (n={mix.loc[c].sum()})' for c in range(1, k+1)]); ax.invert_yaxis()
            ax.axvline(1/3, color='.4', lw=.8, ls=':')
            ax.set_xlim(0, 1); ax.set_xlabel('fraction of the cluster\'s sessions', fontsize=9); style(ax)
            ax.set_title(f'K={k} | {SCALINGS[name]}\nmouse association Cramér\'s V={r.loc[(name, k), "cramers_v_with_mouse"]:.2f}'
                         + ('' if name == 'within_mouse_z' else f' | ARI to frozen={r.loc[(name, k), "ari_to_frozen"]:.2f}'),
                         fontsize=10, loc='left')
    fig.suptitle('Does removing per-mouse centering make clusters sort mice?', fontsize=16, fontweight='bold', x=.02, ha='left')
    fig.text(.02, .89, 'Pooled z keeps between-mouse differences; everything else (clip ±3, RBF median bandwidth, spectral, seed) is frozen. '
             'Dotted line = equal share of three mice.', fontsize=10, color='.3')
    fig.legend(handles=[Patch(color=pv.MOUSE_COLORS[m], label=m) for m in mice], loc='lower center', ncol=3, frameon=False)
    save(fig, path)


def agreement_figure(labels, ref, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(left=.1, right=.97, top=.78, bottom=.14, wspace=.35)
    for ax, k in zip(axes, KS):
        t = pd.crosstab(ref[k], labels['pooled_z', k]).reindex(index=range(1, k+1), columns=range(1, k+1), fill_value=0)
        ax.imshow(t.to_numpy(), cmap='Blues', vmin=0)
        for (a, b), v in np.ndenumerate(t.to_numpy()):
            ax.text(b, a, v, ha='center', va='center', fontsize=12, color='white' if v > t.to_numpy().max()*.6 else '.15')
        ax.set_xticks(range(k), [f'C{c}' for c in range(1, k+1)]); ax.set_yticks(range(k), [f'C{c}' for c in range(1, k+1)])
        ax.set_xlabel('pooled-z cluster'); ax.set_ylabel('frozen cluster'); ax.set_title(f'K={k}', loc='left')
    fig.suptitle('Session counts: frozen clusters vs pooled-z clusters', fontsize=15, fontweight='bold', x=.02, ha='left')
    fig.text(.02, .86, 'Pooled-z labels were matched to frozen labels by maximum overlap. Off-diagonal = sessions that move.',
             fontsize=10, color='.3')
    save(fig, path)


README = """# Scaling sensitivity for the seven-feature RBF clustering

Run `python -m post_analysis.session_clustering.scaling_sensitivity_seven_features`.
Read-only with respect to the frozen analysis: it reads `signed_feature_values.csv` and the frozen K=2/3
labels (sha256 checked), and reruns only the clustering step with the frozen RBF median bandwidth,
spectral settings and seed. No decision models are refitted. The frozen within-mouse-z rerun must
reproduce the frozen labels exactly (ARI = 1), otherwise the script stops.

## Worry (b): scaling removes magnitude differences between features
- `B1_distance_share` - each feature's share of the pairwise squared distance, raw nats vs frozen scaling.
- `B2_raw_nats_clustering` - clusters obtained without any scaling, shown in raw nats.
- `B3_frozen_clusters_in_raw_nats` - the frozen K=2/3 clusters in raw nats, with each cluster's mean z.

## Worry (a): per-mouse centering removes between-mouse differences
- `A1_mouse_composition` - mouse make-up of each cluster, frozen vs pooled z.
- `A2_frozen_vs_pooled_counts` - session counts frozen x pooled-z clusters.

## Tables
- `scaling_sensitivity_summary.csv` - per scaling and K: sizes, ARI to frozen, Cramér's V between cluster
  and mouse, whether every mouse appears in every cluster.
- `cluster_separation_eta_squared.csv` - per scaling, K and feature: share of the raw gain's variance
  explained by the clusters (which gains the clusters actually differ on).
- `feature_distance_share.csv`, `labels_by_scaling.csv`.

ARI measures agreement with the frozen labels (sensitivity), not which scaling is correct.
"""


def run():
    gains, ref = load()
    subjects = gains.subject.to_numpy()
    coords = coordinates(gains)
    labels, summary = cluster_all(coords, ref, subjects)
    share = distance_share(coords)
    for d in (WORRY_A, WORRY_B):
        d.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT/'scaling_sensitivity_summary.csv', index=False)
    share.to_csv(OUT/'feature_distance_share.csv', index=False)
    eta = separation(gains, labels); eta.to_csv(OUT/'cluster_separation_eta_squared.csv', index=False)
    out = gains[features.IDS].copy()
    for (name, k), lab in labels.items():
        out[f'{name}__K{k}'] = lab
    out.to_csv(OUT/'labels_by_scaling.csv', index=False)
    with plt.rc_context(frozen.ILLUSTRATOR):
        share_figure(share, WORRY_B/'B1_distance_share')
        raw_cluster_figure(gains, labels, summary, WORRY_B/'B2_raw_nats_clustering')
        frozen_raw_figure(gains, coords, ref, WORRY_B/'B3_frozen_clusters_in_raw_nats')
        composition_figure(labels, ref, subjects, summary, WORRY_A/'A1_mouse_composition')
        agreement_figure(labels, ref, WORRY_A/'A2_frozen_vs_pooled_counts')
    (OUT/'README.md').write_text(README)
    (OUT/'protocol.json').write_text(json.dumps(dict(source=str(SOURCE), frozen_labels_sha256=FROZEN_SHA, seed=SEED,
                                                     n_init=frozen.N_INIT, K=list(KS), scalings=SCALINGS), indent=2))
    print(summary.round(3).to_string(index=False))
    print(eta.pivot_table(index=['scaling', 'K'], columns='feature', values='eta_squared')[FEATURES].round(2).to_string())
    print(share.pivot(index='feature', columns='scaling', values='share').reindex(FEATURES).round(3).to_string())


if __name__ == '__main__':
    run()
