"""Seven-feature RBF with a star contrast design: every history/context model against the sensory model S.

Run: python -m post_analysis.session_clustering.pattern_rbf_seven_features_star
Identical pipeline, scaling, kernel, spectral settings, seed and reports as
pattern_rbf_seven_features; only the three second-level coordinates change:
L(S+A1)-L(S+AH) -> L(S)-L(S+AH), L(S+W1)-L(S+WH) -> L(S)-L(S+WH),
L(S+Bstatic)-L(S+Bdynamic) -> L(S)-L(S+Bdynamic).
The frozen module and its outputs are not modified; no decision models are refitted.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from . import pattern_exploration as pe, pattern_rbf_seven_features as base

OUT = base.ROOT / 'sliding_window_pattern_exploration_rbf_seven_features_star'
VARIANT = 'rbf_seven_features_star'
TITLE = 'RBF: seven signed predictive gains, star design (all vs sensory)'
FROZEN = base.OUT
# Same model pairs as the frozen run for four coordinates; the other three now share the reduced model S.
DEFINITIONS = (
    ('sensory', 'Sensory-model\nbenefit', 'sensory', 'constant_choice', 'm0_sensory', 'baseline - 1'),
    ('action_single', 'Single-action\nbenefit', 'action', 'm0_sensory', 'm1_previous_action_bias', '1 - 3'),
    ('action_history_total', 'Action-history\nbenefit over S', 'action', 'm0_sensory', 'm1_action_kernel', '1 - 2'),
    ('wsls_single', 'Single-WSLS\nbenefit', 'wsls', 'm0_sensory', 'm1_previous_wsls_bias', '1 - 7'),
    ('wsls_history_total', 'WSLS-history\nbenefit over S', 'wsls', 'm0_sensory', 'm1_stimulus_kernel', '1 - 6'),
    ('static_context', 'Static-context\nbenefit', 'context', 'm0_sensory', 'm2_static', '1 - 10'),
    ('dynamic_context_total', 'Dynamic-context\nbenefit over S', 'context', 'm0_sensory', 'm2_dynamic', '1 - 11'),
)
FEATURES = [d[0] for d in DEFINITIONS]
COLUMN_LABELS = {'sensory': 'Stimulus readout\nvs constant choice', 'action_single': 'Last choice\n(1-back) vs S',
                 'action_history_total': 'Choice kernel\nvs S', 'wsls_single': 'Last rewarded side\n(1-back) vs S',
                 'wsls_history_total': 'Outcome kernel\nvs S', 'static_context': 'Static block\nprior vs S',
                 'dynamic_context_total': 'Dynamic block\nswitching vs S'}
# Exact frozen methods-page text that describes the chain contrasts, and its star replacement.
METHODS_TEXT = {
    r'$L(S{+}A1) - L(S{+}AH)$': r'$L(S) - L(S{+}AH)$',
    r'$L(S{+}W1) - L(S{+}WH)$': r'$L(S) - L(S{+}WH)$',
    r'$L(S{+}B_{static}) - L(S{+}B_{dynamic})$': r'$L(S) - L(S{+}B_{dynamic})$',
    'History contrasts (AH vs A1, WH vs W1)\nchange both memory span and integration rule, so they are architecture comparisons, '
    'not isolated memory effects.':
    'Star design: every history/context model is compared with S directly,\nso S+AH, S+WH and S+Bdynamic are TOTAL gains over S '
    '(they overlap with the S+A1, S+W1 and S+Bstatic coordinates).',
    'How the seven clustering coordinates are calculated': 'How the seven clustering coordinates are calculated (star design)',
}
COMPARISON = '07_comparison_to_frozen'
COMPARISON_FILES = ('label_agreement_with_frozen.csv', 'frozen_vs_star_counts.pdf', 'frozen_vs_star_counts.png')


def methods_page():
    fig = base_methods_page()
    hits = dict.fromkeys(METHODS_TEXT, 0)
    # The suptitle may also be listed in fig.texts; visit each text object once.
    for text in {id(t): t for t in fig.texts + [fig._suptitle]}.values():
        value = text.get_text()
        for old, new in METHODS_TEXT.items():
            if old in value:
                hits[old] += 1; value = value.replace(old, new)
        text.set_text(value)
    if any(n != 1 for n in hits.values()):
        raise RuntimeError(f'Frozen methods-page text changed; star replacements not applied exactly once: {hits}')
    return fig


def frozen_hashes():
    # FROZEN also holds 08_scaling_sensitivity.
    folders = [FROZEN, base.ROOT/'single_session_train_pattern_exploration_rbf_seven_features']
    paths = [p for d in folders for p in d.rglob('*') if p.is_file()] + [base.__file__]
    return {str(p): hashlib.sha256(open(p, 'rb').read()).hexdigest() for p in paths}


def compare_to_frozen(out):
    keys = base.KEYS
    star = pd.read_csv(out/'selected_session_clusters.csv')
    frozen = pd.read_csv(FROZEN/'selected_session_clusters.csv')
    rows, tables = [], []
    for k in sorted(set(star.K) & set(frozen.K)):
        a = frozen[frozen.K.eq(k)].set_index(keys).cluster_label
        b = star[star.K.eq(k)].set_index(keys).cluster_label.reindex(a.index)
        if b.isna().any():
            raise ValueError('Star and frozen runs cover different sessions')
        # Both runs number clusters by overlap with the historical labels, so numbers need not match each other:
        # order star columns by their best match to the frozen clusters, keeping the star's own numbers as labels.
        _, mapping = pe.align_labels(b.to_numpy().astype(int), a.to_numpy(), k)
        columns = sorted(mapping, key=mapping.get)
        t = pd.crosstab(a.to_numpy(), b.to_numpy().astype(int)).reindex(index=range(1, k+1), columns=columns, fill_value=0)
        tables.append((k, t))
        rows.append(dict(K=k, ari_star_vs_frozen=adjusted_rand_score(a, b), sessions_matched=int(np.trace(t.to_numpy())),
                         star_to_frozen_label=json.dumps(mapping),
                         n_sessions=len(a), frozen_sizes='/'.join(map(str, t.sum(axis=1))),
                         star_sizes='/'.join(map(str, t.sum(axis=0).reindex(range(1, k+1))))))
    pd.DataFrame(rows).to_csv(out/COMPARISON_FILES[0], index=False)
    with matplotlib.rc_context(base.ILLUSTRATOR):
        fig, axes = plt.subplots(1, len(tables), figsize=(5.5*len(tables), 5), squeeze=False)
        fig.subplots_adjust(left=.08, right=.97, top=.76, bottom=.14, wspace=.35)
        for ax, (k, t), r in zip(axes[0], tables, rows):
            m = t.to_numpy()
            ax.imshow(m, cmap='Blues', vmin=0)
            for (i, j), v in np.ndenumerate(m):
                ax.text(j, i, v, ha='center', va='center', fontsize=12, color='white' if v > m.max()*.6 else '.15')
            ax.set_xticks(range(k), [f'C{c}' for c in t.columns]); ax.set_yticks(range(k), [f'C{c}' for c in range(1, k+1)])
            ax.set_xlabel('star cluster'); ax.set_ylabel('frozen cluster')
            ax.set_title(f'K={k} | ARI={r["ari_star_vs_frozen"]:.2f} | matched {r["sessions_matched"]}/{r["n_sessions"]}',
                         loc='left', fontsize=11)
        fig.suptitle('Session counts: frozen chain clusters vs star-design clusters', fontsize=15, fontweight='bold', x=.02, ha='left')
        fig.text(.02, .85, 'Both runs share scaling, kernel, spectral settings and seed. Star columns keep their own cluster numbers '
                 'but are ordered by best match\nto the frozen clusters, so the diagonal = matched sessions and off-diagonal = sessions '
                 'that move.', fontsize=10, color='.3')
        fig.savefig(out/COMPARISON_FILES[1]); fig.savefig(out/COMPARISON_FILES[2], dpi=150); plt.close(fig)
    return pd.DataFrame(rows)


def run_analysis(out):
    pages = base_run_analysis(out)
    agreement = compare_to_frozen(out)
    readme = (out/'README.md').read_text()
    old_title, old_run = '# Seven-feature RBF spectral clustering\n\n', 'Run `python -m post_analysis.session_clustering.pattern_rbf_seven_features`.'
    if readme.count(old_title) != 1 or readme.count(old_run) != 1:
        raise RuntimeError('Frozen README layout changed')
    readme = readme.replace(old_title, '# Seven-feature RBF spectral clustering: star design (all vs sensory)\n\n').replace(
        old_run, 'Run `python -m post_analysis.session_clustering.pattern_rbf_seven_features_star`.\n\n'
        'Identical to the frozen seven-feature analysis except three coordinates: L(S)-L(S+AH), L(S)-L(S+WH) and '
        'L(S)-L(S+Bdynamic) replace the chain contrasts A1->AH, W1->WH and Bstatic->Bdynamic. In raw nats each is the sum '
        'of the two frozen coordinates on its path, so no model is refitted; only the clustering geometry changes. '
        'The frozen analysis is untouched; its labels are compared in `07_comparison_to_frozen/`.\n\n'
        + ''.join(f'- K={r.K}: ARI to frozen = {r.ari_star_vs_frozen:.2f}, sessions in the best-matched cluster {r.sessions_matched}/{r.n_sessions} '
                  f'(star->frozen cluster numbers {r.star_to_frozen_label})\n'
                  for r in agreement.itertuples()))
    (out/'README.md').write_text(readme)
    return pages


base_methods_page, base_run_analysis = base.methods_page, base.run_analysis


def run():
    before = frozen_hashes()
    with ExitStack() as context:
        for key, value in [('OUT', OUT), ('VARIANT', VARIANT), ('TITLE', TITLE), ('DEFINITIONS', DEFINITIONS),
                           ('FEATURES', FEATURES), ('COLUMN_LABELS', COLUMN_LABELS), ('methods_page', methods_page),
                           ('run_analysis', run_analysis),
                           ('LAYOUT', base.LAYOUT + ((COMPARISON, 'Star vs frozen chain labels: ARI and session counts', COMPARISON_FILES),))]:
            context.enter_context(patch.object(base, key, value))
        base.run()
    if frozen_hashes() != before:
        raise RuntimeError('Frozen seven-feature outputs changed during the star run')
    print('Frozen seven-feature, single-session and scaling-sensitivity outputs unchanged', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', choices=('seven', 'four'), default='seven',
                        help='Clustering coordinate set; each writes a separate output directory')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.features == 'four':
            from .pattern_rbf_four_features_star import run as run_four
            run_four()
        else:
            run()
