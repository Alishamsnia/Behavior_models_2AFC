"""Repeat the original RBF analysis with the current 13-model registry.

Run: python -m post_analysis.session_clustering.pattern_rbf_13_models
All numerical operations and report panels reuse the original workflow.
"""
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from threadpoolctl import threadpool_limits
from . import features, pattern_exploration as pe, pattern_report as pr
from . import pattern_visuals as pv, behavior, latency
from .cluster import driver_manifest, raster_order
from post_analysis.expanded_model_comparison.config import MODEL_SPECS

ROOT = features.BASE / 'pooled' / 'clustering'
OLD = ROOT / 'version1' / 'sliding_window_final'
OUT = ROOT / 'sliding_window_pattern_exploration_rbf_13_models'
EXPECTED = {
    'sensory_actionHistory_lapse',
    'sensory_learnRatio_learnDynamicBlockSwitch_actionSingle_lapse',
    'sensory_actionSingle_lapse',
    'sensory_learnRatio_learnStaticBlockSwitch_lapse',
    'sensory_learnRatio_learnDynamicBlockSwitch_lapse',
    'sensory_learnRatio_learnDynamicBlockSwitch_wslsSingle_lapse',
    'sensory_wslsHistory_lapse', 'sensory_wslsSingle_lapse', 'sensory_lapse',
    'actionHistory_lapse', 'actionSingle_lapse', 'wslsHistory_lapse', 'wslsSingle_lapse',
}
# Same display conventions; only the displayed model roster changes.
MODELS = [
    ('m1_action_kernel', 'Sensory + action\nhistory', 'History'),
    ('m1_stimulus_kernel', 'Sensory + WSLS\nhistory', 'History'),
    ('m1_previous_action_bias', 'Sensory +\naction single', 'History'),
    ('m1_previous_wsls_bias', 'Sensory +\nWSLS single', 'History'),
    ('extended_action', 'Action\nhistory', 'History'),
    ('pure_action', 'Action\nsingle', 'History'),
    ('extended_wsls', 'WSLS\nhistory', 'History'),
    ('pure_wsls', 'WSLS\nsingle', 'History'),
    ('m2_static', 'Learned ratio\nstatic switch', 'Block context'),
    ('m2_dynamic', 'Learned ratio\ndynamic switch', 'Block context'),
    ('m2_dynamic_repetition_bias', 'Dynamic +\naction single', 'Combined models'),
    ('m2_dynamic_wsls_bias', 'Dynamic +\nWSLS single', 'Combined models'),
]


def rbf_candidate(raw, values, manifest):
    return {'rbf_reference': (values, driver_manifest(manifest), 'rbf')}


def clean_text(fig, n_features, n_readouts):
    """Remove descriptions specific to the previous roster/results, not data."""
    for artist in fig.findobj():
        if not hasattr(artist, 'get_text') or not hasattr(artist, 'set_text'):
            continue
        text = artist.get_text()
        text = text.replace('Existing cluster labels unchanged', '13-model clustering')
        text = text.replace('original 48 model-performance/parameter drivers',
                            f'{n_features} model-performance/parameter drivers')
        text = text.replace('These 12 columns', f'These {n_readouts} columns')
        text = text.replace('Rows keep the previous within-cluster session order.',
                            'Rows follow the original within-cluster ordering method.')
        if text.startswith('Shared heatmap basis'):
            text = ('Same original RBF inputs: test-NLL contrasts and transformed training-fit parameters; 13 requested models.\n'
                    'Within-mouse z scores, clipping at +/-3, equal block/model weights; bias and lapse excluded from distance.\n'
                    'Median squared-distance bandwidth, original spectral-clustering settings and seed; no session exclusions.')
        artist.set_text(text)


def run():
    assert {s.label for s in MODEL_SPECS} == EXPECTED
    assert len(MODEL_SPECS) == 13
    OUT.mkdir(parents=True, exist_ok=True)
    # Configure only this invocation; the archived workflow and outputs stay intact.
    pe.OUT = pr.OUT = pv.OUT = OUT
    pe.LEGACY_OUT = pr.LEGACY_OUT = OLD
    pe.candidates = rbf_candidate
    pr.NAMES['rbf_reference'] = 'RBF: requested 13-model features'
    pr.SHORT.update({s.model_id: s.label.replace('_lapse', '') for s in MODEL_SPECS})
    pv.MODELS = MODELS
    pv.names_for = lambda variant, k: {c: f'Group {c}' for c in range(1, k+1)}
    pe.run()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    protocol.update(reference='13 requested nonoracle models; original bias/lapse exclusions',
                    initial_candidates=['rbf_reference'], additional_controls=[],
                    kernels='Fixed median squared-distance RBF bandwidth; no bandwidth search',
                    selection='RBF requested by user; K=2 and K=3 retained from original workflow',
                    label_alignment_reference=str(OLD / 'session_clusters.csv'),
                    models=[dict(model_id=s.model_id, label=s.label, directory=s.directory) for s in MODEL_SPECS])
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    raw, values, meta, _ = features.collect()
    x, used = features.matrix(values, driver_manifest(meta), raw.subject.to_numpy())
    np.save(OUT / 'clustering_matrix.npy', x)
    labels = pd.read_csv(OUT / 'all_candidate_labels.csv')
    labels.to_csv(OUT / 'selected_session_clusters.csv', index=False)
    pe.export_functional_readouts(raw, labels)
    observed = pd.read_csv(OUT / 'observed_session_metrics.csv')
    d, readout_meta = pv.readouts(raw)
    readout_meta.loc[readout_meta.feature.eq('gain__m1_action_kernel'), 'label'] = 'Action hist.\nbenefit'
    readout_meta.loc[readout_meta.feature.eq('gain__m1_stimulus_kernel'), 'label'] = 'WSLS hist.\nbenefit'
    # Group explanatory columns by display family, as required by session_page.
    order_family = ['Sensory', 'History', 'Block context', 'Combined models']
    readout_meta = pd.concat([readout_meta[readout_meta.family.eq(f)] for f in order_family], ignore_index=True)
    d.to_csv(OUT / 'visual_readouts.csv', index=False)
    readout_meta.to_csv(OUT / 'visual_readout_manifest.csv', index=False)
    order = raster_order(x)
    pages, orders = [], []
    with PdfPages(OUT / 'clusters.pdf') as pdf:
        for k in (3, 2):
            ll = labels[labels.K.eq(k)].copy()
            fig = pv.summary_page(raw, d, readout_meta, ll, observed, 'rbf_reference', k)
            fig.set_size_inches(25, 13)
            clean_text(fig, len(used), len(readout_meta))
            pdf.savefig(fig)
            if k == 3: fig.savefig(OUT / 'clusters_K3.png', dpi=100)
            plt.close(fig)
            pages.append(dict(page=len(pages)+1, K=k, analysis='readable_summary'))
            fig, oo = pv.session_page(d, readout_meta, ll, order, 'rbf_reference', k)
            fig.set_size_inches(25, 12)
            clean_text(fig, len(used), len(readout_meta))
            pdf.savefig(fig); plt.close(fig)
            pages.append(dict(page=len(pages)+1, K=k, analysis='readable_sessions'))
            rows = raw[['subject', 'session_id']].copy()
            rows['variant'], rows['K'], rows['display_rank'] = 'rbf_reference', k, np.argsort(oo)+1
            orders.append(rows)
        ll = labels[labels.K.eq(3)]
        fig = pr.cluster_page(raw, values, used, ll.cluster_label.to_numpy(), order,
                              'rbf_reference', 3, pd.read_csv(OUT / 'behavior_profiles_equal_mouse.csv'))
        fig.set_size_inches(28, 14)
        fig.axes[2].images[0].set_cmap(pv.VALUE_CMAP)
        clean_text(fig, len(used), len(readout_meta))
        # Compact model names in the dense appendix; full names are in protocol.json.
        short = {s.model_id: s.model_id for s in MODEL_SPECS}
        fig.axes[2].set_xticklabels([short[r.model]+': '+('NLL gain' if r.block=='performance' else r.description)
                                     for r in used.itertuples()], rotation=90, fontsize=8)
        pdf.savefig(fig); plt.close(fig)
        pages.append(dict(page=len(pages)+1, K=3, analysis='full_feature_appendix'))
    pd.DataFrame(pages).to_csv(OUT / 'cluster_pages.csv', index=False)
    pd.concat(orders).to_csv(OUT / 'visual_session_order.csv', index=False)
    trials = pd.read_csv(OLD / 'reaction_time_trials.csv').drop(columns='cluster_label', errors='ignore')
    assert not trials.duplicated(['subject', 'session_id', 'trial_index']).any()
    eligible = trials[trials.latency_eligible.eq(True)]
    pages, bins = [], []
    with PdfPages(OUT / 'cluster_behavior_pooled.pdf') as pdf:
        for kind in ('choice_behavior', 'reaction_time'):
            for k in (3, 2):
                ll = labels[labels.K.eq(k)]
                tagged = behavior.attach_labels(trials if kind == 'choice_behavior' else eligible, ll)
                if kind == 'choice_behavior':
                    fig, _, _ = behavior.make_page(tagged, k)
                else:
                    fig, stats, _ = latency.make_page(tagged, k)
                    stats['variant'], stats['K'] = 'rbf_reference', k
                    bins.append(stats)
                fig._suptitle.set_text(f'{kind.replace("_", " ").title()} — RBF: 13 models — K={k}')
                fig._suptitle.set_fontsize(15)
                behavior.annotate_mouse_support(fig, ll)
                pdf.savefig(fig)
                if kind == 'choice_behavior' and k == 3: fig.savefig(OUT / 'behavior_K3.png', dpi=100)
                plt.close(fig)
                pages.append(dict(page=len(pages)+1, K=k, analysis=kind, n_sessions=len(raw)))
    pd.DataFrame(pages).to_csv(OUT / 'behavior_pages.csv', index=False)
    pd.concat(bins).to_csv(OUT / 'reaction_time_binned.csv', index=False)
    shutil.copyfile(OLD / 'reaction_time_input_audit.csv', OUT / 'reaction_time_input_audit.csv')
    (OUT / 'README.md').write_text(
        '# RBF repeat: requested 13 models\n\n'
        'Reproduce with `python -m post_analysis.session_clustering.pattern_rbf_13_models`.\n\n'
        'Same RBF-reference method as version1/sliding_window_pattern_exploration; only the model roster changes. '
        'All 92 sessions retained (MC06: 32, MC07: 30, MC08: 30). K=2 and K=3; no refitting. '
        'Original transformations, bias/lapse exclusions, within-mouse scaling, clip3, equal block/model weights, '
        'median squared-distance bandwidth, spectral settings and seed are reused. '
        'Cluster IDs are aligned to the same legacy K-means reference by overlap.\n\n'
        'clusters.pdf: original summary/session panels updated for the model roster plus full-driver appendix. '
        'cluster_behavior_pooled.pdf: original choice and RT panels. '
        'Model fits and per-session behavioral annotations are existing inputs. '
        'See protocol.json for all model names and candidate_summary.csv for cluster counts.\n')
    print('Reports saved:', OUT, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
