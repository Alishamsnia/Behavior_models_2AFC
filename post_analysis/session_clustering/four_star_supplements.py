"""Reproduce later seven-feature supplements for the four-feature star cohort."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import matplotlib.pyplot as plt
import pandas as pd
from pypdf import PdfReader
from . import pattern_rbf_seven_features as base
from . import pattern_rbf_four_features_star as four
from . import hypothesis_followup as hypothesis
from . import scaling_sensitivity_seven_features as scaling
from . import extend_k3_reaction_time as rt


def run(out):
    out = Path(out)
    labels = pd.read_csv(out/'selected_session_clusters.csv')
    sizes = labels[labels.K.eq(3)].groupby('cluster_label').size().to_dict()
    with patch.object(base, 'OUT', out):
        rt.main()
        summary = pd.read_csv(out/'05_cluster_tables/K3_rt_condition_summary.csv')
        tests = pd.read_csv(out/'05_cluster_tables/K3_rt_pairwise_tests.csv')
        sd = summary.copy(); sd['sem'] = sd['sd']
        fig = rt.mean_sem_page(sd, tests)
        for text in fig.findobj(plt.matplotlib.text.Text):
            value = text.get_text().replace('± SEM', '± SD').replace('±1 SEM', '±1 SD')
            value = value.replace('SEM = sample SD / square root of trial count.', 'SD = sample standard deviation across trials.')
            text.set_text(value)
        fig.savefig(out/'04_clusters_K3/K3_7_reaction_time_mean_sd.png', dpi=100)
        fig.savefig(out/'04_clusters_K3/K3_7_reaction_time_mean_sd.pdf')
        plt.close(fig)
    sensitivity = out/'08_scaling_sensitivity'
    with ExitStack() as context:
        for name, value in dict(SOURCE=out, OUT=sensitivity,
            WORRY_A=sensitivity/'worry_a_between_mouse_offsets',
            WORRY_B=sensitivity/'worry_b_feature_magnitudes',
            FROZEN_SHA=hashlib.sha256((out/'selected_session_clusters.csv').read_bytes()).hexdigest(),
            FEATURES=four.FEATURES, LABELS={d[0]: d[1].replace('\n', ' ') for d in four.DEFINITIONS},
            FAMILY={d[0]: d[2] for d in four.DEFINITIONS},
            README=four.four_text(scaling.README).replace('python -m post_analysis.session_clustering.scaling_sensitivity_seven_features',
                'python -m post_analysis.session_clustering.pattern_rbf_four_features_star')).items():
            context.enter_context(patch.object(scaling, name, value))
        original_save = scaling.save
        def save(fig, path):
            for text in fig.findobj(plt.matplotlib.text.Text):
                value = text.get_text()
                if 'Without scaling, the sensory gain owns most' in value:
                    value = ('Bars show each coordinate\'s measured share of squared distance. '
                             'Compare raw nats with within-mouse standardized gains; no model refits.')
                if '\nUnscaled, the distance is ~80%' in value:
                    value = value.split('\nUnscaled,')[0] + '\nClustering and feature separation are recomputed for this four-coordinate cohort.'
                value = value.replace('every family contributes, unlike the raw-nats clustering.',
                                      'compare the measured feature separation across scalings.')
                text.set_text(four.four_text(value))
            original_save(fig, path)
        context.enter_context(patch.object(scaling, 'save', save))
        scaling.run()
    dest = out/'07_hypothesis_followup'
    original_readme = hypothesis.write_readme
    def write_readme(target, tables, manifest):
        original_readme(target, tables, manifest)
        path = target/'README.md'
        text = four.four_text(path.read_text()).replace('C1=11, C2=25, C3=56', ', '.join(f'C{c}={n}' for c, n in sizes.items()))
        text = text.replace('python -m post_analysis.session_clustering.hypothesis_followup',
                            'python -m post_analysis.session_clustering.pattern_rbf_four_features_star')
        text += ('\nThese are the same exploratory questions and plot assignments as the seven-feature follow-up, '
                 'recomputed with four-feature labels. Cluster numbers do not establish the old strategy interpretations. '
                 'All underlying comparisons are retained for every cluster in the tables.\n')
        path.write_text(text)
    with patch.object(hypothesis, 'PARENT', out), patch.object(hypothesis, 'EXPECTED_SIZES', sizes), \
         patch.object(hypothesis, 'LABELS', {c: f'Exploratory observed-behavior comparisons, group {c}' for c in sizes}), \
         patch.object(hypothesis, 'write_readme', write_readme):
        hypothesis.run(dest)
    completion = json.loads((out/'completion.json').read_text())
    completion['pdf_pages'] = {str(p.relative_to(out)): len(PdfReader(p).pages) for p in sorted(out.rglob('*.pdf'))}
    completion['supplements_complete'] = True
    (out/'completion.json').write_text(json.dumps(completion, indent=2)+'\n')
    readme = out/'README.md'
    readme.write_text(readme.read_text().replace('has six pages:', 'has six core pages (K=3 has two extra RT pages):') + '\n## Later analyses\n\n'
        '`07_hypothesis_followup/`: eight matched observed-behavior follow-ups and all-cluster tables.\n\n'
        '`08_scaling_sensitivity/`: pooled z and raw-nats clustering comparisons, distance budgets and separation.\n\n'
        '`04_clusters_K3/K3_report.pdf`: eight pages including the extended RT plots; separate mean ± SD figure also included. '
        'Pooled-trial Welch tests and SEM reproduce the original descriptive procedure and treat trials as independent.\n')
    manifest_path = out/'06_audit/source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for path in (Path(__file__), Path(four.__file__), Path(hypothesis.__file__), Path(scaling.__file__), Path(rt.__file__)):
        item = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest = [row for row in manifest if row['path'] != item['path']] + [item]
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(four.OUT)
