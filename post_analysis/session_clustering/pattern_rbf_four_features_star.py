"""Four-coordinate star RBF; same complete report pipeline as the seven-feature run.

Run: python -m post_analysis.session_clustering.pattern_rbf_four_features_star
"""
from contextlib import ExitStack
from functools import wraps
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
from threadpoolctl import threadpool_limits

from . import pattern_rbf_seven_features as base
from . import pattern_rbf_seven_features_star as star

OUT = base.ROOT / 'sliding_window_pattern_exploration_rbf_four_features_star'
VARIANT = 'rbf_four_features_star'
TITLE = 'RBF: four signed predictive gains, star design'
DEFINITIONS = tuple(d for d in star.DEFINITIONS if d[0] in
                    ('sensory', 'action_history_total', 'wsls_history_total', 'dynamic_context_total'))
FEATURES = [d[0] for d in DEFINITIONS]
FORMULAS = (r'$L(\mathrm{const}) - L(S)$,  $L(S) - L(S{+}AH)$,  '
            r'$L(S) - L(S{+}WH)$,  $L(S) - L(S{+}B_{dynamic})$')
NOTE = ('All three history/context coordinates measure total gain over S. '
        'Positive = better held-out prediction (nats/trial).\n'
        'The constant-choice probability is fitted on training trials. Only these four losses enter clustering; '
        'other model equations are explanatory.')


def four_text(value):
    return (value.replace('Seven-feature', 'Four-feature').replace('seven-feature', 'four-feature')
            .replace('Seven signed', 'Four signed').replace('seven signed', 'four signed')
            .replace('seven clustering', 'four clustering').replace('seven gains', 'four gains')
            .replace('the seven ', 'the four ').replace("session's seven", "session's four"))


def figure_adapter(function):
    @wraps(function)
    def adapted(*args, **kwargs):
        result = function(*args, **kwargs)
        fig = result[0] if isinstance(result, tuple) else result
        for text in fig.findobj(matplotlib.text.Text):
            value = four_text(text.get_text())
            if value.startswith('Coordinate ='):
                value = 'Coordinate = ' + FORMULAS
            if value.startswith('Positive coordinate =') or value.startswith('Sensory gain is against a constant choice'):
                value = NOTE
            text.set_text(value)
        return result
    return adapted


def run_analysis(out):
    pages = base_run_analysis(out)
    star.compare_to_frozen(out)
    readme = four_text((out/'README.md').read_text()).replace(
        'python -m post_analysis.session_clustering.pattern_rbf_seven_features',
        'python -m post_analysis.session_clustering.pattern_rbf_four_features_star')
    readme += ('\n## Four-coordinate star design\n\n'
               'Exactly L(const)-L(S), L(S)-L(S+AH), L(S)-L(S+WH), L(S)-L(S+Bdynamic). '
               'No single-trial or static-context coordinate enters the distances. '
               'All previous clustering modes and outputs are retained. Comparison to seven-coordinate chain labels '
               'is in `07_comparison_to_frozen/`. The affinity keeps the legacy filename '
               '`affinity_rbf_seven_features.npy` for pipeline compatibility; its input has four columns.\n')
    (out/'README.md').write_text(readme)
    protocol = json.loads((out/'protocol.json').read_text())
    protocol['interpretation'] = NOTE
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    manifest = json.loads((out/'source_manifest.json').read_text())
    for path in (Path(__file__), Path(star.__file__)):
        manifest.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (out/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    sequence_readme = out/'sequence_performance/README.md'
    sequence_readme.write_text(four_text(sequence_readme.read_text()))
    return pages


base_run_analysis = base.run_analysis


def run():
    # Include both previous seven-coordinate runs in the preservation audit.
    protected = [base.OUT, star.OUT]
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in protected for p in folder.rglob('*') if p.is_file()}
    before = hashes()
    with ExitStack() as context:
        overrides = dict(OUT=OUT, VARIANT=VARIANT, TITLE=TITLE, DEFINITIONS=DEFINITIONS,
                         EXTRA_REPORT_K=(4,),
                         FEATURES=FEATURES, COLUMN_LABELS=star.COLUMN_LABELS, run_analysis=run_analysis,
                         LAYOUT=tuple((folder, four_text(description), names) for folder, description, names in base.LAYOUT)
                         + ((star.COMPARISON, 'Four-feature star vs seven-feature chain labels', star.COMPARISON_FILES),))
        for name in ('fingerprint_page', 'gain_detail_page', 'session_heatmap_page', 'methods_page',
                     'choice_behavior_page', 'reaction_time_page'):
            overrides[name] = figure_adapter(getattr(base, name))
        for key, value in overrides.items():
            context.enter_context(patch.object(base, key, value))
        base.run()
        from .four_star_supplements import run as run_supplements
        run_supplements(OUT)
    if hashes() != before:
        raise RuntimeError('Previous seven-coordinate results changed')
    print('Previous seven-coordinate chain and star results unchanged', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
