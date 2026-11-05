"""Seven-feature RBF on single-session TRAINING prefixes, with all prior plots.

python -m post_analysis.session_clustering.pattern_rbf_single_session_train
No fits are run; all old scripts and results remain unchanged.
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from . import pattern_rbf_seven_features as base
from . import features, pattern_exploration as pe, pattern_report as pr, pattern_visuals as pv
from . import behavior, sequence_performance as seq
from .cluster import SEED, mouse_checks
from post_analysis.expanded_model_comparison.config import MODEL_SPECS

ROOT = base.ROOT
SINGLE_ROOT = features.REPO_ROOT/'results_single_session'
OUT = ROOT/'single_session_train_pattern_exploration_rbf_seven_features'
VARIANT = 'rbf_single_session_train'
TITLE = 'RBF: single-session training-prefix gains'
KEYS = ['subject', 'session_id']
EXPECTED_COUNTS = {'MC06': 32, 'MC07': 30, 'MC08': 30}


def bool_mask(series):
    values = series.astype(str).str.lower()
    if not values.isin(['true', 'false']).all():
        raise ValueError('Invalid model evaluation flags')
    return values.eq('true')


def training_population(table):
    """Validate the literal last-two-block split; keep all prefix rows in order."""
    if table.session_id.nunique() != 1 or table.source_trial_index.duplicated().any():
        raise ValueError('Expected unique chronological trials for one session')
    table = table.sort_values('source_trial_index').reset_index(drop=True).copy()
    if not np.array_equal(table.source_trial_index, np.arange(len(table))):
        raise ValueError('Single-session source population is incomplete')
    if set(table.split_role) != {'train', 'test'}:
        raise ValueError('Expected training prefix and held-out final two blocks')
    blocks = table.block_index.drop_duplicates().tolist()
    if len(blocks) < 3 or not table.block_index.is_monotonic_increasing:
        raise ValueError('Invalid chronological block structure')
    expected_train = ~table.block_index.isin(blocks[-2:])
    if not expected_train.equals(table.split_role.eq('train')):
        raise ValueError('Training split is not the prefix before literal final two blocks')
    table['model_evaluated'] = bool_mask(table.model_evaluated)
    return table[expected_train].copy()


def training_score(prefix):
    if not prefix.split_role.eq('train').all():
        raise ValueError('Only training-prefix trials may enter training scores')
    ev = prefix[prefix.model_evaluated]
    if ev.empty or not ev.choice_long.isin([0, 1]).all():
        raise ValueError('No valid binary training decisions')
    p = ev.model_probability_long.to_numpy(float)
    y = ev.choice_long.to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid training choice probabilities')
    # Saved probabilities already obey model-specific clipping; check the
    # standard output contract without ever reading test losses for a score.
    p = np.clip(p, 1e-12, 1-1e-12)
    reconstructed = -y*np.log(p)-(1-y)*np.log1p(-p)
    np.testing.assert_allclose(ev.per_trial_negative_log_likelihood, reconstructed, atol=1e-9, rtol=1e-7)
    return float(ev.per_trial_negative_log_likelihood.mean()), len(ev)


def canonical_sessions(spec, subject):
    root = SINGLE_ROOT/spec.directory/subject/'full_session_matched_v1'
    if not root.exists():
        roots = list(root.parent.rglob('full_session_matched_v1'))
        if len(roots) != 1:
            raise ValueError(f'No unique single-session root: {subject}/{spec.model_id}')
        root = roots[0]
    result = {}
    for path in sorted(root.glob('session_*/single_session_manifest.json')):
        manifest = json.loads(path.read_text())
        sid = manifest['session_id']
        if sid in result:
            raise ValueError(f'Duplicate single-session manifest: {subject}/{sid}')
        result[sid] = path
    if not result:
        raise ValueError(f'No single-session fits: {subject}/{spec.model_id}')
    return result


def collect(out):
    rows, canonical, audits, sources, baseline_rows = [], [], [], {}, []
    identity = ['session_id', 'trial_index', 'source_trial_index', 'block_index', 'split_role',
                'model_evaluated', 'choice_long', 'duration_s', 'category_long', 'warm_up', 'outcome']
    def read(path):
        payload = path.read_bytes(); sources[str(path)] = hashlib.sha256(payload).hexdigest()
        return payload
    for subject in features.SUBJECTS:
        paths = {spec.model_id: canonical_sessions(spec, subject) for spec in MODEL_SPECS}
        cohort = set(paths['m0_sensory'])
        if any(set(p) != cohort for p in paths.values()):
            raise ValueError(f'Canonical single-session cohorts differ across models: {subject}')
        for sid in sorted(cohort):
            row = dict(subject=subject, session_id=sid)
            reference = None
            for spec in MODEL_SPECS:
                path = paths[spec.model_id][sid]
                manifest = json.loads(read(path))
                if manifest.get('status') not in ('complete', 'ineligible'):
                    raise ValueError(f'Incomplete fit artifacts: {path}')
                if manifest['mouse_id'] != subject or manifest['model_id'] != spec.directory:
                    raise ValueError(f'Manifest model/subject mismatch: {path}')
                pars = np.asarray(manifest['best_constrained_parameters'], float)
                if len(pars) != len(spec.parameter_names) or not np.isfinite(pars).all():
                    raise ValueError(f'Invalid fitted vector: {path}')
                full = pd.read_csv(BytesIO(read(path.parent/'trial_predictions.csv')))
                if set(full.session_id) != {sid} or full.chronological_rank.nunique() != 1 or int(full.chronological_rank.iloc[0]) != int(manifest['chronological_rank']):
                    raise ValueError(f'Session identity/rank differs from manifest: {path}')
                prefix = training_population(full)
                if reference is None:
                    reference = prefix[identity].reset_index(drop=True)
                    row['chronological_rank'] = int(manifest['chronological_rank'])
                    # Compatibility aliases for existing explanatory plot helpers,
                    # both identify this single session, not a rolling-window fit.
                    row['window_id'] = row['test_rank'] = row['chronological_rank']
                else:
                    pd.testing.assert_frame_equal(reference, prefix[identity].reset_index(drop=True), check_dtype=False, rtol=0, atol=1e-12)
                loss, n = training_score(prefix)
                metric_paths = [p for p in [path.parent/'prefix_and_heldout_metrics.csv', path.parent/'split_and_session_metrics.csv'] if p.is_file()]
                if not metric_paths:
                    raise ValueError(f'No saved single-session metrics: {path}')
                metrics = pd.read_csv(BytesIO(read(metric_paths[0])))
                met = metrics[metrics.split_role.eq('train') & metrics.session_id.eq('ALL_SESSIONS')]
                if len(met) != 1 or int(met.valid_trial_count.iloc[0]) != n:
                    raise ValueError(f'Training count differs from saved metrics: {path}')
                np.testing.assert_allclose(loss, met.mean_nll.iloc[0], atol=1e-10, rtol=1e-9)
                row['nll__'+spec.model_id] = loss
                row['n__'+spec.model_id] = n
                for name, value in zip(spec.parameter_names, pars):
                    if 'fitted_'+name not in prefix:
                        raise ValueError(f'Missing fitted parameter annotation: {name}/{path}')
                    np.testing.assert_allclose(prefix['fitted_'+name], value, atol=1e-10, rtol=1e-9)
                    row[f'param__{spec.model_id}__{name}'] = value
                audits.append(dict(subject=subject, session_id=sid, model=spec.model_id,
                                   chronological_rank=row['chronological_rank'], status=manifest['status'],
                                   converged=manifest.get('converged'), termination_reason=manifest.get('termination_reason'),
                                   train_rows=len(prefix), evaluated_train_trials=n, heldout_rows=len(full)-len(prefix),
                                   train_mean_nll=loss, parameters_verified=True, paired_population_verified=True))
                if spec.model_id == 'm0_sensory':
                    q, baseline = base.constant_choice_nll(prefix.loc[prefix.model_evaluated, 'choice_long'], prefix.loc[prefix.model_evaluated, 'choice_long'])
                    row['gain_bias__m0_sensory'] = baseline-loss
                    baseline_rows.append(dict(subject=subject, session_id=sid, fitted_on='same_session_train',
                                              scored_on='same_session_train', n_trials=n, p_long=q,
                                              baseline_mean_nll=baseline, sensory_mean_nll=loss, gain=baseline-loss))
                    canonical.append(prefix.assign(subject=subject))
            rows.append(row)
        print('Single-session training fits audited:', subject, len(cohort), flush=True)
    raw = pd.DataFrame(rows).sort_values(['subject', 'chronological_rank']).reset_index(drop=True)
    if raw.groupby('subject').size().to_dict() != EXPECTED_COUNTS or raw.duplicated(KEYS).any():
        raise ValueError('Expected the same 92 unique sessions')
    pd.DataFrame(audits).to_csv(out/'single_session_input_audit.csv', index=False)
    pd.DataFrame(baseline_rows).to_csv(out/'sensory_baseline_audit.csv', index=False)
    # Existing raw-RT cache was independently verified in the prior workflow.
    # Match the current canonical prefixes exactly; never take its old fit mask,
    # labels, full-session metrics, or rolling-window parameters.
    cache_path = base.OLD/'reaction_time_trials.csv'
    cache = pd.read_csv(BytesIO(read(cache_path))).drop(columns='cluster_label', errors='ignore')
    if cache.duplicated(KEYS+['source_trial_index']).any():
        raise ValueError('Duplicate raw-cache trial identity')
    prefix = pd.concat(canonical, ignore_index=True)
    join = prefix.merge(cache, on=KEYS+['source_trial_index'], how='left', suffixes=('', '_cache'),
                        validate='one_to_one', indicator=True)
    if not join._merge.eq('both').all():
        raise ValueError('Training-prefix trials missing from raw RT cache')
    for col in ['trial_index', 'duration_s', 'category_long', 'warm_up', 'choice_long']:
        np.testing.assert_allclose(join[col], join[col+'_cache'], equal_nan=True, rtol=0, atol=1e-10)
    for col in ['true_block', 'outcome']:
        if not join[col].fillna('<NA>').equals(join[col+'_cache'].fillna('<NA>')):
            raise ValueError(f'Raw-cache observation mismatch: {col}')
    for col in ['lick_reaction_time', 'negative_lick', 'latency_s', 'latency_eligible', 'outcome_strict', 'is_common']:
        prefix[col] = join[col].to_numpy()
    prefix = prefix.sort_values(KEYS+['source_trial_index']).reset_index(drop=True)
    prefix.to_csv(out/'training_prefix_trials.csv', index=False)
    prefix.to_csv(out/'reaction_time_trials.csv', index=False)
    rt_audit = prefix.groupby(KEYS).agg(n_training_rows=('trial_index', 'size'), n_retained=('latency_eligible', 'sum')).reset_index()
    rt_audit['population'] = 'single-session training prefix only'
    rt_audit.to_csv(out/'reaction_time_input_audit.csv', index=False)
    values, meta = pd.DataFrame(index=raw.index), []
    for spec in MODEL_SPECS:
        model = spec.model_id
        col = 'performance__'+model
        values[col] = np.log(2)-raw['nll__'+model] if model == 'm0_sensory' else raw.nll__m0_sensory-raw['nll__'+model]
        meta.append(dict(feature=col, block='performance', model=model, oracle=False,
                         description='training gain vs fair coin' if model == 'm0_sensory' else 'training gain vs sensory-only'))
        for name in spec.parameter_names:
            col = f'param__{model}__{name}'
            values[col] = features.transform_parameter(name, raw[col])
            meta.append(dict(feature=col, block=features.parameter_block(name), model=model, oracle=False, description=name))
    return raw, values, pd.DataFrame(meta), prefix, sources


def observed_metrics(trials, raw, out):
    """Original audited interpretation definitions, recomputed on prefixes only."""
    if not trials.split_role.eq('train').all():
        raise ValueError('Behavior summaries require training-prefix trials only')
    t = trials.copy(); t['correct'] = (t.choice_long == t.category_long).astype(float)
    valid = t.model_evaluated & t.choice_long.isin([0, 1]) & t.warm_up.eq(0)
    t['stim_bin'] = -1
    edges = {}
    for cat in [0, 1]:
        _, bins = pd.qcut(t.loc[valid & t.category_long.eq(cat), 'duration_s'], q=4, retbins=True, duplicates='drop')
        if len(bins) < 2:
            raise ValueError('Insufficient category-specific ISI variation')
        bins[0], bins[-1] = -np.inf, np.inf; edges[str(cat)] = bins.tolist()
        ix = t.category_long.eq(cat)
        t.loc[ix, 'stim_bin'] = pd.cut(t.loc[ix, 'duration_s'], bins, labels=False, include_lowest=True).astype(int)+cat*4
    index = pd.MultiIndex.from_frame(raw[KEYS]); result = pd.DataFrame(index=index)
    def add(name, series): result[name] = series.reindex(index)
    v = t[valid]
    add('balanced_accuracy', v.groupby(KEYS+['stim_bin']).correct.mean().groupby(KEYS).mean())
    add('p_long', v.groupby(KEYS).choice_long.mean())
    repeat = v.sort_values(KEYS+['trial_index']).copy()
    repeat['repeat'] = repeat.groupby(KEYS).choice_long.diff().eq(0).where(repeat.groupby(KEYS).cumcount()>0)
    add('choice_repeat_rate', repeat.groupby(KEYS).repeat.mean())
    block = v[v.true_block.isin(['S', 'L'])].groupby(KEYS+['stim_bin', 'true_block']).agg(p_long=('choice_long', 'mean'), n=('choice_long', 'size')).reset_index()
    block = block[block.n >= 2].pivot(index=KEYS+['stim_bin'], columns='true_block', values='p_long').reindex(columns=['S', 'L']).dropna()
    add('block_context_shift', (block.L-block.S).groupby(KEYS).mean())
    rare = v.true_block.isin(['S', 'L']) & v.category_long.ne(v.true_block.map({'S': 0, 'L': 1}))
    rates = v[v.true_block.isin(['S', 'L'])].assign(rare=rare).groupby(KEYS+['rare']).correct.mean().unstack('rare').reindex(columns=[False, True])
    add('rare_accuracy_cost', rates[False]-rates[True])
    rt = t[t.latency_eligible]
    add('rt_stimulus_balanced', rt.groupby(KEYS+['stim_bin']).latency_s.mean().groupby(KEYS).mean())
    event_tables = []
    for alignment in ['adjusted', 'formal']:
        events = []
        for (subject, sid), g in t.groupby(KEYS, sort=False):
            sl = g.sort_values('trial_index'); sl = sl[sl.true_block.isin(['S', 'L'])].reset_index(drop=True)
            b, cat = sl.true_block.to_numpy(), sl.category_long.to_numpy()
            is_rare = ((b == 'S') & (cat == 1)) | ((b == 'L') & (cat == 0))
            run = sl.true_block.ne(sl.true_block.shift()).cumsum()
            early_rare = is_rare & ((sl.groupby(run).cumcount()+1).to_numpy() <= 4)
            ok = sl.model_evaluated.to_numpy() & sl.choice_long.isin([0, 1]).to_numpy()
            for event_id, formal in enumerate(np.flatnonzero(b[1:] != b[:-1])+1):
                kind = 'S2L' if b[formal] == 'L' else 'L2S'
                anchor = behavior.adjusted_start(is_rare, b, formal) if alignment == 'adjusted' else formal
                start, end = formal-1, formal
                while start > 0 and b[start-1] == b[formal-1]: start -= 1
                while end+1 < len(sl) and b[end+1] == b[formal]: end += 1
                for rel in [0, 1, 3, 4]:
                    i = anchor+rel
                    if i < start or i > end or early_rare[i] or not ok[i]: continue
                    events.append(dict(subject=subject, session_id=sid, event_id=event_id, kind=kind,
                                       category=int(cat[i]), phase='early' if rel < 2 else 'late',
                                       stim_bin=int(sl.stim_bin.iloc[i]), correct=float(sl.correct.iloc[i]),
                                       source_trial_index=int(sl.source_trial_index.iloc[i]), alignment=alignment))
        e = pd.DataFrame(events, columns=KEYS+['event_id', 'kind', 'category', 'phase', 'stim_bin', 'correct', 'source_trial_index', 'alignment'])
        event_tables.append(e)
        for kind, cat in [('S2L', 1), ('L2S', 0)]:
            selected = e[e.kind.eq(kind) & e.category.eq(cat)]
            if selected.empty:
                result[f'{alignment}_rapid_{kind}_recovery'] = np.nan
                if alignment == 'adjusted': result[f'{alignment}_rapid_{kind}_stim_matched_recovery'] = np.nan
                continue
            paired = selected.groupby(KEYS+['event_id', 'phase']).correct.mean().unstack('phase').reindex(columns=['early', 'late']).dropna()
            add(f'{alignment}_rapid_{kind}_recovery', (paired.late-paired.early).groupby(KEYS).mean())
            if alignment == 'adjusted':
                matched = selected.groupby(KEYS+['stim_bin', 'phase']).correct.mean().unstack('phase').reindex(columns=['early', 'late']).dropna()
                means = (matched.late-matched.early).groupby(KEYS).mean()
                means[matched.groupby(KEYS).size() < 2] = np.nan
                add(f'{alignment}_rapid_{kind}_stim_matched_recovery', means)
    fit = raw.set_index(KEYS)
    for model in ['m1_action_kernel', 'm1_stimulus_kernel', 'm2_static']:
        result['gain_vs_sensory_'+model] = fit.nll__m0_sensory-fit['nll__'+model]
    result['dynamic_gain'] = fit.nll__m2_static-fit.nll__m2_dynamic
    result['repetition_increment'] = fit.nll__m2_dynamic-fit.nll__m2_dynamic_repetition_bias
    pd.concat(event_tables).to_csv(out/'training_transition_trials.csv', index=False)
    (out/'behavior_bin_edges.json').write_text(json.dumps(edges, indent=2))
    return result[pe.METRICS].reset_index()


def adapt_figure(fig):
    for text in fig.findobj(plt.Text):
        s = text.get_text()
        for old, new in [('test-NLL', 'training-NLL'), ('test NLL', 'training NLL'),
                         ('test trials', 'the same training trials'), ('test sessions', 'training prefixes'),
                         ('test-session', 'training-prefix'), ('test session', 'training prefix'),
                         ('held-out', 'training-prefix'), ('heldout', 'training-prefix'),
                         ('training-window fits', 'single-session training-prefix fits'),
                         ('training-window', 'single-session training-prefix'),
                         ('Fitting window / associated', 'Single-session fit /'),
                         ('final session block omitted', 'final training-prefix block also omitted'),
                         ('per window', 'per session'), ('Seven-feature RBF', 'Single-session training RBF')]:
            s = s.replace(old, new)
        if s.startswith('Seven signed predictive gains'): s = s.replace('Seven signed predictive gains', 'Single-session training: seven signed gains')
        if s.startswith('Actual seven clustering coordinates'): s = s.replace('Actual seven clustering coordinates', 'Seven training-gain clustering coordinates')
        if s.startswith('RBF cluster'): s = s.replace('RBF cluster', 'Training-prefix RBF cluster', 1)
        s = s.replace('L = mean training NLL.', 'L = mean in-sample training NLL.')
        text.set_text(s)


class TrainingPdfPages(PdfPages):
    def savefig(self, figure=None, **kwargs):
        if figure is None: figure = plt.gcf()
        adapt_figure(figure)
        return super().savefig(figure, **kwargs)


def sequence_population(trials, labels):
    if not trials.split_role.eq('train').all():
        raise ValueError('Sequence plots require training-prefix trials only')
    ll = labels[labels.K.eq(3)].copy()
    data = behavior.attach_labels(trials, ll)
    data['block_type'] = data.true_block.map({'N': 0, 'S': 1, 'L': 2})
    data['trial_type'] = data.category_long+1
    data['_source_order'] = data.source_trial_index
    data['change_session'] = data.subject+'|'+data.session_id
    data['correct'] = data.outcome.map({'Reward': 1., 'RewardNaive': 1., 'Punish': 0., 'PunishNaive': 0.})
    data['is_rare'] = data.block_type.isin([1, 2]) & data.trial_type.ne(data.block_type)
    data = data[data.warm_up.eq(0)].sort_values(['change_session', '_source_order']).copy()
    for sid, g in data.groupby('change_session'):
        biased = g[g.block_type.isin([1, 2])]
        if not np.array_equal(np.diff(biased._source_order), np.ones(max(0, len(biased)-1))):
            raise ValueError(f'Internal gap in training sequence population: {sid}')
    return data, ll


def sequence_reports(trials, labels, out):
    data, ll = sequence_population(trials, labels)
    dest = out/'sequence_performance'; dest.mkdir()
    source = seq.source_module()
    e = seq.collect_strict_common_events(data, source); f = source.collect_sequence_events(data, seq.adjusted_block_start)
    ew = seq.tag_events(source.collect_complete_train_sliding_events(e), data)
    fw = seq.tag_events(source.collect_complete_train_sliding_events(f), data)
    summary = pd.concat([seq.summarize(ew, 'E'), seq.summarize(fw, 'F')], ignore_index=True)
    for name, frame in [('common_outcome_window_events', ew), ('rare_sequence_window_events', fw), ('window_summary', summary), ('session_clusters_used', ll)]:
        frame.to_csv(dest/(name+'.csv'), index=False)
    display = ll.copy(); display['subject'] = pd.Categorical(display.subject, categories=seq.SUBJECTS)
    with patch.object(seq, 'PdfPages', TrainingPdfPages): seq.make_figures(summary, display, dest)
    methods = json.loads((ROOT/'version1/sliding_window_pattern_exploration/sequence_performance/methods.json').read_text())
    methods.update(variant=VARIANT, population='Single-session TRAINING prefixes only; final two held-out blocks removed before any event construction.',
                   exclusions='Warm-up/neutral removed as before; final adjusted block of each TRAINING PREFIX also omitted by the original collector. No held-out block contributes.',
                   session_counts=ll.cluster_label.value_counts().sort_index().to_dict(), window_size=source.WINDOW_SIZE,
                   window_step=source.WINDOW_STEP, max_window_center=source.MAX_WINDOW_CENTER,
                   source_sha256=hashlib.sha256((seq.RAW_ROOT/'block_sequence_epoch_analysis.py').read_bytes()).hexdigest(),
                   selected_label_file_sha256=hashlib.sha256((out/'selected_session_clusters.csv').read_bytes()).hexdigest())
    (dest/'methods.json').write_text(json.dumps(methods, indent=2)+'\n')


def analyze(out):
    if {s.label for s in MODEL_SPECS} != base.EXPECTED: raise ValueError('Model roster changed')
    raw, values, manifest, trials, sources = collect(out)
    gains, feature_meta = base.gain_features(raw)
    feature_meta['variant'] = VARIANT
    feature_meta['interpretation'] = feature_meta.interpretation.str.replace('held-out predictive gain', 'in-sample training gain', regex=False)
    z, scaling = base.standardized(gains, raw.subject.to_numpy())
    affinity, bandwidth = pe.gaussian_affinity(z)
    raw.to_csv(out/'session_metrics_parameters.csv', index=False)
    pd.concat([raw[KEYS+['chronological_rank']], gains], axis=1).to_csv(out/'signed_feature_values.csv', index=False)
    pd.concat([raw[KEYS+['chronological_rank']], pd.DataFrame(z, columns=base.FEATURES)], axis=1).to_csv(out/'standardized_feature_values.csv', index=False)
    feature_meta.to_csv(out/'feature_manifest.csv', index=False); scaling.to_csv(out/'feature_scaling.csv', index=False)
    np.save(out/'clustering_matrix.npy', z); np.save(out/'affinity_rbf_single_session_train.npy', affinity)
    obs = observed_metrics(trials, raw, out); obs.to_csv(out/'observed_session_metrics.csv', index=False)
    reference = pd.read_csv(base.OLD/'session_clusters.csv')
    previous = pd.read_csv(base.OUT/'selected_session_clusters.csv')
    keys = pd.MultiIndex.from_frame(raw[KEYS])
    labels, profiles, contrasts, summaries = [], [], [], []
    for k in (2, 3):
        lab = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', n_init=20,
                                 eigen_solver='arpack', random_state=SEED).fit_predict(affinity)+1
        ref = reference[reference.K.eq(k)].set_index(KEYS).reindex(keys).cluster_label.to_numpy()
        prev = previous[previous.K.eq(k)].set_index(KEYS).reindex(keys).cluster_label.to_numpy()
        if not np.isfinite(ref).all() or not np.isfinite(prev).all(): raise ValueError('Missing reference labels')
        lab, mapping = pe.align_labels(lab, ref, k)
        ll = raw[KEYS+['chronological_rank', 'window_id', 'test_rank']].copy()
        ll['variant'], ll['K'], ll['cluster_label'] = VARIANT, k, lab; labels.append(ll)
        tagged = obs.copy(); tagged['cluster_label'] = lab
        p, c = pe.behavior_profiles(tagged, k); p['variant'], p['K'] = VARIANT, k; c['variant'], c['K'] = VARIANT, k
        profiles.append(p); contrasts.append(c)
        sizes = pd.Series(lab).value_counts().reindex(range(1, k+1)).tolist()
        summaries.append(dict(variant=VARIANT, K=k, method='rbf', n_features=7, n_sessions=len(raw), sizes='/'.join(map(str, sizes)),
                              bandwidth=bandwidth, ari_to_sliding_window_seven_features=adjusted_rand_score(prev, lab),
                              label_mapping=json.dumps(mapping), **mouse_checks(lab, raw.subject.to_numpy())))
        print('Single-session TRAIN RBF:', k, sizes, flush=True)
    labels, profiles = pd.concat(labels, ignore_index=True), pd.concat(profiles, ignore_index=True)
    for name in ['all_candidate_labels', 'selected_session_clusters']: labels.to_csv(out/(name+'.csv'), index=False)
    profiles.to_csv(out/'behavior_profiles_equal_mouse.csv', index=False)
    pd.concat(contrasts).to_csv(out/'behavior_contrasts_by_mouse.csv', index=False)
    pd.DataFrame(summaries).to_csv(out/'candidate_summary.csv', index=False)
    # Existing report helper reloads its trial cache, explicitly routed to THIS
    # staged training-prefix cache; it never reads historical full-session data.
    with ExitStack() as context:
        for obj, key, value in [(base, 'OLD', out), (base, 'VARIANT', VARIANT), (base, 'PdfPages', TrainingPdfPages),
                                (pe, 'OUT', out), (pv, 'MODELS', base.MODELS),
                                (pv, 'NAMES', {**pv.NAMES, VARIANT: TITLE}), (pr, 'NAMES', {**pr.NAMES, VARIANT: TITLE}),
                                (pr, 'SHORT', {s.model_id: s.model_id for s in MODEL_SPECS}),
                                (pv, 'names_for', lambda variant, k: {c: f'Group {c}' for c in range(1, k+1)})]:
            context.enter_context(patch.object(obj, key, value))
        pe.export_functional_readouts(raw, labels)
        base.render_reports(raw, values, manifest, gains, z, labels, profiles, obs, out)
    # The legacy exporter has a historical test_ column name; rename it here.
    for name in ['functional_model_readouts.csv', 'selected_functional_parameter_profiles.csv']:
        p = out/name; frame = pd.read_csv(p)
        frame.rename(columns={'test_gain_vs_sensory': 'train_gain_vs_sensory'}).to_csv(p, index=False)
    sequence_reports(trials, labels, out)
    protocol = dict(variant=VARIANT, models=[dict(model_id=s.model_id, label=s.label) for s in MODEL_SPECS],
                    n_sessions=len(raw), counts_by_subject=raw.groupby('subject').size().to_dict(),
                    features=feature_meta.to_dict('records'), scoring='Mean NLL on evaluated single-session TRAINING PREFIX trials only, in nats/trial; signed differences retained.',
                    baseline='Constant p(long) MLE fitted and scored on the SAME evaluated training-prefix choices.',
                    inference='In-sample descriptive fit comparison; no held-out predictive-validation claim.',
                    training_scope='All blocks preceding the literal final two blocks. All chronological prefix rows retained for behavior/sequence continuity.',
                    scaling='Within mouse sample SD (ddof=1), clip +/-3, equal weight=1 per coordinate.',
                    kernel='exp(-squared_distance/(2*median_positive_squared_distance))', bandwidth=bandwidth,
                    spectral=dict(seed=SEED, K=[2, 3], assign_labels='kmeans', n_init=20, eigen_solver='arpack'),
                    fit_policy='All finite saved fits retained, including nonconverged/ineligible diagnostic flags. No model refits.',
                    compatibility_aliases='window_id and test_rank in explanatory helpers equal chronological_rank and identify a SINGLE SESSION, not a rolling-window fit.',
                    behavior='Recomputed on training prefixes only; original matched-ISI-bin/context/recovery definitions.',
                    label_numbering='Overlap alignment to the same legacy reference; reference labels never enter features.')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for path in [Path(__file__), Path(base.__file__), base.OLD/'session_clusters.csv', base.OUT/'selected_session_clusters.csv', seq.RAW_ROOT/'block_sequence_epoch_analysis.py']:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'source_manifest.json').write_text(json.dumps(sources, indent=2)+'\n')
    (out/'README.md').write_text('# Single-session training-prefix RBF\n\nRun `python -m post_analysis.session_clustering.pattern_rbf_single_session_train`.\n\n'
        'Same 13 models and 92 sessions; same seven signed gains, within-mouse scaling, equal weights and spectral RBF K=2/K=3. '
        'NLL and parameter inputs come exclusively from saved SINGLE-SESSION fits. The scoring population is each session\'s training prefix, excluding the final two held-out blocks. '
        'The constant-choice baseline is fitted and scored on that same prefix. These are in-sample fit gains.\n\n'
        'All nine cluster/profile pages, four behavior/RT pages and original K=3 E/F sequence PDFs are regenerated. '
        'Behavior/RT/sequence populations are also training-only. Sequence collectors retain their original final-block omission, now applied within the prefix. '
        'Missing behavioral metrics remain missing. Diagnostic fit flags are exported without excluding rows.\n\n'
        'See protocol.json, single_session_input_audit.csv, sensory_baseline_audit.csv, signed_feature_values.csv and source_manifest.json. '
        'Previous source code, outputs and fit inputs remain unchanged.\n')
    return sources


def protected_hashes():
    paths = [p for p in ROOT.rglob('*') if p.is_file() and OUT not in p.parents and not any('.single_train_staging_' in x for x in p.parts)]
    paths += [p for p in Path(__file__).parent.glob('*.py') if p.resolve() != Path(__file__).resolve()]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run():
    before = protected_hashes()
    stage = Path(tempfile.mkdtemp(prefix='.single_train_staging_', dir=ROOT))
    try:
        sources = analyze(stage)
        if before != protected_hashes(): raise RuntimeError('Historical analysis files changed')
        for path, digest in sources.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest: raise RuntimeError(f'Input changed during run: {path}')
        from pypdf import PdfReader
        # Own report layout (base.render_reports); the sliding-window folder now publishes different reports.
        expected = {'clusters.pdf': 9, 'cluster_behavior_pooled.pdf': 4,
                    'sequence_performance/rbf_sequence_overview.pdf': 3,
                    'sequence_performance/rbf_sequence_by_cluster.pdf': 9,
                    **{f'sequence_performance/rbf_cluster_{c}_sequence_performance.pdf': 3 for c in (1, 2, 3)}}
        for name, count in expected.items():
            if len(PdfReader(stage/name).pages) != count: raise RuntimeError(f'Incomplete PDF: {name}')
        (stage/'preservation_audit.json').write_text(json.dumps(dict(unchanged=True, n_protected_files=len(before), source_files_unchanged=len(sources), sha256=before), indent=2)+'\n')
        (stage/'completion.json').write_text(json.dumps(dict(complete=True, pdf_pages=expected), indent=2)+'\n')
        backup = None
        if OUT.exists():
            backup = Path(tempfile.mkdtemp(prefix='.single_train_previous_', dir=ROOT)); backup.rmdir(); OUT.rename(backup)
        try: stage.rename(OUT)
        except BaseException:
            if backup is not None: backup.rename(OUT)
            raise
        if backup is not None: shutil.rmtree(backup)
        print('PUBLISHED', OUT, flush=True)
    finally:
        if stage.exists(): shutil.rmtree(stage)


if __name__ == '__main__':
    with threadpool_limits(limits=1): run()
