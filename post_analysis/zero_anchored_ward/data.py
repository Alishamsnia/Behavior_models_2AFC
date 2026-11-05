"""Reconstruct independent inputs from original prediction files and raw behavior."""
from io import BytesIO
import ast
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from post_analysis.expanded_model_comparison.config import MODEL_BY_ID, MODEL_SPECS
from .core import FEATURES, MODELS, KEYS

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / 'results_post_analysis/expanded_model_comparison'
RAW_ROOT = REPO.parents[1] / 'Behavior/New_version'
SUBJECTS = ('MC06', 'MC07', 'MC08')
IDS = ['subject', 'window_id', 'test_rank', 'session_id']
IDENTITY = ['session_id', 'trial_index', 'source_trial_index', 'chronological_rank', 'split_role',
            'choice_long', 'category_long', 'duration_s', 'true_block', 'warm_up', 'model_evaluated']
PRED_COLUMNS = IDENTITY + ['per_trial_negative_log_likelihood', 'model_probability_long', 'outcome']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binary_flags(values):
    flags = values.astype(str).str.lower()
    if not flags.isin(['true', 'false']).all():
        raise ValueError('Invalid model_evaluated flags')
    return flags.eq('true')


def constant_loss(train, test):
    a, b = np.asarray(train, float), np.asarray(test, float)
    if not len(a) or not len(b) or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('Constant baseline requires nonempty binary training and test choices')
    p = np.clip(a.mean(), 1e-12, 1-1e-12)
    return float(p), float(np.mean(-b*np.log(p)-(1-b)*np.log1p(-p)))


def prediction_paths(model, subject):
    root = MODEL_BY_ID[model].result_root(subject)
    if not root.exists():
        candidates = sorted(root.parent.rglob('full_session_matched_v1'))
        if len(candidates) != 1:
            raise ValueError(f'No unique prediction root for {model}/{subject}: {candidates}')
        root = candidates[0]
    paths = {}
    for p in sorted(root.glob('window_*/trial_predictions.csv')):
        number = int(p.parent.name.split('_')[1])
        if number in paths:
            raise ValueError(f'Duplicate window {number}: {root}')
        paths[number] = p
    if not paths:
        raise ValueError(f'No predictions: {root}')
    return paths


def assert_paired(reference, candidate, context=''):
    """Compare the complete timeline, including evaluation masks, before any filtering."""
    try:
        pd.testing.assert_frame_equal(reference[IDENTITY].reset_index(drop=True),
                                      candidate[IDENTITY].reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
    except AssertionError as exc:
        raise ValueError(f'Unpaired model trials: {context}') from exc


def collect():
    rows, trials, audits, baseline, sources = [], [], [], [], {}
    for subject in SUBJECTS:
        folder = BASE / subject / 'tables'
        metrics_path, params_path = folder/'sliding_window_metrics.csv', folder/'extended_parameters.csv'
        met, params = pd.read_csv(metrics_path), pd.read_csv(params_path)
        sources.update({str(p.resolve()): sha(p) for p in (metrics_path, params_path)})
        met = met[met.session_id.eq('ALL_SESSIONS')]
        if met.duplicated(['model_id', 'window_id', 'split_role']).any() or params.duplicated(
                ['model_id', 'window_id', 'param_name']).any():
            raise ValueError(f'Duplicate upstream records: {subject}')
        paths = {m: prediction_paths(m, subject) for m in MODELS}
        expected = set(paths[MODELS[0]])
        if any(set(p) != expected for p in paths.values()):
            raise ValueError(f'Model window populations differ: {subject}')
        for wid in sorted(expected):
            reference = None
            record = dict(subject=subject, window_id=wid)
            for model in MODELS:
                path = paths[model][wid]
                payload = path.read_bytes()
                sources[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
                t = pd.read_csv(BytesIO(payload), usecols=PRED_COLUMNS).sort_values(
                    ['chronological_rank', 'source_trial_index']).reset_index(drop=True)
                t['model_evaluated'] = binary_flags(t.model_evaluated)
                if t.duplicated(['session_id', 'trial_index']).any():
                    raise ValueError(f'Duplicate prediction trial: {path}')
                if reference is None:
                    reference = t
                else:
                    assert_paired(reference, t, str(path))
                evaluated = t[t.model_evaluated]
                if not np.isin(evaluated.choice_long, [0, 1]).all():
                    raise ValueError(f'Nonbinary evaluated choice: {path}')
                p = evaluated.model_probability_long.to_numpy(float)
                losses = evaluated.per_trial_negative_log_likelihood.to_numpy(float)
                if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.isfinite(losses).all():
                    raise ValueError(f'Invalid probabilities/losses: {path}')
                p = np.clip(p, 1e-12, 1-1e-12)
                y = evaluated.choice_long.to_numpy(float)
                np.testing.assert_allclose(losses, -y*np.log(p)-(1-y)*np.log1p(-p), atol=1e-9, rtol=1e-8)
                populations = {s: set(t.loc[t.split_role.eq(s), 'session_id']) for s in ('train', 'validation', 'test')}
                if any(populations[a] & populations[b] for a, b in
                       [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
                    raise ValueError(f'Split leakage: {path}')
                for split in ('train', 'validation', 'test'):
                    e = evaluated[evaluated.split_role.eq(split)]
                    match = met[met.model_id.eq(model) & met.window_id.eq(wid) & met.split_role.eq(split)]
                    if len(match) != 1 or e.empty:
                        raise ValueError(f'Missing metrics/population: {path}/{split}')
                    loss = float(e.per_trial_negative_log_likelihood.mean())
                    np.testing.assert_allclose(loss, match.mean_nll.iloc[0], atol=1e-10, rtol=1e-9)
                    if int(match.valid_trial_count.iloc[0]) != len(e):
                        raise ValueError(f'Evaluated count mismatch: {path}/{split}')
                    audits.append(dict(subject=subject, window_id=wid, model=model, split=split,
                                       n_trials=len(e), reconstructed_mean_nll=loss, matched=True))
                    record[f'{split}_nll__{model}'] = loss
                if model == MODELS[0]:
                    test = t[t.split_role.eq('test')].copy()
                    if test.session_id.nunique() != 1 or test.chronological_rank.nunique() != 1:
                        raise ValueError(f'Expected one held-out session: {path}')
                    record.update(session_id=test.session_id.iloc[0], test_rank=int(test.chronological_rank.iloc[0]))
                    tr, te = evaluated[evaluated.split_role.eq('train')], evaluated[evaluated.split_role.eq('test')]
                    q, loss = constant_loss(tr.choice_long, te.choice_long)
                    record['constant_test_nll'] = loss
                    baseline.append(dict(**{k: record[k] for k in IDS}, train_p_long=q, train_n=len(tr),
                                         test_n=len(te), test_p_long=te.choice_long.mean(), baseline_test_mean_nll=loss))
                    trials.append(test.assign(subject=subject, window_id=wid))
            # All registry parameters/performance are annotations only, not clustering coordinates.
            for spec in MODEL_SPECS:
                mm = met[met.model_id.eq(spec.model_id) & met.window_id.eq(wid) & met.split_role.eq('test')]
                if len(mm) != 1:
                    raise ValueError(f'Missing annotation metric: {subject}/{wid}/{spec.model_id}')
                record['nll__'+spec.model_id] = float(mm.mean_nll.iloc[0])
                for name in spec.parameter_names:
                    pp = params[params.model_id.eq(spec.model_id) & params.window_id.eq(wid) & params.param_name.eq(name)]
                    if len(pp) != 1 or not np.isfinite(pp.param_value.iloc[0]):
                        raise ValueError(f'Missing/nonfinite annotation parameter: {subject}/{wid}/{spec.model_id}/{name}')
                    record[f'param__{spec.model_id}__{name}'] = float(pp.param_value.iloc[0])
            rows.append(record)
        print(f'Audited {subject}: {len(expected)} paired held-out sessions', flush=True)
    raw = pd.DataFrame(rows).sort_values(['subject', 'test_rank']).reset_index(drop=True)
    if raw.duplicated(KEYS).any() or len(raw) != 92:
        raise ValueError(f'Expected 92 unique mouse/session pairs, got {len(raw)}')
    gains = np.column_stack([raw.constant_test_nll - raw.test_nll__m0_sensory] + [
        raw.test_nll__m0_sensory - raw['test_nll__'+m] for m in MODELS[1:]])
    if not np.isfinite(gains).all():
        raise ValueError('Nonfinite reconstructed gains')
    return raw, gains, pd.concat(trials, ignore_index=True), pd.DataFrame(audits), pd.DataFrame(baseline), sources


def has_negative_array(value):
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        a = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Some arrays use unquoted nan; parse numeric tokens without executing input.
        a = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value)
    if not isinstance(a, (list, tuple)):
        return False
    return any(float(v) < 0 for v in a if v is not None)


def raw_behavior(trials, raw_root=RAW_ROOT):
    """Join original raw rows by date and source index, verifying stimulus/context/outcome."""
    pieces, audit, sources = [], [], {}
    columns = ['session_idx', 'session_date', 'trial_isi', 'trial_type', 'block_type', 'warm_up', 'outcome',
               'lick_reaction_time', 'all_lick_reaction_times', 'all_lick_start_times_from_choice', 'Contingency_idx']
    for subject, frame in trials.groupby('subject', sort=False):
        path = raw_root/f'{subject}_final_data.csv'
        source = pd.read_csv(path, usecols=columns)
        sources[str(path.resolve())] = sha(path)
        if source[['session_idx', 'session_date']].drop_duplicates().session_date.duplicated().any():
            raise ValueError(f'Ambiguous session dates: {subject}')
        source['_index'] = source.groupby('session_idx', sort=False).cumcount()
        for sid, t in frame.groupby('session_id', sort=False):
            t = t.sort_values('source_trial_index').reset_index(drop=True)
            date = int(sid.rsplit('_', 2)[-2])
            s = source[source.session_date.eq(date)].set_index('_index')
            if len(s) != len(t) or set(s.index) != set(t.source_trial_index):
                raise ValueError(f'Raw trial population mismatch: {sid}')
            s = s.loc[t.source_trial_index].reset_index(drop=True)
            np.testing.assert_allclose(t.duration_s, s.trial_isi, atol=1e-10, rtol=0)
            for a, b in [(t.category_long, s.trial_type-1), (t.true_block, s.block_type.map({0:'N', 1:'S', 2:'L'})),
                         (t.warm_up, s.warm_up), (t.outcome, s.outcome)]:
                if not np.array_equal(a, b):
                    raise ValueError(f'Raw stimulus/context/outcome mismatch: {sid}')
            if not s.Contingency_idx.eq(2).all():
                raise ValueError(f'Unexpected cohort contingency: {sid}')
            t['negative_lick'] = s.all_lick_reaction_times.map(has_negative_array) | s.all_lick_start_times_from_choice.map(has_negative_array)
            t['latency_s'] = pd.to_numeric(s.lick_reaction_time, errors='coerce')
            t['latency_eligible'] = t.warm_up.eq(0) & ~t.negative_lick & np.isfinite(t.latency_s) & t.latency_s.ge(0)
            t['outcome_strict'] = t.outcome.replace({'RewardNaive':'Reward', 'PunishNaive':'Punish'})
            t['correct'] = t.outcome_strict.map({'Reward':1., 'Punish':0.})
            t['choice_eligible'] = t.warm_up.eq(0) & t.model_evaluated & t.choice_long.isin([0, 1])
            t['is_rare'] = t.true_block.isin(['S', 'L']) & t.category_long.ne(t.true_block.map({'S':0, 'L':1}))
            valid = t.choice_eligible & t.correct.notna()
            if not np.array_equal(t.loc[valid, 'correct'], t.loc[valid, 'choice_long'].eq(t.loc[valid, 'category_long']).astype(float)):
                raise ValueError(f'Choice/outcome mismatch: {sid}')
            audit.append(dict(subject=subject, session_id=sid, n_trials=len(t), n_rt_retained=int(t.latency_eligible.sum()),
                              n_negative_lick=int(t.negative_lick.sum()), raw_verified=True))
            pieces.append(t)
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(audit), sources
