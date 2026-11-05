"""Matched sliding-window performance and parameters; no refitting or session-fit inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from post_analysis.common import REPO_ROOT
from post_analysis.expanded_model_comparison.config import MODEL_SPECS

BASE = REPO_ROOT / 'results_post_analysis' / 'expanded_model_comparison'
LEGACY_OUT = BASE / 'pooled' / 'clustering' / 'sliding_window_final'
OUT = BASE / 'pooled' / 'clustering' / 'sliding_window_compact'
SUBJECTS = ('MC06', 'MC07', 'MC08')
IDS = ['subject', 'window_id', 'test_rank', 'session_id']
TRIAL_COLUMNS = ['session_id', 'trial_index', 'category_long', 'duration_s', 'choice_long',
                 'true_block', 'warm_up', 'model_evaluated', 'chronological_rank', 'split_role']


def test_sessions(subject, extra_columns=()):
    """Use only canonical paths; nested copied outputs are never silently duplicated."""
    root = REPO_ROOT / 'results_sliding_window' / 'M0_sensory_v1' / subject / 'full_session_matched_v1'
    for path in sorted(root.glob('window_*/trial_predictions.csv')):
        window_id = int(path.parent.name.split('_')[1])
        t = pd.read_csv(path, usecols=list(dict.fromkeys(TRIAL_COLUMNS + list(extra_columns))))
        t = t[t.split_role.eq('test')].copy()
        if t.empty or t.session_id.nunique() != 1 or t.chronological_rank.nunique() != 1:
            raise ValueError(f'Expected one test session: {path}')
        if t.trial_index.duplicated().any():
            raise ValueError(f'Duplicate test trials: {path}')
        yield window_id, t


def transform_parameter(name, values):
    x = np.asarray(values, float)
    if not np.isfinite(x).all():
        raise ValueError(f'Nonfinite {name}')
    if name in ('w', 's0', 'kappa'):
        if (x <= 0).any():
            raise ValueError(f'Nonpositive {name}')
        return np.log(np.clip(x, 1e-10, None))
    if name in ('ell_short', 'ell_long'):
        if ((x < 0) | (x > 1)).any():
            raise ValueError(f'Invalid lapse probability {name}')
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))
    # a, ah and eta_logit are already unconstrained logits; bh and delta0 stay linear.
    return x


def parameter_block(name):
    if name in ('a', 'eta_logit'):
        return 'history'
    if name in ('ah', 'bh', 'delta0'):
        return 'prior'
    return 'bias' if name == 'kappa' else 'sensory'


def collect(subjects=SUBJECTS):
    """Fail on incomplete/unmatched data, rather than pivoting away duplicates or imputing fits."""
    feature_frames, manifests, trial_frames = [], [], []
    for subject in subjects:
        folder = BASE / subject / 'tables'
        met = pd.read_csv(folder / 'sliding_window_metrics.csv')
        par = pd.read_csv(folder / 'extended_parameters.csv')
        met = met[met.session_id.eq('ALL_SESSIONS')].copy()
        if met.duplicated(['model_id', 'window_id', 'split_role']).any():
            raise ValueError(f'Duplicate model/window/split metrics for {subject}')
        if par.duplicated(['model_id', 'window_id', 'param_name']).any():
            raise ValueError(f'Duplicate model/window parameters for {subject}')
        rows = []
        for window_id, t in test_sessions(subject):
            rows.append(dict(subject=subject, window_id=window_id,
                             test_rank=int(t.chronological_rank.iloc[0]), session_id=t.session_id.iloc[0]))
            trial_frames.append(t.assign(subject=subject, window_id=window_id))
        if not rows:
            raise ValueError(f'No canonical sliding-window test sessions for {subject}')
        frame = pd.DataFrame(rows).set_index('window_id', drop=False)
        columns = {}
        for spec in MODEL_SPECS:
            te = met[met.model_id.eq(spec.model_id) & met.split_role.eq('test')].set_index('window_id')
            tr = met[met.model_id.eq(spec.model_id) & met.split_role.eq('train')].set_index('window_id')
            va = met[met.model_id.eq(spec.model_id) & met.split_role.eq('validation')].set_index('window_id')
            for split, data in [('test', te), ('train', tr), ('validation', va)]:
                if set(data.index) != set(frame.index):
                    raise ValueError(f'{subject}/{spec.model_id}: {split} windows do not match canonical windows')
            if not te.test_rank.reindex(frame.index).equals(frame.test_rank):
                raise ValueError(f'{subject}/{spec.model_id}: test ranks do not match')
            columns[f'n__{spec.model_id}'] = te.valid_trial_count
            columns[f'validation_nll__{spec.model_id}'] = va.mean_nll
            columns[f'train_nll__{spec.model_id}'] = tr.mean_nll
            columns[f'nll__{spec.model_id}'] = te.mean_nll
            # A constant-bias baseline fitted on training choices, evaluated on test choices.
            q = tr.observed_long_rate.clip(1e-12, 1 - 1e-12)
            r = te.observed_long_rate
            columns[f'gain_bias__{spec.model_id}'] = -r * np.log(q) - (1-r) * np.log1p(-q) - te.mean_nll
            for name in spec.parameter_names:
                pp = par[par.model_id.eq(spec.model_id) & par.param_name.eq(name)].set_index('window_id')
                if set(pp.index) != set(frame.index):
                    raise ValueError(f'{subject}/{spec.model_id}/{name}: missing parameter windows')
                columns[f'param__{spec.model_id}__{name}'] = pp.param_value
        frame = pd.concat([frame, pd.DataFrame(columns).reindex(frame.index)], axis=1)
        counts = frame.filter(regex=r'^n__')
        if not counts.nunique(axis=1).eq(1).all():
            raise ValueError(f'Unmatched test trial counts across models: {subject}')
        feature_frames.append(frame.reset_index(drop=True))
    raw = pd.concat(feature_frames, ignore_index=True).sort_values(['subject', 'test_rank']).reset_index(drop=True)
    if raw.duplicated(['subject', 'session_id']).any():
        raise ValueError('A held-out session occurs in multiple windows')
    values = pd.DataFrame(index=raw.index)
    for spec in MODEL_SPECS:
        model = spec.model_id
        # One absolute-performance anchor; other models contribute paired gains, avoiding repeated difficulty columns.
        col = f'performance__{model}'
        values[col] = (np.log(2) - raw[f'nll__{model}'] if model == 'm0_sensory'
                       else raw['nll__m0_sensory'] - raw[f'nll__{model}'])
        manifests.append(dict(feature=col, block='performance', model=model, oracle=spec.oracle,
                              description='gain vs fair coin' if model == 'm0_sensory' else 'gain vs sensory-only'))
        for name in spec.parameter_names:
            col = f'param__{model}__{name}'
            values[col] = transform_parameter(name, raw[col])
            manifests.append(dict(feature=col, block=parameter_block(name), model=model,
                                  oracle=spec.oracle, description=name))
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('Nonfinite fit parameters/performance; repair source fits before clustering')
    return raw, values, pd.DataFrame(manifests), pd.concat(trial_frames, ignore_index=True)


def normalize(values, subjects):
    """Within-mouse mean/SD, capped at +/-3; zero-variance columns do not invent variation."""
    x = np.asarray(values, float)
    subjects = np.asarray(subjects)
    z = np.zeros_like(x)
    for subject in np.unique(subjects):
        ix = subjects == subject
        sd = x[ix].std(axis=0, ddof=1)
        z[ix] = (x[ix] - x[ix].mean(axis=0)) / np.where(sd > 1e-8, sd, 1.)
    return np.clip(z, -3., 3.)


def select_features(values, manifest, subjects):
    """Fixed physical-spread and redundancy rules, refitted on each resample.

    Legacy manifests have no thresholds and retain their original behavior.
    A compact feature needs meaningful 10th–90th percentile spread in >=2 mice.
    Redundancy is assessed after within-mouse scaling, within a feature block only.
    These are variation screens, not parameter-recovery or reliability estimates.
    """
    meta = manifest[~manifest.oracle].copy().reset_index(drop=True)
    meta['selected'] = True
    meta['exclusion_reason'] = ''
    groups = np.asarray(subjects)
    for i, row in meta.iterrows():
        minimum = row.get('min_spread', np.nan)
        if pd.notna(minimum):
            spreads = [np.ptp(np.quantile(values.loc[groups == subject, row.feature], [.1, .9]))
                       for subject in np.unique(groups)]
            if np.sum(np.asarray(spreads) >= minimum) < min(2, len(spreads)):
                meta.loc[i, ['selected', 'exclusion_reason']] = [False, 'small within-mouse 10–90% spread']
    z = normalize(values[meta.feature], groups)
    for i in range(len(meta)):
        if not meta.loc[i, 'selected']:
            continue
        if np.std(z[:, i]) <= 1e-8:
            meta.loc[i, ['selected', 'exclusion_reason']] = [False, 'constant within mouse']
            continue
        threshold = meta.loc[i].get('redundancy_threshold', np.nan)
        if pd.notna(threshold):
            for j in range(i):
                if meta.loc[j, 'selected'] and meta.loc[j, 'block'] == meta.loc[i, 'block']:
                    corr = np.corrcoef(z[:, i], z[:, j])[0, 1]
                    if abs(corr) >= threshold:
                        meta.loc[i, ['selected', 'exclusion_reason']] = [False, f'redundant with {meta.loc[j, "feature"]}']
                        break
    return meta


def matrix(values, manifest, subjects):
    """Equal squared weight per block and per model; select before scaling."""
    audit = select_features(values, manifest, subjects)
    meta = audit[audit.selected].copy().reset_index(drop=True)
    if meta.empty:
        raise ValueError('No varying non-oracle features')
    weights = []
    for row in meta.itertuples():
        block = meta[meta.block.eq(row.block)]
        weights.append(1 / np.sqrt(block.model.nunique() * int(block.model.eq(row.model).sum())))
    meta['weight'] = weights
    return normalize(values[meta.feature], subjects) * np.asarray(weights), meta


def compact_features(raw, performance_split='validation'):
    """Eight candidate features; target-test behavior is never a default driver.

    All functional parameters originate from nine-session training fits. M0 is
    the single sensory reference. Dynamic M2 is deliberately annotation-only.
    """
    from scipy.special import expit
    if performance_split not in ('validation', 'test'):
        raise ValueError('Expected validation or test')
    values, rows = pd.DataFrame(index=raw.index), []

    def add(name, data, block, model, description, spread):
        values[name] = np.asarray(data, float)
        rows.append(dict(feature=name, block=block, model=model, oracle=False,
                         description=description, min_spread=spread, redundancy_threshold=.95))

    for duration, name in [(.8, 'short'), (1.7, 'long')]:
        noise = np.sqrt(raw.param__m0_sensory__w * duration**2 + raw.param__m0_sensory__s0**2)
        add('noise_' + name, np.log(noise), 'sensory', 'm0_sensory',
            f'log effective noise at {duration:g}s', np.log(1.05))
    add('history_alpha', expit(raw.param__m1_action_kernel__a), 'history', 'm1_action_kernel',
        'action-history update rate', .02)
    add('prior_contrast', expit(raw.param__m2_static__delta0) - .5, 'prior', 'm2_static',
        'static block contrast above 0.5', .02)
    add('prior_hazard', expit(raw.param__m2_static__ah), 'prior', 'm2_static',
        'static biased-block switch hazard', .01)
    prefix = 'validation_nll__' if performance_split == 'validation' else 'nll__'
    for model in ('m0_sensory', 'm1_action_kernel', 'm2_static'):
        gain = (np.log(2) - raw[prefix + model] if model == 'm0_sensory'
                else raw[prefix + 'm0_sensory'] - raw[prefix + model])
        description = ('sensory gain vs fair coin' if model == 'm0_sensory' else 'gain vs sensory')
        add('performance__' + model, gain, 'performance', model,
            performance_split + ': ' + description, .002)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('Nonfinite compact features')
    return values, pd.DataFrame(rows)


def dynamic_annotations(raw):
    """No dynamic parameters or target-test scores enter the primary distance."""
    from scipy.special import expit
    data = raw[IDS].copy()
    for split, prefix in [('train', 'train_nll__'), ('validation', 'validation_nll__'), ('test', 'nll__')]:
        data[f'dynamic_gain_{split}'] = raw[prefix+'m2_static'] - raw[prefix+'m2_dynamic']
    for age in (10, 23, 35):
        data[f'dynamic_hazard_{age}'] = expit(raw.param__m2_dynamic__ah + raw.param__m2_dynamic__bh * (age-23))
    data['dynamic_hazard_change_10_to_35'] = data.dynamic_hazard_35 - data.dynamic_hazard_10
    return data
