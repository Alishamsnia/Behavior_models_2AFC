"""Matched test prediction checks. None of these outcomes changes the cluster inputs."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from post_analysis.common import REPO_ROOT
from post_analysis.expanded_model_comparison.config import MODEL_BY_ID

MODELS = ('m0_sensory', 'm1_action_kernel', 'm2_static', 'm2_dynamic')
KEYS = ['subject', 'window_id', 'session_id']


def evaluate(raw, all_labels, out, block=13, repeats=1000, seed=20261025):
    """Four-cell calibration and duration-specific gains, equal weight per mouse/session.

    Accuracy residual CIs resample contiguous windows within each mouse. They
    describe model adequacy, not equivalence, parameter recovery, or independent
    cluster validation. Real block labels/ages are analysis strata only.
    """
    paths = {}
    for model in MODELS:
        for subject in raw.subject.unique():
            root = REPO_ROOT / 'results_sliding_window' / MODEL_BY_ID[model].directory / subject
            for path in sorted(root.rglob('trial_predictions.csv')):
                if 'full_session_matched_v1' not in path.parts or not path.parent.name.startswith('window_'):
                    continue
                key = model, subject, int(path.parent.name.split('_')[1])
                if key not in paths or path.parent.parent == root / 'full_session_matched_v1':
                    paths[key] = path
    columns = ['session_id','trial_index','split_role','model_evaluated','warm_up','choice_long',
               'category_long','duration_s','true_block','true_run_length','model_probability_long']

    def job(row):
        frames, metrics, duration = {}, [], []
        for model in MODELS:
            d = pd.read_csv(paths[model,row.subject,int(row.window_id)], usecols=columns)
            d = d[d.split_role.eq('test') & d.model_evaluated.eq(True) & d.choice_long.isin([0,1])].copy()
            if d.session_id.nunique() != 1 or d.session_id.iloc[0] != row.session_id:
                raise ValueError('Unmatched test session in prediction diagnostic')
            d = d.sort_values('trial_index').reset_index(drop=True)
            if frames:
                ref = next(iter(frames.values()))
                matched_columns = [c for c in columns if c != 'model_probability_long']
                pd.testing.assert_frame_equal(d[matched_columns], ref[matched_columns], check_dtype=False)
            p = np.clip(d.model_probability_long.to_numpy(), 1e-12, 1-1e-12)
            y, category = d.choice_long.to_numpy(), d.category_long.to_numpy()
            d['nll'] = -(y*np.log(p)+(1-y)*np.log1p(-p))
            d['accuracy_mouse'] = (y==category).astype(float)
            d['accuracy_model'] = np.where(category==1,p,1-p)
            d['accuracy_residual'] = d.accuracy_mouse-d.accuracy_model
            frames[model] = d
            good = d.warm_up.eq(0)
            masks = {'all':good, 'common_short':good & d.true_block.eq('S') & d.category_long.eq(0),
                     'rare_long':good & d.true_block.eq('S') & d.category_long.eq(1),
                     'common_long':good & d.true_block.eq('L') & d.category_long.eq(1),
                     'rare_short':good & d.true_block.eq('L') & d.category_long.eq(0)}
            for subset, mask in masks.items():
                g = d[mask]
                metrics.append(dict(subject=row.subject,window_id=row.window_id,session_id=row.session_id,
                                    model=model,subset=subset,n_trials=len(g),
                                    **g[['nll','accuracy_mouse','accuracy_model','accuracy_residual']].mean().to_dict()))
        a,b = frames['m2_static'],frames['m2_dynamic']
        gain = a.nll-b.nll
        for name, mask in [('1–10',a.true_run_length.between(1,10)),
                           ('11–25',a.true_run_length.between(11,25)),('26+',a.true_run_length.ge(26))]:
            mask &= a.warm_up.eq(0) & a.true_block.isin(['S','L'])
            duration.append(dict(subject=row.subject,window_id=row.window_id,session_id=row.session_id,
                                 age_bin=name,n_trials=int(mask.sum()),dynamic_gain=float(gain[mask].mean())))
        return metrics,duration

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(job,raw.itertuples(index=False)))
    metrics = pd.DataFrame([r for ms,_ in results for r in ms])
    ages = pd.DataFrame([r for _,aa in results for r in aa])
    metrics.to_csv(out/'test_model_category_metrics.csv',index=False)
    ages.to_csv(out/'test_dynamic_age_metrics.csv',index=False)
    ages.groupby(['subject','age_bin']).dynamic_gain.mean().to_csv(out/'dynamic_age_by_mouse.csv')
    metrics.merge(all_labels[['K',*KEYS,'cluster_label']],on=KEYS,validate='many_to_many').groupby(
        ['K','cluster_label','subject','model','subset'])[['nll','accuracy_mouse','accuracy_model','accuracy_residual']].mean().to_csv(
            out/'test_model_profiles_by_cluster_mouse.csv')
    # One shared resampling sequence across all model/category comparisons.
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        indices=[]
        for subject, g in raw.groupby('subject',sort=True):
            g=g.sort_values('window_id'); n=len(g); width=min(block,n)
            starts=rng.integers(0,n-width+1,size=int(np.ceil(n/width)))
            ix=np.concatenate([np.arange(i,i+width) for i in starts])[:n]
            indices.extend((subject,int(w)) for w in g.window_id.to_numpy()[ix])
        draws.append(indices)
    summary=[]
    for (model,subset),g in metrics.groupby(['model','subset']):
        residual=g.set_index(['subject','window_id']).accuracy_residual
        boot=[]
        for draw in draws:
            sampled=pd.DataFrame(draw,columns=['subject','window_id'])
            sampled['residual']=residual.reindex(pd.MultiIndex.from_frame(sampled)).to_numpy()
            boot.append(sampled.groupby('subject').residual.mean().mean())
        means=g.groupby('subject')[['nll','accuracy_mouse','accuracy_model','accuracy_residual']].mean().mean()
        summary.append(dict(model=model,subset=subset,n_windows=int(len(g)),**means.to_dict(),
                            residual_ci_low=float(np.nanquantile(boot,.025)),residual_ci_high=float(np.nanquantile(boot,.975))))
    pd.DataFrame(summary).to_csv(out/'test_model_category_adequacy.csv',index=False)
    return dict(models=list(MODELS),matched_prediction_files=len(raw)*len(MODELS),warmup_excluded=True,
                accuracy_residual='observed minus predicted probability of correct choice; equal mouse then session weight',
                ci='descriptive 95% moving-block percentile intervals; not equivalence tests',block=block,repeats=repeats)
