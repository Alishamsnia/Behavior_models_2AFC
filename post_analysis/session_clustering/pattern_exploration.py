"""All-session, all-model reference exploration for interpretable behavioral patterns.

No model refits, calendar exclusions, forced-fit weighting or behavior-derived
features. Test-NLL inputs and behavioral interpretation are intentionally exploratory.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.spatial.distance import pdist,squareform
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from .features import LEGACY_OUT,collect,matrix,normalize
from .cluster import driver_manifest,partition,mouse_checks,SEED
from post_analysis.expanded_model_comparison.config import MODEL_SPECS

OUT=LEGACY_OUT.parent/'sliding_window_pattern_exploration'
SUBSET=('m0_sensory','m1_action_kernel','m1_stimulus_kernel','m1_hmm_true','m2_static','m2_dynamic','m2_dynamic_repetition_bias')
METRICS=['balanced_accuracy','block_context_shift','rare_accuracy_cost','adjusted_rapid_S2L_recovery',
         'adjusted_rapid_L2S_recovery','formal_rapid_S2L_recovery','formal_rapid_L2S_recovery',
         'rt_stimulus_balanced','choice_repeat_rate','p_long',
         'adjusted_rapid_S2L_stim_matched_recovery','adjusted_rapid_L2S_stim_matched_recovery','gain_vs_sensory_m1_action_kernel','gain_vs_sensory_m1_stimulus_kernel',
         'gain_vs_sensory_m2_static','dynamic_gain','repetition_increment']


def functional_features(raw,values,meta,rich=False):
    """Transform existing fits. Kernel lags count valid history updates, not elapsed trials."""
    v=values.copy();m=meta.copy();new=[];remove=[]
    def add(feature,data,model,block,description):
        v[feature]=np.asarray(data,float)
        new.append(dict(feature=feature,model=model,block=block,oracle=False,description=description))
    for spec in MODEL_SPECS:
        if spec.oracle:continue
        model=spec.model_id;names=spec.parameter_names
        if 'w' in names:
            remove.extend([f'param__{model}__w',f'param__{model}__s0'])
            for t in (.8,1.7):
                sd=np.sqrt(raw[f'param__{model}__w']*t*t+raw[f'param__{model}__s0']**2)
                add(f'noise__{model}__{t}',np.log(sd),model,'sensory',f'log effective SD {t:g}s')
        if not rich:continue
        if 'a' in names:
            remove.append(f'param__{model}__a');alpha=expit(raw[f'param__{model}__a'])
            for lag in (1,3,8):
                add(f'history__{model}__{lag}',alpha*(1-alpha)**(lag-1),model,'history',f'history weight: update lag {lag}')
        if 'ah' in names:
            remove.append(f'param__{model}__ah');bh=raw[f'param__{model}__bh'] if 'bh' in names else 0.
            if 'bh' in names:remove.append(f'param__{model}__bh')
            for age in ((10,23,35) if 'bh' in names else (23,)):
                add(f'hazard__{model}__{age}',expit(raw[f'param__{model}__ah']+bh*(age-23)),model,'prior',f'switch hazard: age {age}')
        if 'delta0' in names:
            remove.append(f'param__{model}__delta0')
            add(f'contrast__{model}',expit(raw[f'param__{model}__delta0'])-.5,model,'prior','subjective block contrast')
        if 'eta_logit' in names:
            remove.append(f'param__{model}__eta_logit')
            add(f'mixing__{model}',expit(raw[f'param__{model}__eta_logit']),model,'history','one-step history mixing weight')
    m=pd.concat([m[~m.feature.isin(remove)],pd.DataFrame(new)],ignore_index=True)
    if not np.isfinite(v[m.feature].to_numpy()).all():raise ValueError('Nonfinite functional feature')
    return v,m


def candidates(raw,values,manifest):
    base=driver_manifest(manifest);base=base[~base.oracle].copy()
    noise,nm=functional_features(raw,values,base)
    rich,rm=functional_features(raw,values,base,rich=True)
    return {
        'reference_all_models':(values,base,'kmeans'),
        'effective_noise':(noise,nm,'kmeans'),
        'richer_history_prior':(rich,rm,'kmeans'),
        'seven_models':(noise,nm[nm.model.isin(SUBSET)].copy(),'kmeans'),
        'performance_only':(values,base[base.block.eq('performance')].copy(),'kmeans'),
        'parameters_only':(noise,nm[~nm.block.eq('performance')].copy(),'kmeans'),
        'rbf_richer':(rich,rm,'rbf'),
        'multiview_richer':(rich,rm,'multiview'),
        # Targeted follow-up controls: distinguish feature transforms from the
        # nonlinear similarity itself, after the first eight candidates.
        'rbf_reference':(values,base,'rbf'),
        'rbf_effective_noise':(noise,nm,'rbf'),
    }


def gaussian_affinity(x):
    squared=squareform(pdist(x,metric='sqeuclidean'))
    positive=squared[np.triu_indices_from(squared,1)];positive=positive[positive>1e-12]
    bandwidth=float(np.median(positive)) if len(positive) else 1.
    return np.exp(-squared/(2*bandwidth)),bandwidth


def affinity(x,used,kind):
    if kind=='rbf':
        a,b=gaussian_affinity(x);return a,{'all':b}
    if kind!='multiview':raise ValueError(kind)
    mats=[];widths={}
    for block in used.block.unique():
        a,b=gaussian_affinity(x[:,used.block.eq(block).to_numpy()]);mats.append(a);widths[block]=b
    return np.mean(mats,axis=0),widths


def align_labels(labels,reference,k):
    table=np.zeros((k,k),int)
    for a,b in zip(reference,labels):table[a-1,b-1]+=1
    aa,bb=linear_sum_assignment(-table);mapping={int(b+1):int(a+1) for a,b in zip(aa,bb)}
    return np.array([mapping[int(x)] for x in labels]),mapping


def session_metrics(raw):
    """Reuse audited per-session metrics; old labels never enter these calculations."""
    source=LEGACY_OUT/'interpretation_K3'/'session_metrics.csv'
    d=pd.read_csv(source)
    if d.duplicated(['subject','session_id']).any():raise ValueError('Duplicate per-session interpretation metrics')
    merged=raw[['subject','session_id']].merge(d[['subject','session_id',*METRICS]],on=['subject','session_id'],how='left',validate='one_to_one')
    if merged.balanced_accuracy.isna().any():raise ValueError('Missing session behavior metrics')
    return merged


def behavior_profiles(tagged,k):
    """Equal sessions within mouse, then strict equal-mouse means; missing mice remain missing."""
    per_mouse=tagged.groupby(['subject','cluster_label'])[METRICS].mean()
    subjects=sorted(tagged.subject.unique());index=pd.MultiIndex.from_product([subjects,range(1,k+1)],names=['subject','cluster_label'])
    per_mouse=per_mouse.reindex(index)
    counts=tagged.groupby(['subject','cluster_label'])[METRICS].count().reindex(index).fillna(0)
    rows=[];contrasts=[]
    for metric in METRICS:
        cells=per_mouse[metric].unstack('subject').reindex(range(1,k+1))
        means=np.mean(cells.to_numpy(),axis=1)
        for c in range(1,k+1):
            rows.append(dict(metric=metric,cluster_label=c,equal_mouse_mean=float(means[c-1]),
                             n_mice_available=int(cells.loc[c].notna().sum()),
                             n_sessions_contributing=int(counts.xs(c,level='cluster_label')[metric].sum()),
                             **{'n_'+s:int(counts.loc[(s,c),metric]) for s in subjects},
                             **{s:float(cells.loc[c,s]) for s in subjects}))
        for a in range(1,k+1):
            for b in range(a+1,k+1):
                dif=cells.loc[a]-cells.loc[b];mean=float(np.mean(dif.to_numpy()));sign=np.sign(mean)
                contrasts.append(dict(metric=metric,contrast=f'C{a}-C{b}',difference=mean,
                                      n_mice_available=int(dif.notna().sum()),
                                      n_mice_same_direction=int(((np.sign(dif)==sign)&dif.notna()).sum()),
                                      **{s:float(dif[s]) for s in subjects}))
    return pd.DataFrame(rows),pd.DataFrame(contrasts)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(goal='Descriptive, intuitive model-defined patterns; retain all sessions',K=[2,3],
                  no_exclusions='No calendar, convergence or forced-fit filtering/weighting',
                  performance='Original test-NLL contrasts intentionally retained; behavior is exploratory interpretation',
                  scaling='Original within-mouse z scores, clip3, equal block/model weights',
                  kernels='Fixed median squared-distance bandwidth; no bandwidth search; equal mean of four block kernels for multiview',
                  reference='11 nonoracle models; original oracle/bias/lapse exclusions',
                  history='Fitted EMA coefficients at valid-update lags1,3,8, not additional fitted kernels or independent evidence',
                  selection='Review model profiles, behavior and per-mouse support; not a numerical stability contest; report all candidates',
                  initial_candidates=['reference_all_models','effective_noise','richer_history_prior','seven_models','performance_only','parameters_only','rbf_richer','multiview_richer'],
                  additional_controls=['rbf_reference','rbf_effective_noise'],seed=SEED)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    raw,values,meta,_=collect();raw.to_csv(OUT/'window_metrics_parameters.csv',index=False)
    options=candidates(raw,values,meta);ref=pd.read_csv(LEGACY_OUT/'session_clusters.csv')
    obs=session_metrics(raw);all_labels=[];summaries=[];profiles=[];contrasts=[];manifests=[];model_profiles=[]
    for name,(v,m,method) in options.items():
        x,used=matrix(v,m,raw.subject.to_numpy());used.insert(0,'variant',name);manifests.append(used)
        bandwidth={};a=None
        if method!='kmeans':
            a,bandwidth=affinity(x,used,method);np.save(OUT/f'affinity_{name}.npy',a)
        for k in (2,3):
            if method=='kmeans':labels=partition(x,k)
            else:labels=SpectralClustering(n_clusters=k,affinity='precomputed',assign_labels='kmeans',n_init=20,
                                          eigen_solver='arpack',random_state=SEED).fit_predict(a)+1
            reference=ref[ref.K.eq(k)].set_index(['subject','session_id']).reindex(pd.MultiIndex.from_frame(raw[['subject','session_id']])).cluster_label.to_numpy()
            if name=='reference_all_models' and adjusted_rand_score(reference,labels)!=1.:raise ValueError('Reference labels did not reproduce')
            labels,mapping=align_labels(labels,reference,k)
            label=raw[['subject','window_id','test_rank','session_id']].copy();label['variant'],label['K'],label['cluster_label']=name,k,labels;all_labels.append(label)
            tagged=obs.copy();tagged['cluster_label']=labels
            profile,con=behavior_profiles(tagged,k);profile['variant'],profile['K']=name,k;con['variant'],con['K']=name,k;profiles.append(profile);contrasts.append(con)
            row=dict(variant=name,K=k,method=method,n_features=len(used),n_models=used.model.nunique(),
                     sizes='/'.join(map(str,pd.Series(labels).value_counts().reindex(range(1,k+1)).tolist())),
                     ari_to_reference=float(adjusted_rand_score(reference,labels)),bandwidth=json.dumps(bandwidth),**mouse_checks(labels,raw.subject.to_numpy()))
            for metric in ('block_context_shift','rare_accuracy_cost','adjusted_rapid_S2L_recovery','adjusted_rapid_L2S_recovery','rt_stimulus_balanced','balanced_accuracy'):
                p=profile[profile.metric.eq(metric)].equal_mouse_mean.to_numpy();row['range__'+metric]=float(np.ptp(p))
            summaries.append(row)
            # All original model/parameter columns remain available for explanation.
            group=raw.copy();group['cluster_label']=labels
            for c,g in group.groupby('cluster_label'):
                for col in raw.columns:
                    if col.startswith(('nll__','param__')):
                        model_profiles.append(dict(variant=name,K=k,cluster_label=c,feature=col,median=float(g[col].median()),q25=float(g[col].quantile(.25)),q75=float(g[col].quantile(.75))))
            print(name,k,row['sizes'],'mouse NMI',round(row['mouse_nmi'],3),'ARIref',round(row['ari_to_reference'],3),flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'candidate_summary.csv',index=False)
    pd.concat(all_labels).to_csv(OUT/'all_candidate_labels.csv',index=False)
    pd.concat(profiles).to_csv(OUT/'behavior_profiles_equal_mouse.csv',index=False)
    pd.concat(contrasts).to_csv(OUT/'behavior_contrasts_by_mouse.csv',index=False)
    pd.concat(manifests).to_csv(OUT/'feature_manifest.csv',index=False)
    pd.DataFrame(model_profiles).to_csv(OUT/'model_parameter_profiles.csv',index=False)
    obs.to_csv(OUT/'observed_session_metrics.csv',index=False)
    print('DONE',OUT,flush=True)

def export_functional_readouts(raw,labels):
    """Physical-unit annotations of each model separately; never averaged across models."""
    rows=[]
    for spec in MODEL_SPECS:
        if spec.oracle:continue
        model=spec.model_id
        for _,r in raw.iterrows():
            d=dict(subject=r.subject,session_id=r.session_id,window_id=r.window_id,model=model,
                   test_gain_vs_sensory=r.nll__m0_sensory-r['nll__'+model])
            p=lambda name:r['param__'+model+'__'+name]
            if 'w' in spec.parameter_names:
                d.update(noise_sd_08=float(np.sqrt(p('w')*.8**2+p('s0')**2)),
                         noise_sd_17=float(np.sqrt(p('w')*1.7**2+p('s0')**2)),decision_bias_kappa=p('kappa'))
            if 'a' in spec.parameter_names:
                alpha=expit(p('a'));d['history_alpha']=alpha;d['mean_history_update_lag']=1/alpha
                for lag in (1,3,8):d['history_weight_'+str(lag)]=alpha*(1-alpha)**(lag-1)
            if 'ah' in spec.parameter_names:
                bh=p('bh') if 'bh' in spec.parameter_names else 0.
                for age in (10,23,35):d['hazard_'+str(age)]=expit(p('ah')+bh*(age-23))
            if 'delta0' in spec.parameter_names:d['subjective_block_contrast']=expit(p('delta0'))-.5
            if 'eta_logit' in spec.parameter_names:d['one_step_history_mixing']=expit(p('eta_logit'))
            rows.append(d)
    f=pd.DataFrame(rows);f.to_csv(OUT/'functional_model_readouts.csv',index=False)
    ll=labels[['variant','K','cluster_label','subject','session_id','window_id']]
    x=ll.merge(f,on=['subject','session_id','window_id'],validate='many_to_many').drop(columns='window_id')
    x.groupby(['variant','K','cluster_label','subject','model']).median(numeric_only=True).to_csv(
        OUT/'selected_functional_parameter_profiles.csv')


def refresh_annotations():
    """Add descriptive checks to fixed candidates without re-estimating any labels."""
    raw=pd.read_csv(OUT/'window_metrics_parameters.csv');obs=session_metrics(raw)
    labels=pd.read_csv(OUT/'all_candidate_labels.csv');profiles=[];contrasts=[]
    for (variant,k),lab in labels.groupby(['variant','K'],sort=False):
        t=obs.merge(lab[['subject','session_id','cluster_label']],on=['subject','session_id'],validate='one_to_one')
        p,c=behavior_profiles(t,int(k));p['variant'],p['K']=variant,k;c['variant'],c['K']=variant,k
        profiles.append(p);contrasts.append(c)
    pd.concat(profiles).to_csv(OUT/'behavior_profiles_equal_mouse.csv',index=False)
    pd.concat(contrasts).to_csv(OUT/'behavior_contrasts_by_mouse.csv',index=False)
    obs.to_csv(OUT/'observed_session_metrics.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
