"""Audited staged publication for the independent Ward analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader
from threadpoolctl import threadpool_limits

from . import core, data, behavior, plots, followup, sequences, sensitivity
from post_analysis.expanded_model_comparison.config import MODEL_SPECS

NAME='sliding_window_pattern_exploration_zero_anchored_colstd_rowl2_ward_four_features_star'
ROOT=data.BASE/'pooled/clustering'
DEFAULT_OUT=ROOT/NAME
REFERENCE=ROOT/'sliding_window_pattern_exploration_rbf_four_features_star'
FOLDERS=['01_clustering_input','02_cluster_labels','03_number_of_clusters','05_cluster_tables','06_audit',
         '07_comparison_to_frozen','07_hypothesis_followup','08_scaling_sensitivity']


def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=lambda v: v.item() if isinstance(v,np.generic) else str(v),allow_nan=False)+'\n')


def file_hashes(folder):
    return {str(p.resolve()):data.sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def protected():
    # All pre-existing clustering results and code are read-only, except THIS new result.
    result={}
    if ROOT.exists():
        for folder in ROOT.iterdir():
            if folder.is_dir() and folder.name!=NAME and not folder.name.startswith(NAME+'.'):
                result.update(file_hashes(folder))
    result.update(file_hashes(data.REPO/'post_analysis/session_clustering'))
    return result


def readouts(raw,labels,tables):
    rows=[]
    for spec in MODEL_SPECS:
        for r in raw.to_dict('records'):
            def value(name):return r[f'param__{spec.model_id}__{name}']
            result={key:r[key] for key in data.IDS};result['model']=spec.model_id
            result['test_gain_vs_sensory']=r['nll__m0_sensory']-r['nll__'+spec.model_id]
            if 'w' in spec.parameter_names:
                result.update(noise_sd_08=np.sqrt(value('w')*.8**2+value('s0')**2),
                              noise_sd_17=np.sqrt(value('w')*1.7**2+value('s0')**2),decision_bias_kappa=value('kappa'))
            if 'a' in spec.parameter_names:
                alpha=expit(value('a'));result.update(history_alpha=alpha,mean_history_update_lag=1/alpha)
                for lag in (1,3,8):result[f'history_weight_{lag}']=alpha*(1-alpha)**(lag-1)
            if 'ah' in spec.parameter_names:
                slope=value('bh') if 'bh' in spec.parameter_names else 0
                for age in (10,23,35):result[f'hazard_{age}']=expit(value('ah')+slope*(age-23))
            if 'delta0' in spec.parameter_names:result['subjective_block_contrast']=expit(value('delta0'))-.5
            if 'eta_logit' in spec.parameter_names:result['one_step_history_mixing']=expit(value('eta_logit'))
            rows.append(result)
    f=pd.DataFrame(rows);f.to_csv(tables/'functional_model_readouts.csv',index=False)
    tagged=labels.merge(f,on=data.IDS,validate='many_to_many')
    cols=[c for c in f if c not in data.IDS+['model']]
    tagged.groupby(['K','cluster_label','subject','model'])[cols].median().to_csv(tables/'selected_functional_parameter_profiles.csv')
    parameters=[]
    for k,ll in labels.groupby('K'):
        group=raw.merge(ll[core.KEYS+['cluster_label']],on=core.KEYS,validate='one_to_one')
        for c,g in group[group.cluster_label.gt(0)].groupby('cluster_label'):
            for column in raw.filter(regex='^(param__|nll__)'):
                values=g[column]
                parameters.append(dict(K=k,cluster_label=c,feature=column,median=values.median(),q25=values.quantile(.25),q75=values.quantile(.75)))
    pd.DataFrame(parameters).to_csv(tables/'model_parameter_profiles.csv',index=False)


def rt_transitions(trans,out,tables,pdf):
    fig,axes=plots.shell('K=3: reaction time aligned to block transitions',3,4,size=(22,15),
                         note='Same adjusted events as the choice page; rows split by current outcome. Pooled event mean +/- SEM; y scales vary.')
    records=[]
    for row,outcome in enumerate(['All outcomes','Reward','Punish']):
        for col,(kind,cat) in enumerate([('S2L',0),('S2L',1),('L2S',0),('L2S',1)]):
            e=trans[trans.kind.eq(kind)&trans.category_long.eq(cat)]
            if outcome!='All outcomes':e=e[e.outcome.eq(outcome)]
            records.extend(plots.plot_event(axes[row,col],e,3,f'{kind} / category {cat} / {outcome}',value='latency_s'))
    plots.footer(fig,'Warm-up, pre-choice negative lick times and invalid scalar RT excluded without removing timeline positions.\nOpen -1 marker copies previous eligible matching-category trial; overlapping events and pooled SEM are descriptive.')
    plots.save(fig,out/'K3_8_reaction_time_transitions',pdf)
    trans.to_csv(tables/'K3_rt_transition_events.csv',index=False)
    pd.concat(records,ignore_index=True).to_csv(tables/'K3_rt_transition_summary.csv',index=False)


def report(raw,gains,geom,tree,labels,trials,observed,out):
    tables=out/'05_cluster_tables';pages=[];profiles=[];raw_profiles=[];beh_profiles=[];orders=[];bins=[];choices=[];contrasts=[];support=[]
    readouts(raw,labels,tables)
    for k,ll in labels.groupby('K',sort=True):
        k=int(k);ll=ll.reset_index(drop=True);dest=out/f'04_clusters_K{k}';dest.mkdir()
        tagged=behavior.attach(trials,ll)
        signed=pd.DataFrame(gains,columns=core.FEATURES).assign(subject=raw.subject,cluster_label=ll.cluster_label)
        signed=signed[signed.cluster_label.gt(0)]
        raw_profiles.append(plots.means_equal_mouse(signed,list(core.FEATURES)).reset_index().assign(K=k))
        trans,rare=behavior.event_tables(tagged)
        trans.to_csv(tables/f'K{k}_choice_transition_events.csv',index=False);rare.to_csv(tables/f'K{k}_choice_rare_events.csv',index=False)
        with PdfPages(dest/f'K{k}_report.pdf') as pdf:
            fig,pr,bpr=plots.fingerprint(raw,gains,geom,ll,observed,k);profiles.append(pr);beh_profiles.append(bpr)
            plots.save(fig,dest/f'K{k}_1_fingerprint',pdf)
            plots.save(plots.methods(k),dest/f'K{k}_2_methods',pdf)
            plots.save(plots.gain_details(raw,gains,ll,k),dest/f'K{k}_3_gains_raw_units',pdf)
            fig,order=plots.session_heatmap(raw,gains,geom,ll,tree,k);orders.append(order)
            plots.save(fig,dest/f'K{k}_4_session_heatmap',pdf)
            # Separate explicitly named editable PDF for reference navigation.
            shutil.copyfile(dest/f'K{k}_4_session_heatmap.pdf',dest/f'K{k}_4_session_heatmap_illustrator.pdf')
            titles=['fingerprint','methods','gains_raw_units','session_heatmap']
            if k==4:
                from . import k4_figures
                titles+=k4_figures.render(tagged,trans,rare,dest,tables,pdf)
                choices.append(pd.read_csv(tables/'K4_choice_behavior_plotted.csv'))
                for condition,mask in behavior.rt_masks(tagged).items():
                    for c in range(1,k+1):
                        frame=tagged[mask & tagged.cluster_label.eq(c) & tagged.latency_eligible]
                        b=behavior.binned(frame,'latency_s',session_weighted=True)
                        if not b.empty:bins.append(b.assign(K=k,cluster_label=c,condition=condition))
            else:
                fig,cstats=plots.choice_page(tagged,k,trans,rare);choices.append(cstats.assign(K=k));plots.save(fig,dest/f'K{k}_5_choice_behavior',pdf)
                fig,b=plots.reaction_page(tagged,k);bins.append(b);plots.save(fig,dest/f'K{k}_6_reaction_time',pdf)
                titles+=['choice_behavior','reaction_time']
            if k==3:
                plots.rt_extension(tagged,k,dest,tables,pdf)
                rt_transitions(trans,dest,tables,pdf)
                titles+=['reaction_time_mean_sem','reaction_time_shared_bins','reaction_time_transitions']
            for i,title in enumerate(titles,1):pages.append(dict(K=k,page=i,analysis=title,pdf=f'04_clusters_K{k}/K{k}_report.pdf'))
        # Cross-cluster effects per mouse, never pooling absent mice into zeros.
        ob=observed.merge(ll[core.KEYS+['cluster_label']],on=core.KEYS,validate='one_to_one')
        cols=observed.select_dtypes('number').columns.tolist()
        mouse=ob[ob.cluster_label.gt(0)].groupby(['subject','cluster_label'])[cols].mean()
        for c,group in ob[ob.cluster_label.gt(0)].groupby('cluster_label'):
            for metric in cols:
                available=group.dropna(subset=[metric])
                support.append(dict(K=k,cluster_label=c,metric=metric,n_sessions=len(available),n_mice=available.subject.nunique(),
                                    mice='/'.join(sorted(available.subject.unique()))))
        for metric in cols:
            wide=mouse[metric].unstack('cluster_label').reindex(columns=range(1,k+1))
            for a in range(1,k):
                for b in range(a+1,k+1):
                    for subject,row in wide.iterrows():
                        contrasts.append(dict(K=k,subject=subject,metric=metric,cluster_a=a,cluster_b=b,difference=row[a]-row[b]))
        print(f'Rendered Ward K={k}: '+str(ll.cluster_label.value_counts().sort_index().to_dict()),flush=True)
    for name,frame in [('report_pages',pd.DataFrame(pages)),('normalized_profiles_equal_mouse',pd.concat(profiles)),
                       ('gain_profiles_equal_mouse',pd.concat(raw_profiles)),('behavior_profile_support',pd.DataFrame(support)),
                       ('behavior_profiles_equal_mouse',pd.concat(beh_profiles)),('session_heatmap_order',pd.concat(orders)),
                       ('reaction_time_binned',pd.concat(bins)),('choice_behavior_binned',pd.concat(choices)),
                       ('behavior_contrasts_by_mouse',pd.DataFrame(contrasts))]:
        frame.to_csv(tables/f'{name}.csv',index=False)
    return pd.DataFrame(pages)


def analysis(out,replicates=100,raw_root=data.RAW_ROOT):
    for folder in FOLDERS:(out/folder).mkdir(parents=True)
    raw,gains,trials,pred_audit,baseline,sources=data.collect()
    trials,rt_audit,raw_sources=data.raw_behavior(trials,raw_root);sources.update(raw_sources)
    geom,tree,candidates,selection,draws,loo,optimal=core.clustering(gains,raw.subject.to_numpy(),replicates=replicates)
    ks=sorted({2,3,4,optimal}&set(candidates))
    if not {2,3,4}.issubset(ks):raise ValueError('Reference report family requires distinct K2/K3/K4 partitions')
    input_dir=out/'01_clustering_input';audit_dir=out/'06_audit';tables=out/'05_cluster_tables';k_dir=out/'03_number_of_clusters'
    for name,values in [('signed_feature_values',gains),('positive_feature_values',geom.positive),
                        ('column_scaled_feature_values',geom.scaled),('normalized_feature_values',geom.z)]:
        pd.concat([raw[data.IDS],pd.DataFrame(values,columns=core.FEATURES)],axis=1).to_csv(input_dir/f'{name}.csv',index=False)
    raw.to_csv(input_dir/'window_metrics_parameters.csv',index=False)
    trials.to_csv(audit_dir/'reaction_time_trials.csv',index=False)
    pred_audit.to_csv(audit_dir/'prediction_input_audit.csv',index=False);baseline.to_csv(audit_dir/'sensory_baseline_audit.csv',index=False)
    rt_audit.to_csv(tables/'reaction_time_input_audit.csv',index=False)
    np.save(input_dir/'clustering_matrix.npy',geom.z);np.save(input_dir/'euclidean_distances.npy',core.distance_matrix(geom.z[geom.eligible]))
    np.save(input_dir/'ward_linkage.npy',tree)
    pd.DataFrame(dict(feature=core.FEATURES,global_sample_sd=geom.positive.std(axis=0,ddof=1),denominator=geom.scale,
                      negative_clamped_count=np.sum(gains<0,axis=0))).to_csv(input_dir/'feature_scaling.csv',index=False)
    row_audit=raw[data.IDS].assign(pre_l2_norm=geom.norm,final_l2_norm=np.linalg.norm(geom.z,axis=1),
                                 eligible=geom.eligible,positive_coordinate_count=np.sum(geom.positive>0,axis=1))
    row_audit.to_csv(input_dir/'session_normalization_audit.csv',index=False)
    pd.DataFrame(dict(feature=core.FEATURES,label=core.LABELS,formula=core.FORMULAS,
                      reduced_model=['training_fitted_constant',*([core.MODELS[0]]*3)],full_model=core.MODELS,
                      raw_units='nats/trial',clustering_units='dimensionless',selected=True)).to_csv(input_dir/'feature_manifest.csv',index=False)
    pd.DataFrame(gains,columns=core.FEATURES).corr().to_csv(input_dir/'raw_feature_correlations.csv')
    labels=pd.concat([raw[data.IDS].assign(variant='zero_anchored_colstd_rowl2_ward_four_features_star',K=k,cluster_label=candidates[k]) for k in ks],ignore_index=True)
    all_labels=pd.concat([raw[data.IDS].assign(K=k,cluster_label=lab) for k,lab in candidates.items()],ignore_index=True)
    labels.to_csv(out/'selected_session_clusters.csv',index=False)
    all_labels.to_csv(out/'02_cluster_labels/all_candidate_labels.csv',index=False)
    selection.to_csv(k_dir/'k_selection.csv',index=False);selection[selection.K.isin(ks)].to_csv(out/'02_cluster_labels/candidate_summary.csv',index=False)
    draws.to_csv(k_dir/'subsample_stability.csv',index=False);loo.to_csv(k_dir/'leave_one_mouse_out.csv',index=False)
    print(f'Descriptive selected K={optimal}; silhouette={selection.loc[selection.K.eq(optimal),"silhouette"].iloc[0]:.4f}',flush=True)
    plots.save(plots.k_figure(selection,tree,raw,geom,candidates,optimal),k_dir/'k_selection')
    plots.save(plots.normalization_figure(raw,gains,geom),input_dir/'normalization_audit')
    # Full labeled dendrogram and mouse-removal audit are separate from the compact K page.
    fig,axes=plots.shell('Ward hierarchy: all held-out sessions',1,1,size=(22,8),note='Ward merge distances; labels show mouse and chronological held-out rank.')
    from scipy.cluster.hierarchy import dendrogram
    dendrogram(tree,ax=axes[0,0],labels=[f'{r.subject}:{r.test_rank}' for r in raw[geom.eligible].itertuples()],leaf_rotation=90,leaf_font_size=6)
    axes[0,0].set_ylabel('Ward merge distance');plots.save(fig,k_dir/'ward_dendrogram')
    fig,axes=plots.shell('Sensitivity to omitting each mouse',1,1,size=(11,6),note='Scaling and Ward refitted on remaining mice; ARI compares assignments on retained sessions.')
    for subject,g in loo.groupby('omitted_mouse'):axes[0,0].plot(g.K,g.ari,'o-',label=subject)
    axes[0,0].set_ylabel('ARI');axes[0,0].set_xlabel('K');axes[0,0].legend();plots.save(fig,k_dir/'leave_one_mouse_out')
    observed=behavior.interpretation_metrics(trials);observed.to_csv(tables/'observed_session_metrics.csv',index=False)
    report(raw,gains,geom,tree,labels,trials,observed,out)
    ll3=labels[labels.K.eq(3)].reset_index(drop=True);tagged=behavior.attach(trials,ll3)
    sequences.run(tagged,ll3,out/'04_clusters_K3/sequence_performance')
    followup.run(tagged,ll3,out/'07_hypothesis_followup')
    rt_audit.to_csv(out/'07_hypothesis_followup/tables/raw_session_audit.csv',index=False)
    sensitivity.run(raw,gains,labels,out/'08_scaling_sensitivity')
    old_labels=REFERENCE/'selected_session_clusters.csv'
    if old_labels.exists():
        sensitivity.historical(labels,old_labels,out/'07_comparison_to_frozen');sources[str(old_labels.resolve())]=data.sha(old_labels)
    else:raise FileNotFoundError('Reference label CSV required for requested comparison figures: '+str(old_labels))
    protocol=dict(name=NAME,features=list(core.FEATURES),formulas=list(core.FORMULAS),n_sessions=len(raw),n_trials=len(trials),
                  normalization=dict(clamp='max(0, signed gain)',column_scale='Global sample SD of positive gains, ddof=1; no centering',
                                     row_scale='u / sqrt(sum(u**2) + epsilon**2)',epsilon=core.EPSILON,
                                     zero_columns='Denominator 1 for all-zero coordinate; reject nonzero constant coordinate',
                                     zero_rows='Excluded from Ward; retained as unresolved label 0',
                                     weak_rows='Norms exported; no arbitrary reliability cutoff or claim of gain uncertainty'),
                  clustering=dict(method='Ward',metric='Euclidean',input='Normalized coordinates directly',label_numbering='Descending size; first session breaks ties'),
                  k_selection=dict(candidate_K=selection.K.tolist(),selected_K=optimal,reported_K=ks,
                                   rule='Maximum Euclidean silhouette, smaller K on ties; descriptive only; no all-mice constraint',
                                   stability_replicates=replicates,subsample_fraction=.8,seed=core.SEED,refit_scaling=True),
                  baseline='Training-only evaluated choice mean, probability clipped to [1e-12,1-1e-12], scored on evaluated test choices',
                  annotations='All parameters and observed behavior are descriptive, never used in distances or K selection',
                  caveats=['Gains are predictive contrasts, not additive strategy fractions','Clamping pools negative and zero estimates',
                           'Row normalization discards magnitude, including meaningful magnitude differences',
                           'Overlapping training windows/sessions and behavioral events are dependent','No trial bootstrap uncertainty was estimated'],
                  versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    json_write(out/'02_cluster_labels/protocol.json',protocol)
    for source in Path(__file__).parent.glob('*.py'):sources[str(source.resolve())]=data.sha(source)
    config_path=data.REPO/'post_analysis/expanded_model_comparison/config.py';sources[str(config_path)]=data.sha(config_path)
    json_write(audit_dir/'source_manifest.json',[dict(path=path,sha256=value) for path,value in sorted(sources.items())])
    write_readme(out,protocol,selection,loo)
    return protocol


def write_readme(out,protocol,selection,loo):
    k=protocol['k_selection']['selected_K'];row=selection[selection.K.eq(k)].iloc[0]
    text=f'''# Zero-anchored column-SD + row-L2 + Ward, four-feature star

Run: `python -m post_analysis.zero_anchored_ward`

Independent code lives in `post_analysis/zero_anchored_ward/`; no imports from previous clustering modules. Inputs are original model predictions, upstream model-comparison metrics/parameters, and original behavior CSVs. The previous label CSV is read only for the explicit historical comparison.

## Coordinates and normalization

Exactly L(const)-L(S), L(S)-L(S+AH), L(S)-L(S+WH), L(S)-L(S+Bdynamic). L is mean held-out NLL in nats/trial. The constant-choice probability uses evaluated training choices only. The four model timelines and evaluation masks are verified before reconstructing losses and checking upstream metrics.

For clustering: clamp negative gains to zero; divide each column by the global sample SD of the clamped column (ddof=1), without centering; divide each row by sqrt(sum(u^2)+1e-24). No within-mouse normalization, +/-3 clipping, RBF or spectral embedding. Ward receives the four normalized coordinates with Euclidean distance. Original signed gains and all intermediate coordinates remain available. All-zero columns use denominator 1; nonzero constant columns fail for review. Zero rows retain unresolved label 0 and are excluded from clustering. Global SD weights features, so directions are scaled gain profiles, not raw gain ratios or behavioral mixture fractions.

## Results and interpretation

{protocol['n_sessions']} sessions. Descriptive selected K={k}, silhouette={row.silhouette:.4f}, sizes {row.sizes}; median subset ARI={row.stability_ari_median:.3f}. K2/K3/K4 reports are retained. Selection is the maximum silhouette over K2..8, not a statistical proof of discrete states. Cluster numbers are assigned independently by size. Mouse composition is not constrained; missing mice remain visible. See leave-one-mouse-out results before inferring cross-mouse reproducibility. Row norm is recorded but gain-estimation uncertainty is not evaluated.

## Output structure and figures

- `01_clustering_input/`: signed/clamped/column-scaled/normalized gains, clustering matrix, Euclidean distances, Ward linkage, per-column scales, per-session norms and normalization audit figure.
- `02_cluster_labels/`: labels for all candidate K, per-K summary, exact protocol. Root `selected_session_clusters.csv` contains the report partitions.
- `03_number_of_clusters/`: silhouette, CH, DB, subset ARI, minimum sizes and dendrogram in the selection figure; full dendrogram; leave-one-mouse-out figure and tables.
- `04_clusters_K2/`, `04_clusters_K3/`, `04_clusters_K4/`: core six-page reports: fingerprints, methods, raw signed gains, session heatmap, choice behavior, RT. Every page has PNG and editable-text PDF. K3 adds pooled RT mean/SEM, shared-duration RT and transition RT; mean/SD is separate.
- K4 has eight pages: pages 5-8 follow the reference extended layout (choice behavior, shared-duration pooled RT with binwise statistics, pooled mean/SEM with six pairwise tests per condition, transition RT). Its separate mean/SD figure and statistical CSVs are also exported. Pages 6-7 use trial-pooled SEM, explicitly distinct from the session-mean RT summaries retained in `reaction_time_binned.csv`.
- `04_clusters_K3/sequence_performance/`: E/F overview (3 pages), per-cluster/mouse/context details (9 pages), separate per-cluster reports, sequence identities and window summaries.
- `05_cluster_tables/`: observed session effects, equal-mouse summaries, per-mouse contrasts, fitted model/functional readouts, choice events, RT bins, pairwise reference tests and plot support.
- `06_audit/`: paired prediction and constant-baseline audits, verified raw trials, source SHA256 manifest and preservation audit.
- `07_comparison_to_frozen/`: previous four-feature star RBF vs new Ward labels; both preprocessing and algorithm change.
- `07_hypothesis_followup/`: all eight reference follow-up figure purposes, showing all new K3 groups, with matched-support estimates, conditional block intervals and raw/retained event tables. Subfolder names preserve navigation only, not old cluster interpretations.
- `08_scaling_sensitivity/`: five Ward geometries, mouse composition, ARI agreement, distance share and raw-gain profile figures.
- `figure_coverage.csv`: explicit mapping of requested reference figure purposes to new figures.
- `completion.json`: validation status, output hashes and PDF page counts.

Behavior is descriptive on the same sessions used to define clusters. Core RT uses session-mean SEM; extended RT reproduces pooled-trial SD/SEM and Welch tests with Holm correction, which assume independent trials and are not confirmatory repeated-measures inference. Follow-up histories use the intact timeline and matched support; sequence E/F excludes incomplete final blocks and uses complete trains inside seven-trial sliding windows. Exact procedures are documented in each supplement.
'''
    (out/'README.md').write_text(text)


def coverage(out):
    """Every reference family is mapped to its new scientific equivalent, not copied artifacts."""
    mapping=[]
    for k in (2,3,4):
        for index,name in enumerate(['fingerprint','methods','gains_raw_units','session_heatmap','choice_behavior','reaction_time'],1):
            mapping.append(dict(reference_family=f'K{k}/{name}',new_figure=f'04_clusters_K{k}/K{k}_{index}_{name}.png',adaptation='Fresh Ward labels, zero-anchored directional geometry'))
    mapping += [dict(reference_family=name,new_figure=path,adaptation=note) for name,path,note in [
        ('K selection','03_number_of_clusters/k_selection.png','Ward separation/stability replaces spectral eigengap'),
        ('K3 RT SEM','04_clusters_K3/K3_7_reaction_time_mean_sem.png','Pooled-trial descriptive SEM'),
        ('K3 RT SD','04_clusters_K3/K3_7_reaction_time_mean_sd.png','Pooled-trial SD'),
        ('K3 RT shared bins','04_clusters_K3/K3_7_reaction_time_shared_bins.png','Shared physical-duration edges'),
        ('K3 RT transitions','04_clusters_K3/K3_8_reaction_time_transitions.png','Fresh labels and intact timelines'),
        ('E/F sequence overview','04_clusters_K3/sequence_performance/overview_All_biased_blocks.png','Three contexts and all cluster/mouse details also rendered'),
        ('Historical comparison','07_comparison_to_frozen/previous_rbf_vs_ward_counts.png','Compare previous four-feature RBF labels only'),
        ('Mouse composition sensitivity','08_scaling_sensitivity/worry_a_between_mouse_offsets/A1_mouse_composition.png','All alternatives use Ward'),
        ('Scaling agreement','08_scaling_sensitivity/worry_a_between_mouse_offsets/A2_primary_vs_alternative_agreement.png','ARI across normalizations'),
        ('Feature distance budget','08_scaling_sensitivity/worry_b_feature_magnitudes/B1_distance_share.png','Actual coordinate squared distances'),
        ('Raw-nats clusters','08_scaling_sensitivity/worry_b_feature_magnitudes/B2_raw_nats_clustering.png','Positive raw Ward groups'),
        ('Primary groups in raw nats','08_scaling_sensitivity/worry_b_feature_magnitudes/B3_primary_clusters_in_raw_nats.png','Original signed gain distributions')]]
    for path in sorted((out/'07_hypothesis_followup').rglob('*.png')):
        mapping.append(dict(reference_family='Hypothesis follow-up: '+path.stem,new_figure=str(path.relative_to(out)),adaptation='All new K3 groups; no inherited strategy identities'))
    for c in (1,2,3):
        mapping.append(dict(reference_family=f'E/F cluster {c} overview',
                            new_figure=f'04_clusters_K3/sequence_performance/cluster_{c}_overview.png',
                            adaptation='E/F pooled event means across all/short/long contexts, using new Ward labels'))
    mapping.append(dict(reference_family='Scaling cross-tabulation',
                        new_figure='08_scaling_sensitivity/worry_a_between_mouse_offsets/A2_primary_vs_alternative_counts.png',
                        adaptation='Primary vs alternative Ward label contingency tables'))
    for suffix in ['7_reaction_time_mean_sem','7_reaction_time_mean_sd','8_reaction_time_transitions']:
        mapping.append(dict(reference_family='K4 extended RT: '+suffix,
                            new_figure=f'04_clusters_K4/K4_{suffix}.png',
                            adaptation='Four-cluster version of the reference extended reaction-time report'))
    for row in mapping:
        if not (out/row['new_figure']).exists():raise RuntimeError('Missing figure family: '+str(row))
    pd.DataFrame(mapping).to_csv(out/'figure_coverage.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--raw-root',type=Path,default=data.RAW_ROOT)
    parser.add_argument('--stability-replicates',type=int,default=100)
    parser.add_argument('--replace',action='store_true',help='Replace only a previously published run of this new analysis, retaining its backup')
    args=parser.parse_args();out=args.output.resolve()
    if args.stability_replicates<1:parser.error('At least one stability replicate required')
    if out==REFERENCE.resolve() or (ROOT.resolve() in out.parents and (out.parent!=ROOT.resolve() or out.name!=NAME)):
        parser.error('Inside the historical clustering directory, output must use the new analysis name')
    if out.exists() and not args.replace:parser.error('Output already exists; use --replace to retain a backup and publish a new run')
    if out.exists():
        protocol_path=out/'02_cluster_labels/protocol.json'
        if not protocol_path.exists() or json.loads(protocol_path.read_text()).get('name')!=NAME:
            parser.error('Refusing replacement of a directory not owned by this analysis')
    out.parent.mkdir(parents=True,exist_ok=True)
    before=protected();own_source=file_hashes(Path(__file__).parent)
    stage=Path(tempfile.mkdtemp(prefix=NAME+'.staging_',dir=out.parent))
    print('STAGING',stage,flush=True)
    try:
        with threadpool_limits(limits=1):protocol=analysis(stage,args.stability_replicates,args.raw_root)
        coverage(stage)
        if own_source!=file_hashes(Path(__file__).parent):raise RuntimeError('Analysis code changed during the run; rerun the final source version')
        if before!=protected():raise RuntimeError('Historical result/code preservation check failed')
        json_write(stage/'06_audit/preservation_audit.json',dict(unchanged=True,n_protected_files=len(before),sha256=before))
        pdfs={str(path.relative_to(stage)):len(PdfReader(path).pages) for path in sorted(stage.rglob('*.pdf'))}
        for k in protocol['k_selection']['reported_K']:
            expected=9 if k==3 else 8 if k==4 else 6
            if pdfs[f'04_clusters_K{k}/K{k}_report.pdf']!=expected:raise RuntimeError('Incorrect cluster report page count')
        if pdfs['07_hypothesis_followup/hypothesis_followup_all_clusters.pdf']!=8:raise RuntimeError('Incomplete followup report')
        if pdfs['04_clusters_K3/sequence_performance/ward_sequence_overview.pdf']!=3:raise RuntimeError('Incomplete sequence overview')
        if pdfs['04_clusters_K3/sequence_performance/ward_sequence_by_cluster.pdf']!=9:raise RuntimeError('Incomplete sequence details')
        output_hashes={str(p.relative_to(stage)):data.sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        json_write(stage/'completion.json',dict(complete=True,selected_K=protocol['k_selection']['selected_K'],
                                             n_sessions=protocol['n_sessions'],pdf_pages=pdfs,sha256=output_hashes))
        backup=None
        if out.exists():
            backup=out.with_name(out.name+'.previous_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.rename(backup)
        try:stage.rename(out)
        except BaseException:
            if backup is not None:backup.rename(out)
            raise
        print('PUBLISHED',out,flush=True)
        if backup:print('Previous run retained at',backup,flush=True)
    except BaseException:
        print('Incomplete staging retained for diagnosis:',stage,flush=True)
        raise
