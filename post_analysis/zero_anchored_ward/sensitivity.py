"""Scaling and historical-label comparisons; all fresh clusterings use Ward."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score, silhouette_score
from .core import FEATURES, LABELS, KEYS, normalize, ward_tree, partition
from . import plots as p


def cramers_v(subjects,labels):
    table=pd.crosstab(subjects,labels)
    if min(table.shape)<2:return 0.
    return float(np.sqrt(chi2_contingency(table,correction=False)[0]/(table.to_numpy().sum()*(min(table.shape)-1))))


def coordinates(gains,subjects):
    g=normalize(gains)
    result={'colstd_rowl2':g.z,'rowl2_only':g.positive/np.hypot(np.linalg.norm(g.positive,axis=1),1e-12)[:,None],
            'colstd_only':g.scaled,'positive_raw':g.positive}
    legacy=np.zeros_like(gains)
    for subject in np.unique(subjects):
        ix=np.asarray(subjects)==subject;a=gains[ix];sd=a.std(axis=0,ddof=1);sd[sd<=1e-8]=1
        legacy[ix]=np.clip((a-a.mean(axis=0))/sd,-3,3)
    result['within_mouse_z']=legacy
    return result


def run(raw,gains,labels,out):
    out.mkdir(parents=True,exist_ok=True);coords=coordinates(gains,raw.subject)
    # Keep common nonzero-session population for fair ARI comparisons.
    active=normalize(gains).eligible;rows=[];records=[];share=[];etas=[]
    ks=sorted(labels.K.unique())
    all_labels={}
    for name,z in coords.items():
        zz=z[active];tree=ward_tree(zz)
        totals=np.array([np.sum(pdist(zz[:,j,None],metric='sqeuclidean')) for j in range(4)])
        for j in range(4):share.append(dict(scaling=name,feature=FEATURES[j],distance_share=totals[j]/totals.sum() if totals.sum() else np.nan))
        for k in ks:
            ref=labels[labels.K.eq(k)].cluster_label.to_numpy()[active]
            lab=partition(tree,k);all_labels[(name,k)]=lab
            rows.append(dict(scaling=name,K=k,ari_to_primary=adjusted_rand_score(ref,lab),silhouette=silhouette_score(zz,lab),
                             mouse_cramers_v=cramers_v(raw.subject[active].to_numpy(),lab),sizes='/'.join(str(np.sum(lab==c)) for c in range(1,k+1))))
            records.append(raw.loc[active,KEYS].assign(scaling=name,K=k,cluster_label=lab))
            for j in range(4):
                x=gains[active,j];total=np.sum((x-x.mean())**2)
                between=sum(np.sum(lab==c)*(x[lab==c].mean()-x.mean())**2 for c in np.unique(lab))
                etas.append(dict(scaling=name,K=k,feature=FEATURES[j],eta_squared=between/total if total else np.nan))
    summary=pd.DataFrame(rows);budget=pd.DataFrame(share)
    summary.to_csv(out/'scaling_sensitivity_summary.csv',index=False);budget.to_csv(out/'feature_distance_share.csv',index=False)
    pd.concat(records,ignore_index=True).to_csv(out/'labels_by_scaling.csv',index=False)
    pd.DataFrame(etas).to_csv(out/'cluster_separation_eta_squared.csv',index=False)
    folder_a=out/'worry_a_between_mouse_offsets';folder_b=out/'worry_b_feature_magnitudes'
    fig,axes=p.shell('Ward sensitivity: mouse composition across normalizations',len(ks),len(coords),size=(23,4*len(ks)+2),
                     note='All variants use Euclidean Ward. Within-mouse Z is a comparison only; cluster numbers are local to each partition.')
    for row,k in enumerate(ks):
        for col,name in enumerate(coords):
            lab=all_labels[(name,k)]
            t=pd.crosstab(pd.Series(lab),pd.Series(raw.subject[active].to_numpy())).reindex(range(1,k+1),fill_value=0)
            p.matrix(axes[row,col],t,t.columns,[f'C{c}' for c in range(1,k+1)],f'{name} / K={k}',0,None,'Blues','.0f')
    p.footer(fig,'Mouse association is descriptive. Subject-specific clusters are not automatically invalid and are not prevented by the algorithm.')
    p.save(fig,folder_a/'A1_mouse_composition')
    fig,axes=p.shell('Ward sensitivity: partition agreement with primary normalization',1,1,size=(12,6),note='Adjusted Rand index is invariant to label numbering.')
    for name in coords:
        s=summary[summary.scaling.eq(name)];axes[0,0].plot(s.K,s.ari_to_primary,'o-',label=name)
    axes[0,0].set_xticks(ks);axes[0,0].set_xlabel('K');axes[0,0].set_ylabel('ARI to column-SD + row-L2');axes[0,0].legend(fontsize=9)
    p.save(fig,folder_a/'A2_primary_vs_alternative_agreement')
    fig,axes=p.shell('K=3: primary vs alternative Ward session counts',1,len(coords)-1,size=(22,6),
                     note='Rows are primary clusters; columns are local alternative labels. Numbers are sessions, without label alignment.')
    primary=labels[labels.K.eq(3)].cluster_label.to_numpy()[active]
    for ax,name in zip(axes[0],[name for name in coords if name!='colstd_rowl2']):
        counts=pd.crosstab(pd.Series(primary),pd.Series(all_labels[(name,3)])).reindex(index=range(1,4),columns=range(1,4),fill_value=0)
        p.matrix(ax,counts,[f'Alt C{c}' for c in range(1,4)],[f'Primary C{c}' for c in range(1,4)],name,0,None,'Blues','.0f')
    p.save(fig,folder_a/'A2_primary_vs_alternative_counts')
    fig,axes=p.shell('Which coordinates contribute to squared Euclidean distance?',1,1,size=(12,6),note='Pairwise squared-distance totals in each actual clustering space; no claim of equal contributions after row normalization.')
    bottom=np.zeros(len(coords));names=list(coords)
    for j,feature in enumerate(FEATURES):
        vals=budget[budget.feature.eq(feature)].set_index('scaling').reindex(names).distance_share.to_numpy()
        axes[0,0].bar(names,vals,bottom=bottom,label=LABELS[j],color=p.COLORS[j]);bottom+=vals
    axes[0,0].set_ylabel('Share of total squared distance');axes[0,0].legend(fontsize=9,ncol=2);axes[0,0].tick_params(axis='x',rotation=15)
    p.save(fig,folder_b/'B1_distance_share')
    fig,axes=p.shell('Raw positive-gain Ward clusters, viewed in signed nats/trial',1,4,size=(18,6),note='K=3; alternative clustering removes both SD scaling and row normalization.')
    alternative=all_labels[('positive_raw',3)]
    for j,ax in enumerate(axes[0]):
        ax.boxplot([gains[active,j][alternative==c] for c in (1,2,3)],tick_labels=['C1','C2','C3']);ax.axhline(0,color='.6',ls=':');ax.set_title(LABELS[j]);ax.set_ylabel('Signed gain (nats/trial)')
    p.save(fig,folder_b/'B2_raw_nats_clustering')
    fig,axes=p.shell('Primary directional Ward groups, viewed in signed nats/trial',1,4,size=(18,6),note='K=3 primary labels; these are original signed gains, not normalized coordinates.')
    primary=labels[labels.K.eq(3)].cluster_label.to_numpy()[active]
    for j,ax in enumerate(axes[0]):
        ax.boxplot([gains[active,j][primary==c] for c in (1,2,3)],tick_labels=['C1','C2','C3']);ax.axhline(0,color='.6',ls=':');ax.set_title(LABELS[j]);ax.set_ylabel('Signed gain (nats/trial)')
    p.save(fig,folder_b/'B3_primary_clusters_in_raw_nats')
    (out/'protocol.json').write_text(json.dumps(dict(primary='colstd_rowl2',variants=list(coords),clustering='Euclidean Ward for every variant',
        common_nonzero_population=int(active.sum()),within_mouse_z='signed gains, within-mouse sample SD and centering, clip +/-3; comparison only'),indent=2)+'\n')
    (out/'README.md').write_text('# Normalization sensitivity\n\nAll five variants use independently computed Euclidean Ward labels on the common nonzero-session population. The primary geometry is clamped gains / global sample SD followed by row L2. Alternatives remove column scaling, row normalization, or both; within-mouse signed Z is included only as a sensitivity comparison. Labels are local to each partition. ARI does not depend on cluster numbering. Cramer V quantifies descriptive mouse association. Distance shares use actual squared coordinate differences. Separation eta-squared is computed on original signed gains and is descriptive.\n')


def historical(labels,path,out):
    old=pd.read_csv(path);out.mkdir(parents=True,exist_ok=True);rows=[];tables=[]
    for k in sorted(set(old.K)&set(labels.K)):
        a=old[old.K.eq(k)][KEYS+['cluster_label']].rename(columns={'cluster_label':'old_cluster'})
        b=labels[labels.K.eq(k)][KEYS+['cluster_label']]
        t=a.merge(b,on=KEYS,how='outer',validate='one_to_one',indicator=True)
        if not t._merge.eq('both').all():raise ValueError('Historical comparison session populations differ')
        t=t[t.cluster_label.gt(0)]
        counts=pd.crosstab(t.old_cluster,t.cluster_label).reindex(index=range(1,k+1),columns=range(1,k+1),fill_value=0)
        ri,ci=linear_sum_assignment(-counts.to_numpy())
        rows.append(dict(K=k,ari_to_previous_rbf=adjusted_rand_score(t.old_cluster,t.cluster_label),n_sessions=len(t),
                         best_matched_sessions=int(counts.to_numpy()[ri,ci].sum())))
        tables.append((k,counts))
    pd.DataFrame(rows).to_csv(out/'label_agreement_with_frozen.csv',index=False)
    fig,axes=p.shell('Previous four-feature RBF vs new zero-anchored Ward labels',1,len(tables),size=(6*len(tables),6),
                     note='Both normalization and clustering change. Differences cannot be attributed to Ward alone; labels are not renumbered to match.')
    for ax,(k,t),r in zip(axes[0],tables,rows):
        p.matrix(ax,t,[f'Ward C{c}' for c in t.columns],[f'RBF C{c}' for c in t.index],f'K={k}: ARI={r["ari_to_previous_rbf"]:.2f}',0,None,'Blues','.0f')
    p.save(fig,out/'previous_rbf_vs_ward_counts')
    (out/'README.md').write_text('Read-only label comparison to the previous four-feature star analysis. Only its label CSV is read; no previous code, normalized inputs or behavior caches are used. Cluster numbering is independent. Both preprocessing and clustering differ.\n')
