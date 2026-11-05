"""Predeclared scale/method sensitivity audit; never overwrites the compact results.

python -m post_analysis.session_clustering.scale_audit --bootstrap 200
No behavioral-model refits, test-outcome selection or equal-size constraints.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score,silhouette_score
from threadpoolctl import threadpool_limits
from .features import OUT,compact_features,select_features
from .cluster import SEED,BLOCK,block_sample,mouse_checks

SCALES=('own_z','pooled_z','pooled_robust')
METHODS=('kmeans','ward','gmm_tied')
AUDIT_OUT=OUT.parent/'sliding_window_scale_audit'


def geometry(values,meta,subjects,scale):
    """Select on donors/resample; equal block weights; shared scale fits donors only."""
    if scale not in SCALES:raise ValueError(scale)
    audit=select_features(values,meta,subjects)
    used=audit[audit.selected].copy().reset_index(drop=True)
    if used.empty:raise ValueError('No usable features')
    weights=[]
    for row in used.itertuples():
        group=used[used.block.eq(row.block)]
        weights.append(1/np.sqrt(group.model.nunique()*group.model.eq(row.model).sum()))
    data=values[used.feature].to_numpy(float);subjects=np.asarray(subjects)
    variances=[];iqrs=[]
    for subject in np.unique(subjects):
        d=data[subjects==subject]
        variances.append(d.var(axis=0,ddof=1));iqrs.append(np.diff(np.quantile(d,[.25,.75],axis=0),axis=0)[0]/1.3489795)
    pooled=np.sqrt(np.mean(variances,axis=0))
    denominator=(np.maximum(np.median(iqrs,axis=0),.5*pooled) if scale=='pooled_robust' else pooled)
    model=dict(features=used.feature.tolist(),weights=np.array(weights),scale=scale,
               denominator=np.where(denominator>1e-8,denominator,1.))
    return transform(values,subjects,model),model,used


def transform(values,subjects,model):
    data=values[model['features']].to_numpy(float);subjects=np.asarray(subjects);z=np.zeros_like(data)
    for subject in np.unique(subjects):
        ix=subjects==subject;d=data[ix]
        if model['scale']=='own_z':
            denominator=d.std(axis=0,ddof=1);denominator=np.where(denominator>1e-8,denominator,1.)
        else:denominator=model['denominator']
        center=np.median(d,axis=0) if model['scale']=='pooled_robust' else d.mean(axis=0)
        z[ix]=(d-center)/denominator
    return np.clip(z,-3,3)*model['weights']


class PAM:
    """Deterministic BUILD + best SWAP k-medoids with Euclidean (not squared) cost.

    Used only as a robust sensitivity: every session retains a label. No trimming,
    minimum-size constraints, mouse labels or test outcomes enter its objective.
    """
    def __init__(self,k):self.k=k

    def fit(self,x):
        distances=cdist(x,x);medoids=[int(distances.sum(axis=0).argmin())]
        while len(medoids)<self.k:
            current=distances[:,medoids].min(axis=1)
            costs=np.minimum(current[:,None],distances).sum(axis=0);costs[medoids]=np.inf
            medoids.append(int(costs.argmin()))
        for _ in range(100):
            d=distances[:,medoids];nearest=d.argmin(axis=1);best=d.min(axis=1);second=np.partition(d,1,axis=1)[:,1]
            current=best.sum();choice=None;best_cost=current
            candidates=np.array([i for i in range(len(x)) if i not in medoids])
            for slot in range(self.k):
                base=np.where(nearest==slot,second,best)
                costs=np.minimum(base[:,None],distances[:,candidates]).sum(axis=0)
                j=int(costs.argmin())
                if costs[j]<best_cost-1e-10:best_cost=float(costs[j]);choice=(slot,int(candidates[j]))
            if choice is None:break
            medoids[choice[0]]=choice[1]
        else:raise RuntimeError('PAM swap iterations did not converge')
        self.medoid_indices_=np.array(medoids);self.centers_=np.asarray(x)[medoids].copy()
        self.labels_=self.predict(x);self.cost_=float(cdist(x,self.centers_).min(axis=1).sum())
        return self

    def predict(self,x):return cdist(x,self.centers_).argmin(axis=1)


def fit_method(x,k,method):
    if method=='kmeans':
        model=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit(x);labels=model.labels_
    elif method=='pam':
        model=PAM(k).fit(x);labels=model.labels_
    elif method=='ward':
        model=AgglomerativeClustering(n_clusters=k,linkage='ward').fit(x);labels=model.labels_
    elif method=='gmm_tied':
        model=GaussianMixture(n_components=k,covariance_type='tied',reg_covar=.05,n_init=10,
                              max_iter=500,random_state=SEED).fit(x);labels=model.predict(x)
        if not model.converged_:raise RuntimeError('GMM failed to converge')
    else:raise ValueError(method)
    return labels+1,model


def predict(x,donor_x,donor_labels,model,method):
    if method in ('kmeans','gmm_tied','pam'):return model.predict(x)+1
    classes=np.unique(donor_labels)
    centroids=np.vstack([donor_x[donor_labels==c].mean(axis=0) for c in classes])
    return classes[cdist(x,centroids).argmin(axis=1)]


def matched_jaccard(reference,replicate,k):
    """Match labels on sampled unique rows. Missing reference groups stay NaN."""
    table=np.zeros((k,k),int)
    for a,b in zip(reference,replicate):table[a-1,b-1]+=1
    aa,bb=linear_sum_assignment(-table)
    out=np.full(k,np.nan)
    for a,b in zip(aa,bb):
        if table[a].sum()>0:
            out[a]=table[a,b]/(table[a].sum()+table[:,b].sum()-table[a,b])
    return out


def coverage(raw,labels,values,meta,out):
    subjects=raw.subject.to_numpy();counts=np.zeros(len(raw));rng=np.random.default_rng(SEED);rows=[]
    small=int(pd.Series(labels).value_counts().idxmin());rare=np.flatnonzero(labels==small)
    for i in range(10000):
        ix=np.unique(block_sample(subjects,rng));counts[ix]+=1
        rows.append(dict(draw=i,n_small_group_rows=int(np.isin(rare,ix).sum())))
    pd.DataFrame(rows).to_csv(out/'baseline_resampling_coverage.csv',index=False)
    data=raw[['subject','window_id','session_id']].copy();data['inclusion_probability']=counts/10000
    data['baseline_K3_cluster']=labels;data.to_csv(out/'window_resampling_inclusion.csv',index=False)
    x,_,used=geometry(values,meta,subjects,'own_z')
    diff=(x[labels==small].mean(axis=0)-x[labels!=small].mean(axis=0))**2
    pd.DataFrame(dict(feature=used.feature,block=used.block,squared_centroid_contribution=diff,
                      fraction_of_squared_separation=diff/diff.sum())).to_csv(out/'tiny_group_feature_contributions.csv',index=False)


def run(repeats=200,out=AUDIT_OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(OUT/'window_metrics_parameters.csv');values,meta=compact_features(raw)
    subjects=raw.subject.to_numpy();baseline=pd.read_csv(OUT/'session_clusters.csv')
    refs={};summaries=[];labels_rows=[];lomo_rows=[];boot_rows=[];feature_rows=[];block_rows=[]
    # Independent of test behavior. Save this protocol before any candidate is fitted.
    protocol=dict(scales=list(SCALES),methods=list(METHODS),K=[2,3],seed=SEED,repeats=repeats,block=BLOCK,
                  clipping=3,gmm=dict(covariance='tied',reg_covar=.05,n_init=10),
                  primary_candidate='pooled_z + kmeans (scale intervention only; declared before results)',
                  selection='No tuning by behavior, desired group sizes, or mouse gate; all candidates reported',
                  resampling='Existing nonwrapping moving blocks; report coverage and conditional stability, do not conceal unconditional ARI',
                  limitations='Retrospective within-mouse centering; overlapping source sessions; Ward LOMO uses nearest-centroid surrogate')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for scale in SCALES:
        x,geom,used=geometry(values,meta,subjects,scale)
        for i,feature in enumerate(geom['features']):
            feature_rows.append(dict(scale=scale,feature=feature,shared_denominator=geom['denominator'][i],weight=geom['weights'][i]))
        for method in METHODS:
            for k in (2,3):
                labels,model=fit_method(x,k,method);refs[scale,method,k]=labels
                row=dict(scale=scale,method=method,K=k,n_features=x.shape[1],
                         sizes='/'.join(map(str,np.sort(np.bincount(labels)[1:]))),
                         smallest_cluster=int(np.bincount(labels)[1:].min()),silhouette=float(silhouette_score(x,labels)),
                         ari_to_current=float(adjusted_rand_score(baseline[baseline.K.eq(k)].cluster_label,labels)),
                         **mouse_checks(labels,subjects))
                if method=='gmm_tied':
                    row['mean_max_membership']=float(model.predict_proba(x).max(axis=1).mean())
                summaries.append(row)
                f=raw[['subject','window_id','session_id']].copy();f['scale'],f['method'],f['K'],f['cluster_label']=scale,method,k,labels
                labels_rows.append(f)
                for subject in np.unique(subjects):
                    donor=subjects!=subject
                    xd,gd,_=geometry(values.loc[donor].reset_index(drop=True),meta,subjects[donor],scale)
                    xt=transform(values.loc[~donor],subjects[~donor],gd)
                    ld,md=fit_method(xd,k,method);pred=predict(xt,xd,ld,md,method)
                    true=labels[~donor];valid=len(np.unique(true))>1 and len(np.unique(pred))>1
                    lomo_rows.append(dict(scale=scale,method=method,K=k,subject=subject,
                                          ari=float(adjusted_rand_score(true,pred)),valid_multigroup=valid,
                                          reference_groups=int(len(np.unique(true))),predicted_groups=int(len(np.unique(pred)))))
        # Whole-block deletion is a sensitivity check, never a strategy-selection criterion.
        for block in meta.block.unique():
            reduced=meta[meta.block.ne(block)]
            xr,_,_=geometry(values,reduced,subjects,scale)
            for method in METHODS:
                for k in (2,3):
                    ll,_=fit_method(xr,k,method)
                    block_rows.append(dict(scale=scale,method=method,K=k,removed_block=block,
                                           ari=float(adjusted_rand_score(refs[scale,method,k],ll)),
                                           smallest_cluster=int(np.bincount(ll)[1:].min()),**mouse_checks(ll,subjects)))
    # Shared draws across all methods/scales/K. Geometry refits inside each draw.
    rng=np.random.default_rng(SEED)
    for draw in range(repeats):
        ix=block_sample(subjects,rng);unique,first=np.unique(ix,return_index=True)
        for scale in SCALES:
            xb,_,_=geometry(values.iloc[ix].reset_index(drop=True),meta,subjects[ix],scale)
            for method in METHODS:
                for k in (2,3):
                    lb,_=fit_method(xb,k,method);ref=refs[scale,method,k][unique]
                    j=matched_jaccard(ref,lb[first],k)
                    boot_rows.append(dict(draw=draw,scale=scale,method=method,K=k,
                                          ari=float(adjusted_rand_score(ref,lb[first])),
                                          all_reference_groups_present=len(np.unique(ref))==k,
                                          minimum_present_group_jaccard=float(np.nanmin(j)),
                                          **{f'jaccard_C{c+1}':v for c,v in enumerate(j)}))
        if (draw+1)%20==0:
            print(f'Finished {draw+1}/{repeats} paired bootstrap draws',flush=True)
    boot=pd.DataFrame(boot_rows);lomo=pd.DataFrame(lomo_rows)
    for row in summaries:
        mask=lambda d:d.scale.eq(row['scale']) & d.method.eq(row['method']) & d.K.eq(row['K'])
        b=boot[mask(boot)];l=lomo[mask(lomo)];covered=b[b.all_reference_groups_present]
        row.update(bootstrap_ari_median=float(b.ari.median()),bootstrap_ari_p10=float(b.ari.quantile(.1)),
                   all_group_coverage=float(b.all_reference_groups_present.mean()),
                   conditional_ari_median=float(covered.ari.median()) if len(covered) else np.nan,
                   lomo_ari_min=float(l.ari.min()),all_lomo_multigroup=bool(l.valid_multigroup.all()),
                   min_present_group_jaccard_median=float(b.minimum_present_group_jaccard.median()))
    pd.DataFrame(summaries).to_csv(out/'scale_method_comparison.csv',index=False)
    pd.concat(labels_rows).to_csv(out/'candidate_labels.csv',index=False)
    boot.to_csv(out/'bootstrap_draws.csv',index=False);lomo.to_csv(out/'leave_mouse_out.csv',index=False)
    pd.DataFrame(block_rows).to_csv(out/'leave_feature_block_out.csv',index=False)
    pd.DataFrame(feature_rows).to_csv(out/'scale_denominators.csv',index=False)
    coverage(raw,baseline[baseline.K.eq(3)].cluster_label.to_numpy(),values,meta,out)
    print(pd.DataFrame(summaries).round(3).to_string(index=False),flush=True)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bootstrap',type=int,default=200)
    args=parser.parse_args()
    if args.bootstrap<20:parser.error('At least 20 draws are required')
    with threadpool_limits(limits=1):run(args.bootstrap)

if __name__=='__main__':main()
