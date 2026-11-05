"""Robust-medoid, edge-balanced-resampling and chronology sensitivities.

All original rows remain in the main comparisons. No strategy transitions are fitted.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score,silhouette_score
from threadpoolctl import threadpool_limits
from .features import OUT,compact_features,REPO_ROOT
from .cluster import SEED,BLOCK,block_sample,mouse_checks
from .scale_audit import AUDIT_OUT,SCALES,METHODS,geometry,transform,fit_method,predict,matched_jaccard


def circular_sample(subjects,rng):
    """Equal-probability boundary sensitivity; wraps sample blocks, not fit histories.

    This assumes circular stationarity, which is questionable with chronology
    drift. It supplements rather than replaces the nonwrapping diagnostic.
    """
    rows=[]
    for subject in np.unique(subjects):
        ix=np.flatnonzero(subjects==subject);n=len(ix)
        starts=rng.integers(0,n,size=int(np.ceil(n/BLOCK)))
        rows.extend(ix[np.concatenate([(start+np.arange(BLOCK))%n for start in starts])[:n]])
    return np.array(rows)


def source_audit(raw,out):
    rows=[];sources=[]
    for row in raw.itertuples(index=False):
        root=REPO_ROOT/'results_sliding_window'/'M0_sensory_v1'/row.subject/'full_session_matched_v1'
        paths=list(root.glob(f'window_{int(row.window_id):03d}_*/trial_predictions.csv'))
        if len(paths)!=1:raise ValueError('Expected one canonical window file')
        d=pd.read_csv(paths[0],usecols=['session_id','session_date','split_role','chronological_rank','duration_s','category_long','model_evaluated'])
        sessions=d[['session_id','session_date','split_role','chronological_rank']].drop_duplicates().copy()
        sessions['date']=pd.to_datetime(sessions.session_date);sessions=sessions.sort_values('chronological_rank')
        if sessions.groupby('split_role').size().to_dict() != {'test':1,'train':9,'validation':3}:
            raise ValueError('Unexpected source-session split counts')
        if sessions.loc[sessions.split_role.eq('test'),'session_id'].iloc[0] != row.session_id:
            raise ValueError('Unexpected target-session identity')
        train=sessions[sessions.split_role.eq('train')];test=sessions[sessions.split_role.eq('test')].date.iloc[0]
        # Chronological epochs are metadata diagnostics, never weights on forced fits.
        gaps=sessions.date.diff().dt.total_seconds()/86400
        cutoff=sessions.loc[gaps.gt(30),'date'].max() if gaps.gt(30).any() else sessions.date.min()
        for part in sessions.itertuples(index=False):
            sources.append(dict(subject=row.subject,window_id=row.window_id,source_session_id=part.session_id,
                                split_role=part.split_role,source_rank=part.chronological_rank,date=part.date))
        rows.append(dict(subject=row.subject,window_id=row.window_id,session_id=row.session_id,
                         train_start=train.date.min(),train_end=train.date.max(),test_date=test,
                         train_sessions_in_target_epoch=int(train.date.ge(cutoff).sum()),
                         train_span_days=(train.date.max()-train.date.min()).total_seconds()/86400,
                         train_end_to_test_days=(test-train.date.max()).total_seconds()/86400,
                         max_source_gap_days=float(gaps.max()),crosses_30day_gap=bool(gaps.gt(30).any()),
                         train_evaluated_short_trials=int((d.split_role.eq('train') & d.model_evaluated.eq(True)&d.category_long.eq(0)).sum())))
    audit=pd.DataFrame(rows);audit.to_csv(out/'window_calendar_audit.csv',index=False)
    source_sessions=pd.DataFrame(sources)
    source_sessions.to_csv(out/'window_source_sessions.csv',index=False)
    raw_gap_context(source_sessions,out)
    return audit


def raw_gap_context(sources,out):
    """Distinguish gaps in fit-eligible sessions from complete recording pauses."""
    from .latency import RAW_ROOT
    rows=[]
    for subject,d in sources.groupby('subject'):
        d=d[['source_session_id','date']].drop_duplicates().sort_values('date')
        dates=pd.to_datetime(d.date);i=dates.diff().dt.total_seconds().idxmax();pos=d.index.get_loc(i)
        lo,hi=dates.iloc[pos-1],dates.iloc[pos]
        raw=pd.read_csv(RAW_ROOT/f'{subject}_final_data.csv',usecols=['session_date','session_idx']).drop_duplicates()
        rd=pd.to_datetime(raw.session_date.astype(str),format='%Y%m%d')
        between=raw.loc[rd.dt.normalize().gt(lo.normalize())&rd.dt.normalize().lt(hi.normalize()),'session_date'].tolist()
        rows.append(dict(subject=subject,source_gap_start=str(lo),source_gap_end=str(hi),
                         gap_days=(hi-lo).total_seconds()/86400,raw_dates_between=json.dumps(between)))
    pd.DataFrame(rows).to_csv(out/'raw_session_gap_context.csv',index=False)


def run(repeats=200):
    out=AUDIT_OUT;raw=pd.read_csv(OUT/'window_metrics_parameters.csv');v,m=compact_features(raw);subjects=raw.subject.to_numpy()
    protocol=json.loads((out/'protocol.json').read_text())
    protocol['supplement']={'additional_method':'Euclidean PAM BUILD+SWAP, robust unsquared-distance objective; no trimming',
                            'resampling':'circular 13-window boundary sensitivity, all methods/scales; keep original nonwrapping results',
                            'calendar':'predeclared >30 day recording-gap sensitivity on existing windows only',
                            'null':'99 independent within-mouse shifts of whole feature blocks; synchrony reference, not a universal no-cluster null'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    methods=(*METHODS,'pam');refs={};summary=[];lomos=[];frames=[];clipped=[];chrono=[]
    for scale in SCALES:
        x,g,u=geometry(v,m,subjects,scale)
        for j,f in enumerate(g['features']):
            clipped.append(dict(scale=scale,feature=f,fraction_clipped=float(np.mean(np.abs(x[:,j]/g['weights'][j])>=3-1e-10))))
        for method in methods:
            for k in (2,3):
                labels,model=fit_method(x,k,method);refs[scale,method,k]=labels
                summary.append(dict(scale=scale,method=method,K=k,sizes='/'.join(map(str,np.sort(np.bincount(labels,minlength=k+1)[1:]))),
                                    silhouette=float(silhouette_score(x,labels)),**mouse_checks(labels,subjects)))
                ff=raw[['subject','window_id','session_id']].copy();ff['scale'],ff['method'],ff['K'],ff['cluster_label']=scale,method,k,labels;frames.append(ff)
                time_labels=np.zeros(len(raw),int)
                for subject in np.unique(subjects):
                    ix=np.flatnonzero(subjects==subject);time_labels[ix]=np.minimum(np.arange(len(ix))*k//len(ix),k-1)+1
                    keep=subjects!=subject
                    xd,gd,_=geometry(v.loc[keep].reset_index(drop=True),m,subjects[keep],scale)
                    xt=transform(v.loc[~keep],subjects[~keep],gd);ld,md=fit_method(xd,k,method);pred=predict(xt,xd,ld,md,method)
                    true=labels[~keep];valid=len(np.unique(true))>1 and len(np.unique(pred))>1
                    table=pd.crosstab(pd.Series(true,name='reference'),pd.Series(pred,name='predicted'))
                    for c in range(1,k+1):
                        for p in range(1,k+1):
                            lomos.append(dict(scale=scale,method=method,K=k,subject=subject,reference=c,predicted=p,
                                              n=int(table.loc[c,p]) if c in table.index and p in table.columns else 0,
                                              ari=float(adjusted_rand_score(true,pred)),valid_multigroup=valid))
                chrono.append(dict(scale=scale,method=method,K=k,ari_to_equal_chronology_partition=float(adjusted_rand_score(labels,time_labels))))
    pd.DataFrame(clipped).to_csv(out/'scale_clipping.csv',index=False)
    pd.DataFrame(chrono).to_csv(out/'chronology_agreement.csv',index=False)
    pd.DataFrame(lomos).to_csv(out/'lomo_contingencies.csv',index=False)
    pd.concat(frames).to_csv(out/'all_candidate_labels.csv',index=False)
    boot=[]
    # Same random draws for every candidate and method. Original PAM is the sole
    # new method on the old resampler; the circular sensitivity includes all four.
    for scheme in ('nonwrapping_pam','circular'):
        rng=np.random.default_rng(SEED)
        for draw in range(repeats):
            ix=(block_sample(subjects,rng) if scheme=='nonwrapping_pam' else circular_sample(subjects,rng))
            unique,first=np.unique(ix,return_index=True)
            for scale in SCALES:
                xb,_,_=geometry(v.iloc[ix].reset_index(drop=True),m,subjects[ix],scale)
                for method in (('pam',) if scheme=='nonwrapping_pam' else methods):
                    for k in (2,3):
                        lb,_=fit_method(xb,k,method);reference=refs[scale,method,k][unique]
                        j=matched_jaccard(reference,lb[first],k)
                        boot.append(dict(scheme=scheme,draw=draw,scale=scale,method=method,K=k,
                                         ari=float(adjusted_rand_score(reference,lb[first])),all_groups_present=len(np.unique(reference))==k,
                                         min_present_jaccard=float(np.nanmin(j)),**{f'jaccard_C{i+1}':x for i,x in enumerate(j)}))
            if (draw+1)%50==0:print(scheme,draw+1,flush=True)
    b=pd.DataFrame(boot);b.to_csv(out/'supplement_bootstrap_draws.csv',index=False)
    old=pd.read_csv(out/'scale_method_comparison.csv');lm=pd.DataFrame(lomos).drop_duplicates(['scale','method','K','subject'])
    for row in summary:
        mask=lambda d:d.scale.eq(row['scale']) & d.method.eq(row['method']) & d.K.eq(row['K'])
        draw=b[mask(b)&b.scheme.eq('circular')];l=lm[mask(lm)]
        row.update(circular_ari_median=float(draw.ari.median()),circular_ari_p10=float(draw.ari.quantile(.1)),
                   circular_all_group_coverage=float(draw.all_groups_present.mean()),
                   circular_min_present_jaccard_median=float(draw.min_present_jaccard.median()),
                   lomo_ari_min=float(l.ari.min()),all_lomo_multigroup=bool(l.valid_multigroup.all()))
        if row['method']=='pam':
            old_draw=b[mask(b)&b.scheme.eq('nonwrapping_pam')]
            row['nonwrapping_ari_median']=float(old_draw.ari.median())
        else:row['nonwrapping_ari_median']=float(old[mask(old)].bootstrap_ari_median.iloc[0])
    pd.DataFrame(summary).to_csv(out/'all_method_summary.csv',index=False)
    calendar=source_audit(raw,out)
    # A calendar sensitivity, not deletion from the original population or a default.
    keep=~calendar.crosses_30day_gap.to_numpy();epoch=[]
    for scale in ('own_z','pooled_z'):
        if min(pd.Series(subjects[keep]).value_counts())<13:continue
        vv=v.loc[keep].reset_index(drop=True);ss=subjects[keep];xx,_,_=geometry(vv,m,ss,scale)
        for method in ('kmeans','pam'):
            for k in (2,3):
                labels,_=fit_method(xx,k,method);frame=raw.loc[keep,['subject','window_id','session_id']].copy()
                frame['scale'],frame['method'],frame['K'],frame['cluster_label']=scale,method,k,labels;epoch.append(frame)
    if epoch:pd.concat(epoch).to_csv(out/'calendar_sensitivity_labels.csv',index=False)
    # Shift families together to preserve sensory/prior/performance internal structure.
    null=[];rng=np.random.default_rng(SEED+142)
    for draw in range(99):
        shifted=v.copy()
        for subject in np.unique(subjects):
            ix=np.flatnonzero(subjects==subject)
            for block,g in m.groupby('block'):
                shift=int(rng.integers(0,len(ix)))
                shifted.loc[ix,g.feature]=np.roll(v.loc[ix,g.feature].to_numpy(),shift,axis=0)
        xx,_,_=geometry(shifted,m,subjects,'pooled_z')
        for k in (2,3):
            ll,_=fit_method(xx,k,'kmeans');null.append(dict(draw=draw,K=k,silhouette=float(silhouette_score(xx,ll))))
    pd.DataFrame(null).to_csv(out/'block_synchrony_reference.csv',index=False)
    print(pd.DataFrame(summary).round(3).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
