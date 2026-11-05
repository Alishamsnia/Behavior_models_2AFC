"""Validate predeclared recording-gap sensitivity; do not alter the main cohort."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score,silhouette_score
from threadpoolctl import threadpool_limits
from .features import OUT,compact_features
from .cluster import SEED,block_sample,mouse_checks
from .scale_audit import AUDIT_OUT,geometry,transform,fit_method,predict,matched_jaccard
from .scale_supplement import circular_sample


def run(repeats=200):
    raw=pd.read_csv(OUT/'window_metrics_parameters.csv');calendar=pd.read_csv(AUDIT_OUT/'window_calendar_audit.csv')
    keep=~calendar.crosses_30day_gap.to_numpy();v,m=compact_features(raw);v=v.loc[keep].reset_index(drop=True)
    raw=raw.loc[keep].reset_index(drop=True);subjects=raw.subject.to_numpy()
    refs={};rows=[];draws=[]
    for scale in ('own_z','pooled_z'):
        x,_,_=geometry(v,m,subjects,scale)
        for method in ('kmeans','pam'):
            for k in (2,3):
                labels,_=fit_method(x,k,method);refs[scale,method,k]=labels
                transfers=[];valid=[]
                for subject in np.unique(subjects):
                    donor=subjects!=subject;xd,g,_=geometry(v.loc[donor].reset_index(drop=True),m,subjects[donor],scale)
                    xt=transform(v.loc[~donor],subjects[~donor],g);ld,model=fit_method(xd,k,method)
                    pp=predict(xt,xd,ld,model,method);true=labels[~donor]
                    transfers.append(adjusted_rand_score(true,pp));valid.append(len(np.unique(true))>1 and len(np.unique(pp))>1)
                rows.append(dict(scale=scale,method=method,K=k,n_windows=len(raw),
                                 sizes='/'.join(map(str,np.sort(np.bincount(labels,minlength=k+1)[1:]))),
                                 silhouette=float(silhouette_score(x,labels)),lomo_ari_min=float(min(transfers)),
                                 all_lomo_multigroup=all(valid),**mouse_checks(labels,subjects)))
    for scheme,sampler in [('nonwrapping',block_sample),('circular',circular_sample)]:
        rng=np.random.default_rng(SEED)
        for draw in range(repeats):
            ix=sampler(subjects,rng);unique,first=np.unique(ix,return_index=True)
            for scale in ('own_z','pooled_z'):
                x,_,_=geometry(v.iloc[ix].reset_index(drop=True),m,subjects[ix],scale)
                for method in ('kmeans','pam'):
                    for k in (2,3):
                        ll,_=fit_method(x,k,method);ref=refs[scale,method,k][unique];j=matched_jaccard(ref,ll[first],k)
                        draws.append(dict(scheme=scheme,scale=scale,method=method,K=k,draw=draw,
                                          ari=float(adjusted_rand_score(ref,ll[first])),min_present_jaccard=float(np.nanmin(j))))
            if (draw+1)%100==0:print('calendar',scheme,draw+1,flush=True)
    b=pd.DataFrame(draws)
    for row in rows:
        for scheme in ('nonwrapping','circular'):
            d=b[b.scale.eq(row['scale'])&b.method.eq(row['method'])&b.K.eq(row['K'])&b.scheme.eq(scheme)]
            row[scheme+'_ari_median']=float(d.ari.median());row[scheme+'_ari_p10']=float(d.ari.quantile(.1))
            row[scheme+'_min_present_jaccard_median']=float(d.min_present_jaccard.median())
    pd.DataFrame(rows).to_csv(AUDIT_OUT/'calendar_sensitivity_metrics.csv',index=False)
    b.to_csv(AUDIT_OUT/'calendar_bootstrap_draws.csv',index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
