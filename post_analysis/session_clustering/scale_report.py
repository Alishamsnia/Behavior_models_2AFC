"""Two candidate PDFs: cluster/audit pages and existing observed behavior/RT layouts."""
from pathlib import Path
import json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .features import OUT,compact_features
from .scale_audit import AUDIT_OUT,geometry,fit_method
from .cluster import cluster_figure,raster_order,GATES,SEED
from .behavior import figures

CANDIDATE_OUT=OUT.parent/'sliding_window_epoch_candidate'


def comparison_page():
    d=pd.read_csv(AUDIT_OUT/'all_method_summary.csv')
    names={'own_z':'Own SD','pooled_z':'Shared SD','pooled_robust':'Shared robust','kmeans':'K-means','ward':'Ward','gmm_tied':'Tied GMM','pam':'K-medoids'}
    rows=[]
    for r in d.itertuples():
        rows.append([names[r.scale],names[r.method],r.K,r.sizes,f'{r.nonwrapping_ari_median:.3f}',
                     f'{r.circular_ari_median:.3f}',f'{r.lomo_ari_min:.3f}',r.min_per_mouse,f'{r.mouse_nmi:.3f}'])
    fig=plt.figure(figsize=(16,11));ax=fig.add_axes([.04,.16,.92,.68]);ax.axis('off')
    table=ax.table(cellText=rows,colLabels=['Scale','Method','K','Group sizes','ARI: nonwrap','ARI: circular','Min LOMO ARI','Min / mouse','Mouse NMI'],
                   cellLoc='center',bbox=[0,0,1,1],colWidths=[.13,.12,.04,.12,.13,.13,.13,.10,.10])
    table.auto_set_font_size(False);table.set_fontsize(9)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('.8');cell.set_linewidth(.4)
        if r==0:cell.set_facecolor('#e5ecf3');cell.set_text_props(weight='bold')
        elif rows[r-1][2]==3 and rows[r-1][-2]==0:cell.set_facecolor('#fff1ef')
    fig.suptitle('Scale and method audit — all 92 windows',fontsize=19,y=.96)
    fig.text(.5,.91,'Fixed feature panel; three scales; four methods; K=2/3. No trimming or balanced-size constraints.',ha='center',fontsize=12)
    fig.text(.04,.10,'200 paired resamples per scheme; all preprocessing refitted. Higher ARI is better; lower mouse NMI is better.\n'
             'Nonwrapping blocks under-cover boundary windows. Circular blocks balance coverage but assume circular stationarity.\n'
             'Ward transfer uses a nearest-centroid surrogate. Tied GMM changes the geometry. K-medoids is exploratory.\n'
             'Larger clusters, low mouse NMI or high silhouette alone do not establish shared strategies.',fontsize=11,va='top')
    return fig


def calendar_page(raw,v):
    d=pd.read_csv(AUDIT_OUT/'calendar_sensitivity_metrics.csv');a=pd.read_csv(AUDIT_OUT/'window_calendar_audit.csv')
    fig=plt.figure(figsize=(16,11));fig.suptitle('Source-session gap sensitivity — stronger evidence for K=2, not K=3',fontsize=18,y=.97)
    fig.text(.5,.925,'Existing windows with no >30-day source-session gap: 69 retained (MC06 22, MC07 25, MC08 22); 23 marked unclassified.',ha='center',fontsize=11)
    rows=raw.subject.eq('MC06') & raw.window_id.le(12);small=raw.window_id.le(3)[rows]
    for j,(name,title) in enumerate([('history_alpha','MC06: action-history update rate'),('noise_short','MC06: effective sensory SD at 0.8 s')]):
        ax=fig.add_axes([.08+j*.47,.61,.38,.23]);y=v.loc[rows,name].to_numpy();y=np.exp(y) if name=='noise_short' else y
        ax.scatter(raw.loc[rows,'window_id'],y,c=np.where(small,'#b52828','#4477aa'),s=45)
        ax.set_title(title,fontsize=12);ax.set_xlabel('Historical fitting window');ax.set_ylabel('Update rate' if name=='history_alpha' else 'SD (s)')
        ax.spines[['top','right']].set_visible(False)
    fig.text(.08,.50,'First two MC06 training sets end 104 days before their January test sessions. The third has only one January training session.\n'
             'Red points are the three-session group. These are training-fit profiles, not fits to the January test sessions.\n'
             'Calendar mismatch is a plausible contributor; task changes and parameter uncertainty remain untested.',fontsize=11)
    body=[]
    for r in d.itertuples():
        body.append(['Own SD' if r.scale=='own_z' else 'Shared SD','K-means' if r.method=='kmeans' else 'K-medoids',r.K,r.sizes,
                     f'{r.nonwrapping_ari_median:.3f}',f'{r.circular_ari_median:.3f}',f'{r.lomo_ari_min:.3f}',r.min_per_mouse])
    ax=fig.add_axes([.06,.21,.88,.24]);ax.axis('off');table=ax.table(cellText=body,
            colLabels=['Scale','Method','K','Sizes','ARI: nonwrap','ARI: circular','Min LOMO ARI','Min / mouse'],cellLoc='center',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('.8');cell.set_linewidth(.4)
        if r==0:cell.set_facecolor('#e5ecf3');cell.set_text_props(weight='bold')
    fig.text(.06,.14,'This sensitivity changes the eligible population; it is not an apples-to-apples improvement over all 92 windows.\n'
             'The 30-day threshold is a calendar eligibility rule, not an optimized clustering parameter. K=3 is still scale-sensitive.\n'
             'Overlapping windows and only three mice limit interpretation. Neither panel establishes discrete biological strategies.\n'
             'Gaps refer to sessions included in saved fits; some intervening raw recordings exist.',fontsize=11,va='top')
    return fig


def render():
    out=CANDIDATE_OUT;out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(OUT/'window_metrics_parameters.csv');v,m=compact_features(raw)
    calendar=pd.read_csv(AUDIT_OUT/'window_calendar_audit.csv');keep=~calendar.crosses_30day_gap.to_numpy()
    cohort=calendar.copy();cohort['candidate_eligible']=keep;cohort['reason']=np.where(keep,'within recording epoch','unclassified: historical window crosses >30-day gap')
    cohort.to_csv(out/'cohort_membership.csv',index=False)
    r=raw.loc[keep].reset_index(drop=True);vv=v.loc[keep].reset_index(drop=True)
    x,g,used=geometry(vv,m,r.subject.to_numpy(),'pooled_z');used['weight']=g['weights']
    order=raster_order(x);checks=[];metrics=pd.read_csv(AUDIT_OUT/'calendar_sensitivity_metrics.csv');label_frames=[]
    with PdfPages(out/'clusters.pdf') as pdf:
        for k in (2,3):
            labels,_=fit_method(x,k,'kmeans');check=metrics[metrics.scale.eq('pooled_z')&metrics.method.eq('kmeans')&metrics.K.eq(k)].iloc[0].to_dict()
            check.update(bootstrap_ari_median=check['nonwrapping_ari_median'],k=k)
            check['passes_diagnostics']=bool(check['mouse_gate'] and check['all_lomo_multigroup'] and check['silhouette']>=GATES['min_silhouette'] and
                                            check['nonwrapping_ari_median']>=GATES['min_bootstrap_ari'] and check['lomo_ari_min']>=GATES['min_lomo_ari'])
            fig,display=cluster_figure(r,vv,used,labels,order,check,display_z=x/g['weights'])
            checks.append(check)
            for text in fig.texts:
                if text.get_text().startswith('Requested K='):text.set_position((.11,.14))
            fig.suptitle(f'Candidate within source-session epochs — K={k}; shared feature scales',fontsize=18)
            fig.text(.11,.03,'69 eligible windows; 23 source-gap windows unclassified in this candidate. Original data and labels preserved.\n'
                     'Heatmap: within-mouse means removed, shared within-mouse SD; clipped at +/-3. Candidate remains exploratory.',fontsize=9)
            pdf.savefig(fig)
            if k==2:fig.savefig(out/'clusters.png',dpi=150)
            plt.close(fig)
            lab=r[['subject','window_id','test_rank','session_id']].copy();lab['K'],lab['cluster_label'],lab['display_rank']=k,labels,np.argsort(display)+1;label_frames.append(lab)
        for fig in [comparison_page(),calendar_page(raw,v)]:pdf.savefig(fig);plt.close(fig)
    labels=pd.concat(label_frames);labels.to_csv(out/'session_clusters.csv',index=False)
    r.to_csv(out/'window_metrics_parameters.csv',index=False);vv.to_csv(out/'feature_values.csv',index=False);used.to_csv(out/'clustering_features.csv',index=False)
    pd.DataFrame(x,columns=used.feature).to_csv(out/'clustering_matrix.csv',index=False)
    counts=[]
    for k in (2,3):
        d=labels[labels.K.eq(k)]
        c=pd.crosstab(d.subject,d.cluster_label).rename_axis(columns='cluster_label').stack().rename('n_sessions').reset_index()
        c.insert(0,'K',k);counts.append(c)
    pd.concat(counts).to_csv(out/'cluster_by_mouse.csv',index=False)
    pages=figures(labels,out,allow_subset=True)
    (out/'summary.json').write_text(json.dumps(dict(n_windows=69,n_unclassified=23,eligibility='No >30-day gap in source train/validation/test sessions',
                solutions=checks, algorithm='KMeans',scaling='Within-mouse mean; equal-mouse pooled within-mouse SD',seed=SEED,
                target_test_used_in_features=False,forced_fit_handling='No special weighting or exclusion',
                interpretation='K2 candidate only; K3 still fails mouse support and transfer. Calendar sensitivity changes the population.',
                behavior_pages=pages,full_metrics=str(AUDIT_OUT/'calendar_sensitivity_metrics.csv')),indent=2))
    for source,name in [('.codex-consult/claude_clustering_review.txt','claude_review_initial.md'),
                        ('.codex-consult/claude_clustering_followup_review.txt','claude_review_followup.md')]:
        (AUDIT_OUT/name).write_text('Claude reviewed supplied statistical briefs, without source-code access or independent data analysis.\n\n'+Path(source).read_text())
    print('Wrote two candidate PDFs:',out,flush=True)

if __name__=='__main__':render()
