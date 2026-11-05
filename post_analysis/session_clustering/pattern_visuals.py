"""Readable profiles of frozen clusters; does not estimate or alter any labels.

python -m post_analysis.session_clustering.pattern_visuals
"""
from pathlib import Path
import hashlib,json,shutil,textwrap
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,ListedColormap,LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from .pattern_exploration import OUT
from .pattern_report import CASES,NAMES
from .features import normalize
from .cluster import COLORS

VALUE_CMAP=LinearSegmentedColormap.from_list('readable_orange_blue',['#b85c28','#f0bc94','#fafafa','#9bc8db','#205d86'])

MOUSE_COLORS={'MC06':'#1b9e77','MC07':'#d95f02','MC08':'#7570b3'}
FAMILY_COLORS={'Sensory':'#7564a8','History':'#257b78','Block context':'#a57a29','Combined models':'#52758d'}
MODELS=[('m1_action_kernel','Action\nhistory','History'),
        ('m1_stimulus_kernel','Stimulus / WSLS\nhistory','History'),
        ('m1_hmm_true','Block: fixed\ncategory ratio','Block context'),
        ('m2_static','Block: learned\ncategory ratio','Block context'),
        ('m2_dynamic','Block: duration-\naware switches','Block context'),
        ('m2_dynamic_repetition_bias','Block +\nlast choice','Combined models'),
        ('m2_dynamic_wsls_bias','Block +\nWSLS','Combined models')]
BEHAVIOR=[('block_context_shift','Block-context\nchoice shift'),('rare_accuracy_cost','Common–rare\naccuracy gap'),
          ('adjusted_rapid_S2L_recovery','Short → long\nrecovery'),('adjusted_rapid_L2S_recovery','Long → short\nrecovery')]


def readouts(raw):
    """Physical-unit display fields; all original clustering inputs/labels stay fixed."""
    d=raw[['subject','session_id','window_id','test_rank']].copy();meta=[]
    def add(name,value,label,family,unit,definition):
        d[name]=value;meta.append(dict(feature=name,label=label,family=family,unit=unit,definition=definition))
    add('noise_short',np.sqrt(raw.param__m0_sensory__w*.8**2+raw.param__m0_sensory__s0**2),'Noise\nshort (s)','Sensory','seconds','M0 effective sensory SD at 0.8 s')
    add('noise_long',np.sqrt(raw.param__m0_sensory__w*1.7**2+raw.param__m0_sensory__s0**2),'Noise\nlong (s)','Sensory','seconds','M0 effective sensory SD at 1.7 s')
    add('memory',1/expit(raw.param__m1_action_kernel__a),'History\nmemory','History','valid updates','M1 action EMA mean lag = 1/alpha')
    gains={}
    for model,label,family in MODELS:
        if (raw.nll__m0_sensory<=0).any():raise ValueError('Positive reference loss required')
        gains[model]=100*(raw.nll__m0_sensory-raw['nll__'+model])/raw.nll__m0_sensory
    for model,label,family in MODELS[:2]:
        add('gain__'+model,gains[model],label.replace('history','benefit'),family,'% loss reduction','100*(M0 NLL - model NLL)/M0 NLL, per window')
    add('contrast',100*(expit(raw.param__m2_static__delta0)-.5),'Prior\ncontrast','Block context','percentage points','M2 static subjective P(long|long block) minus 0.5')
    add('hazard',100*expit(raw.param__m2_static__ah),'Switch\nchance','Block context','percent per update','M2 static biased-block switch hazard')
    for model,label,family in MODELS[2:]:
        add('gain__'+model,gains[model],{'m1_hmm_true':'Fixed-ratio\nbenefit','m2_static':'Learned-prior\nbenefit','m2_dynamic':'Duration-aware\nbenefit'}.get(model,label),family,'% loss reduction','100*(M0 NLL - model NLL)/M0 NLL, per window')
    return d,pd.DataFrame(meta)


def balanced(data,labels,columns,k):
    """Equal sessions within mouse, then all mice equally; never fill absent mice."""
    t=data.merge(labels[['subject','session_id','cluster_label']],on=['subject','session_id'],validate='one_to_one')
    mice=sorted(data.subject.unique());index=pd.MultiIndex.from_product([mice,range(1,k+1)],names=['subject','cluster_label'])
    by_mouse=t.groupby(['subject','cluster_label'])[columns].mean().reindex(index)
    arr=np.stack([by_mouse.xs(s).reindex(range(1,k+1)).to_numpy() for s in mice])
    return np.mean(arr,axis=0),by_mouse


def matrix_plot(ax,values,columns,rows,limit,fmt,missing='—'):
    values=np.asarray(values,float)
    im=ax.imshow(np.ma.masked_invalid(values),cmap=VALUE_CMAP,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect='auto')
    ax.set_xticks(np.arange(len(columns)),columns,fontsize=10);ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=12)
    ax.set_yticks(np.arange(len(rows)),rows,fontsize=11);ax.tick_params(axis='y',length=0,pad=12)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value=values[i,j];text=fmt(value) if np.isfinite(value) else missing
            ax.text(j,i,text,ha='center',va='center',fontsize=15,fontweight='bold',
                    color='white' if np.isfinite(value) and abs(value)>.66*limit else '#25313b')
    ax.set_xticks(np.arange(-.5,len(columns),1),minor=True);ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=3);ax.tick_params(which='minor',length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    return im


def names_for(variant,k):
    if k!=3:return {c:f'Group {c}' for c in range(1,k+1)}
    if variant=='performance_only':return {1:'History-heavy / lower accuracy',2:'Stronger block context',3:'Higher accuracy / weaker context'}
    return {1:'Stronger history / context',2:'Intermediate context',3:'Weaker block context'}


def summary_page(raw,d,meta,labels,observed,variant,k):
    fig=plt.figure(figsize=(17,13),facecolor='white')
    fig.suptitle(f'What distinguishes these {k} groups?',fontsize=23,fontweight='bold',x=.06,ha='left',y=.975)
    fig.text(.06,.935,f'{NAMES[variant]}  ·  All 92 sessions  ·  Existing cluster labels unchanged',fontsize=12,color='#52606d')
    counts=labels.cluster_label.value_counts().reindex(range(1,k+1));rows=[f'C{c} · {counts[c]} sessions' for c in range(1,k+1)]
    model_cols=['gain__'+m for m,_,_ in MODELS];means,_=balanced(d,labels,model_cols,k)
    fig.text(.06,.88,'1  Which models explain choices better?',fontsize=15,fontweight='bold')
    fig.text(.06,.857,'Blue = improvement over sensory-only; orange = worse. Numbers are percent reduction in prediction loss.',fontsize=10,color='#52606d')
    ax=fig.add_axes([.15,.645,.63,.15]);im=matrix_plot(ax,means,[label for _,label,_ in MODELS],rows,35,lambda x:f'{x:+.1f}%')
    cax=fig.add_axes([.795,.645,.012,.15]);cb=fig.colorbar(im,cax=cax,extend='both');cb.set_ticks([-35,0,35]);cb.ax.tick_params(labelsize=9)
    cb.set_label('Loss reduction (%)',fontsize=9)
    # Mouse counts and accuracy preserve the small-sample context without dense diagnostics.
    sidebar=fig.add_axes([.865,.65,.125,.16]);sidebar.axis('off');table=pd.crosstab(labels.cluster_label,labels.subject).reindex(range(1,k+1))
    acc,_=balanced(observed,labels,['balanced_accuracy'],k)
    body=[[str(int(table.loc[c,s])) for s in ('MC06','MC07','MC08')]+[f'{acc[c-1,0]*100:.1f}'] for c in range(1,k+1)]
    tab=sidebar.table(cellText=body,colLabels=['06','07','08','Acc.%'],rowLabels=[f'C{c}' for c in range(1,k+1)],cellLoc='center',bbox=[0,.10,1,.70])
    tab.auto_set_font_size(False);tab.set_fontsize(10)
    sidebar.text(.5,1.05,'Mouse counts\n& balanced accuracy',ha='center',va='bottom',fontsize=10,fontweight='bold')
    desc=names_for(variant,k)
    for c in range(1,k+1):sidebar.text(-.05,-.13-(c-1)*.15,f'C{c}  '+textwrap.fill(desc[c],width=26),color=COLORS[c-1],fontsize=9,ha='left')
    # Behavior uses common physical units, not another layer of arbitrary z scores.
    fig.text(.06,.575,'2  What do the mice actually do?',fontsize=15,fontweight='bold')
    fig.text(.06,.552,'Larger blue values = larger context or adaptation effects; numbers are percentage points.',fontsize=10,color='#52606d')
    behavior_cols=[m for m,_ in BEHAVIOR];bm,_=balanced(observed,labels,behavior_cols,k)
    ax=fig.add_axes([.15,.355,.63,.135]);im=matrix_plot(ax,bm*100,[n for _,n in BEHAVIOR],rows,30,lambda x:f'{x:.1f}')
    cax=fig.add_axes([.795,.355,.012,.135]);cb=fig.colorbar(im,cax=cax,extend='both');cb.set_ticks([-30,0,30]);cb.ax.tick_params(labelsize=9);cb.set_label('Effect size (pp)',fontsize=9)
    fig.text(.865,.48,'Read across a row\nto see one group.\n\nRead down a column\nto compare groups.',fontsize=11,color='#52606d',va='top')
    # Comparable physical-unit parameters in separate panels, not mixed into one color scale.
    fig.text(.06,.285,'3  A few fitted parameters in their own units',fontsize=15,fontweight='bold')
    columns=[('noise_short','Sensory noise · M0','SD at 0.8 s (s)'),('memory','History memory · M1 action','Mean lag (valid updates)'),
             ('contrast','Prior contrast · M2 static','Above neutral prior (pp)'),('hazard','Switch chance · M2 static','Biased-block hazard (%)')]
    for j,(col,title,xlabel) in enumerate(columns):
        ax=fig.add_axes([.075+j*.235,.105,.185,.125]);mean,by=balanced(d,labels,[col],k)
        for c in range(1,k+1):
            for n,s in enumerate(sorted(raw.subject.unique())):
                value=by.loc[(s,c),col]
                ax.plot(value,c+(n-1)*.12,marker=['o','^','s'][n],ms=5,color='#9ca3af',alpha=.85,zorder=2,linestyle='none')
            ax.plot(mean[c-1,0],c,'o',ms=9,color=COLORS[c-1],zorder=3)
        ax.set_yticks(range(1,k+1),[f'C{c}' for c in range(1,k+1)]);ax.set_ylim(k+.45,.55)
        ax.set_title(title,fontsize=10,fontweight='bold');ax.set_xlabel(xlabel,fontsize=9);ax.tick_params(labelsize=9)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',color='#e5e7eb',lw=.6)
    fig.text(.06,.055,'Colored dots = equal-mouse means. Gray circles = MC06; triangles = MC07; squares = MC08. Parameters summarize training-window fits.',fontsize=9,color='#52606d')
    fig.text(.06,.035,'Other panels describe associated test sessions. Prediction loss = mean NLL; model gains are relative to sensory-only.',fontsize=9,color='#52606d')
    fig.text(.06,.015,'Context compares matched stimulus bins; recovery compares newly common trials 3–4 with 0–1 after adjusted transitions.',fontsize=9,color='#52606d')
    return fig


def family_sections(meta):
    sections=[];start=0
    for family in meta.family.unique():
        n=int(meta.family.eq(family).sum());sections.append((family,start,n));start+=n
    return sections


def session_page(d,meta,labels,order,variant,k):
    # Coherent readout families; the same source session appears once and labels never change.
    order=np.asarray(order,int);order=order[np.argsort(labels.cluster_label.to_numpy()[order],kind='stable')]
    z=normalize(d[meta.feature],d.subject.to_numpy())
    fig=plt.figure(figsize=(17,12),facecolor='white');fig.suptitle('Where are those patterns in individual sessions?',fontsize=21,fontweight='bold',x=.06,ha='left',y=.97)
    fig.text(.06,.93,f'{NAMES[variant]} · K={k} · Selected explanatory readouts; the clustering calculation is unchanged.',fontsize=11,color='#52606d')
    # Use generous gaps between feature families, horizontal readable labels and one direction legend.
    left=.20;right=.90;gap=.018;total=right-left-3*gap
    sections=family_sections(meta);axes=[]
    for family,start,n in sections:
        width=total*n/len(meta);ax=fig.add_axes([left,.19,width,.64]);axes.append(ax)
        ax.imshow(z[order,start:start+n],aspect='auto',cmap=VALUE_CMAP,vmin=-2.5,vmax=2.5,interpolation='nearest')
        for border in np.flatnonzero(np.diff(labels.cluster_label.to_numpy()[order]))+.5:ax.axhline(border,color='#28313b',lw=2)
        ax.set_xticks(np.arange(n),meta.label.iloc[start:start+n],fontsize=9);ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=12)
        ax.set_yticks([])
        # Header bands have category colors; heatmap colors always encode relative magnitude.
        fig.patches.append(Rectangle((left,.892),width,.022,transform=fig.transFigure,facecolor=FAMILY_COLORS[family],clip_on=False))
        fig.text(left+width/2,.903,family,ha='center',va='center',color='white',fontsize=11,fontweight='bold')
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_xticks(np.arange(-.5,n,1),minor=True);ax.grid(which='minor',axis='x',color='white',lw=.9);ax.tick_params(which='minor',length=0)
        left+=width+gap
    lab=labels.cluster_label.to_numpy();subjects=sorted(d.subject.unique());mcolors=[MOUSE_COLORS[s] for s in subjects]
    ax=fig.add_axes([.16,.19,.014,.64]);ax.imshow(np.array([subjects.index(s) for s in d.subject.iloc[order]])[:,None],aspect='auto',cmap=ListedColormap(mcolors),vmin=0,vmax=2);ax.set_xticks([]);ax.set_yticks([]);ax.set_title('Mouse',fontsize=9,pad=12)
    # Replace row-number clutter by explicit cluster brackets and session counts.
    for c in range(1,k+1):
        positions=np.flatnonzero(lab[order]==c);top=positions[0];bottom=positions[-1]+1
        yc=.19+.64*(1-(top+bottom)/(2*len(order)))
        fig.text(.055,yc,f'C{c}\n{len(positions)} sessions',ha='left',va='center',fontsize=12,fontweight='bold',color=COLORS[c-1])
        y0=.19+.64*(1-bottom/len(order));height=.64*len(positions)/len(order)
        fig.patches.append(Rectangle((.135,y0),.007,height,transform=fig.transFigure,facecolor=COLORS[c-1],clip_on=False))
    cax=fig.add_axes([.935,.37,.014,.30]);im=axes[0].images[0];cb=fig.colorbar(im,cax=cax,extend='both');cb.set_ticks([-2.5,0,2.5]);cb.ax.set_yticklabels(['Lower','Usual','Higher']);cb.ax.tick_params(labelsize=10)
    fig.text(.955,.71,'Within-mouse\nrelative value',fontsize=10,ha='center')
    for i,s in enumerate(subjects):fig.text(.20+i*.15,.12,'■ '+s,color=MOUSE_COLORS[s],fontsize=11)
    fig.text(.06,.075,'Blue = above that mouse’s usual value; orange = below. Higher noise means more sensory uncertainty; higher model benefit means better predictions.',fontsize=10,color='#52606d')
    fig.text(.06,.05,'The first columns use M0 sensory, M1 action-history and M2 static-prior fits as fixed references. The remaining columns show model benefits versus sensory-only.',fontsize=9,color='#52606d')
    fig.text(.06,.03,'Rows keep the previous within-cluster session order. These 12 columns summarize the groups; they were not substituted for each variant’s full clustering feature set.',fontsize=9,color='#52606d')
    return fig,order


def run(refresh_archive=False):
    labels_path=OUT/'selected_session_clusters.csv';before=hashlib.sha256(labels_path.read_bytes()).hexdigest()
    raw=pd.read_csv(OUT/'window_metrics_parameters.csv');all_labels=pd.read_csv(OUT/'all_candidate_labels.csv');observed=pd.read_csv(OUT/'observed_session_metrics.csv')
    d,meta=readouts(raw);d.to_csv(OUT/'visual_readouts.csv',index=False);meta.to_csv(OUT/'visual_readout_manifest.csv',index=False)
    # Retain the existing full parameter display as a final appendix page.
    previous=OUT/'clusters.pdf';archive=OUT/'clusters_dense_archive.pdf'
    if refresh_archive or not archive.exists():shutil.copyfile(previous,archive)
    temporary=OUT/'clusters_readable.tmp.pdf';pages=[];summary_rows=[];orders=[]
    old_order=pd.read_csv(OUT/'visual_session_order.csv') if (OUT/'visual_session_order.csv').exists() else None
    # Reconstruct the original display from its saved inputs, without rereading fits.
    from .features import matrix,transform_parameter
    from .cluster import raster_order
    from .pattern_exploration import candidates
    from .pattern_report import cluster_page
    original_meta=pd.read_csv(OUT/'feature_manifest.csv')
    original_meta=original_meta[original_meta.variant.eq('reference_all_models')].copy()
    values=pd.DataFrame(index=raw.index)
    for row in original_meta.itertuples():
        if row.block=='performance':
            values[row.feature]=np.log(2)-raw.nll__m0_sensory if row.model=='m0_sensory' else raw.nll__m0_sensory-raw['nll__'+row.model]
        else:values[row.feature]=transform_parameter(row.description,raw[row.feature])
    options=candidates(raw,values,original_meta)
    with PdfPages(temporary) as pdf:
        for variant,k in (('reference_all_models',3),*CASES):
            ll=all_labels[all_labels.variant.eq(variant)&all_labels.K.eq(k)].copy()
            ll=raw[['subject','session_id']].merge(ll,on=['subject','session_id'],validate='one_to_one')
            fig=summary_page(raw,d,meta,ll,observed,variant,k);pdf.savefig(fig)
            if variant=='rbf_reference' and k==3:
                fig.savefig(OUT/'clusters_K3.png',dpi=140)
            if variant=='reference_all_models' and k==3:
                fig.savefig(OUT/'clusters_original_K3.png',dpi=140)
            plt.close(fig);pages.append(dict(page=len(pages)+1,variant=variant,K=k,analysis='readable_summary'))
            if old_order is not None and (old_order.variant.eq(variant)&old_order.K.eq(k)).any():
                oo=old_order[old_order.variant.eq(variant)&old_order.K.eq(k)].sort_values('display_rank')
                keys=pd.MultiIndex.from_frame(raw[['subject','session_id']]);order=keys.get_indexer(pd.MultiIndex.from_frame(oo[['subject','session_id']]))
            else:
                v,m,_=options[variant];x,_=matrix(v,m,raw.subject.to_numpy());order=raster_order(x)
            if len(order)!=len(raw) or (np.asarray(order)<0).any() or len(np.unique(order))!=len(raw):
                raise ValueError('Display order does not match all source sessions')
            fig,order=session_page(d,meta,ll,order,variant,k);pdf.savefig(fig);plt.close(fig)
            pages.append(dict(page=len(pages)+1,variant=variant,K=k,analysis='readable_sessions'))
            o=raw[['subject','session_id']].copy();o['variant'],o['K'],o['display_rank']=variant,k,np.argsort(order)+1;orders.append(o)
            cols=meta.feature.tolist();means,_=balanced(d,ll,cols,k)
            for c in range(1,k+1):summary_rows.append(dict(variant=variant,K=k,cluster_label=c,**dict(zip(cols,means[c-1]))))
        primary=all_labels[all_labels.variant.eq('rbf_reference')&all_labels.K.eq(3)].copy()
        primary=raw[['subject','session_id']].merge(primary,on=['subject','session_id'],validate='one_to_one')
        appendix_order=np.argsort(next(o for o in orders if o.variant.iloc[0]=='rbf_reference' and int(o.K.iloc[0])==3).display_rank.to_numpy())
        fig=cluster_page(raw,values,original_meta,primary.cluster_label.to_numpy(),appendix_order,
                         'rbf_reference',3,pd.read_csv(OUT/'behavior_profiles_equal_mouse.csv'))
        fig.axes[2].images[0].set_cmap(VALUE_CMAP)
        fig.suptitle('Appendix: all 48 original driver columns — same frozen clusters',fontsize=18)
        fig.text(.09,.93,'Blue = higher within-mouse value; orange = lower. The overview pages use selected readable summaries.',fontsize=11)
        pdf.savefig(fig);plt.close(fig)
        pages.append(dict(page=len(pages)+1,variant='rbf_reference',K=3,analysis='full_48_feature_appendix'))
    temporary.replace(previous)
    pd.DataFrame(pages).to_csv(OUT/'cluster_pages.csv',index=False)
    pd.concat(orders).to_csv(OUT/'visual_session_order.csv',index=False)
    pd.DataFrame(summary_rows).to_csv(OUT/'visual_cluster_readouts.csv',index=False)
    after=hashlib.sha256(labels_path.read_bytes()).hexdigest()
    if before!=after:raise RuntimeError('Visualization changed labels')
    (OUT/'visualization_audit.json').write_text(json.dumps(dict(labels_sha256=after,labels_unchanged=True,n_sessions=len(raw),
                summary='Actual-unit equal-mouse means; model colors are percent reduction in NLL; behavior colors are pp',
                sessions='Selected readouts normalized within mouse for display only; original all-model feature set unchanged',
                archive=str(archive),pages=len(pages)),indent=2))
    print('Updated clusters.pdf:',len(pages),'pages. Frozen labels unchanged.',flush=True)

if __name__=='__main__':run()
