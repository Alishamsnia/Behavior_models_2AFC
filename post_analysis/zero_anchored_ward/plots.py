"""Fresh report figures for four-coordinate Ward clustering."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, leaves_list
import numpy as np
import pandas as pd

from .core import FEATURES, LABELS, FORMULAS, MODELS, KEYS
from . import behavior as bh

COLORS = ['#4477AA','#EE6677','#228833','#AA3377','#CCBB44','#66CCEE','#EE7733','#777777']
MOUSE_MARKERS = {'MC06':'o','MC07':'s','MC08':'^'}
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                     'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def shell(title, rows=1, cols=1, size=None, note=''):
    fig,axes=plt.subplots(rows,cols,figsize=size or (6*cols,4*rows+1.4),squeeze=False)
    # Reserve physical space for tick labels, axis labels and two-line footnotes.
    bottom=max(.12,1.1/fig.get_size_inches()[1])
    fig.subplots_adjust(left=.075,right=.97,top=.86,bottom=bottom,hspace=.50,wspace=.35)
    fig.suptitle(title,x=.055,y=.975,ha='left',fontsize=17,weight='bold')
    if note:
        fig.text(.055,.92,note,fontsize=9,color='.3',va='top')
    return fig,axes


def footer(fig,text):
    fig.text(.055,.025,text,fontsize=8,color='.3',va='bottom')


def save(fig,path,pdf=None,svg=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.png'),dpi=140,facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),facecolor='white')
    if svg:
        fig.savefig(path.with_suffix('.svg'),facecolor='white')
    if pdf is not None:
        pdf.savefig(fig,facecolor='white')
    plt.close(fig)


def means_equal_mouse(frame,columns):
    return frame.groupby(['cluster_label','subject'])[columns].mean().groupby('cluster_label').mean()


def matrix(ax,arr,xlabels,ylabels,title,vmin=None,vmax=None,cmap='viridis',fmt='.2f',annotate=False):
    a=np.asarray(arr,float)
    im=ax.imshow(a,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(xlabels)),xlabels,rotation=20,ha='right',fontsize=9)
    ax.set_yticks(range(len(ylabels)),ylabels)
    ax.set_title(title,loc='left',fontsize=11)
    if a.size <= 100 or annotate:
        for (i,j),v in np.ndenumerate(a):
            ax.text(j,i,format(v,fmt) if np.isfinite(v) else 'NA',ha='center',va='center',fontsize=6 if a.size>100 else 9,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.6,pad=1))
    return im


def k_figure(selection,tree,raw,geom,labels,optimal):
    fig,ax=shell(f'Ward: descriptive cluster-count diagnostics (selected K={optimal})',2,3,size=(18,11),
                note='K=2..8; selected by maximum Euclidean silhouette, smaller K on ties. A candidate partition is not evidence of discrete states.')
    s=selection
    ax[0,0].plot(s.K,s.silhouette,'o-');ax[0,0].set_ylabel('Euclidean silhouette');ax[0,0].set_title('Separation of normalized directions')
    ax[0,1].plot(s.K,s.stability_ari_median,'o-');ax[0,1].fill_between(s.K,s.stability_ari_q10,s.stability_ari_q90,alpha=.2)
    ax[0,1].set_ylabel('ARI');ax[0,1].set_title('80% subsets: median and 10-90% range')
    ax[0,2].plot(s.K,s.calinski_harabasz,'o-');ax[0,2].set_title('Calinski-Harabasz');ax[0,2].set_ylabel('Score')
    ax[1,0].plot(s.K,s.davies_bouldin,'o-');ax[1,0].set_title('Davies-Bouldin (lower is better)')
    sizes=[np.bincount(labels[k])[1:].min() for k in s.K]
    ax[1,1].plot(s.K,sizes,'o-');ax[1,1].set_title('Smallest cluster');ax[1,1].set_ylabel('Sessions')
    dendrogram(tree,ax=ax[1,2],no_labels=True,color_threshold=0,above_threshold_color='.35')
    ax[1,2].set_title('Euclidean Ward dendrogram');ax[1,2].set_ylabel('Ward merge distance')
    for a in ax.flat:
        if a is not ax[1,2]:
            a.set_xlabel('K');a.set_xticks(s.K);a.axvline(optimal,color='.6',ls=':',lw=1)
    footer(fig,'Subsets refit column scales and Ward. These perturbations assess partition stability, not uncertainty in the estimated gains.\nMouse composition is audited separately; no requirement forces all mice into every group.')
    return fig


def normalization_figure(raw,gains,geom):
    fig,axes=shell('Zero-anchored normalization: input and geometry audit',2,3,size=(17,10),
                   note='Coordinates: sensory | action history | WSLS history | dynamic context. Raw signed gains remain in the exported tables.')
    for k,label in enumerate(LABELS):
        axes[0,0].hist(gains[:,k],bins=20,alpha=.4,label=label)
        axes[0,1].hist(geom.positive[:,k],bins=20,alpha=.4,label=label)
    axes[0,0].set_title('Signed gains');axes[0,1].set_title('After max(g, 0)')
    axes[0,0].set_xlabel('nats/trial');axes[0,1].set_xlabel('nats/trial')
    axes[0,0].legend(fontsize=8);axes[0,0].axvline(0,color='black',lw=.7)
    axes[0,2].bar(range(4),geom.scale);axes[0,2].set_xticks(range(4),LABELS,rotation=25,ha='right')
    axes[0,2].set_title('Global sample SD (ddof=1)');axes[0,2].set_ylabel('nats/trial')
    for mouse,ix in raw.groupby('subject').groups.items():
        axes[1,0].scatter(raw.loc[ix,'test_rank'],geom.norm[ix],s=22,label=mouse)
    axes[1,0].set_title('Magnitude retained for interpretation');axes[1,0].set_xlabel('Held-out session rank');axes[1,0].set_ylabel('Pre-L2 norm');axes[1,0].legend()
    axes[1,1].hist(np.linalg.norm(geom.z,axis=1),bins=10,range=(0,1.05));axes[1,1].set_title('Final vector length');axes[1,1].set_xlabel('L2 norm')
    axes[1,2].scatter(geom.z[:,0],geom.z[:,1],c=geom.norm,cmap='viridis',s=25)
    axes[1,2].set_xlabel('Sensory direction');axes[1,2].set_ylabel('Action-history direction');axes[1,2].set_title('Example projection; color = pre-L2 norm')
    footer(fig,'No centering or +/-3 clipping. Row L2 discards magnitude for clustering, while magnitudes remain in the audit.\nZero rows have no direction and are excluded; small norms are not claimed to be reliable merely because their final length is near one.')
    return fig


def fingerprint(raw,gains,geom,ll,observed,k):
    active=ll.cluster_label.gt(0).to_numpy(); lab=ll.cluster_label.to_numpy()
    directions=pd.DataFrame(geom.z,columns=FEATURES).assign(subject=raw.subject,cluster_label=lab).loc[active]
    obs=observed.merge(ll[KEYS+['cluster_label']],on=KEYS,validate='one_to_one');obs=obs[obs.cluster_label.gt(0)]
    columns=['balanced_accuracy','choose_long','common_accuracy','rare_accuracy']
    a=means_equal_mouse(directions,list(FEATURES)).reindex(range(1,k+1))
    b=means_equal_mouse(obs,columns).reindex(range(1,k+1))
    fig,axes=shell(f'K={k}: cluster fingerprints',2,3,size=(18,11),
                   note='Cells average session values within each mouse, then average available mice equally. Centroids are not renormalized.')
    matrix(axes[0,0],a,LABELS,[f'C{c}' for c in a.index],'Normalized gain directions',0,1)
    matrix(axes[0,1],b,['Balanced accuracy','P(long)','Common accuracy','Rare accuracy'],[f'C{c}' for c in b.index],'Observed behavior (descriptive)',0,1)
    counts=pd.crosstab(ll.cluster_label,ll.subject).reindex(range(1,k+1),fill_value=0)
    matrix(axes[0,2],counts,counts.columns,[f'C{c}' for c in counts.index],'Mouse/session support',0,None,'Blues','.0f')
    signed=pd.DataFrame(gains,columns=FEATURES).assign(subject=raw.subject,cluster_label=lab).loc[active]
    raw_means=means_equal_mouse(signed,list(FEATURES)).reindex(range(1,k+1))
    lim=max(float(np.nanmax(np.abs(raw_means))),1e-12)
    matrix(axes[1,0],raw_means,LABELS,[f'C{c}' for c in raw_means.index],
           'Original signed gains (nats/trial)',-lim,lim,'RdBu_r','.3f')
    effects=['block_context_shift','rare_accuracy_cost','adjusted_rapid_S2L_recovery','adjusted_rapid_L2S_recovery']
    em=means_equal_mouse(obs,effects).reindex(range(1,k+1))*100
    matrix(axes[1,1],em,['Context shift','Common-rare gap','S2L recovery','L2S recovery'],
           [f'C{c}' for c in em.index],'Observed effects (percentage points)',-30,30,'RdBu_r','.1f')
    norms=pd.DataFrame(dict(subject=raw.subject,cluster_label=lab,norm=geom.norm)).loc[active]
    nm=means_equal_mouse(norms,['norm']).reindex(range(1,k+1))
    axes[1,2].bar(range(1,k+1),nm.norm,color=COLORS[:k])
    axes[1,2].set_xticks(range(1,k+1),[f'C{c}' for c in range(1,k+1)])
    axes[1,2].set_title('Pre-L2 magnitude (equal-mouse mean)');axes[1,2].set_ylabel('Norm of column-scaled positive gains')
    footer(fig,'Groups describe relative scaled predictive gains. They are not estimates of strategy-use probabilities.\nBehavior is measured in the same held-out sessions and is not independent validation.')
    # Export all observed metrics rather than only those selected for the display.
    all_observed=means_equal_mouse(obs,observed.select_dtypes('number').columns.tolist())
    return fig,a.reset_index().assign(K=k),all_observed.reset_index().assign(K=k)


def methods(k):
    fig=plt.figure(figsize=(16,11));fig.suptitle(f'K={k}: four coordinates, zero anchoring and Ward',x=.055,y=.965,ha='left',fontsize=19,weight='bold')
    sections=[
        ('1  Model comparisons and held-out loss',
         'L(M) = mean negative log-likelihood on the same evaluated held-out test trials (nats/trial).\n'
         'const: Bernoulli choice probability fitted only on evaluated training choices, clipped to [1e-12, 1-1e-12].\n'
         + '\n'.join(f'{name}: {formula}' for name,formula in zip(LABELS,FORMULAS))),
        ('2  What the four fitted models represent',
         'S: sensory Bayesian choice model, fixed prior 0.5, with fitted sensory noise, decision bias and lapses.\n'
         'S+AH: exponential prior update toward past choices; alpha = sigmoid(a).\n'
         'S+WH: exponential prior update toward past rewarded sides (stay after reward, shift after error).\n'
         'History update: prior(t+1) = prior(t) + alpha * [history_target(t) - prior(t)].\n'
         'S+Bdynamic: latent block prior with run-length-dependent switching hazard\n'
         'h_B(r) = sigmoid(ah + bh * (r - 23)); block probabilities use past feedback.'),
        ('3  Normalization used for distances',
         'g_positive = max(g, 0)\n'
         'u(i,k) = g_positive(i,k) / SD_global(g_positive(:,k)), sample SD with ddof=1; no centering.\n'
         'z(i,:) = u(i,:) / sqrt(sum(u(i,:)^2) + epsilon^2), epsilon = 1e-12.\n'
         'All-zero columns stay zero; nonzero constant columns fail for review. All-zero sessions are unresolved (label 0).'),
        ('4  Clustering and interpretation',
         'Euclidean Ward on z directly. No RBF affinity, spectral embedding or cosine-distance input.\n'
         'Merge cost = n_A*n_B/(n_A+n_B) * squared Euclidean distance between cluster centroids.\n'
         'K=2..8 inspected; descriptive selected K maximizes silhouette. K=2,3,4 reports always retained when feasible.\n'
         'Cluster numbers: decreasing size, first session breaks ties; no historical label alignment.\n'
         'Negative held-out gains are retained in raw exports; clipping defines retained positive gain, not true strategy absence.')]
    ys=[.88,.65,.415,.20]
    for (title,body),y in zip(sections,ys):
        fig.text(.055,y,title,fontsize=13,weight='bold',va='top')
        fig.text(.065,y-.035,body,fontsize=10.5,linespacing=1.6,va='top')
    return fig


def gain_details(raw,gains,ll,k):
    fig,axes=shell(f'K={k}: signed predictive gains in original units',1,4,size=(18,6.7),
                   note='Each point is one held-out session; markers identify mice. Black horizontal segments are pooled cluster medians.')
    rng=np.random.default_rng(11)
    for j,ax in enumerate(axes[0]):
        for c in range(1,k+1):
            for mouse,marker in MOUSE_MARKERS.items():
                ix=np.flatnonzero(ll.cluster_label.eq(c)&raw.subject.eq(mouse))
                ax.scatter(c+rng.uniform(-.15,.15,len(ix)),gains[ix,j],marker=marker,color=COLORS[c-1],s=25,alpha=.8,
                           label=mouse if c==1 else None)
            vals=gains[ll.cluster_label.eq(c),j]
            ax.plot([c-.2,c+.2],[np.median(vals)]*2,color='black',lw=2)
        ax.axhline(0,color='.4',ls=':');ax.set_xticks(range(1,k+1),[f'C{c}' for c in range(1,k+1)])
        ax.set_title(LABELS[j]);ax.set_ylabel('Signed gain (nats/trial)')
    axes[0,0].legend(fontsize=8)
    footer(fig,'Negative values mean worse held-out prediction. These signed values are shown for interpretation; Ward receives max(g,0), global SD scaling and row L2.')
    return fig


def session_heatmap(raw,gains,geom,ll,tree,k):
    leaf=leaves_list(tree);active=np.flatnonzero(geom.eligible);order=active[leaf]
    # Keep clusters contiguous while retaining dendrogram order within each group.
    order=np.array(sorted(order,key=lambda i:ll.cluster_label.iloc[i]),int)
    fig,axes=shell(f'K={k}: every session, signed gains and normalized coordinates',1,3,size=(17,15),
                   note='Rows follow cluster number and Ward leaf order within cluster. Zero means no retained positive gain in directional coordinates.')
    limit=max(float(np.max(np.abs(gains))),1e-12)
    im=matrix(axes[0,0],gains[order],LABELS,[],'Original signed gains (nats/trial)',-limit,limit,'RdBu_r')
    fig.colorbar(im,ax=axes[0,0],shrink=.4)
    im=matrix(axes[0,1],geom.z[order],LABELS,[],'Actual clustering coordinates',0,1)
    fig.colorbar(im,ax=axes[0,1],shrink=.4)
    annotation=np.column_stack([ll.cluster_label.iloc[order],raw.subject.iloc[order].map({'MC06':1,'MC07':2,'MC08':3}),raw.test_rank.iloc[order]])
    matrix(axes[0,2],annotation,['Cluster','Mouse code','Session rank'],[],'Mouse: 1=MC06, 2=MC07, 3=MC08',fmt='.0f',annotate=True)
    for ax in axes[0]:
        ax.set_yticks(range(len(order)),[f'{raw.subject.iloc[i]} / {raw.test_rank.iloc[i]}' for i in order],fontsize=6)
        lab=ll.cluster_label.iloc[order].to_numpy()
        for y in np.flatnonzero(lab[1:]!=lab[:-1])+.5:
            ax.axhline(y,color='black',lw=1)
    footer(fig,'Raw and normalized panels use different explicit scales. Magnitude is discarded only in the directional panel; no z-scoring is applied for display.')
    return fig,raw.iloc[order][KEYS+['test_rank']].assign(K=k,cluster_label=ll.cluster_label.iloc[order].to_numpy(),row=np.arange(len(order)))


def plot_psych(ax,g,k,title,records):
    for c in range(1,k+1):
        t=g[g.cluster_label.eq(c)]
        b=bh.binned(t,'choice_long')
        if b.empty: continue
        ax.errorbar(b.duration_s,b['mean'],yerr=b['sem'],fmt='o',color=COLORS[c-1],capsize=2,label=f'C{c} (n={len(t)})',ms=4)
        p=bh.psychometric(t)
        if p is not None:
            x=np.linspace(.15,2.35,200);ax.plot(x,bh.curve(x,*p),color=COLORS[c-1])
        records.append(b.assign(cluster_label=c,panel=title))
    ax.set_title(title);ax.set_ylim(-.03,1.03);ax.axhline(.5,color='.6',ls=':',lw=.8);ax.axvline(1.25,color='.6',ls=':',lw=.8)
    ax.set_xlabel('Duration (s)');ax.set_ylabel('P(choose long)');ax.legend(fontsize=7,frameon=False)


def plot_event(ax,events,k,title,value='correct',transition=True):
    rows=[]
    for c in range(1,k+1):
        g=events[events.cluster_label.eq(c)].dropna(subset=[value])
        if transition: g=bh.bridge_events(g,value)
        if g.empty: continue
        s=g.groupby('rel')[value].agg(['mean','sem','size']).reset_index()
        xx=np.arange(-20,20) if transition else np.arange(-2,3)
        ss=s.set_index('rel').reindex(xx)
        ax.plot(xx,ss['mean'],color=COLORS[c-1],label=f'C{c}',lw=1.5)
        ax.fill_between(xx,ss['mean']-ss['sem'],ss['mean']+ss['sem'],color=COLORS[c-1],alpha=.15)
        if g.bridge.any() and -1 in s.rel.values:
            ax.scatter([-1],[s.set_index('rel').loc[-1,'mean']],facecolors='white',edgecolors=COLORS[c-1],s=25,zorder=5)
        rows.append(s.assign(cluster_label=c,panel=title))
    ax.axvline(0,color='.4',ls='--',lw=.8);ax.set_title(title,fontsize=10)
    ax.set_xlabel('Trial relative to adjusted switch' if transition else 'Trial relative to rare stimulus')
    ax.set_ylabel('P(correct)' if value=='correct' else 'Reaction time (s)')
    ax.legend(frameon=False,fontsize=8)
    return rows


def choice_page(tagged,k,trans,rare):
    fig=plt.figure(figsize=(20,22));gs=fig.add_gridspec(5,12,left=.06,right=.98,top=.91,bottom=.07,hspace=.55,wspace=1.3)
    fig.suptitle(f'K={k}: observed choices, block transitions and rare stimuli',x=.055,ha='left',y=.98,fontsize=19,weight='bold')
    fig.text(.055,.95,'Psychometric points: four rank bins per category; fitted four-parameter logistic curves. Errors: pooled trial/event SEM.',fontsize=10)
    valid=tagged[tagged.choice_eligible];records=[]
    for j,(name,g) in enumerate([('All blocks',valid),('Short block',valid[valid.true_block.eq('S')]),('Long block',valid[valid.true_block.eq('L')])]):
        plot_psych(fig.add_subplot(gs[0,j*4:(j+1)*4]),g,k,name,records)
    for c in range(1,k+1):
        ax=fig.add_subplot(gs[1,(c-1)*12//k:c*12//k]);g=valid[valid.cluster_label.eq(c)]
        for block,ls in [('S','--'),('L','-')]:
            t=g[g.true_block.eq(block)];b=bh.binned(t,'choice_long')
            if b.empty: continue
            ax.errorbar(b.duration_s,b['mean'],yerr=b['sem'],fmt='o',color=COLORS[c-1],mfc='white' if block=='S' else COLORS[c-1],ms=4)
            p=bh.psychometric(t)
            if p is not None:
                x=np.linspace(.15,2.35,200);ax.plot(x,bh.curve(x,*p),color=COLORS[c-1],ls=ls,label=block)
            records.append(b.assign(cluster_label=c,panel='Within cluster block '+block))
        ax.axhline(.5,color='.6',ls=':');ax.axvline(1.25,color='.6',ls=':');ax.set_ylim(-.03,1.03)
        ax.set_title(f'C{c}: short vs long block');ax.set_xlabel('Duration (s)');ax.set_ylabel('P(long)');ax.legend(fontsize=8)
    for j,(kind,cat) in enumerate([('S2L',0),('S2L',1),('L2S',0),('L2S',1)]):
        ax=fig.add_subplot(gs[2+j//2,(j%2)*6:(j%2+1)*6])
        records.extend(plot_event(ax,trans[trans.kind.eq(kind)&trans.category_long.eq(cat)],k,f'{kind} / category {cat}'))
    for j,kind in enumerate(['rare_long','rare_short']):
        ax=fig.add_subplot(gs[4,j*6:(j+1)*6]);records.extend(plot_event(ax,rare[rare.kind.eq(kind)],k,kind,transition=False))
    footer(fig,'Observed held-out sessions; descriptive pooled summaries. Histories/events are built before filtering invalid responses.\nOpen switch markers at -1 copy the previous eligible matching-category trial; event windows overlap.')
    return fig,pd.concat(records,ignore_index=True)


def reaction_page(tagged,k):
    fig,axes=shell(f'K={k}: reaction time by stimulus duration',4,4,size=(21,20),
                   note='Mean +/- SEM across within-session means. Four rank bins within each category per cluster/condition; y scales vary by panel.')
    records=[]
    for ax,(name,mask) in zip(axes.flat,bh.rt_masks(tagged).items()):
        for c in range(1,k+1):
            t=tagged[mask&tagged.cluster_label.eq(c)&tagged.latency_eligible]
            b=bh.binned(t,'latency_s',session_weighted=True)
            if b.empty:continue
            ax.errorbar(b.duration_s,b['mean'],yerr=b['sem'],color=COLORS[c-1],fmt='o-',ms=3,capsize=2,label=f'C{c}')
            records.append(b.assign(K=k,cluster_label=c,condition=name))
        ax.set_title(name);ax.axvline(1.25,color='.6',ls=':',lw=.8);ax.set_xlabel('Duration (s)');ax.set_ylabel('RT (s)')
    axes[0,0].legend(fontsize=8)
    footer(fig,'Saved first lick at/after choice-window onset. Warm-up, negative all-lick times and invalid/negative scalar latency excluded; no trimming.\nNeutral/block/common/rare labels refer to the current trial. No raw observations enter the clustering.')
    return fig,pd.concat(records,ignore_index=True)


def rt_extension(tagged,k,out,tables,report):
    """K3 pooled-trial SD/SEM and Welch comparisons; dependence explicitly stated."""
    from itertools import combinations
    from scipy.stats import ttest_ind
    rows,tests,duration,bin_tests=[],[],[],[]
    eligible=tagged[tagged.latency_eligible]
    def pairs(g,metadata):
        result=[]
        for a,b in combinations(range(1,k+1),2):
            x=g.loc[g.cluster_label.eq(a),'latency_s'];y=g.loc[g.cluster_label.eq(b),'latency_s']
            test=ttest_ind(x,y,equal_var=False) if min(len(x),len(y))>=2 else None
            result.append(dict(**metadata,cluster_a=a,cluster_b=b,n_a=len(x),n_b=len(y),delta_s=x.mean()-y.mean(),
                               p_raw=test.pvalue if test else np.nan,t=test.statistic if test else np.nan))
        return result
    def holm(records):
        frame=pd.DataFrame(records);frame['p_holm']=np.nan
        good=frame.p_raw.dropna().sort_values();m=len(good)
        if m:frame.loc[good.index,'p_holm']=np.minimum(1,np.maximum.accumulate(good.to_numpy()*(m-np.arange(m))))
        return frame
    for name,mask in bh.rt_masks(eligible).items():
        g=eligible[mask]
        for c,x in g.groupby('cluster_label'):
            rows.append(dict(condition=name,cluster_label=c,mean=x.latency_s.mean(),sd=x.latency_s.std(),sem=x.latency_s.sem(),n_trials=len(x)))
        tests.extend(pairs(g,dict(condition=name)))
        b=bh.duration_bins(g,shared_edges=True)
        for bin_id,h in b.groupby('duration_bin'):
            bin_tests.extend(pairs(h,dict(condition=name,duration_bin=bin_id)))
            for c,x in h.groupby('cluster_label'):
                duration.append(dict(condition=name,duration_bin=bin_id,cluster_label=c,duration_s=x.duration_s.mean(),
                                     mean=x.latency_s.mean(),sem=x.latency_s.sem(),n_trials=len(x)))
    summary=pd.DataFrame(rows);dur=pd.DataFrame(duration)
    summary.to_csv(tables/'K3_rt_condition_summary.csv',index=False)
    holm(tests).to_csv(tables/'K3_rt_pairwise_tests.csv',index=False)
    holm(bin_tests).to_csv(tables/'K3_rt_binwise_pairwise_tests.csv',index=False)
    dur.to_csv(tables/'K3_rt_duration_binned_extended.csv',index=False)
    for error in ['sem','sd']:
        fig,axes=shell(f'K={k}: pooled-trial reaction time, mean +/- {error.upper()}',4,4,size=(19,17),
                       note='Pooled individual trials, matching the reference supplement. This figure uses different weighting from the core session-mean RT page.')
        for ax,name in zip(axes.flat,bh.rt_masks(eligible)):
            g=summary[summary.condition.eq(name)]
            for r in g.itertuples():
                ax.errorbar(r.cluster_label,r.mean,yerr=getattr(r,error),fmt='o',color=COLORS[int(r.cluster_label)-1],capsize=4)
                ax.annotate(f'n={r.n_trials}',(r.cluster_label,r.mean),xytext=(5,8),textcoords='offset points',fontsize=7)
            ax.set_xticks(range(1,k+1),[f'C{c}' for c in range(1,k+1)]);ax.set_title(name);ax.set_ylabel('RT (s)')
        footer(fig,'Trial-level Welch tests with Holm adjustment are exported as descriptive reference comparisons; repeated measurements are not modeled.\nSD describes trial spread. SEM assumes independent trials and should not be read as animal-level uncertainty.')
        save(fig,out/f'K3_7_reaction_time_mean_{error}',report if error=='sem' else None)
    fig,axes=shell('K=3: reaction time with shared duration bins',4,4,size=(20,19),
                   note='Shared physical-duration bin edges across clusters within condition/category. Pooled-trial mean +/- SEM; y scales vary.')
    for ax,name in zip(axes.flat,bh.rt_masks(eligible)):
        for c in range(1,k+1):
            g=dur[dur.condition.eq(name)&dur.cluster_label.eq(c)].sort_values('duration_s')
            ax.errorbar(g.duration_s,g['mean'],yerr=g['sem'],fmt='o-',color=COLORS[c-1],label=f'C{c}',ms=3)
        ax.set_title(name);ax.set_xlabel('Duration (s)');ax.set_ylabel('RT (s)')
    footer(fig,'Binwise Welch tests and Holm-adjusted p values are in K3_rt_binwise_pairwise_tests.csv. Tests treat trials as independent; exploratory only.')
    save(fig,out/'K3_7_reaction_time_shared_bins',report)
