"""Limited multipage figures for all-session pattern candidates."""
import json,hashlib,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from .features import LEGACY_OUT,collect,matrix,normalize
from .cluster import driver_manifest,raster_order,COLORS
from .pattern_exploration import OUT,candidates,export_functional_readouts
from . import behavior,latency

CASES=(('rbf_reference',3),('rbf_reference',2),('rbf_richer',3),('performance_only',3))
SELECTED=tuple(dict.fromkeys(v for v,k in CASES))
NAMES={'reference_all_models':'Original all-model K-means','effective_noise':'Effective noise only',
       'richer_history_prior':'Functional history/prior','seven_models':'Seven representative models',
       'performance_only':'Model performances only','parameters_only':'Parameters only',
       'rbf_richer':'RBF: functional history/prior','multiview_richer':'Separate feature-block kernels',
       'rbf_reference':'RBF: original all-model features','rbf_effective_noise':'RBF: effective noise only'}
SHORT={'pure_wsls':'WSLS','extended_wsls':'WSLS-hist','m0_sensory':'M0','m1_action_kernel':'Action',
       'm1_stimulus_kernel':'WSLS-history','m1_hmm_true':'M1-static','m1_dynamic':'M1-dynamic',
       'm2_static':'M2-static','m2_dynamic':'M2-dynamic','m2_dynamic_wsls_bias':'M2+WSLS',
       'm2_dynamic_repetition_bias':'M2+action'}


def overview(summary):
    fig=plt.figure(figsize=(17,12));ax=fig.add_axes([.04,.20,.92,.64]);ax.axis('off');rows=[]
    for r in summary.itertuples():
        rows.append([NAMES[r.variant],r.K,r.n_features,r.sizes,f'{r.ari_to_reference:.2f}',r.min_per_mouse,f'{r.mouse_nmi:.3f}'])
    table=ax.table(cellText=rows,colLabels=['Feature set / similarity','K','Features','Group sizes (C1/C2/C3)','Agreement with original','Min sessions / mouse / group','Mouse association'],
                   cellLoc='center',bbox=[0,0,1,1],colWidths=[.29,.04,.06,.17,.17,.18,.09])
    table.auto_set_font_size(False);table.set_fontsize(9)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('.8');cell.set_linewidth(.4)
        if r==0:cell.set_facecolor('#e5edf5');cell.set_text_props(weight='bold')
        elif (summary.iloc[r-1].variant,int(summary.iloc[r-1].K)) in CASES:cell.set_facecolor('#eff6ee')
    fig.suptitle('Exploring clearer model-defined session patterns — all 92 sessions',fontsize=18,y=.97)
    fig.text(.5,.91,'Original all-model grouping retained as reference. No calendar exclusions, forced-fit weights or cluster-size constraints.',ha='center',fontsize=12)
    fig.text(.04,.15,'Green rows are the exploratory comparisons shown in this file. They were selected after descriptive profile inspection; all alternatives are retained.\n'
             'The first eight feature sets were planned together; the two RBF controls were added to separate feature changes from kernel effects.\n'
             '"All models" uses the original 11 nonoracle models and nuisance-parameter exclusions. Cluster IDs are matched by overlap, not biological identity.\n'
             'Behavior is used to interpret these exploratory groups. Motion/pupil data have not been used to choose them.',fontsize=11,va='top')
    return fig


def profile_page(profiles):
    shown=('reference_all_models','rbf_reference','rbf_richer','performance_only')
    metrics=['block_context_shift','rare_accuracy_cost','adjusted_rapid_S2L_recovery','adjusted_rapid_L2S_recovery',
             'gain_vs_sensory_m1_action_kernel','gain_vs_sensory_m2_static']
    rows=[]
    for variant in shown:
        p=profiles[profiles.variant.eq(variant)&profiles.K.eq(3)].pivot(index='cluster_label',columns='metric',values='equal_mouse_mean')
        for c in (1,2,3):rows.append([NAMES[variant],f'C{c}',*[f'{p.loc[c,m]*100:.1f}' if i<4 else f'{p.loc[c,m]:.3f}' for i,m in enumerate(metrics)]])
    fig=plt.figure(figsize=(17,11));ax=fig.add_axes([.04,.30,.92,.50]);ax.axis('off')
    table=ax.table(cellText=rows,colLabels=['Grouping','Cluster','Context shift (pp)','Common−rare cost (pp)','S→L recovery (pp)','L→S recovery (pp)','Action gain','Static-prior gain'],
                   cellLoc='center',bbox=[0,0,1,1],colWidths=[.27,.06,.11,.13,.12,.12,.095,.095])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('.8');cell.set_linewidth(.4)
        if r==0:cell.set_facecolor('#e5edf5');cell.set_text_props(weight='bold')
    fig.suptitle('What distinguishes the K=3 groups?',fontsize=19,y=.96)
    fig.text(.5,.89,'Equal mouse weight, then equal session weight within mouse. These summaries describe the observed patterns.',ha='center',fontsize=12)
    fig.text(.04,.22,'Context shift: difference in P(long) between long/short blocks at matched stimulus bins.\n'
             'Recovery: newly common stimulus accuracy, trials 3–4 minus 0–1, paired within adjusted transition events.\n'
             'Model gains: original test NLL improvement over sensory-only; positive favors added history/context.\n\n'
             'RBF on the original features preserves the original weak-context C3 sessions, and adds seven MC07 sessions to C1.\n'
             'C1 has greater L→S recovery than C2/C3 in each mouse; C2 has greater S→L recovery than C3 in each mouse.\n'
             'C3 retains action-history benefits: "weak context / relatively sensory-led" does not mean pure sensory or best sensory precision.\n'
             'The performance-only side result separates a lower-accuracy, history-heavy group; it is a different question, not a replacement.',fontsize=11,va='top')
    return fig


def cluster_page(raw,values,meta,labels,order,variant,k,profiles):
    order=order[np.argsort(labels[order],kind='stable')];z=normalize(values[meta.feature],raw.subject.to_numpy())
    fig=plt.figure(figsize=(25,12));gs=fig.add_gridspec(1,4,width_ratios=[.35,.35,18,4],left=.04,right=.97,bottom=.28,top=.89,wspace=.12)
    subjects=list(raw.subject.unique());mouse_colors=['#1b9e77','#d95f02','#7570b3']
    ax=fig.add_subplot(gs[0,0]);ax.imshow(np.array([subjects.index(s) for s in raw.subject.iloc[order]])[:,None],aspect='auto',cmap=ListedColormap(mouse_colors),vmin=0,vmax=2)
    ax.set_title('Mouse',fontsize=9);ax.set_xticks([]);ax.set_yticks([])
    ax=fig.add_subplot(gs[0,1]);ax.imshow(labels[order,None],aspect='auto',cmap=ListedColormap(COLORS[:k]),vmin=1,vmax=k)
    ax.set_title('Group',fontsize=9);ax.set_xticks([]);ax.set_yticks([])
    ax=fig.add_subplot(gs[0,2]);im=ax.imshow(z[order],aspect='auto',cmap='RdBu_r',vmin=-3,vmax=3,interpolation='nearest')
    for border in np.flatnonzero(np.diff(labels[order]))+.5:ax.axhline(border,color='black',lw=2)
    for border in np.flatnonzero(meta.model.to_numpy()[1:]!=meta.model.to_numpy()[:-1])+.5:ax.axvline(border,color='black',lw=.6)
    names=[f'{SHORT[r.model]}: '+('NLL gain' if r.block=='performance' else r.description) for r in meta.itertuples()]
    ax.set_xticks(np.arange(len(names)),names,rotation=90,fontsize=8);ax.set_ylabel('Fitting window / associated test session')
    ax.set_title('Common display: original 48 model-performance/parameter drivers',fontsize=13)
    fig.colorbar(im,ax=ax,fraction=.012,pad=.01,label='Within-mouse z score (clipped)')
    ax=fig.add_subplot(gs[0,3]);ax.axis('off');table=pd.crosstab(raw.subject,labels)
    tab=ax.table(cellText=table.to_numpy(),rowLabels=table.index,colLabels=[f'C{c}' for c in table.columns],loc='upper center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,1.6)
    for i,s in enumerate(subjects):ax.text(0,.79-i*.035,'■ '+s,color=mouse_colors[i],fontsize=10)
    p=profiles[profiles.variant.eq(variant)&profiles.K.eq(k)].pivot(index='cluster_label',columns='metric',values='equal_mouse_mean')
    lines=['Descriptive behavior guide','', 'Context shift / S→L / L→S recovery']
    for c in range(1,k+1):
        r=p.loc[c];lines.append(f'C{c}: {r.block_context_shift*100:.1f} / {r.adjusted_rapid_S2L_recovery*100:.1f} / {r.adjusted_rapid_L2S_recovery*100:.1f} pp')
    lines+=['','All 92 sessions included.','Forced fits treated equally.','','Labels matched to reference','by session overlap.']
    ax.text(0,.65,'\n'.join(lines),va='top',fontsize=10)
    fig.suptitle(f'{NAMES[variant]} — K={k}',fontsize=19)
    fig.text(.09,.095,'Shared heatmap basis makes the groupings comparable; the functional RBF candidate uses 58 transformed drivers to define similarity.\n'
             'Sensory w/s0 are replaced by effective noise only in the functional candidate. History weights use valid-update lags 1/3/8; hazards use ages 10/23/35.\n'
             'Parameters come from training-window fits; performance and behavior refer to the associated test session. Exploratory descriptions for follow-up.',fontsize=11,va='top')
    return fig


def render():
    raw,v,manifest,_=collect();display=driver_manifest(manifest);display=display[~display.oracle].reset_index(drop=True)
    opts=candidates(raw,v,manifest);labels=pd.read_csv(OUT/'all_candidate_labels.csv');summary=pd.read_csv(OUT/'candidate_summary.csv');profiles=pd.read_csv(OUT/'behavior_profiles_equal_mouse.csv')
    selected=pd.concat([labels[labels.variant.eq(v)&labels.K.eq(k)] for v,k in CASES],ignore_index=True)
    selected.to_csv(OUT/'selected_session_clusters.csv',index=False)
    export_functional_readouts(raw,selected)
    pages=[]
    with PdfPages(OUT/'clusters.pdf') as pdf:
        order_cache={}
        for variant,k in CASES:
            if variant not in order_cache:
                values,meta,method=opts[variant];x,_=matrix(values,meta,raw.subject.to_numpy());order_cache[variant]=raster_order(x)
            order=order_cache[variant]
            lab=labels[labels.variant.eq(variant)&labels.K.eq(k)].cluster_label.to_numpy()
            fig=cluster_page(raw,v,display,lab,order,variant,k,profiles);pdf.savefig(fig)
            if variant==SELECTED[0] and k==3:fig.savefig(OUT/'clusters_K3.png',dpi=140)
            plt.close(fig);pages.append(dict(page=len(pages)+1,analysis='heatmap',variant=variant,K=k))
        for kind,fig in [('profiles',profile_page(profiles)),('overview',overview(summary))]:
            pdf.savefig(fig);plt.close(fig);pages.append(dict(page=len(pages)+1,analysis=kind))
    # Reuse the original independently audited trial extraction, not old labels.
    source=LEGACY_OUT/'reaction_time_trials.csv';trials=pd.read_csv(source)
    if 'cluster_label' in trials:trials=trials.drop(columns='cluster_label')
    if trials.duplicated(['subject','session_id','trial_index']).any():raise ValueError('Duplicated source trials')
    eligible=trials[trials.latency_eligible.eq(True)].copy();bp=[];bins=[]
    with PdfPages(OUT/'cluster_behavior_pooled.pdf') as pdf:
        for variant,k in CASES:
            ll=labels[labels.variant.eq(variant)&labels.K.eq(k)]
            tagged=behavior.attach_labels(trials,ll);fig,trans,rare=behavior.make_page(tagged,k)
            fig._suptitle.set_text(f'Mouse behavior — {NAMES[variant]} — K={k}');fig._suptitle.set_fontsize(15)
            behavior.annotate_mouse_support(fig,ll);pdf.savefig(fig)
            if variant==SELECTED[0] and k==3:fig.savefig(OUT/'behavior_K3.png',dpi=120)
            plt.close(fig);bp.append(dict(page=len(bp)+1,variant=variant,K=k,analysis='choice_behavior',n_sessions=92))
        for variant,k in CASES:
            ll=labels[labels.variant.eq(variant)&labels.K.eq(k)]
            tagged=behavior.attach_labels(eligible,ll);fig,stats,sizes=latency.make_page(tagged,k)
            fig._suptitle.set_text(f'Reaction time — {NAMES[variant]} — K={k}');fig._suptitle.set_fontsize(15)
            behavior.annotate_mouse_support(fig,ll);pdf.savefig(fig);plt.close(fig)
            stats['variant'],stats['K']=variant,k;bins.append(stats)
            bp.append(dict(page=len(bp)+1,variant=variant,K=k,analysis='reaction_time',n_sessions=92,n_trials=len(eligible)))
    pd.concat(bins).to_csv(OUT/'reaction_time_binned.csv',index=False)
    pd.DataFrame(bp).to_csv(OUT/'behavior_pages.csv',index=False);pd.DataFrame(pages).to_csv(OUT/'cluster_pages.csv',index=False)
    for name in ['reaction_time_input_audit.csv']:
        shutil.copyfile(LEGACY_OUT/name,OUT/name)
    freeze=dict(selected=[dict(variant=v,K=k) for v,k in CASES],primary_reference='reference_all_models',
                rationale='RBF reference preserves original C3 and improves cross-mouse C1 profile; richer version tests functional geometry; performance-only is a separate history-heavy/low-accuracy hypothesis.',
                selection='Exploratory: chosen after inspecting descriptive model/behavior profiles. No independent validation claim.',
                labels_sha256=hashlib.sha256((OUT/'selected_session_clusters.csv').read_bytes()).hexdigest(),
                source_trials=str(source),source_trials_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                all_sessions=92,forced_fit_handling='No special weights or exclusions',future_followup='Pupil/motion not used in this exploration')
    (OUT/'followup_candidates.json').write_text(json.dumps(freeze,indent=2))
    print('Rendered 6 cluster/profile pages and 8 behavior/RT pages; original plots unchanged:',OUT,flush=True)

if __name__=='__main__':
    render()
    from .pattern_visuals import run as readable_report
    readable_report(refresh_archive=True)
