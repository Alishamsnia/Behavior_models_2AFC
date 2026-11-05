"""Independent complete-train E/F sequence collectors and figures."""
import json
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from .core import KEYS
from .behavior import adjusted_start
from . import plots as p

SUBJECTS=('MC06','MC07','MC08')
CONTEXTS=('All biased blocks','Short block','Long block')
TRACES={'E': [('rewarded_common','After rewarded common','#009E73'),('punished_common','After punished common','#D55E00')],
        'F': [('pre','Common before rare','#0072B2'),('rare','Rare','#D55E00'),('post','Common after rare','#009E73')]}


def collect(tagged):
    rows={'E':[],'F':[]};audits=[]
    for key,full in tagged.groupby(KEYS,sort=False):
        g=full[full.warm_up.eq(0)&full.true_block.isin(['S','L'])].sort_values('source_trial_index').reset_index(drop=True)
        if not len(g):continue
        if not np.all(np.diff(g.source_trial_index)==1):
            raise ValueError('Sequence timeline has gaps')
        block=g.true_block.to_numpy();rare=g.is_rare.to_numpy(bool);correct=g.correct.to_numpy(float)
        switches=np.flatnonzero(block[1:]!=block[:-1])+1
        starts=[0]+[adjusted_start(rare,block,int(i)) for i in switches]
        if np.any(np.diff(starts)<=0):
            raise ValueError('Adjusted complete blocks collapse; refuse to delete a block')
        audits.append(dict(subject=key[0],session_id=key[1],n_trials=len(g),n_complete_blocks=max(0,len(starts)-1),raw_verified=True))
        # Last block is incomplete and omitted, as in reference E/F.
        for number,(lo,hi) in enumerate(zip(starts[:-1],starts[1:])):
            def record(panel,anchor,train_start,train_end,trace,value):
                base=dict(subject=key[0],session_id=key[1],cluster_label=int(g.cluster_label.iloc[0]),
                          block_id=f'{key[0]}|{key[1]}|B{number}',sequence_id=f'{key[0]}|{key[1]}|B{number}|{panel}{anchor-lo}',
                          block_context='Short block' if block[anchor]=='S' else 'Long block',
                          anchor_position=anchor-lo,train_start_position=train_start-lo,train_end_position=train_end-lo,
                          block_length=hi-lo,trace=trace,performance=value)
                for start in range(0,hi-lo-6,3):
                    center=start+4
                    if center>25:break
                    if train_start-lo>=start and train_end-lo<start+7:
                        rows[panel].append(dict(**base,window_start=start,window_center=center))
            for target in range(lo+2,hi):
                if not rare[target-2:target+1].any() and np.isfinite(correct[target-1:target+1]).all():
                    trace='rewarded_common' if correct[target-1]==1 else 'punished_common'
                    record('E',target,target-2,target,trace,correct[target])
            for anchor in range(lo+2,hi-1):
                if rare[anchor-2:anchor+2].tolist()==[False,False,True,False] and np.isfinite(correct[anchor-1:anchor+2]).all():
                    for trace,offset in [('pre',-1),('rare',0),('post',1)]:
                        record('F',anchor,anchor-2,anchor+1,trace,correct[anchor+offset])
    columns=[*KEYS,'cluster_label','block_id','sequence_id','block_context','anchor_position',
             'train_start_position','train_end_position','block_length','trace','performance','window_start','window_center']
    return {key:pd.DataFrame(value,columns=columns) for key,value in rows.items()},pd.DataFrame(audits)


def summarize(events):
    records=[]
    for panel,frame in events.items():
        for c in (1,2,3):
            for subject in (*SUBJECTS,'All subjects pooled'):
                for context in CONTEXTS:
                    g=frame[frame.cluster_label.eq(c)]
                    if subject in SUBJECTS:g=g[g.subject.eq(subject)]
                    if context!=CONTEXTS[0]:g=g[g.block_context.eq(context)]
                    for (center,trace),e in g.groupby(['window_center','trace']):
                        records.append(dict(panel=panel,cluster_label=c,subject=subject,block_context=context,
                                            window_center=center,trace=trace,mean=e.performance.mean(),sem=e.performance.sem(),
                                            n_sequences=len(e),n_sessions=len(e[KEYS].drop_duplicates()),n_subjects=e.subject.nunique()))
    return pd.DataFrame(records,columns=['panel','cluster_label','subject','block_context','window_center','trace','mean','sem','n_sequences','n_sessions','n_subjects'])


def panel(ax,s,c,subject,context,kind):
    g=s[s.cluster_label.eq(c)&s.subject.eq(subject)&s.block_context.eq(context)&s.panel.eq(kind)]
    for trace,label,color in TRACES[kind]:
        e=g[g.trace.eq(trace)].sort_values('window_center')
        ax.errorbar(e.window_center,e['mean'],yerr=e['sem'],fmt='o-',color=color,lw=1.3,ms=3,capsize=2,label=label)
    ax.set_title(f'C{c} / {subject} / {kind}',fontsize=10);ax.set_ylim(-.03,1.03);ax.set_xlim(3,26)
    ax.set_xlabel('7-trial window center (block trial)');ax.set_ylabel('P(correct)');ax.legend(fontsize=6,frameon=False)
    if g.empty:ax.text(.5,.5,'No qualifying sequences',transform=ax.transAxes,ha='center',fontsize=8)


def run(tagged,labels,out):
    out.mkdir(parents=True,exist_ok=True)
    events,audit=collect(tagged);s=summarize(events)
    audit.to_csv(out/'session_input_audit.csv',index=False);labels.to_csv(out/'session_clusters_used.csv',index=False)
    events['E'].to_csv(out/'common_outcome_window_events.csv',index=False)
    events['F'].to_csv(out/'rare_sequence_window_events.csv',index=False);s.to_csv(out/'window_summary.csv',index=False)
    foot='E: C-C-C target after rewarded/punished middle common. F: C-C-R-C, measuring pre/rare/post correctness.\nComplete adjusted blocks only, final block excluded. Whole train must fit each 7-trial window (step 3, centers <=25). Pooled event SEM; overlapping windows are dependent.'
    with PdfPages(out/'ward_sequence_overview.pdf') as pdf:
        for c in (1,2,3):
            fig,axes=p.shell(f'K=3 / C{c}: complete-sequence overview',2,3,size=(18,10),
                             note='Pooled event summaries in all biased blocks, short blocks and long blocks. New Ward labels.')
            for col,context in enumerate(CONTEXTS):
                for row,kind in enumerate(['E','F']):
                    panel(axes[row,col],s,c,'All subjects pooled',context,kind)
                    axes[row,col].set_title(f'{context} / {kind}')
            p.footer(fig,foot);p.save(fig,out/f'cluster_{c}_overview',pdf)
    with PdfPages(out/'ward_sequence_context_overview.pdf') as pdf:
        for context in CONTEXTS:
            fig,axes=p.shell(f'K=3 sequence overview: {context}',3,2,size=(15,14),note='New Ward labels. Sequence definitions match the reference; no reaction-time filtering is applied.')
            for c in (1,2,3):
                for j,kind in enumerate(['E','F']):panel(axes[c-1,j],s,c,'All subjects pooled',context,kind)
            p.footer(fig,foot);p.save(fig,out/('overview_'+context.replace(' ','_')),pdf)
    with PdfPages(out/'ward_sequence_by_cluster.pdf') as all_pdf:
        for c in (1,2,3):
            with PdfPages(out/f'ward_cluster_{c}_sequence_performance.pdf') as cluster_pdf:
                for context in CONTEXTS:
                    fig,axes=p.shell(f'K=3 / C{c}: {context}',2,4,size=(20,10),note='Pooled and individual-mouse event summaries; missing mouse/cluster support remains empty.')
                    for j,subject in enumerate(['All subjects pooled',*SUBJECTS]):
                        for row,kind in enumerate(['E','F']):panel(axes[row,j],s,c,subject,context,kind)
                    p.footer(fig,foot);all_pdf.savefig(fig);cluster_pdf.savefig(fig)
                    p.save(fig,out/f'C{c}_{context.replace(" ","_")}')
    (out/'methods.json').write_text(json.dumps(dict(K=3,window_size=7,window_step=3,max_window_center=25,
        weighting='Pooled sequence/event means and event SEM; windows overlap',
        exclusions='Warm-up/neutral trials; incomplete final block; missing correctness in required response phases',
        E='Three common stimuli; final correctness, split by middle reward/punishment',F='C-C-R-C; pre/rare/post correctness'),indent=2)+'\n')
    (out/'README.md').write_text('# Sequence performance\n\nIndependent reconstruction from verified raw trial timelines. Overview PDF: three contexts, three pages. Detailed PDF: three clusters x three contexts, nine pages; also split by cluster. Event identities, window membership and support are exported. No previous clustering code or trial caches are loaded.\n')
