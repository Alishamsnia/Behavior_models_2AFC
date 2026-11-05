"""Matched observed-behavior follow-ups with freshly constructed histories."""
import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from matplotlib.backends.backend_pdf import PdfPages
from .core import KEYS
from . import plots as p


def prepare(trials):
    chunks=[]
    for _,source in trials.groupby(KEYS,sort=False):
        g=source.sort_values('source_trial_index').reset_index(drop=True).copy()
        if not np.all(np.diff(g.source_trial_index)==1):
            raise ValueError('History construction requires the complete trial timeline')
        g['block_number']=g.true_block.ne(g.true_block.shift()).cumsum()
        g['block_uid']=g.subject+'|'+g.session_id+'|'+g.block_number.astype(str)
        g['block_age']=g.groupby('block_number').cumcount()+1
        g['block_length']=g.groupby('block_number').block_number.transform('size')
        g['complete_block']=g.block_number.ne(g.block_number.iloc[-1])
        g['eligible']=g.choice_eligible & g.correct.notna() & g.true_block.isin(['S','L'])
        g['duration_key']=g.duration_s.round(3)
        g['difficulty']=(abs(g.duration_s-1.25)<.60).astype(int)
        g['age_band']=np.select([g.block_age.le(15),g.block_age.ge(21)],[0,2],default=1)
        for lag in (1,2,3):
            for col in ['choice_long','correct','is_rare','eligible','difficulty','block_number']:
                g[f'lag{lag}_{col}']=g[col].shift(lag)
            g[f'valid{lag}']=g.eligible & g[f'lag{lag}_eligible'].eq(True) & g.block_number.eq(g[f'lag{lag}_block_number'])
        g['valid2'] &= g.valid1
        g['valid3'] &= g.valid2
        run=0;previous=None;lengths=[]
        for row in g.itertuples():
            if not row.eligible:
                run=0;previous=None
            else:
                run=run+1 if row.valid1 and previous==row.choice_long else 1
                previous=row.choice_long
            lengths.append(run)
        g['prior_run']=pd.Series(lengths).shift().clip(upper=3)
        g['repeat']=g.choice_long.eq(g.lag1_choice_long).astype(float)
        g['against_block']=np.where(g.true_block.eq('L'),1-g.choice_long,g.choice_long)
        chunks.append(g)
    return pd.concat(chunks,ignore_index=True)


def analyses(g):
    specs={}
    for outcome,name in [(1,'after_reward'),(0,'after_error')]:
        e=g[g.valid1 & g.lag1_correct.eq(outcome)].copy()
        e['condition']=e.lag1_choice_long.astype(int);e['response']=e.choice_long
        specs[name]=(e,[0,1],['true_block','lag1_difficulty'],'duration_key')
    e=g[g.valid1].copy();e['condition']=e.prior_run.astype(int);e['response']=e.repeat
    specs['action_run']=(e,[1,2,3],['duration_key','true_block','lag1_correct','lag1_choice_long'],None)
    one=g.valid2 & g.lag1_is_rare.eq(True) & g.lag2_is_rare.eq(False)
    two=g.valid3 & g.lag1_is_rare.eq(True) & g.lag2_is_rare.eq(True) & g.lag3_is_rare.eq(False)
    e=g[one|two].copy();e['condition']=np.where(two.loc[e.index],2,1);e['response']=e.against_block
    specs['contradictory_sequence']=(e,[1,2],['duration_key','true_block','lag1_correct','age_band'],None)
    isolated=g.valid3 & g.lag1_is_rare.eq(True) & g.lag2_is_rare.eq(False) & g.lag3_is_rare.eq(False) & ~g.is_rare
    anchor_age=g.block_age-1
    stage=pd.Series(np.select([anchor_age.between(6,15),anchor_age.between(21,27)],[0,1],default=-1),index=g.index)
    e=g[isolated & stage.ge(0)].copy();e['condition']=stage.loc[e.index];e['response']=e.against_block;e['anchor_age']=e.block_age-1
    specs['block_age']=(e,[0,1],['difficulty','true_block','lag1_correct','lag1_difficulty'],None)
    specs['block_age_exact_stimulus']=(e,[0,1],['duration_key','true_block','lag1_correct','lag1_difficulty'],None)
    specs['block_age_paired_block']=(e,[0,1],['block_uid'],None)
    return specs


def matched_estimates(events,conditions,strata,facet):
    """Within-session overlap cells, weighted by minimum condition count.

    Each shared cell gets the same weight for every condition. Uncertainty is
    conditional on these selected cells/weights, with formal-block influence sums.
    """
    rows=[];retained=[]
    grouping=KEYS+['cluster_label']+([facet] if facet else [])
    schema=grouping+[f'mean_{c}' for c in conditions]+[f'n_{c}' for c in conditions]+[
        'n_cells','n_retained','n_eligible','n_blocks','contrast','se_block','interval_supported','ci_low','ci_high']
    for key,g in events.groupby(grouping,sort=False,observed=True):
        e=g.dropna(subset=strata+['condition','response']).copy()
        if e.empty: continue
        e['cell']=list(map(tuple,e[strata].to_numpy()))
        counts=e.groupby(['cell','condition'],observed=True).size().unstack('condition',fill_value=0).reindex(columns=conditions,fill_value=0)
        counts=counts[counts.gt(0).all(axis=1)]
        if counts.empty:continue
        weights=counts.min(axis=1).astype(float);weights=weights/weights.sum()
        e=e[e.cell.isin(counts.index)].copy();e['cell_weight']=e.cell.map(weights)
        mu=e.groupby(['cell','condition'],observed=True).response.mean().unstack('condition').reindex(index=counts.index,columns=conditions)
        averages=mu.mul(weights,axis=0).sum()
        row=dict(zip(grouping,key));row.update({f'mean_{c}':averages[c] for c in conditions})
        row.update({f'n_{c}':int(e.condition.eq(c).sum()) for c in conditions})
        row.update(n_cells=len(counts),n_retained=len(e),n_eligible=len(g),n_blocks=e.block_uid.nunique(),interval_supported=False)
        if len(conditions)==2:
            a,b=conditions;row['contrast']=float(averages[b]-averages[a])
            means={(cell,c):v for cell,r in mu.iterrows() for c,v in r.items()}
            sizes={(cell,c):v for cell,r in counts.iterrows() for c,v in r.items()}
            scores=[]
            for r in e.itertuples():
                sign=1 if r.condition==b else -1
                scores.append(sign*r.cell_weight*(r.response-means[(r.cell,r.condition)])/sizes[(r.cell,r.condition)])
            blocks=pd.Series(scores,index=e.index).groupby(e.block_uid).sum();n=len(blocks)
            se=float(np.sqrt(n/(n-1)*np.sum(blocks**2))) if n>1 else np.nan
            enough=n>=6 and min(row[f'n_{a}'],row[f'n_{b}'])>=8 and counts.ge(2).all().all() and se>0
            row['se_block']=se;row['interval_supported']=bool(enough)
            half=float(student_t.ppf(.975,n-1)*se) if enough else np.nan
            row['ci_low']=max(-1,row['contrast']-half) if enough else np.nan
            row['ci_high']=min(1,row['contrast']+half) if enough else np.nan
        rows.append(row);retained.append(e.drop(columns='cell'))
    return pd.DataFrame(rows,columns=schema),pd.concat(retained,ignore_index=True) if retained else events.iloc[:0]


def psych_axes(ax,table,c,title):
    g=table[table.cluster_label.eq(c)]
    for condition,ls in [(0,'--'),(1,'-')]:
        col=f'mean_{condition}'
        if g.empty: continue
        # Session means within mouse, then equal-mouse mean for each physical duration.
        m=g.groupby(['subject','duration_key'])[col].mean().reset_index()
        avg=m.groupby('duration_key')[col].mean()
        ax.plot(avg.index,avg.values,ls=ls,color=p.COLORS[c-1],marker='o',ms=3,label='Previous short' if condition==0 else 'Previous long')
        for mouse,x in m.groupby('subject'):
            ax.scatter(x.duration_key,x[col],marker=p.MOUSE_MARKERS[mouse],s=12,color=p.COLORS[c-1],alpha=.35)
    ax.set_title(title,fontsize=11);ax.set_ylim(-.03,1.03);ax.set_xlabel('Current duration (s)');ax.set_ylabel('P(choose long)')
    ax.axvline(1.25,color='.6',ls=':');ax.axhline(.5,color='.6',ls=':');ax.legend(fontsize=7,frameon=False)
    if g.empty: ax.text(.5,.5,'No matched support',transform=ax.transAxes,ha='center')


def points_axes(ax,table,c,conditions,title):
    g=table[table.cluster_label.eq(c)]
    columns=[f'mean_{v}' for v in conditions]
    for mouse,x in g.groupby('subject'):
        ax.plot(conditions,x[columns].mean(),marker=p.MOUSE_MARKERS[mouse],color=p.COLORS[c-1],alpha=.55,label=mouse)
    if not g.empty:
        avg=g.groupby('subject')[columns].mean().mean()
        ax.plot(conditions,avg,color='black',lw=2,marker='o',label='Equal mouse')
    ax.set_title(title);ax.set_xticks(conditions);ax.set_ylim(-.03,1.03);ax.set_ylabel('Matched response probability');ax.legend(fontsize=7,frameon=False)
    if g.empty: ax.text(.5,.5,'No matched support',transform=ax.transAxes,ha='center')


def run(tagged,labels,out):
    out.mkdir(parents=True,exist_ok=True);tables=out/'tables';tables.mkdir()
    g=prepare(tagged);g.to_csv(tables/'annotated_trials.csv',index=False)
    labels.to_csv(tables/'session_clusters_used.csv',index=False)
    results={};support=[]
    for name,(events,conditions,strata,facet) in analyses(g).items():
        estimates,keep=matched_estimates(events,conditions,strata,facet);results[name]=estimates
        events.to_csv(tables/f'{name}_eligible_events.csv',index=False)
        keep.to_csv(tables/f'{name}_retained_events.csv',index=False)
        estimates.to_csv(tables/f'{name}_session_estimates.csv',index=False)
        for c in (1,2,3):
            support.append(dict(analysis=name,cluster_label=c,n_eligible=int(events.cluster_label.eq(c).sum()),
                                n_retained=int(keep.cluster_label.eq(c).sum()),
                                n_sessions=len(estimates[estimates.cluster_label.eq(c)][KEYS].drop_duplicates())))
    pd.DataFrame(support).to_csv(tables/'support_summary.csv',index=False)
    foot='Matched overlap cells within each session; fixed cell weights = minimum condition count. Lines average sessions within mouse, then available mice equally.\nMarkers identify mice; no confidence band is claimed across three animals. Histories never cross invalid trials or formal block boundaries.'
    with PdfPages(out/'hypothesis_followup_all_clusters.pdf') as pdf:
        # Preserve the eight requested figure purposes; show every new cluster in each.
        fig,ax=p.shell('After error: previous short vs previous long choices',1,3,size=(17,6),note='All Ward K=3 groups; physical current duration is held fixed.')
        for c in (1,2,3):psych_axes(ax[0,c-1],results['after_error'],c,f'C{c}: after error')
        p.footer(fig,foot);p.save(fig,out/'cross_cluster/01_after_error_comparison',pdf,svg=True)
        fig,ax=p.shell('Action history after reward and error',2,3,size=(17,10),note='All clusters shown; new cluster numbers carry no inherited strategy interpretation.')
        for row,name in enumerate(['after_reward','after_error']):
            for c in (1,2,3):psych_axes(ax[row,c-1],results[name],c,f'C{c}: {name.replace("_"," ")}')
        p.footer(fig,foot);p.save(fig,out/'cluster_1/01_action_after_reward_and_error',pdf,svg=True)
        for name,path,title,conditions in [
            ('action_run','cluster_1/02_action_run_length','Action repetition by preceding run length',[1,2,3]),
            ('contradictory_sequence','cluster_2/02_contradictory_sequence','Against-block choice after one vs two rare stimuli',[1,2]),
            ('block_age','cluster_3/02_block_age','Isolated rare stimulus: early vs late block age',[0,1])]:
            fig,ax=p.shell(title,1,3,size=(17,6),note='Same-session observed behavior; available matched support differs across groups and comparisons.')
            for c in (1,2,3):
                axis=ax[0,c-1]
                points_axes(axis,results[name],c,conditions,f'C{c}')
                if name=='block_age':
                    axis.set_xticks([0,1],['Early (6-15)','Late (21-27)']);axis.set_xlabel('Rare-anchor block age')
                elif name=='action_run':
                    axis.set_xticks([1,2,3],['1','2','3+']);axis.set_xlabel('Preceding choice run length')
                else:
                    axis.set_xlabel('Consecutive preceding rare stimuli')
            p.footer(fig,foot);p.save(fig,out/path,pdf,svg=True)
        fig,ax=p.shell('Feedback after error: all Ward groups',1,3,size=(17,6),note='Dashed: previous short choice; solid: previous long choice. Outcomes refer to the previous trial.')
        for c in (1,2,3):psych_axes(ax[0,c-1],results['after_error'],c,f'C{c}')
        p.footer(fig,foot);p.save(fig,out/'cluster_2/01_feedback_after_error',pdf,svg=True)
        fig,ax=p.shell('Sensory and history readouts at matched physical durations',2,3,size=(17,10),note='Observed current choices by previous choice and outcome; these panels do not decompose model predictions.')
        for row,name in enumerate(['after_reward','after_error']):
            for c in (1,2,3):psych_axes(ax[row,c-1],results[name],c,f'C{c}: {name.replace("_"," ")}')
        p.footer(fig,foot);p.save(fig,out/'cluster_3/01_sensory_and_history',pdf,svg=True)
        fig,ax=p.shell('Session chronology and matched block-age contrasts',1,2,size=(15,6),note='Block-age contrasts require matched support; absent estimates remain missing.')
        for c in (1,2,3):
            for mouse,gp in labels[labels.cluster_label.eq(c)].groupby('subject'):
                ax[0,0].scatter(gp.test_rank,[mouse]*len(gp),color=p.COLORS[c-1],marker=p.MOUSE_MARKERS[mouse],s=30)
            t=results['block_age'];t=t[t.cluster_label.eq(c)].merge(labels[KEYS+['test_rank']],on=KEYS,validate='one_to_one')
            ax[0,1].scatter(t.test_rank,t.contrast,color=p.COLORS[c-1],s=25,label=f'C{c}')
            supported=t[t.interval_supported]
            ax[0,1].vlines(supported.test_rank,supported.ci_low,supported.ci_high,color=p.COLORS[c-1],alpha=.3)
        ax[0,0].set_xlabel('Held-out session rank');ax[0,0].set_title('Cluster chronology')
        ax[0,1].set_xlabel('Held-out session rank');ax[0,1].set_ylabel('Late - early against-block probability');ax[0,1].axhline(0,color='.6',ls=':');ax[0,1].legend()
        p.footer(fig,'Intervals: conditional 95% t intervals using formal-block influence sums, only with >=6 blocks, >=8 events/condition, repeated cells and nonzero SE.\nThe final block remains present in formal-block-age histories, matching the reference; complete-block exclusion is specific to sequence E/F.')
        p.save(fig,out/'cross_cluster/02_session_block_age',pdf,svg=True)
    (out/'README.md').write_text('''# Observed-behavior follow-ups under Ward K=3

Eight reference figure purposes are retained, with all three new clusters shown in each relevant comparison. Folder names preserve the reference navigation, not strategy identities. No old strategy names or cluster counts are inherited.

All source trials remain in place while constructing histories. An eligible target and every history trial must be valid, non-warm-up, in the same formal biased block. after_reward/error compare previous short/long at fixed current duration, block and previous difficulty. action_run compares run length 1/2/3+ at shared duration, block, previous outcome and choice. contradictory_sequence compares C-R-target with C-R-R-target. block_age compares C-C-R-C targets with rare-anchor age 6..15 versus 21..27, with fixed current difficulty, block, previous outcome/difficulty; exact-duration and paired-block sensitivities are exported.

Within each session/facet, retain only strata having every condition; cell weight is the minimum condition count normalized over shared cells. Formal-block influence sums provide conditional intervals only with adequate support. No trial-level independence assumption is used for these intervals. The figure lines show available-mouse means, with individual mouse markers. Same-session behavioral associations are descriptive.
''')
    (out/'analysis_manifest.json').write_text(json.dumps(dict(K=3,n_trials=len(g),n_figures=8,all_clusters_in_each_comparison=True,
                                                           cluster_sizes=labels.cluster_label.value_counts().to_dict()),indent=2)+'\n')
