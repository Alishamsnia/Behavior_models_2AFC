"""Refresh K4 pages 5-8 using the published, unchanged Ward assignments.

Run: python -m post_analysis.zero_anchored_ward.refresh_k4_report
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter
from threadpoolctl import threadpool_limits

from . import behavior, data, k4_figures
from .pipeline import DEFAULT_OUT, NAME, coverage, file_hashes, json_write, protected


def refresh(out):
    out=Path(out).resolve()
    completion=json.loads((out/'completion.json').read_text())
    protocol=json.loads((out/'02_cluster_labels/protocol.json').read_text())
    if not completion.get('complete') or protocol.get('name')!=NAME:
        raise ValueError('Requires a completed zero-anchored Ward analysis')
    # These immutable inputs are sufficient for a presentation-only refresh.
    inputs=['selected_session_clusters.csv','01_clustering_input/clustering_matrix.npy',
            '01_clustering_input/ward_linkage.npy','06_audit/reaction_time_trials.csv']
    for name in inputs:
        if data.sha(out/name)!=completion['sha256'][name]:
            raise ValueError('Published report input changed: '+name)
    labels=pd.read_csv(out/inputs[0]);labels=labels[labels.K.eq(4)].reset_index(drop=True)
    if len(labels)!=92 or labels.duplicated(['subject','session_id']).any() or set(labels.cluster_label)!={1,2,3,4}:
        raise ValueError('Expected the 92 fixed Ward K4 assignments')
    trials=pd.read_csv(out/inputs[-1])
    for column in ['model_evaluated','negative_lick','latency_eligible','choice_eligible','is_rare']:
        trials[column]=data.binary_flags(trials[column])
    tagged=behavior.attach(trials,labels)
    before=file_hashes(out);historical=protected()
    source_before=file_hashes(Path(__file__).parent)
    stage=Path(tempfile.mkdtemp(prefix=NAME+'.k4_report_staging_',dir=out.parent))
    try:
        shutil.copytree(out,stage,dirs_exist_ok=True)
        dest=stage/'04_clusters_K4';tables=stage/'05_cluster_tables'
        trans,rare=behavior.event_tables(tagged)
        with threadpool_limits(limits=1), PdfPages(dest/'K4_updated_pages_5_to_8.pdf') as pdf:
            titles=k4_figures.render(tagged,trans,rare,dest,tables,pdf)
        writer=PdfWriter()
        previous=PdfReader(out/'04_clusters_K4/K4_report.pdf')
        for page in previous.pages[:4]:writer.add_page(page)
        writer.append(dest/'K4_updated_pages_5_to_8.pdf')
        with (dest/'K4_report.pdf').open('wb') as stream:writer.write(stream)
        updated=PdfReader(dest/'K4_report.pdf')
        if len(updated.pages)!=8:
            raise RuntimeError('K4 report must have eight pages')
        for i in range(4):
            if previous.pages[i].get_contents().get_data()!=updated.pages[i].get_contents().get_data():
                raise RuntimeError('Unrequested page changed: '+str(i+1))
        pages=pd.read_csv(tables/'report_pages.csv')
        pages=pages[~(pages.K.eq(4)&pages.page.ge(5))]
        pages=pd.concat([pages,pd.DataFrame([dict(K=4,page=i,analysis=title,pdf='04_clusters_K4/K4_report.pdf')
                                          for i,title in enumerate(titles,5)])],ignore_index=True)
        pages.sort_values(['K','page']).to_csv(tables/'report_pages.csv',index=False)
        methods=json.loads((tables/'K4_rt_extension_methods.json').read_text())
        protocol['K4_report_revision']=dict(pages=[5,6,7,8],label_sha256=data.sha(out/inputs[0]),
                                          normalization_and_clustering_unchanged=True,methods=methods)
        json_write(stage/'02_cluster_labels/protocol.json',protocol)
        text=(stage/'README.md').read_text()
        marker='\n## K4 extended report revision\n'
        text=text.split(marker)[0]
        text+=marker+'''
`04_clusters_K4/K4_report.pdf` now has eight pages. Pages 1-4 retain their original content. Pages 5-8 match the reference extended report layout using the unchanged Ward K4 labels:

5. Choice behavior: three pooled psychometrics, four within-cluster block comparisons, four transitions, two rare-stimulus panels; sample counts, SEM bands and bridge markers.
6. Pooled-trial RT by shared physical-duration bins, with significance tables for all six cluster pairs (Holm-adjusted across the binwise family).
7. Pooled mean RT +/- SEM for 16 conditions, with sample counts and all six pairwise Welch tests (Holm across 96 condition-level comparisons). A separate SD figure is included.
8. Transition-aligned RT split by current outcome, with four clusters, event counts, SEM shading and bridge markers.

Pages 6-7 follow the reference's trial-pooled statistics; trials are treated as independent and the comparisons are exploratory. The earlier session-mean RT table remains available in `05_cluster_tables/reaction_time_binned.csv`; the plotted page-6 values are in `K4_rt_duration_binned_extended.csv`. Exact statistical procedures are in `K4_rt_extension_methods.json`. No models, normalization, Ward hierarchy or assignments were recomputed. The K2/K3 reports are unchanged.
'''
        (stage/'README.md').write_text(text)
        coverage(stage)
        # The refresh has its own source manifest; the original fit/cluster manifest
        # continues to describe the source versions that produced the fixed labels.
        json_write(stage/'06_audit/K4_report_source_manifest.json',
                   [dict(path=str(path.resolve()),sha256=data.sha(path)) for path in sorted(Path(__file__).parent.glob('*.py'))])
        # Limit mutation to the requested report and its metadata/statistical tables.
        allowed={'README.md','completion.json','figure_coverage.csv','02_cluster_labels/protocol.json','05_cluster_tables/report_pages.csv',
                 '06_audit/K4_report_source_manifest.json','06_audit/K4_report_refresh_audit.json'}
        for absolute, expected in before.items():
            relative=str(Path(absolute).relative_to(out))
            if relative in allowed or relative.startswith('04_clusters_K4/') or relative.startswith('05_cluster_tables/K4_'):
                continue
            if data.sha(stage/relative)!=expected:
                raise RuntimeError('Unrequested output changed: '+relative)
        if historical!=protected():raise RuntimeError('Historical analyses changed during the refresh')
        if source_before!=file_hashes(Path(__file__).parent):raise RuntimeError('Report code changed during rendering')
        json_write(stage/'06_audit/K4_report_refresh_audit.json',dict(
            unchanged_inputs={name:data.sha(stage/name) for name in inputs},
            first_four_page_content_streams_unchanged=True,K2_K3_reports_unchanged=True,
            n_historical_files_verified=len(historical),
            previous_report_sha256=data.sha(out/'04_clusters_K4/K4_report.pdf'),
            refreshed_report_sha256=data.sha(dest/'K4_report.pdf')))
        pdfs={str(path.relative_to(stage)):len(PdfReader(path).pages) for path in sorted(stage.rglob('*.pdf'))}
        hashes={str(path.relative_to(stage)):data.sha(path) for path in sorted(stage.rglob('*')) if path.is_file() and path.name!='completion.json'}
        completion.update(pdf_pages=pdfs,sha256=hashes,K4_report_refreshed=True)
        json_write(stage/'completion.json',completion)
        backup=out.with_name(out.name+'.previous_k4_report_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        out.rename(backup)
        try:stage.rename(out)
        except BaseException:
            backup.rename(out)
            raise
        print('UPDATED',out/'04_clusters_K4/K4_report.pdf',flush=True)
        print('BACKUP',backup,flush=True)
        return out
    except BaseException:
        print('Incomplete refresh retained at',stage,flush=True)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    refresh(args.output)


if __name__=='__main__':main()
