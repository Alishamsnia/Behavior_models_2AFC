#!/usr/bin/env python3
"""Run a bounded M0-versus-M1 synthetic model-recovery experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from m0_sensory_v1.m1_registry import load_m1_model_registry  # noqa: E402
from m0_sensory_v1.registry import load_model_registry  # noqa: E402
from m0_sensory_v1.model import (  # noqa: E402
    M1SessionTrials, SessionTrials, evaluate_split_metrics,
    fit_single_initialization, simulate_m0_sessions, simulate_m1_hmm_sessions,
)


def _as_m0(records):
    return tuple(SessionTrials(
        x.mouse_id, x.session_id, x.duration_s, x.category_long,
        x.choice_long, x.split_role
    ) for x in records)


def _replace_choices(templates, choices):
    return tuple(M1SessionTrials(
        t.mouse_id,t.session_id,t.duration_s,c.choice_long,t.category_long,t.feedback_valid,t.split_role
    ) for t,c in zip(templates,choices,strict=True))


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--m0-registry",type=Path); p.add_argument("--m1-registry",type=Path)
    p.add_argument("--epochs",type=int,default=500); p.add_argument("--sessions",type=int,default=6); p.add_argument("--trials",type=int,default=300)
    a=p.parse_args(); m0=load_model_registry(a.m0_registry); m1=load_m1_model_registry(a.m1_registry)
    levels=torch.tensor([.2,.35,.5,.65,.8,1.7,1.85,2.,2.15,2.3],dtype=torch.float64)
    durations=tuple(levels.repeat(max(1,a.trials//10)) for _ in range(a.sessions)); nval=max(1,a.sessions//3)
    m1_generate=m1.initializations[4].constrained_tensor()
    m1_train,_=simulate_m1_hmm_sessions(durations,m1_generate,m1.task_moments,random_seed=901,split_role="train")
    m1_val,_=simulate_m1_hmm_sessions(durations[:nval],m1_generate,m1.task_moments,random_seed=902,split_role="validation")
    templates_train,_=simulate_m1_hmm_sessions(durations,m1_generate,m1.task_moments,random_seed=903,split_role="train")
    templates_val,_=simulate_m1_hmm_sessions(durations[:nval],m1_generate,m1.task_moments,random_seed=904,split_role="validation")
    m0_generate=m0.initializations[0].constrained_tensor()
    m0_train_choice=simulate_m0_sessions(durations,m0_generate,m0.task_moments,random_seed=905,split_role="train")
    m0_val_choice=simulate_m0_sessions(durations[:nval],m0_generate,m0.task_moments,random_seed=906,split_role="validation")
    cases={"generated_M0":(_replace_choices(templates_train,m0_train_choice),_replace_choices(templates_val,m0_val_choice)),"generated_M1":(m1_train,m1_val)}
    rows=[]
    for label,(train1,val1) in cases.items():
        train0,val0=_as_m0(train1),_as_m0(val1)
        fit0=fit_single_initialization(train0,val0,m0,m0.initializations[0],max_epochs_override=a.epochs)
        fit1=fit_single_initialization(train1,val1,m1,m1.initializations[0],max_epochs_override=a.epochs)
        score0=evaluate_split_metrics(val0,torch.tensor(fit0.best_raw_parameters,dtype=torch.float64),m0,required_split_role="validation")
        score1=evaluate_split_metrics(val1,torch.tensor(fit1.best_raw_parameters,dtype=torch.float64),m1,required_split_role="validation")
        delta=(score1.elpd-score0.elpd)/(score0.valid_trial_count*torch.log(torch.tensor(2.)).item())
        rows.append({"generating_model":label,"m0_elpd":score0.elpd,"m1_elpd":score1.elpd,"delta_bits_per_trial_m1_minus_m0":delta})
        print(label,rows[-1],flush=True)
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);(out/"model_recovery.json").write_text(json.dumps({"cases":rows},indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(7,5));ax.bar([r["generating_model"] for r in rows],[r["delta_bits_per_trial_m1_minus_m0"] for r in rows],color=["#607d8b","#c62828"]);ax.axhline(0,color="black",lw=1);ax.set(ylabel="Δ bits/trial (M1 − M0)",title="M0/M1 model recovery");ax.grid(axis="y",alpha=.25)
    for ext in ("png","pdf"):fig.savefig(out/f"model_recovery.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig);return 0


if __name__=="__main__":raise SystemExit(main())
