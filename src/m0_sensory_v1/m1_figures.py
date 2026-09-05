"""M1 belief, calibration, switch, hazard, and M0-comparison figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, output: Path, stem: str) -> list[str]:
    names=[]
    for ext in ("png","pdf"):
        path=output/f"{stem}.{ext}";fig.savefig(path,dpi=180,bbox_inches="tight");names.append(path.name)
    plt.close(fig);return names


def generate_m1_diagnostic_figures(
    m1_predictions: pd.DataFrame,
    output_directory: str | Path,
    *,
    m0_predictions: pd.DataFrame | None = None,
) -> dict:
    """Create registered M1 diagnostics and optional paired M0 comparisons."""

    output=Path(output_directory).expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    data=m1_predictions.copy();files={};summaries=[]
    evaluated=data["model_evaluated"].astype(bool)
    # Example causal belief trajectory.
    sid=str(data["session_id"].iloc[0]);example=data[data["session_id"].astype(str).eq(sid)].sort_values("trial_index")
    x=np.arange(len(example));fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True)
    axes[0].plot(x,example["model_prior_long"],color="black",label="π(long)");axes[0].axhline(.5,color="gray",ls="--");axes[0].set(ylabel="Prior",ylim=(0,1),title=f"M1 causal prior: {sid}")
    for state,color in zip(("N","S","L"),("#777777","#1565c0","#c62828")):
        axes[1].plot(x,example[f"model_pre_choice_probability_{state}"],label=state,color=color)
    axes[1].set(xlabel="Task trial",ylabel="Pre-choice belief",ylim=(0,1));axes[1].legend(ncol=3);[ax.grid(alpha=.2) for ax in axes]
    files["belief_trajectory"]=_save(fig,output,"m1_belief_trajectory")
    # Calibration and observed/predicted choices.
    valid=data[evaluated].copy();valid["calibration_bin"]=pd.cut(valid["model_probability_long"],np.linspace(0,1,11),include_lowest=True)
    calibration=valid.groupby("calibration_bin",observed=False).agg(mean_predicted=("model_probability_long","mean"),observed_long=("choice_long","mean"),n=("choice_long","size")).dropna().reset_index()
    calibration.to_csv(output/"m1_calibration.csv",index=False);fig,ax=plt.subplots(figsize=(6,6));ax.plot([0,1],[0,1],"k--");ax.plot(calibration["mean_predicted"],calibration["observed_long"],"o-",color="#c62828");ax.set(xlabel="Predicted P(long)",ylabel="Observed long rate",title="M1 calibration",xlim=(0,1),ylim=(0,1));ax.grid(alpha=.25);files["calibration"]=_save(fig,output,"m1_calibration")
    # Switch-aligned prior and expected correctness.
    aligned=[]
    for session_id,frame in data.groupby("session_id",sort=False):
        frame=frame.sort_values("trial_index").reset_index(drop=True);switch=np.flatnonzero(frame["true_block"].astype(str).ne(frame["true_block"].shift()).to_numpy())[1:]
        for event,center in enumerate(switch):
            for i in range(max(0,center-10),min(len(frame),center+11)):
                aligned.append({"session_id":session_id,"event":event,"relative_trial":i-center,"prior_long":frame.loc[i,"model_prior_long"],"probability_correct":frame.loc[i,"model_probability_correct"]})
    aligned_frame=pd.DataFrame(aligned)
    if not aligned_frame.empty:
        switch_summary=aligned_frame.groupby("relative_trial").agg(prior_long=("prior_long","mean"),probability_correct=("probability_correct","mean"),n=("event","size")).reset_index();switch_summary.to_csv(output/"m1_switch_aligned.csv",index=False)
        fig,axes=plt.subplots(1,2,figsize=(12,4));axes[0].plot(switch_summary.relative_trial,switch_summary.prior_long,color="#c62828");axes[1].plot(switch_summary.relative_trial,switch_summary.probability_correct,color="#1565c0");axes[0].set(ylabel="Mean π(long)");axes[1].set(ylabel="Mean expected P(correct)");[ax.axvline(0,color="black",ls="--") for ax in axes];[ax.set(xlabel="Trials from true switch") for ax in axes];files["switch_aligned"]=_save(fig,output,"m1_switch_aligned")
    # Fitted hazard and implied duration.
    hazard=float(data["fitted_hazard"].iloc[0]);fig,ax=plt.subplots(figsize=(7,4));ax.bar(["h_B","h_N=h_B/2","1/h_B (/100)","1/h_N (/100)"],[hazard,hazard/2,(1/hazard)/100,(2/hazard)/100],color=["#c62828","#2e7d32","#607d8b","#90a4ae"]);ax.set(title=f"Biased mean={1/hazard:.2f}; neutral mean={2/hazard:.2f} trials",ylabel="Value");ax.grid(axis="y",alpha=.25);files["hazard"]=_save(fig,output,"m1_fitted_hazard")
    # Optional paired improvement analyses.
    if m0_predictions is not None:
        keys=["mouse_id","session_id","trial_index","split_role"]
        m0=m0_predictions[keys+["per_trial_negative_log_likelihood"]].rename(columns={"per_trial_negative_log_likelihood":"m0_nll"})
        paired=data.merge(m0,on=keys,validate="one_to_one");paired=paired[paired["model_evaluated"].astype(bool)].copy();paired["delta_elpd"]=paired["m0_nll"]-paired["per_trial_negative_log_likelihood"]
        session=paired.groupby("session_id").delta_elpd.sum().sort_values();session.to_csv(output/"m1_vs_m0_per_session_delta_elpd.csv")
        fig,ax=plt.subplots(figsize=(max(7,len(session)*.3),4));ax.bar(np.arange(len(session)),session,color=np.where(session>=0,"#2e7d32","#c62828"));ax.axhline(0,color="black");ax.set(xlabel="Session",ylabel="ΔELPD M1−M0",title="Paired validation improvement by session");files["per_session_comparison"]=_save(fig,output,"m1_vs_m0_per_session_delta_elpd")
        paired["block_run_position"]=paired.groupby("session_id")["true_block"].transform(lambda s:s.groupby(s.ne(s.shift()).cumsum()).cumcount()+1)
        paired["timing"]=np.where(paired.block_run_position<=10,"early_1_10","late_11_plus")
        paired["congruence"]=np.where(((paired.true_block=="L")&(paired.category_long==1))|((paired.true_block=="S")&(paired.category_long==0)),"common","rare")
        context=paired.groupby(["timing","congruence"]).agg(delta_elpd=("delta_elpd","sum"),n=("delta_elpd","size")).reset_index();context["delta_bits_per_trial"]=context.delta_elpd/(context.n*np.log(2));context.to_csv(output/"m1_vs_m0_context_improvement.csv",index=False)
        fig,ax=plt.subplots(figsize=(8,4));labels=context.timing+"\n"+context.congruence;ax.bar(labels,context.delta_bits_per_trial,color="#6a1b9a");ax.axhline(0,color="black");ax.set(ylabel="Δ bits/trial",title="M1 improvement: early/late and rare/common");files["context_comparison"]=_save(fig,output,"m1_vs_m0_context_improvement")
    manifest={"analysis":"M1 causal diagnostics","files":files,"m0_comparison_included":m0_predictions is not None}
    (output/"m1_figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");return manifest
