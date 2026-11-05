"""Parameter-recovery study for the action-history model (M1_action_kernel_v1), all six parameters.

For EVERY sliding window of a mouse, the window's already-fitted (winner) parameters are used as
ground truth, choices are re-simulated from the frozen model on the REAL training-session stimulus
sequences, and the model is refitted with the frozen fitter. This is repeated ``--reps`` times per
window with different seeds. Nothing in ``src/`` is modified: simulation reuses the model's own
readout functions, fitting calls ``fit_m1_action_kernel_single_initialization`` with the
production L-BFGS settings from the registry (250 steps by default).

What is and is not tested
-------------------------
* LEVEL recovery per window and parameter (bias, replicate SD, RMSE, calibration slope).
* "Adjacent change" columns compare differences of replicate-MEAN recovered values with differences
  of truths, i.e. changes in mean recovery. Every window is simulated independently, so the
  estimation-error covariance between adjacent (8-of-9-shared-session) real windows is NOT
  reproduced; the noise of real window-to-window differences is therefore not assessed here.
* Conclusions are conditional on the model being the generator (no misspecification) and on the
  real omission pattern and stimulus sequences.

Fits per simulated dataset
--------------------------
* blind starts (``--start-ids``, registry starts): the estimate is the lowest-training-NLL
  CONVERGED blind start (production rule); if none converged it falls back to the lowest finite NLL
  and is flagged ``winner_converged = False``. Production's escalation ladder is not replicated.
* one truth start (warm start at the generating parameters): a diagnostic. If the blind winner's NLL
  is clearly worse than the truth start's, the blind fit is stuck in a worse optimum
  (``blind_stuck``), i.e. an optimiser failure rather than an identifiability limit.

Usage (from the repo root)
--------------------------
python -m post_analysis.identifiability.action_history_alpha_recovery --subjects MC08 --benchmark
python -m post_analysis.identifiability.action_history_alpha_recovery --subjects MC08 --reps 10 --budget-hours 22
python -m post_analysis.identifiability.action_history_alpha_recovery --subjects MC08 --analyze-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = str(REPO_ROOT / "src")
# This machine also has an older sibling checkout installed under the same package names;
# put THIS repo's src/ first so it wins (same precaution as identifiability/collect.py).
if SRC_ROOT in sys.path:
    sys.path.remove(SRC_ROOT)
sys.path.insert(0, SRC_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from m0_sensory_v1.model import LBFGSSettings  # noqa: E402
from m0_sensory_v1.model.bayesian import lapse_mixture  # noqa: E402
from m0_sensory_v1.model.m1_bayesian import m1_bayesian_choice_probability  # noqa: E402
from m0_sensory_v1.model.m1_session import m1_session_trials_from_frame  # noqa: E402
from m1_action_kernel_sliding_window_v1 import build_sliding_windows, prepare_sliding_data  # noqa: E402
from m1_action_kernel_v1.fit import fit_m1_action_kernel_single_initialization  # noqa: E402
from m1_action_kernel_v1.initialization import M1ActionKernelInitializationSpec  # noqa: E402
from m1_action_kernel_v1.parameters import decode_m1_action_kernel_raw_parameters  # noqa: E402
from m1_action_kernel_v1.registry import load_m1_action_kernel_model_registry  # noqa: E402
from m1_action_kernel_v1.session import evaluate_m1_action_kernel_mouse_objective  # noqa: E402

SUBJECTS = ("MC06", "MC07", "MC08")
DATA_ROOT = Path("/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior")
PARAM_NAMES = ("w", "s0", "ell_short", "ell_long", "kappa", "alpha")
OUT_ROOT = REPO_ROOT / "results_post_analysis" / "identifiability" / "action_history_param_recovery"
FIT_ROOT = REPO_ROOT / "results_sliding_window" / "M1_action_kernel_v1"
TRUTH_START_ID = 900  # warm start at the generating parameters (diagnostic only)
NLL_STUCK_TOLERANCE = 1e-6  # mean NLL per valid trial
SCHEMA = "action_history_param_recovery_v2"


# --------------------------------------------------------------------------- helpers
def registry_path(subject: str) -> Path:
    specific = REPO_ROOT / "config" / f"m1_action_kernel_v1_{subject.lower()}.json"
    return specific if specific.is_file() else REPO_ROOT / "config" / "m1_action_kernel_v1.json"


def winner_raw(subject: str, directory_name: str) -> list[float]:
    # MC06/MC07 are nested one level deeper (<mouse>/<model>/<mouse>/...); MC08 is flat.
    candidates = [FIT_ROOT / subject / "M1_action_kernel_v1" / subject / "full_session_matched_v1",
                  FIT_ROOT / subject / "full_session_matched_v1"]
    path = next(c / directory_name / "fit_result.json" for c in candidates
                if (c / directory_name / "fit_result.json").is_file())
    summary = json.loads(path.read_text(encoding="utf-8"))
    winner = next(r for r in summary["runs"] if r["start_id"] == summary["winner_start_id"])
    return [float(v) for v in winner["best_raw_parameters"]]


def decode(raw: list[float], registry):
    return decode_m1_action_kernel_raw_parameters(
        torch.tensor(raw, dtype=torch.float64),
        lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon)


def constrained(raw: list[float], registry) -> dict[str, float]:
    decoded = decode(raw, registry)
    return {name: float(getattr(decoded, name)) for name in PARAM_NAMES}


def lbfgs_settings(registry, steps: int | None) -> LBFGSSettings:
    f = registry.fitting
    return LBFGSSettings(
        maximum_steps=f.lbfgs_maximum_steps if steps is None else steps,
        learning_rate=f.lbfgs_learning_rate, history_size=f.lbfgs_history_size,
        maximum_evaluations_per_step=f.lbfgs_maximum_evaluations_per_step,
        tolerance_grad=f.lbfgs_tolerance_grad, tolerance_change=f.lbfgs_tolerance_change,
        line_search_fn=f.lbfgs_line_search_fn)


def truth_start(raw: list[float], registry) -> M1ActionKernelInitializationSpec:
    d = decode(raw, registry)
    return M1ActionKernelInitializationSpec(
        start_id=TRUTH_START_ID, w=float(d.w), s0=float(d.s0), ell_short=float(d.ell_short),
        ell_long=float(d.ell_long), kappa=float(d.kappa), a=float(raw[5]),
        purpose="recovery_diagnostic_warm_start_at_generating_parameters")


def simulate_session(session, raw: list[float], registry, generator: torch.Generator):
    """Sample choices from the frozen model, strictly causally (own sampled action drives pi).

    Mirrors the frozen evaluator's validity rules (src/m1_action_kernel_v1/session.py): a trial is
    evaluated when duration and choice are both valid; pi is updated on ANY valid binary action,
    even when the duration is invalid. For that last kind of trial the likelihood cannot be
    computed, so the REAL recorded action is kept (it still drives pi, exactly as in the real fit).
    Trials with an invalid real choice stay missing and leave pi unchanged.
    Returns (session, number_of_kept_real_actions).
    """
    parameters = decode(raw, registry)
    alpha = float(parameters.alpha)
    action_valid = (torch.isfinite(session.choice_long)
                    & ((session.choice_long == 0.0) | (session.choice_long == 1.0)))
    duration_valid = torch.isfinite(session.duration_s) & (session.duration_s > 0.0)
    pi = registry.initial_pi  # pi resets at every session start
    choices = torch.full_like(session.choice_long, float("nan"))
    kept_real = 0
    with torch.no_grad():
        for t in range(session.duration_s.numel()):
            if not bool(action_valid[t]):
                continue
            if bool(duration_valid[t]):
                prior = session.duration_s.new_tensor(pi)
                result = m1_bayesian_choice_probability(
                    session.duration_s[t], parameters.common(), registry.task_moments, prior,
                    prior_probability_epsilon=registry.prior_probability_epsilon,
                    quadratic_tolerance=registry.quadratic_tolerance,
                    discriminant_tolerance=registry.discriminant_tolerance)
                p = float(lapse_mixture(result.probability_long, session.category_long[t],
                                        parameters.common()))
                action = float(torch.rand((), generator=generator, dtype=torch.float64) < p)
            else:
                action = float(session.choice_long[t])
                kept_real += 1
            choices[t] = action
            pi = pi + alpha * (action - pi)
    return replace(session, choice_long=choices), kept_real


def nll(sessions, raw: list[float], registry) -> float:
    with torch.no_grad():
        result = evaluate_m1_action_kernel_mouse_objective(
            sessions, torch.tensor(raw, dtype=torch.float64), registry.task_moments,
            objective_weighting=registry.objective_weighting, required_split_role="train",
            lapse_max=registry.lapse_max, parameter_epsilon=registry.parameter_epsilon,
            log_probability_epsilon=registry.log_probability_epsilon,
            prior_probability_epsilon=registry.prior_probability_epsilon,
            quadratic_tolerance=registry.quadratic_tolerance,
            discriminant_tolerance=registry.discriminant_tolerance,
            initial_pi=registry.initial_pi)
    return float(result.loss)


def hash_seed(subject: str, window_id: int, rep: int) -> int:
    return zlib.crc32(f"{SCHEMA}|{subject}|{window_id}|{rep}".encode()) & 0x7FFFFFFF


def data_digest(sessions) -> str:
    h = hashlib.sha256()
    for s in sessions:
        for t in (s.duration_s, s.choice_long, s.category_long):
            h.update(t.detach().cpu().numpy().astype("float64").tobytes())
    return h.hexdigest()[:16]


def fingerprint(job: dict) -> str:
    """Identity of everything that determines a job's result except the replicate count."""
    registry_bytes = Path(job["registry"]).read_bytes()
    payload = json.dumps({
        "schema": SCHEMA, "start_ids": job["start_ids"], "lbfgs_steps": job["lbfgs_steps"],
        "true_raw": job["true_raw"], "registry_sha": hashlib.sha256(registry_bytes).hexdigest(),
        "sessions": job["session_ids"], "data_sha": job["data_sha"]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- worker
def _init_worker(threads: int) -> None:
    torch.set_num_threads(threads)


def _fit_record(run) -> dict:
    return {
        "start_id": run.start_id, "converged": bool(run.converged),
        "termination_reason": run.termination_reason, "exception_message": run.exception_message,
        "train_nll": float(run.recalculated_best_training_mean_nll),
        "gradient_norm": float(run.gradient_norm_at_restored_best),
        "steps": int(run.optimizer_step_count), "best_raw": [float(v) for v in run.best_raw_parameters],
    }


def run_job(job: dict) -> dict:
    """Simulate one dataset from ``job['true_raw']`` and refit it (blind starts + truth start)."""
    started = time.monotonic()
    registry = load_m1_action_kernel_model_registry(job["registry"])
    generator = torch.Generator().manual_seed(int(job["seed"]))
    simulated, kept = [], 0
    for s in job["sessions"]:
        sim, k = simulate_session(s, job["true_raw"], registry, generator)
        simulated.append(sim); kept += k
    simulated = tuple(simulated)
    nll_truth = nll(simulated, job["true_raw"], registry)
    settings = lbfgs_settings(registry, job["lbfgs_steps"])
    inits = {i.start_id: i for i in registry.initializations}
    specs = [inits[i] for i in job["start_ids"]] + [truth_start(job["true_raw"], registry)]
    fits = [_fit_record(fit_m1_action_kernel_single_initialization(
        simulated, (), registry, spec, optimizer_strategy="lbfgs", lbfgs_settings=settings))
        for spec in specs]
    blind = [f for f in fits if f["start_id"] != TRUTH_START_ID and np.isfinite(f["train_nll"])]
    converged = [f for f in blind if f["converged"]]
    pool = converged or blind
    best = min(pool, key=lambda f: f["train_nll"]) if pool else None
    tfit = next((f for f in fits if f["start_id"] == TRUTH_START_ID
                 and np.isfinite(f["train_nll"])), None)
    est = constrained(best["best_raw"], registry) if best else None
    spread = {}
    if len(converged) > 1:
        cs = [constrained(f["best_raw"], registry) for f in converged]
        spread = {n: float(max(c[n] for c in cs) - min(c[n] for c in cs)) for n in PARAM_NAMES}
    return {
        "schema": SCHEMA, "fingerprint": job["fingerprint"], "job_id": job["job_id"],
        "settings": {"start_ids": list(job["start_ids"]), "lbfgs_steps": job["lbfgs_steps"]},
        "subject": job["subject"], "window_id": job["window_id"], "rep": job["rep"],
        "seed": job["seed"], "true_raw": job["true_raw"], "true": constrained(job["true_raw"], registry),
        "est": est, "est_truthstart": constrained(tfit["best_raw"], registry) if tfit else None,
        "n_valid_choices": int(sum(int(torch.isfinite(s.choice_long).sum()) for s in simulated)),
        "kept_real_actions": kept, "nll_at_truth": nll_truth,
        "nll_fit": best["train_nll"] if best else None,
        "nll_truthstart": tfit["train_nll"] if tfit else None,
        "winner_converged": bool(converged), "best_start_id": best["start_id"] if best else None,
        "n_blind_converged": len(converged), "start_spread": spread,
        "fits": fits, "seconds": time.monotonic() - started,
    }


# --------------------------------------------------------------------------- job building
def build_jobs(args) -> list[dict]:
    jobs: list[dict] = []
    for subject in args.subjects:
        registry = load_m1_action_kernel_model_registry(registry_path(subject))
        prepared = prepare_sliding_data(DATA_ROOT / f"SCHR_{subject}", registry)
        windows = build_sliding_windows(prepared.assignments)
        for w in windows:
            if args.windows and w.window_id not in args.windows:
                continue
            raw = winner_raw(subject, w.directory_name)
            sessions = tuple(m1_session_trials_from_frame(prepared.frame(sid, "full", "train"),
                                                          split_role="train")
                             for sid in w.training_session_ids)
            digest = data_digest(sessions)
            for rep in range(args.reps):
                job = dict(data_sha=digest, job_id=f"{subject}_w{w.window_id:03d}_r{rep:03d}", subject=subject,
                           window_id=w.window_id, rep=rep, true_raw=raw, sessions=sessions,
                           session_ids=list(w.training_session_ids),
                           registry=str(registry_path(subject)),
                           seed=hash_seed(subject, w.window_id, rep),
                           start_ids=args.start_ids, lbfgs_steps=args.lbfgs_steps)
                job["fingerprint"] = fingerprint(job)
                jobs.append(job)
    # round-robin by replicate so a time-limited run covers every window evenly
    jobs.sort(key=lambda j: (j["rep"], j["subject"], j["window_id"]))
    return jobs


# --------------------------------------------------------------------------- driver
def _saved_fingerprint(path: Path) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("fingerprint")
    except (OSError, ValueError):
        return None


def run_all(args) -> None:
    jobs_dir = OUT_ROOT / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    todo = []
    for j in jobs:
        path = jobs_dir / f"{j['job_id']}.json"
        if not path.is_file():
            todo.append(j)
        elif _saved_fingerprint(path) != j["fingerprint"]:
            raise SystemExit(f"{path.name} was produced with different settings/truth/registry "
                             f"(fingerprint mismatch). Use a new --out-root or delete the old jobs.")
    print(f"{len(jobs)} jobs planned, {len(jobs) - len(todo)} already done, {len(todo)} to run",
          flush=True)
    if args.benchmark:
        job = dict(todo[0] if todo else jobs[0])
        t0 = time.monotonic(); res = run_job(job)
        print(f"BENCHMARK one dataset ({len(job['start_ids'])} blind starts + truth start): "
              f"{(time.monotonic() - t0) / 60:.1f} min")
        for f in res["fits"]:
            print(f"  start {f['start_id']}: steps={f['steps']} converged={f['converged']} "
                  f"nll={f['train_nll']:.6f} grad={f['gradient_norm']:.2e}")
        print(f"  nll_at_truth={res['nll_at_truth']:.6f}; alpha true={res['true']['alpha']:.3f} "
              f"est={res['est']['alpha'] if res['est'] else None}")
        print(f"  kept real actions on invalid-duration trials: {res['kept_real_actions']}")
        print(f"Full-run estimate: {len(todo)} jobs x ~{(time.monotonic() - t0) / 3600:.2f} h "
              f"/ {args.workers} workers = {len(todo) * (time.monotonic() - t0) / 3600 / args.workers:.1f} h")
        return
    deadline = time.monotonic() + args.budget_hours * 3600.0
    longest = args.expected_job_hours * 3600.0  # until real runtimes are seen; budget is soft
    pending: dict = {}
    queue = list(todo)
    seen_one = False
    # spawn (not fork): the parent has already used torch, and forking after that can hang
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.threads,),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        while queue or pending:
            while queue and len(pending) < args.workers and \
                    time.monotonic() + longest < deadline:
                job = queue.pop(0)
                pending[pool.submit(run_job, job)] = job["job_id"]
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job_id = pending.pop(future)
                try:
                    res = future.result()
                except Exception as error:  # record and continue; never lose the whole run
                    print(f"FAILED {job_id}: {type(error).__name__}: {error}", flush=True)
                    continue
                longest = max(longest if seen_one else 0.0, res["seconds"]); seen_one = True
                tmp = jobs_dir / f"{job_id}.json.tmp"
                tmp.write_text(json.dumps(res), encoding="utf-8")
                tmp.replace(jobs_dir / f"{job_id}.json")
                est = res["est"]["alpha"] if res["est"] else float("nan")
                print(f"done {job_id} in {res['seconds'] / 60:.1f} min "
                      f"alpha {res['true']['alpha']:.3f}->{est:.3f} "
                      f"converged={res['winner_converged']}", flush=True)
            if queue and time.monotonic() + longest >= deadline:
                print(f"time budget reached: {len(queue)} jobs not started", flush=True)
                queue = []


# --------------------------------------------------------------------------- analysis
def load_results() -> pd.DataFrame:
    rows, settings = [], set()
    for path in sorted((OUT_ROOT / "jobs").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if r.get("schema") != SCHEMA:
            continue
        settings.add(json.dumps(r["settings"], sort_keys=True))
        if r["est"] is None:  # every blind start failed: keep it so failures are counted
            rows.append(dict(job_id=r["job_id"], subject=r["subject"], window_id=r["window_id"],
                             rep=r["rep"], fit_failed=True,
                             **{f"true_{n}": r["true"][n] for n in PARAM_NAMES}))
            continue
        row = dict(fit_failed=False, job_id=r["job_id"], subject=r["subject"], window_id=r["window_id"],
                   rep=r["rep"], nll_fit=r["nll_fit"], nll_truthstart=r["nll_truthstart"],
                   nll_at_truth=r["nll_at_truth"], n_valid=r["n_valid_choices"],
                   winner_converged=r["winner_converged"], n_blind_converged=r["n_blind_converged"],
                   kept_real_actions=r["kept_real_actions"], seconds=r["seconds"])
        # blind winner clearly worse than the warm start at the truth -> optimiser stuck,
        # which is an optimisation failure, not an identifiability statement
        # NaN (unknown) when the truth-start diagnostic itself failed
        row["blind_stuck"] = (float(r["nll_fit"] > r["nll_truthstart"] + NLL_STUCK_TOLERANCE)
                              if r["nll_truthstart"] is not None else np.nan)
        for name in PARAM_NAMES:
            row[f"true_{name}"] = r["true"][name]
            row[f"est_{name}"] = r["est"][name]
            row[f"estT_{name}"] = r["est_truthstart"][name] if r["est_truthstart"] else np.nan
            row[f"spread_{name}"] = r["start_spread"].get(name, np.nan)
        rows.append(row)
    if len(settings) > 1:
        raise SystemExit(f"jobs in {OUT_ROOT} were run with different settings {sorted(settings)}; "
                         "use a separate --out-root per configuration")
    return pd.DataFrame(rows)


def window_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per subject x window x parameter: bias, replicate SD, RMSE (the per-window trust table)."""
    rows = []
    for (subject, wid), g_all in df.groupby(["subject", "window_id"]):
        g = g_all[~g_all.fit_failed]
        if g.empty:  # every replicate failed: keep the window, with NaN recovery metrics
            for name in PARAM_NAMES:
                rows.append(dict(subject=subject, window_id=wid, parameter=name, n_reps=0,
                                 frac_fit_failed=1.0, truth=g_all[f"true_{name}"].iloc[0]))
            continue
        for name in PARAM_NAMES:
            err = g[f"est_{name}"] - g[f"true_{name}"]
            rows.append(dict(
                subject=subject, window_id=wid, parameter=name, n_reps=len(g),
                frac_fit_failed=float(g_all.fit_failed.mean()),
                truth=g[f"true_{name}"].iloc[0], est_mean=g[f"est_{name}"].mean(),
                bias=err.mean(), rep_sd=g[f"est_{name}"].std(ddof=1) if len(g) > 1 else np.nan,
                rmse=float(np.sqrt((err ** 2).mean())),
                frac_winner_converged=g.winner_converged.mean(),
                frac_blind_stuck=g.blind_stuck.mean(),  # NaN-skipping: unknown diagnostics excluded
                mean_start_spread=g[f"spread_{name}"].mean()))
    return pd.DataFrame(rows)


def summarize(wt: pd.DataFrame) -> pd.DataFrame:
    """Per subject x parameter, from the per-window table. 'reliability-like' numbers are
    descriptive: truths are noisy fitted values, errors are heteroskedastic, and bias depends on
    the truth, so no additive-noise reliability claim is made."""
    rows = []
    for (subject, name), g_all in wt.groupby(["subject", "parameter"]):
        g = g_all[g_all.n_reps > 0].sort_values("window_id")
        if g.empty:
            continue
        t, m = g.truth.to_numpy(), g.est_mean.to_numpy()
        sd_t = float(np.std(t, ddof=1)) if len(g) > 1 else np.nan
        ok_corr = len(g) >= 3 and np.std(t) > 0 and np.std(m) > 0
        ok_slope = len(g) >= 2 and np.std(t) > 0
        # adjacent-window change (independent simulation => optimistic about change noise)
        adj = np.diff(g.window_id.to_numpy()) == 1
        dt, dm = np.diff(t)[adj], np.diff(m)[adj]
        rows.append(dict(
            subject=subject, parameter=name, n_windows=len(g), reps_per_window=g.n_reps.median(),
            truth_sd_across_windows=sd_t,
            corr_level=float(np.corrcoef(t, m)[0, 1]) if ok_corr else np.nan,
            calibration_slope=float(np.polyfit(t, m, 1)[0]) if ok_slope else np.nan,
            mean_bias=float(g.bias.mean()), pooled_rmse=float(np.sqrt((g.rmse ** 2).mean())),
            mean_rep_sd=float(g.rep_sd.mean()),
            rmse_over_truth_sd=float(np.sqrt((g.rmse ** 2).mean()) / sd_t) if sd_t > 0 else np.nan,
            corr_adjacent_change=float(np.corrcoef(dt, dm)[0, 1])
            if len(dt) >= 3 and np.std(dt) > 0 and np.std(dm) > 0 else np.nan,
            rmse_adjacent_change=float(np.sqrt(np.mean((dm - dt) ** 2))) if len(dt) else np.nan,
            frac_fit_failed=float(g_all.frac_fit_failed.mean()),
            frac_winner_converged=float(g.frac_winner_converged.mean()),
            frac_blind_stuck=float(g.frac_blind_stuck.mean())))
    return pd.DataFrame(rows)


def make_figures(df: pd.DataFrame, wt: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    sys.path.insert(0, str(REPO_ROOT / "post_analysis" / "expanded_model_comparison"))
    from action_history_param_means import COLORS, DESC  # same colours, names and hints

    subjects = [s for s in SUBJECTS if s in set(df.subject)]
    titles = {n: DESC[n][0].replace("\n", " ") for n in PARAM_NAMES}

    def style(ax):
        ax.grid(False); ax.spines[["top", "right"]].set_visible(False)

    with PdfPages(OUT_ROOT / "action_history_param_recovery.pdf") as pdf:
        # page 1: recovered vs true, every replicate (faint) and per-window mean +- SD
        fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
        for ax, name in zip(axes.flat, PARAM_NAMES):
            for s in subjects:
                d, w = df[df.subject.eq(s)], wt[(wt.subject.eq(s)) & (wt.parameter.eq(name))]
                ax.scatter(d[f"true_{name}"], d[f"est_{name}"], s=8, alpha=.25, color=COLORS[s])
                ax.errorbar(w.truth, w.est_mean, yerr=w.rep_sd.fillna(0), fmt="o", ms=4,
                            color=COLORS[s], alpha=.9, lw=.8, label=s)
            lo = min(df[f"true_{name}"].min(), df[f"est_{name}"].min())
            hi = max(df[f"true_{name}"].max(), df[f"est_{name}"].max())
            pad = .05 * (hi - lo or 1)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.5", lw=1.2)
            ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
            s_ = summary[summary.parameter.eq(name)]
            ax.set_title(f"{titles[name]}\nmean r={s_.corr_level.mean():.2f}, "
                         f"slope={s_.calibration_slope.mean():.2f}, "
                         f"RMSE/truth SD={s_.rmse_over_truth_sd.mean():.2f}", fontsize=9)
            ax.set_xlabel("true (generating) value", fontsize=8)
            ax.set_ylabel("recovered value", fontsize=8); style(ax)
        axes.flat[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Recovery per window: dots = window mean +- replicate SD, faint = replicates, "
                     "grey = perfect")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # pages 2..: trajectories across windows, truth vs recovered (does the trend survive?)
        for s in subjects:
            fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True)
            for ax, name in zip(axes.flat, PARAM_NAMES):
                w = wt[(wt.subject.eq(s)) & (wt.parameter.eq(name))].sort_values("window_id")
                ax.fill_between(w.window_id, w.est_mean - w.rep_sd.fillna(0),
                                w.est_mean + w.rep_sd.fillna(0), color=COLORS[s], alpha=.25, lw=0)
                ax.plot(w.window_id, w.est_mean, "-o", ms=3, color=COLORS[s], label="recovered mean +- SD")
                ax.plot(w.window_id, w.truth, "-", color="k", lw=1.4, label="true (generating)")
                ax.set_title(titles[name], fontsize=9); ax.set_xlabel("window", fontsize=8); style(ax)
            axes.flat[0].legend(frameon=False, fontsize=8)
            fig.suptitle(f"{s}: generating vs recovered parameters across windows")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # error correlations (trade-offs)
        errs = pd.DataFrame({n: df[f"est_{n}"] - df[f"true_{n}"] for n in PARAM_NAMES})
        corr = errs.corr().to_numpy()
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(6), PARAM_NAMES, rotation=40, ha="right"); ax.set_yticks(range(6), PARAM_NAMES)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=.8)
        ax.set_title("Correlation of recovery errors (strong off-diagonal = trade-off)", fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # numbers
        fig = plt.figure(figsize=(13, 8.5)); fig.text(.03, .97, "Summary", va="top", fontsize=12)
        text = summary.round(3).to_string(index=False)
        text += (f"\n\n{len(df)} refitted datasets over {df.groupby(['subject', 'window_id']).ngroups} windows; "
                 f"blind winner not converged: {int((~df.winner_converged).sum())}; "
                 f"blind winner stuck in a worse optimum than the truth start: {int(df.blind_stuck.sum())}."
                 "\n*_adjacent_change compare replicate-MEAN differences from INDEPENDENT window simulations;"
                 " real adjacent windows share 8 of 9 sessions, so real change noise is not assessed."
                 "\nrmse_over_truth_sd is descriptive only (truths are noisy fitted values).")
        fig.text(.03, .93, text, va="top", family="monospace", fontsize=6)
        pdf.savefig(fig); plt.close(fig)


def analyze() -> None:
    df = load_results()
    if df.empty:
        print("no finished jobs found"); return
    df.to_csv(OUT_ROOT / "recovery_results.csv", index=False)
    n_failed = int(df.fit_failed.sum())
    if n_failed == len(df):
        print(f"all {n_failed} datasets failed to fit"); return
    wt = window_table(df)  # sees the failed rows so per-window failure fractions are counted
    df = df[~df.fit_failed].astype({"winner_converged": bool})
    wt.to_csv(OUT_ROOT / "window_param_recovery.csv", index=False)
    summary = summarize(wt)
    summary.to_csv(OUT_ROOT / "recovery_summary.csv", index=False)
    make_figures(df, wt, summary)
    print(summary.round(3).to_string(index=False))
    print(f"datasets: {len(df)} (+{n_failed} with every blind start failed); winner not converged: {int((~df.winner_converged).sum())}; "
          f"blind stuck: {int(df.blind_stuck.sum())} "
          f"(unknown: {int(df.blind_stuck.isna().sum())}); "
          f"kept real actions on invalid-duration trials (total): {int(df.kept_real_actions.sum())}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    p.add_argument("--windows", nargs="+", type=int, default=None, help="restrict to these window ids")
    p.add_argument("--reps", type=int, default=10, help="simulated datasets per window")
    p.add_argument("--start-ids", type=int, nargs="+", default=[2, 7], help="blind registry starts")
    p.add_argument("--lbfgs-steps", type=int, default=None, help="default: registry value (250)")
    p.add_argument("--workers", type=int, default=36)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--budget-hours", type=float, default=22.0,
                   help="soft: no job is started that is expected to end after this many hours")
    p.add_argument("--expected-job-hours", type=float, default=4.0,
                   help="runtime assumed for admission until real jobs have finished")
    p.add_argument("--out-root", type=Path, default=None, help="override output folder")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    return p.parse_args()


def main() -> None:
    global OUT_ROOT
    args = parse_args()
    if args.out_root is not None:
        OUT_ROOT = args.out_root
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        run_all(args)
        if args.benchmark:
            return
    analyze()


if __name__ == "__main__":
    main()
