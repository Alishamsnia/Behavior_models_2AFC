#!/usr/bin/env bash
# Fit the five new models for every subject with the standard 32-thread layout:
#   4 windows in parallel x 8 starts per window x 1 Torch thread = 32 threads
#   (rolling stage), then 32 one-thread single-session fine-tunes.
# Model/mouse jobs run ONE AT A TIME: each runner enforces the 32-CPU budget
# only for itself, so running two jobs concurrently would use 64 threads.
#
# Usage (from anywhere):
#   bash scripts/run_five_new_models.sh                        # all models, all mice
#   bash scripts/run_five_new_models.sh fit  "MC06"            # all models, one mouse
#   bash scripts/run_five_new_models.sh fit  "MC06 MC07" "pure_action extended_action"
#   bash scripts/run_five_new_models.sh force                  # phase 2 (see below)
#
# Phase 1 ("fit", default): sliding windows + single-session fine-tuning with
#   --resume. A window whose eight starts do not converge is saved as
#   status=no_winner and the run continues with the other windows.
# Phase 2 ("force"): for every model/mouse, list the no-winner windows, stage
#   the lowest-training-NLL start for them with
#   extra_tools/forced_best_train_nll.py (no retraining), install, and re-run
#   the runner with --resume so the newly resolved windows' test sessions get
#   their single-session fine-tunes. Inspect the staging tree before phase 2's
#   install if you want to review: set FORCE_INSTALL=0 to stop after staging.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DATA_ROOT="${DATA_ROOT:-/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior}"
PYTHON="${PYTHON:-python}"
PHASE="${1:-fit}"
MICE="${2:-MC06 MC07 MC08}"
MODELS="${3:-pure_action extended_action m1_previous_action_bias m1_previous_wsls_bias m2_dynamic_action_kernel_bias}"
FORCE_INSTALL="${FORCE_INSTALL:-1}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# model key -> results directory (= registry model_id) and forced-best runner key
declare -A MODEL_ID=(
  [pure_action]=Pure_action_v1
  [extended_action]=Extended_action_v1
  [m1_previous_action_bias]=M1_previous_action_bias_v1
  [m1_previous_wsls_bias]=M1_previous_wsls_bias_v1
  [m2_dynamic_action_kernel_bias]=M2_dynamic_action_kernel_bias_v1
)
declare -A FORCE_KEY=(
  [pure_action]=pure_action
  [extended_action]=extended_action
  [m1_previous_action_bias]=previous_action_bias
  [m1_previous_wsls_bias]=previous_wsls_bias
  [m2_dynamic_action_kernel_bias]=action_kernel_bias
)
# MC08 uses the base registry; MC06/MC07 use their subject overlays.
config_for() {
  local model="$1" mouse="$2"
  case "$mouse" in
    MC08) echo "config/${model}_v1.json" ;;
    *)    echo "config/${model}_v1_$(echo "$mouse" | tr '[:upper:]' '[:lower:]').json" ;;
  esac
}

# Optimizer flags: identical in phase 1 and phase 2 (the forced tool refuses a
# window whose saved optimizer settings differ).
OPTIMIZER_ARGS=(--optimizer lbfgs)
CPU_ARGS=(
  --window-parallel-workers 4
  --rolling-workers 8
  --rolling-threads-per-worker 1
  --single-session-workers 32
  --single-session-threads-per-worker 1
)

run_fit() {
  local model="$1" mouse="$2"
  local model_id="${MODEL_ID[$model]}"
  local log_dir="results_sliding_window/${model_id}/${mouse}"
  mkdir -p "$log_dir"
  echo "=== fit ${model_id} ${mouse} $(date '+%F %T') ==="
  "$PYTHON" -u "scripts/run_${model}_sliding_window.py" \
    --raw-data-root "${DATA_ROOT}/SCHR_${mouse}" \
    --registry "$(config_for "$model" "$mouse")" \
    "${OPTIMIZER_ARGS[@]}" "${CPU_ARGS[@]}" \
    --resume \
    2>&1 | tee -a "${log_dir}/full_run.log"
}

run_force() {
  local model="$1" mouse="$2"
  local model_id="${MODEL_ID[$model]}"
  local windows
  windows="$("$PYTHON" extra_tools/list_no_winner.py --model "$model_id" --mouse "$mouse")"
  if [[ -z "$windows" ]]; then
    # Nothing left to force, but still resume: if an earlier force run
    # installed windows and was interrupted before its fine-tunes finished,
    # this completes them (finished windows/sessions are resume-skipped).
    echo "=== force ${model_id} ${mouse}: no no-winner windows; resuming fine-tunes ==="
    run_fit "$model" "$mouse"
    return 0
  fi
  # Fresh staging folder per invocation: --install installs every staged
  # window it finds, so a reused folder would retry already-installed ones.
  local staging="extra_tools_output/forced_best_staging/${model_id}_${mouse}_$(date +%Y%m%d_%H%M%S)"
  echo "=== force ${model_id} ${mouse}: windows ${windows} ==="
  # shellcheck disable=SC2086
  "$PYTHON" extra_tools/forced_best_train_nll.py \
    --runner "${FORCE_KEY[$model]}" --windows $windows \
    --original-sliding-dir "$REPO/results_sliding_window" \
    --staging-dir "$REPO/$staging" \
    -- --raw-data-root "${DATA_ROOT}/SCHR_${mouse}" \
       --registry "$(config_for "$model" "$mouse")" \
       "${OPTIMIZER_ARGS[@]}" "${CPU_ARGS[@]}" \
       --sliding-only --resume
  if [[ "$FORCE_INSTALL" != "1" ]]; then
    echo "staged only (FORCE_INSTALL=0): inspect $staging, then re-run with FORCE_INSTALL=1"
    return 0
  fi
  "$PYTHON" extra_tools/forced_best_train_nll.py --install \
    --original-sliding-dir "$REPO/results_sliding_window" \
    --staging-dir "$REPO/$staging"
  # Re-run with --resume: complete windows are skipped, and the test sessions
  # of the newly forced windows now get their single-session fine-tunes.
  run_fit "$model" "$mouse"
}

for mouse in $MICE; do
  for model in $MODELS; do
    [[ -n "${MODEL_ID[$model]:-}" ]] || { echo "unknown model key: $model" >&2; exit 2; }
    case "$PHASE" in
      fit)   run_fit "$model" "$mouse" ;;
      force) run_force "$model" "$mouse" ;;
      *)     echo "unknown phase: $PHASE (use fit or force)" >&2; exit 2 ;;
    esac
  done
done
echo "=== ${PHASE} finished $(date '+%F %T') ==="
