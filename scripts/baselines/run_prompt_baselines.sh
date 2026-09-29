#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail
source "$(dirname "$0")/common.sh"

num_gpus="${NUM_GPUS:-4}"
tp="${TP:-2}"
seed="${SEED:-20260818}"
model_key="${BASELINE_MODEL_KEY:-qwen35_4b}"
run_direct_qa="${RUN_DIRECT_QA:-1}"
run_preloaded_skill="${RUN_PRELOADED_SKILL:-1}"
resolve_baseline_model "$model_key"
model_display_name="$BASELINE_MODEL_DISPLAY_NAME"
model_config="$BASELINE_MODEL_CONFIG"
hf_checkpoint="$BASELINE_HF_CHECKPOINT"
ref_load="$BASELINE_REF_LOAD"
artifact_root="${ARTIFACT_ROOT:-$baseline_root/data/artifacts/prompt_baselines/$model_key}"
wandb_dir="${WANDB_DIR:-$baseline_root/data/wandb_offline/prompt_baselines/$model_key}"
require_paths "$model_config" "$hf_checkpoint" "$ref_load"
[[ "$num_gpus" -gt 0 && "$tp" -gt 0 && $((num_gpus % tp)) -eq 0 ]] || {
  echo "[ERROR] NUM_GPUS=$num_gpus must be divisible by TP=$tp" >&2
  exit 1
}
for enabled in "$run_direct_qa" "$run_preloaded_skill"; do
  [[ "$enabled" == "0" || "$enabled" == "1" ]] || {
    echo "[ERROR] RUN_DIRECT_QA and RUN_PRELOADED_SKILL must be 0 or 1" >&2
    exit 1
  }
done
[[ "$run_direct_qa" == "1" || "$run_preloaded_skill" == "1" ]] || {
  echo "[ERROR] at least one prompt-baseline condition must be enabled" >&2
  exit 1
}

activate_env harnessr1
configure_cuda_runtime
prepare_baseline_data
"$python_bin" -m skill_following_baselines.self_test
mkdir -p "$artifact_root" "$wandb_dir"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[READY] prompt baselines: direct_qa=$run_direct_qa preloaded_skill=$run_preloaded_skill"
  echo "[READY] model=$model_display_name key=$model_key GPUs=$num_gpus TP=$tp datasets=2 Math + 7 Search"
  echo "[READY] artifact_root=$artifact_root"
  exit 0
fi
trap stop_retriever EXIT
start_retriever "$artifact_root"
activate_env harnessr1
configure_cuda_runtime
start_ray "$num_gpus"
source "$model_config"

run_one() {
  local condition="$1"
  local task_mode="$2"
  local eval_config="$3"
  local custom_generate="$4"
  local condition_dir="$artifact_root/$condition/$task_mode"
  mkdir -p "$condition_dir"
  export BASELINE_CONDITION="$condition"
  export BASELINE_EVAL_TRACE_PATH="$condition_dir/trace.jsonl"
  export BASELINE_EVAL_SUMMARY_PATH="$condition_dir/summary.json"
  export SF_EXP_ID="${condition}_${model_key}"
  export SF_EXP_TASK_MODE="$task_mode"
  export SF_EXP_SKILL_VARIANT=full
  export SF_EXP_STATE_AWARE_HINTS=1
  export SF_EXP_TRANSITION_CREDIT=0
  export SF_EXP_RECOVERY_FACTOR=0.9
  export SF_EXP_OBSERVATION_MODE=normal
  export SF_EXP_SEED="$seed"
  local runtime_env
  runtime_env="{\"env_vars\":{\"PYTHONPATH\":\"$PYTHONPATH\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"CUDA_HOME\":\"$CUDA_HOME\",\"CUDA_PATH\":\"$CUDA_PATH\",\"LD_LIBRARY_PATH\":\"$LD_LIBRARY_PATH\",\"TILELANG_CACHE_DIR\":\"$TILELANG_CACHE_DIR\",\"TILELANG_TMP_DIR\":\"$TILELANG_TMP_DIR\",\"BASELINE_MODEL_KEY\":\"$model_key\",\"BASELINE_CONDITION\":\"$BASELINE_CONDITION\",\"BASELINE_EVAL_TRACE_PATH\":\"$BASELINE_EVAL_TRACE_PATH\",\"BASELINE_EVAL_SUMMARY_PATH\":\"$BASELINE_EVAL_SUMMARY_PATH\",\"SF_EXP_ID\":\"$SF_EXP_ID\",\"SF_EXP_TASK_MODE\":\"$SF_EXP_TASK_MODE\",\"SF_EXP_SKILL_VARIANT\":\"full\",\"SF_EXP_STATE_AWARE_HINTS\":\"1\",\"SF_EXP_TRANSITION_CREDIT\":\"0\",\"SF_EXP_RECOVERY_FACTOR\":\"0.9\",\"SF_EXP_OBSERVATION_MODE\":\"normal\",\"SF_EXP_SEED\":\"$seed\",\"UNIFIED_SEARCH_URL\":\"$UNIFIED_SEARCH_URL\"}}"
  local custom_args=(
    --custom-rm-path skill_following_baselines.metrics.reward_func
    --custom-eval-rollout-log-function-path skill_following_baselines.metrics.log_eval_rollout_data
  )
  if [[ -n "$custom_generate" ]]; then
    custom_args+=(--custom-generate-function-path "$custom_generate")
  fi
  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="$runtime_env" \
    -- python3 "$root_path/train.py" \
    --actor-num-nodes 1 --actor-num-gpus-per-node "$num_gpus" --rollout-num-gpus "$num_gpus" --colocate \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$hf_checkpoint" --ref-load "$ref_load" --no-load-optim --no-load-rng \
    --prompt-data "$baseline_root/data/prepared/direct_qa/eval_math.parquet" \
    --input-key prompt --label-key label --metadata-key extra_info --apply-chat-template \
    --reward-key score --num-rollout 0 --rollout-batch-size 128 --n-samples-per-prompt 1 \
    --rollout-max-prompt-len 4096 --rollout-max-response-len 8192 --rollout-temperature 1 \
    --eval-interval 1 --eval-config "$eval_config" --eval-input-key prompt --eval-label-key label \
    --n-samples-per-eval-prompt 1 --eval-max-prompt-len 4096 --eval-max-response-len 8192 \
    --global-batch-size 128 --seed "$seed" --optimizer adam --lr 1e-6 --lr-decay-style constant \
    --lr-decay-iters 1 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28 \
    --tensor-model-parallel-size "$tp" --sequence-parallel --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
    --use-dynamic-batch-size --max-tokens-per-gpu 16384 --rollout-num-gpus-per-engine "$tp" \
    --sglang-mem-fraction-static 0.60 --attention-dropout 0.0 --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash \
    --use-wandb --wandb-mode offline --wandb-dir "$wandb_dir" \
    --wandb-team "${WANDB_TEAM:-anonymous}" --wandb-project "${WANDB_PROJECT:-grounded-skill-following}" \
    --wandb-group "baseline-${model_key}-${condition}-${task_mode}" \
    "${custom_args[@]}" 2>&1 | tee "$condition_dir/eval.log"
}

if [[ "$run_direct_qa" == "1" ]]; then
  run_one direct_qa math "$baseline_root/data/prepared/direct_qa/math_eval_config.yaml" ""
  run_one direct_qa search "$baseline_root/data/prepared/direct_qa/search_eval_config.yaml" ""
fi
if [[ "$run_preloaded_skill" == "1" ]]; then
  run_one preloaded_skill math "$baseline_root/data/prepared/preloaded_skill/math_eval_config.yaml" skill_following_baselines.preloaded_skill.generate
  run_one preloaded_skill search "$baseline_root/data/prepared/preloaded_skill/search_eval_config.yaml" skill_following_baselines.preloaded_skill.generate
fi
