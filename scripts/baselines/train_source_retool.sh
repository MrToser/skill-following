#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail
source "$(dirname "$0")/common.sh"

num_gpus="${NUM_GPUS:-4}"
tp="${TP:-2}"
seed="${SEED:-20260818}"
model_key="${BASELINE_MODEL_KEY:-qwen35_4b}"
resolve_baseline_model "$model_key"
model_slug="${model_key//_/-}"
checkpoint="${CHECKPOINT:-$baseline_root/data/checkpoints/source_retool_${model_key}}"
default_artifact_root="$baseline_root/data/artifacts/source_retool_${model_key}"
default_wandb_dir="$baseline_root/data/wandb_offline/source_retool_${model_key}"
artifact_root="${ARTIFACT_ROOT:-$default_artifact_root}"
wandb_dir="${WANDB_DIR:-$default_wandb_dir}"
train_data="$baseline_root/data/prepared/source_retool/train_math_gsm8k_14973.jsonl"
monitor_data="$baseline_root/data/prepared/source_retool/monitor_math_gsm8k_500.jsonl"
full_config="$baseline_root/data/prepared/source_retool/full_eval_config.yaml"
custom_generate_function_path="generate_with_retool.generate"
base_train="$baseline_root/scripts/baselines/retool_train_impl.sh"
model_config="$BASELINE_MODEL_CONFIG"
hf_checkpoint="$BASELINE_HF_CHECKPOINT"
ref_load="$BASELINE_REF_LOAD"
require_paths "$base_train" "$model_config" "$hf_checkpoint" "$ref_load" "$root_path/examples/retool/generate_with_retool.py"

activate_env harnessr1
configure_cuda_runtime
prepare_baseline_data
if [[ "$model_key" == "qwen35_4b" ]]; then
  "$python_bin" -m skill_following_baselines.data \
    --root "$baseline_root" \
    --retool-processor-view-only
  train_data="$baseline_root/data/prepared/source_retool_messages/train_math_gsm8k_14973.jsonl"
  monitor_data="$baseline_root/data/prepared/source_retool_messages/monitor_math_gsm8k_500.jsonl"
  full_config="$baseline_root/data/prepared/source_retool_messages/full_eval_config.yaml"
  custom_generate_function_path="skill_following_baselines.source_retool.generate"
fi
require_paths "$train_data" "$monitor_data" "$full_config"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[READY] Source ReTool generator=$custom_generate_function_path"
  echo "[READY] model=$BASELINE_MODEL_DISPLAY_NAME GPUs=$num_gpus TP=$tp rollout=200 batch=72 samples=5 global_batch=72"
  echo "[READY] checkpoint=$checkpoint"
  echo "[READY] artifact_root=$artifact_root"
  exit 0
fi
mkdir -p "$checkpoint" "$artifact_root" "$wandb_dir"

ROOT_PATH="$root_path" \
TRAIN_DATA="$train_data" \
EVAL_MATH_DATA="$monitor_data" \
AUTO_PREPARE_DATA=0 \
MODEL_CONFIG="$model_config" \
HF_CHECKPOINT="$hf_checkpoint" \
REF_LOAD="$ref_load" \
SAVE_PATH="$checkpoint" \
NUM_GPUS="$num_gpus" \
TP="$tp" \
ROLLOUT_NUM_GPUS_PER_ENGINE="$tp" \
NUM_ROLLOUT=200 \
ROLLOUT_BATCH_SIZE=72 \
N_SAMPLES_PER_PROMPT=5 \
GLOBAL_BATCH_SIZE=72 \
EVAL_INTERVAL=25 \
N_SAMPLES_PER_EVAL_PROMPT=1 \
FINAL_EVAL_N_SAMPLES_PER_PROMPT=0 \
SAVE_INTERVAL=200 \
SAVE_ONLY_AT_END=1 \
SAVE_MODEL_ONLY=1 \
APPLY_CHAT_TEMPLATE=0 \
CUSTOM_GENERATE_FUNCTION_PATH="$custom_generate_function_path" \
CUSTOM_RM_PATH=generate_with_retool.reward_func \
WANDB_MODE=offline \
WANDB_DIR="$wandb_dir" \
WANDB_GROUP="baseline-source-retool-${model_slug}" \
EVAL_CONFIG_PATH="$baseline_root/data/runtime/source_retool_monitor_eval_config.yaml" \
bash "$base_train" 2>&1 | tee "$artifact_root/train.log"

activate_env harnessr1
configure_cuda_runtime
source "$model_config"
export BASELINE_CONDITION=source_retool
export BASELINE_EVAL_TRACE_PATH="$artifact_root/full_eval_trace.jsonl"
export BASELINE_EVAL_SUMMARY_PATH="$artifact_root/summary.json"
runtime_env="{\"env_vars\":{\"PYTHONPATH\":\"$PYTHONPATH\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"CUDA_HOME\":\"$CUDA_HOME\",\"CUDA_PATH\":\"$CUDA_PATH\",\"LD_LIBRARY_PATH\":\"$LD_LIBRARY_PATH\",\"TILELANG_CACHE_DIR\":\"$TILELANG_CACHE_DIR\",\"TILELANG_TMP_DIR\":\"$TILELANG_TMP_DIR\",\"BASELINE_MODEL_KEY\":\"$model_key\",\"BASELINE_CONDITION\":\"$BASELINE_CONDITION\",\"BASELINE_EVAL_TRACE_PATH\":\"$BASELINE_EVAL_TRACE_PATH\",\"BASELINE_EVAL_SUMMARY_PATH\":\"$BASELINE_EVAL_SUMMARY_PATH\"}}"
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="$runtime_env" \
  -- python3 "$root_path/train.py" \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$num_gpus" --rollout-num-gpus "$num_gpus" --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$hf_checkpoint" --ref-load "$ref_load" --load "$checkpoint" --no-load-optim --no-load-rng \
  --prompt-data "$train_data" --input-key prompt --label-key label --metadata-key metadata \
  --num-rollout 0 --rollout-batch-size 72 --n-samples-per-prompt 1 --rollout-max-prompt-len 4096 \
  --rollout-max-response-len 8192 --eval-interval 1 --eval-config "$full_config" --eval-input-key prompt \
  --eval-label-key label --n-samples-per-eval-prompt 1 --eval-max-prompt-len 4096 \
  --eval-max-response-len 8192 --global-batch-size 72 --optimizer adam --lr 1e-6 \
  --lr-decay-style constant --lr-decay-iters 1 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28 --tensor-model-parallel-size "$tp" \
  --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 --use-dynamic-batch-size \
  --max-tokens-per-gpu 9216 --rollout-num-gpus-per-engine "$tp" --sglang-mem-fraction-static 0.70 \
  --attention-dropout 0.0 --hidden-dropout 0.0 --attention-backend flash \
  --use-wandb --wandb-mode offline --wandb-dir "$wandb_dir" \
  --wandb-team "${WANDB_TEAM:-anonymous}" --wandb-project "${WANDB_PROJECT:-grounded-skill-following}" \
  --wandb-group "baseline-source-retool-${model_slug}-full-eval" \
  --custom-generate-function-path "$custom_generate_function_path" \
  --custom-rm-path skill_following_baselines.metrics.reward_func \
  --custom-eval-rollout-log-function-path skill_following_baselines.metrics.log_eval_rollout_data \
  2>&1 | tee "$artifact_root/full_eval.log"
