#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail
source "$(dirname "$0")/common.sh"

num_gpus="${NUM_GPUS:-2}"
tp="${TP:-2}"
model_key="${BASELINE_MODEL_KEY:-qwen35_4b}"
resolve_baseline_model "$model_key"
model_slug="${model_key//_/-}"
checkpoint="${CHECKPOINT:?CHECKPOINT is required}"
artifact_root="${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}"
wandb_dir="${WANDB_DIR:-$baseline_root/data/wandb_offline/source_retool_${model_key}_recovery_eval}"
train_data="$baseline_root/data/prepared/source_retool/train_math_gsm8k_14973.jsonl"
full_config="$baseline_root/data/prepared/source_retool/full_eval_config.yaml"
custom_generate_function_path="generate_with_retool.generate"

if [[ "$model_key" == "qwen35_4b" ]]; then
  train_data="$baseline_root/data/prepared/source_retool_messages/train_math_gsm8k_14973.jsonl"
  full_config="$baseline_root/data/prepared/source_retool_messages/full_eval_config.yaml"
  custom_generate_function_path="skill_following_baselines.source_retool.generate"
fi

require_paths "$BASELINE_MODEL_CONFIG" "$BASELINE_HF_CHECKPOINT" "$BASELINE_REF_LOAD" \
  "$checkpoint" "$train_data" "$full_config"
assert_checkpoint_marker="$checkpoint/latest_checkpointed_iteration.txt"
require_paths "$assert_checkpoint_marker"
[[ "$num_gpus" -eq 2 && "$tp" -eq 2 ]] || {
  echo "[ERROR] recovery evaluation requires NUM_GPUS=2 and TP=2" >&2
  exit 1
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[READY] Source ReTool eval model=$BASELINE_MODEL_DISPLAY_NAME GPUs=$num_gpus TP=$tp"
  echo "[READY] checkpoint=$checkpoint"
  echo "[READY] eval_config=$full_config"
  echo "[READY] artifact_root=$artifact_root"
  exit 0
fi

# 数据：已完成训练的 Source ReTool checkpoint 与完整 Math/GSM8K 评测集。算法：只加载模型并运行 pass@1，不恢复训练状态。
mkdir -p "$artifact_root" "$wandb_dir"
activate_env harnessr1
configure_cuda_runtime
"$python_bin" -m skill_following_baselines.self_test
source "$BASELINE_MODEL_CONFIG"
start_ray "$num_gpus"

export BASELINE_CONDITION=source_retool
export BASELINE_MODEL_KEY="$model_key"
export BASELINE_EVAL_TRACE_PATH="$artifact_root/full_eval_trace.jsonl"
export BASELINE_EVAL_SUMMARY_PATH="$artifact_root/summary.json"
runtime_env="{\"env_vars\":{\"PYTHONPATH\":\"$PYTHONPATH\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"CUDA_HOME\":\"$CUDA_HOME\",\"CUDA_PATH\":\"$CUDA_PATH\",\"LD_LIBRARY_PATH\":\"$LD_LIBRARY_PATH\",\"TILELANG_CACHE_DIR\":\"$TILELANG_CACHE_DIR\",\"TILELANG_TMP_DIR\":\"$TILELANG_TMP_DIR\",\"BASELINE_MODEL_KEY\":\"$model_key\",\"BASELINE_CONDITION\":\"$BASELINE_CONDITION\",\"BASELINE_EVAL_TRACE_PATH\":\"$BASELINE_EVAL_TRACE_PATH\",\"BASELINE_EVAL_SUMMARY_PATH\":\"$BASELINE_EVAL_SUMMARY_PATH\"}}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="$runtime_env" \
  -- python3 "$root_path/train.py" \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$num_gpus" --rollout-num-gpus "$num_gpus" --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$BASELINE_HF_CHECKPOINT" --ref-load "$BASELINE_REF_LOAD" \
  --load "$checkpoint" --no-load-optim --no-load-rng \
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
  --wandb-group "baseline-source-retool-${model_slug}-recovery-full-eval" \
  --custom-generate-function-path "$custom_generate_function_path" \
  --custom-rm-path skill_following_baselines.metrics.reward_func \
  --custom-eval-rollout-log-function-path skill_following_baselines.metrics.log_eval_rollout_data \
  2>&1 | tee "$artifact_root/full_eval.log"
