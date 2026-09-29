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
checkpoint="${CHECKPOINT:-$baseline_root/data/checkpoints/source_search_r1_${model_key}}"
default_artifact_root="$baseline_root/data/artifacts/source_search_r1_${model_key}"
default_wandb_dir="$baseline_root/data/wandb_offline/source_search_r1_${model_key}"
artifact_root="${ARTIFACT_ROOT:-$default_artifact_root}"
wandb_dir="${WANDB_DIR:-$default_wandb_dir}"
train_data="$root_path/Search-R1/data/nq_hotpotqa_train/train.parquet"
monitor_config="$baseline_root/data/prepared/source_search_r1/monitor_eval_config.yaml"
full_config="$baseline_root/data/prepared/source_search_r1/full_eval_config.yaml"
model_config="$BASELINE_MODEL_CONFIG"
hf_checkpoint="$BASELINE_HF_CHECKPOINT"
ref_load="$BASELINE_REF_LOAD"
require_paths "$train_data" "$model_config" "$hf_checkpoint" "$ref_load" "$root_path/examples/search-r1/generate_with_search.py"

activate_env harnessr1
configure_cuda_runtime
prepare_baseline_data
require_paths "$monitor_config" "$full_config"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[READY] Source Search-R1 uses examples/search-r1/generate_with_search.py"
  echo "[READY] model=$BASELINE_MODEL_DISPLAY_NAME GPUs=$num_gpus TP=$tp rollout=200 batch=512 samples=5 global_batch=256"
  echo "[READY] checkpoint=$checkpoint"
  echo "[READY] artifact_root=$artifact_root"
  exit 0
fi
mkdir -p "$checkpoint" "$artifact_root" "$wandb_dir"
trap stop_retriever EXIT
start_retriever "$artifact_root"
activate_env harnessr1
configure_cuda_runtime
start_ray "$num_gpus"
source "$model_config"
runtime_env="{\"env_vars\":{\"PYTHONPATH\":\"$PYTHONPATH\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"CUDA_HOME\":\"$CUDA_HOME\",\"CUDA_PATH\":\"$CUDA_PATH\",\"LD_LIBRARY_PATH\":\"$LD_LIBRARY_PATH\",\"TILELANG_CACHE_DIR\":\"$TILELANG_CACHE_DIR\",\"TILELANG_TMP_DIR\":\"$TILELANG_TMP_DIR\",\"BASELINE_MODEL_KEY\":\"$model_key\",\"UNIFIED_SEARCH_URL\":\"$UNIFIED_SEARCH_URL\"}}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="$runtime_env" \
  -- python3 "$root_path/train.py" \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$num_gpus" --rollout-num-gpus "$num_gpus" --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$hf_checkpoint" --ref-load "$ref_load" --save "$checkpoint" --save-interval 200 --save-only-at-end \
  --no-save-optim --no-save-rng \
  --prompt-data "$train_data" --input-key prompt --label-key reward_model --apply-chat-template --rollout-shuffle \
  --num-rollout 200 --rollout-batch-size 512 --n-samples-per-prompt 5 --rollout-max-prompt-len 4096 \
  --rollout-max-response-len 500 --rollout-temperature 1 --eval-interval 25 --eval-config "$monitor_config" \
  --eval-input-key prompt --eval-label-key reward_model --n-samples-per-eval-prompt 1 \
  --eval-max-prompt-len 4096 --eval-max-response-len 500 --global-batch-size 256 --balance-data \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --lr-warmup-fraction 0.285 --weight-decay 0.01 \
  --adam-beta1 0.9 --adam-beta2 0.98 --advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.001 \
  --kl-loss-type low_var_kl --entropy-coef 0.001 --eps-clip 0.2 --eps-clip-high 0.28 \
  --tensor-model-parallel-size "$tp" --sequence-parallel --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 16384 --rollout-num-gpus-per-engine "$tp" \
  --sglang-mem-fraction-static 0.60 --attention-dropout 0.0 --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash \
  --use-wandb --wandb-mode offline --wandb-dir "$wandb_dir" \
  --wandb-team "${WANDB_TEAM:-anonymous}" --wandb-project "${WANDB_PROJECT:-grounded-skill-following}" \
  --wandb-group "baseline-source-search-r1-${model_slug}" \
  --custom-generate-function-path generate_with_search.generate \
  --custom-rm-path generate_with_search.reward_func 2>&1 | tee "$artifact_root/train.log"

export BASELINE_CONDITION=source_search_r1
export BASELINE_EVAL_TRACE_PATH="$artifact_root/full_eval_trace.jsonl"
export BASELINE_EVAL_SUMMARY_PATH="$artifact_root/summary.json"
eval_runtime_env="{\"env_vars\":{\"PYTHONPATH\":\"$PYTHONPATH\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"CUDA_HOME\":\"$CUDA_HOME\",\"CUDA_PATH\":\"$CUDA_PATH\",\"LD_LIBRARY_PATH\":\"$LD_LIBRARY_PATH\",\"TILELANG_CACHE_DIR\":\"$TILELANG_CACHE_DIR\",\"TILELANG_TMP_DIR\":\"$TILELANG_TMP_DIR\",\"BASELINE_MODEL_KEY\":\"$model_key\",\"UNIFIED_SEARCH_URL\":\"$UNIFIED_SEARCH_URL\",\"BASELINE_CONDITION\":\"$BASELINE_CONDITION\",\"BASELINE_EVAL_TRACE_PATH\":\"$BASELINE_EVAL_TRACE_PATH\",\"BASELINE_EVAL_SUMMARY_PATH\":\"$BASELINE_EVAL_SUMMARY_PATH\"}}"
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="$eval_runtime_env" \
  -- python3 "$root_path/train.py" \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$num_gpus" --rollout-num-gpus "$num_gpus" --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$hf_checkpoint" --ref-load "$ref_load" --load "$checkpoint" --no-load-optim --no-load-rng \
  --prompt-data "$train_data" --input-key prompt --label-key reward_model --apply-chat-template \
  --num-rollout 0 --rollout-batch-size 128 --n-samples-per-prompt 1 --rollout-max-prompt-len 4096 \
  --rollout-max-response-len 500 --eval-interval 1 --eval-config "$full_config" --eval-input-key prompt \
  --eval-label-key reward_model --n-samples-per-eval-prompt 1 --eval-max-prompt-len 4096 \
  --eval-max-response-len 500 --global-batch-size 128 --optimizer adam --lr 1e-6 \
  --lr-decay-style constant --lr-decay-iters 1 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 \
  --advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28 --tensor-model-parallel-size "$tp" \
  --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 --use-dynamic-batch-size \
  --max-tokens-per-gpu 16384 --rollout-num-gpus-per-engine "$tp" --sglang-mem-fraction-static 0.60 \
  --attention-dropout 0.0 --hidden-dropout 0.0 --attention-backend flash \
  --use-wandb --wandb-mode offline --wandb-dir "$wandb_dir" \
  --wandb-team "${WANDB_TEAM:-anonymous}" --wandb-project "${WANDB_PROJECT:-grounded-skill-following}" \
  --wandb-group "baseline-source-search-r1-${model_slug}-full-eval" \
  --custom-generate-function-path generate_with_search.generate \
  --custom-rm-path skill_following_baselines.metrics.reward_func \
  --custom-eval-rollout-log-function-path skill_following_baselines.metrics.log_eval_rollout_data \
  2>&1 | tee "$artifact_root/full_eval.log"
