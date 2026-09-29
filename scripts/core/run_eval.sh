#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail
source "$(dirname "$0")/common.sh"

experiment_id="${1:?usage: run_eval.sh EXPERIMENT_ID [SEED]}"
seed="${2:-20260814}"
load_experiment_environment "$experiment_id" eval "$seed"
[[ "$SF_EXP_KIND" == "train" || "$SF_EXP_KIND" == "eval" || "$SF_EXP_KIND" == "alias" ]] || {
  echo "[ERROR] $experiment_id cannot be evaluated because kind=$SF_EXP_KIND" >&2
  exit 1
}

set +e
run_preflight "$experiment_id" eval "$seed"
preflight_status=$?
set -e
if [[ "$preflight_status" != "0" ]]; then
  [[ "${DRY_RUN:-0}" == "1" && "$preflight_status" == "2" ]] && exit 0
  exit "$preflight_status"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[READY] eval experiment=$experiment_id seed=$seed"
  echo "[READY] model=$SF_MODEL_KEY task=$SF_EXP_TASK_MODE GPUs=$NUM_GPUS TP=$TP"
  echo "[READY] skill=$SF_EXP_SKILL_VARIANT hints=$SF_EXP_STATE_AWARE_HINTS credit=$SF_EXP_TRANSITION_CREDIT first_research_reward=$SF_EXP_FIRST_RESEARCH_REWARD observation=$SF_EXP_OBSERVATION_MODE"
  echo "[READY] checkpoint=$CHECKPOINT"
  echo "[READY] eval_config=$EVAL_CONFIG_PATH"
  exit 0
fi

mkdir -p "$TRACE_DIR" "$WANDB_DIR" "$ARTIFACT_ROOT/eval"
eval_config_for_run="$(prepare_eval_config "$EVAL_CONFIG_PATH" eval)"
echo "[INFO] using release-local eval config: $eval_config_for_run"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
unset LD_PRELOAD || true

"$python_bin" -m "$package_name.tests.self_test"
trap stop_retriever EXIT
start_retriever_if_needed
activate_env harnessr1
configure_cuda_runtime
source "$MODEL_CONFIG"

ray stop --force >/dev/null 2>&1 || true
ray start --head \
  --node-ip-address "${MASTER_ADDR:-127.0.0.1}" \
  --num-gpus "$NUM_GPUS" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

runtime_env_json="$("$python_bin" -m "$package_name.registry" runtime-env "$experiment_id" --phase eval --seed "$seed")"
eval_log="$TRACE_DIR/eval.log"
: > "$eval_log"
eval_batch_size="${EVAL_BATCH_SIZE:-128}"

ray job submit --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
  --runtime-env-json="$runtime_env_json" \
  -- python3 "$root_path/train.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$NUM_GPUS" \
  --rollout-num-gpus "$NUM_GPUS" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --ref-load "$REF_LOAD" \
  --load "$CHECKPOINT" \
  --no-load-optim \
  --no-load-rng \
  --prompt-data "$TRAIN_DATA" \
  --input-key prompt \
  --label-key label \
  --metadata-key extra_info \
  --apply-chat-template \
  --reward-key score \
  --num-rollout 0 \
  --rollout-batch-size "$eval_batch_size" \
  --n-samples-per-prompt 1 \
  --rollout-max-prompt-len 4096 \
  --rollout-max-response-len "$EVAL_MAX_RESPONSE_LEN" \
  --rollout-temperature 1 \
  --eval-interval 1 \
  --eval-config "$eval_config_for_run" \
  --eval-input-key prompt \
  --eval-label-key label \
  --eval-max-prompt-len 4096 \
  --eval-max-response-len "$EVAL_MAX_RESPONSE_LEN" \
  --n-samples-per-eval-prompt 1 \
  --global-batch-size "$eval_batch_size" \
  --seed "$seed" \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --lr-decay-iters 1 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-coef 0.0 \
  --entropy-coef 0.0 \
  --eps-clip 0.20 \
  --eps-clip-high 0.30 \
  --tensor-model-parallel-size "$TP" \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}" \
  --rollout-num-gpus-per-engine "$TP" \
  --sglang-server-concurrency "$SGLANG_SERVER_CONCURRENCY" \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.60}" \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --use-wandb \
  --wandb-mode "$WANDB_MODE" \
  --wandb-dir "$WANDB_DIR" \
  --wandb-team "${WANDB_TEAM:-anonymous}" \
  --wandb-project "${WANDB_PROJECT:-grounded-skill-following}" \
  --wandb-group "${WANDB_GROUP}-eval" \
  --custom-config-path "$(dirname "$EVAL_CONFIG_PATH")/runtime_config.yaml" \
  --custom-generate-function-path "$package_name.entrypoint.generate" \
  --custom-rm-path "$package_name.entrypoint.reward_func" \
  --custom-eval-rollout-log-function-path "$package_name.entrypoint.log_eval_rollout_data" \
  2>&1 | tee "$eval_log"
