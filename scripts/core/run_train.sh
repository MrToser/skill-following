#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail
source "$(dirname "$0")/common.sh"

experiment_id="${1:?usage: run_train.sh EXPERIMENT_ID [SEED]}"
seed="${2:-20260814}"
load_experiment_environment "$experiment_id" train "$seed"
[[ "$SF_EXP_KIND" == "train" ]] || { echo "[ERROR] $experiment_id is kind=$SF_EXP_KIND, not train" >&2; exit 1; }

set +e
run_preflight "$experiment_id" train "$seed"
preflight_status=$?
set -e
if [[ "$preflight_status" != "0" ]]; then
  [[ "${DRY_RUN:-0}" == "1" && "$preflight_status" == "2" ]] && exit 0
  exit "$preflight_status"
fi

for integer_value in "$NUM_GPUS" "$TP" "$NUM_ROLLOUT" "$ROLLOUT_BATCH_SIZE" "$N_SAMPLES_PER_PROMPT" "$NUM_STEPS_PER_ROLLOUT" "$GLOBAL_BATCH_SIZE" "$EVAL_INTERVAL" "$SAVE_INTERVAL" "$ROLLOUT_MAX_RESPONSE_LEN" "$EVAL_MAX_RESPONSE_LEN" "$SGLANG_SERVER_CONCURRENCY"; do
  [[ "$integer_value" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] invalid positive integer: $integer_value" >&2; exit 1; }
done
(( NUM_GPUS % TP == 0 )) || { echo "[ERROR] NUM_GPUS must be divisible by TP" >&2; exit 1; }

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[READY] train experiment=$experiment_id seed=$seed"
  echo "[READY] model=$SF_MODEL_KEY task=$SF_EXP_TASK_MODE GPUs=$NUM_GPUS TP=$TP steps=$NUM_ROLLOUT"
  echo "[READY] rollout_batch=$ROLLOUT_BATCH_SIZE samples_per_prompt=$N_SAMPLES_PER_PROMPT optimizer_steps_per_rollout=$NUM_STEPS_PER_ROLLOUT global_batch=$GLOBAL_BATCH_SIZE"
  echo "[READY] eval_interval=$EVAL_INTERVAL save_interval=$SAVE_INTERVAL"
  echo "[READY] rollout_response_len=$ROLLOUT_MAX_RESPONSE_LEN eval_response_len=$EVAL_MAX_RESPONSE_LEN sglang_concurrency=$SGLANG_SERVER_CONCURRENCY warmup=$LR_WARMUP_FRACTION weight_decay=$WEIGHT_DECAY entropy=$ENTROPY_COEF clip=$EPS_CLIP/$EPS_CLIP_HIGH grad_clip=$CLIP_GRAD kl=$KL_LOSS_COEF tis=$TIS_CLIP"
  echo "[READY] skill=$SF_EXP_SKILL_VARIANT hints=$SF_EXP_STATE_AWARE_HINTS credit=$SF_EXP_TRANSITION_CREDIT recovery=$SF_EXP_RECOVERY_FACTOR first_research_reward=$SF_EXP_FIRST_RESEARCH_REWARD"
  echo "[READY] train_data=$TRAIN_DATA eval_config=$EVAL_CONFIG_PATH"
  echo "[READY] save_path=$SAVE_PATH"
  echo "[READY] checkpoint=model-only no_save_optim=1 no_save_rng=1"
  exit 0
fi

if [[ -d "$SAVE_PATH" && "${ALLOW_EXISTING_SAVE_PATH:-0}" != "1" ]]; then
  existing_entry="$(find "$SAVE_PATH" -mindepth 1 -maxdepth 1 -print -quit)"
  [[ -z "$existing_entry" ]] || { echo "[ERROR] nonempty SAVE_PATH: $SAVE_PATH" >&2; exit 1; }
fi

mkdir -p "$SAVE_PATH" "$TRACE_DIR" "$WANDB_DIR" "$ARTIFACT_ROOT/eval"
eval_config_for_run="$(prepare_eval_config "$EVAL_CONFIG_PATH" train)"
echo "[INFO] using release-local eval config: $eval_config_for_run"
export SF_TRACE_TRAIN_TOKENS="${SF_TRACE_TRAIN_TOKENS:-1}"
export SF_TRACE_MAX_SAMPLES="${SF_TRACE_MAX_SAMPLES:-5}"
export SF_TRACE_MAX_CHARS="${SF_TRACE_MAX_CHARS:-12000}"
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

rollout_batch_size="$ROLLOUT_BATCH_SIZE"
n_samples_per_prompt="$N_SAMPLES_PER_PROMPT"
num_steps_per_rollout="$NUM_STEPS_PER_ROLLOUT"
global_batch_size="$GLOBAL_BATCH_SIZE"
(( rollout_batch_size * n_samples_per_prompt == global_batch_size * num_steps_per_rollout )) || {
  echo "[ERROR] rollout batch times samples must equal global batch times optimizer steps" >&2
  exit 1
}
eval_start_args=()
if [[ "$SF_SKIP_EVAL_BEFORE_TRAIN" == "1" ]]; then
  eval_start_args+=(--skip-eval-before-train)
fi

ray stop --force >/dev/null 2>&1 || true
ray start --head \
  --node-ip-address "${MASTER_ADDR:-127.0.0.1}" \
  --num-gpus "$NUM_GPUS" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

runtime_env_json="$("$python_bin" -m "$package_name.registry" runtime-env "$experiment_id" --phase train --seed "$seed")"
train_log="$TRACE_DIR/train.log"
: > "$train_log"

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
  --save "$SAVE_PATH" \
  --save-interval "$SAVE_INTERVAL" \
  --save-only-at-end \
  --no-save-optim \
  --no-save-rng \
  --prompt-data "$TRAIN_DATA" \
  --input-key prompt \
  --label-key label \
  --metadata-key extra_info \
  --apply-chat-template \
  --rollout-shuffle \
  --reward-key score \
  --num-rollout "$NUM_ROLLOUT" \
  --rollout-batch-size "$rollout_batch_size" \
  --n-samples-per-prompt "$n_samples_per_prompt" \
  --num-steps-per-rollout "$num_steps_per_rollout" \
  --rollout-max-prompt-len 4096 \
  --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" \
  --rollout-temperature 1 \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-config "$eval_config_for_run" \
  --eval-input-key prompt \
  --eval-label-key label \
  --eval-max-prompt-len 4096 \
  --eval-max-response-len "$EVAL_MAX_RESPONSE_LEN" \
  --n-samples-per-eval-prompt 1 \
  "${eval_start_args[@]}" \
  --global-batch-size "$global_batch_size" \
  --balance-data \
  --seed "$seed" \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --lr-warmup-fraction "$LR_WARMUP_FRACTION" \
  --weight-decay "$WEIGHT_DECAY" \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-coef "$KL_LOSS_COEF" \
  --kl-loss-type low_var_kl \
  --entropy-coef "$ENTROPY_COEF" \
  --eps-clip "$EPS_CLIP" \
  --eps-clip-high "$EPS_CLIP_HIGH" \
  --clip-grad "$CLIP_GRAD" \
  --use-tis \
  --tis-clip "$TIS_CLIP" \
  --tensor-model-parallel-size "$TP" \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}" \
  --rollout-num-gpus-per-engine 1 \
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
  --wandb-group "$WANDB_GROUP" \
  --custom-config-path "$(dirname "$EVAL_CONFIG_PATH")/runtime_config.yaml" \
  --custom-generate-function-path "$package_name.entrypoint.generate" \
  --custom-rm-path "$package_name.entrypoint.reward_func" \
  --custom-eval-rollout-log-function-path "$package_name.entrypoint.log_eval_rollout_data" \
  2>&1 | tee "$train_log"
