#!/bin/bash
# LOCKED: false

set -euo pipefail

root_path="${ROOT_PATH:-$PWD}"
opensource_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RETOOL_SCRIPT_DIR="$root_path/examples/retool"
RELEASE_RETOOL_CODE_DIR="$opensource_root/code/legacy_retool"
DATA_DIR="${DATA_DIR:-$opensource_root/data/retool_skill_protocol}"
BASE_DATA_DIR="${BASE_DATA_DIR:-$DATA_DIR/retool_base}"
TRAIN_SOURCE_DATA="${TRAIN_SOURCE_DATA:-$DATA_DIR/math_train.jsonl}"
EVAL_SOURCE_DATA="${EVAL_SOURCE_DATA:-$DATA_DIR/math_test.jsonl}"
TRAIN_DATA="${TRAIN_DATA:-$BASE_DATA_DIR/math_train_base.jsonl}"
EVAL_MATH_DATA="${EVAL_MATH_DATA:-$BASE_DATA_DIR/math_test_base.jsonl}"
AUTO_PREPARE_DATA="${AUTO_PREPARE_DATA:-1}"

MODEL_CONFIG="${MODEL_CONFIG:-$root_path/scripts/models/qwen3.5-4B.sh}"
HF_CHECKPOINT="${HF_CHECKPOINT:-$root_path/model/Qwen3.5-4B/}"
REF_LOAD="${REF_LOAD:-$root_path/model/Qwen3.5-4B_torch_dist/}"
SAVE_PATH="${SAVE_PATH:-$root_path/model/Qwen3.5-4B_retool_math_base_slime/}"
ROTARY_BASE="${ROTARY_BASE:-}"
NUM_GPUS="${NUM_GPUS:-4}"
TP="${TP:-2}"
CP="${CP:-1}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.00}"
ENTROPY_COEF="${ENTROPY_COEF:-0.00}"
USE_TIS="${USE_TIS:-0}"
TIS_CLIP="${TIS_CLIP:-4.0}"
EPS_CLIP="${EPS_CLIP:-0.20}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"

NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-4096}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-16}"
FINAL_EVAL_N_SAMPLES_PER_PROMPT="${FINAL_EVAL_N_SAMPLES_PER_PROMPT:-0}"
EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-4096}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-16384}"
EVAL_TOP_P="${EVAL_TOP_P:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"
METADATA_KEY="${METADATA_KEY:-metadata}"
CUSTOM_GENERATE_FUNCTION_PATH="${CUSTOM_GENERATE_FUNCTION_PATH:-generate_with_retool.generate}"
CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-generate_with_retool.reward_func}"
CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH="${CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH:-}"

SAVE_ONLY_AT_END="${SAVE_ONLY_AT_END:-1}"
SAVE_MODEL_ONLY="${SAVE_MODEL_ONLY:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DIR="${WANDB_DIR:-$opensource_root/data/wandb_offline}"
WANDB_TEAM="${WANDB_TEAM:-anonymous}"
WANDB_PROJECT="${WANDB_PROJECT:-grounded-skill-following}"
WANDB_GROUP="${WANDB_GROUP:-retool-math-base_qwen3.5-4B}"
FINAL_EVAL_WANDB_GROUP="${FINAL_EVAL_WANDB_GROUP:-${WANDB_GROUP}-final-eval}"
EVAL_CONFIG_PATH="${EVAL_CONFIG_PATH:-$opensource_root/data/runtime/retool_math_base_eval_config.yaml}"
FINAL_EVAL_CONFIG_PATH="${FINAL_EVAL_CONFIG_PATH:-$opensource_root/data/runtime/retool_math_base_final_eval_config.yaml}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
CLEANUP_BEFORE_RUN="${CLEANUP_BEFORE_RUN:-0}"
REQUIRE_MODEL_PATHS="${REQUIRE_MODEL_PATHS:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "$RETOOL_SCRIPT_DIR" ]]; then
  echo "[ERROR] missing Retool example directory: $RETOOL_SCRIPT_DIR"
  exit 1
fi

if [[ ! -f "$RELEASE_RETOOL_CODE_DIR/prepare_retool_base_data.py" ]]; then
  echo "[ERROR] missing base data adapter: $RELEASE_RETOOL_CODE_DIR/prepare_retool_base_data.py"
  exit 1
fi

if [[ "$AUTO_PREPARE_DATA" == "1" ]]; then
  if [[ ! -f "$TRAIN_DATA" ]]; then
    python "$RELEASE_RETOOL_CODE_DIR/prepare_retool_base_data.py" \
      --source-path "$TRAIN_SOURCE_DATA" \
      --output-path "$TRAIN_DATA" \
      --manifest-path "$BASE_DATA_DIR/math_train_base_manifest.json"
  fi
  if [[ ! -f "$EVAL_MATH_DATA" ]]; then
    python "$RELEASE_RETOOL_CODE_DIR/prepare_retool_base_data.py" \
      --source-path "$EVAL_SOURCE_DATA" \
      --output-path "$EVAL_MATH_DATA" \
      --manifest-path "$BASE_DATA_DIR/math_test_base_manifest.json"
  fi
fi

for data_file in "$TRAIN_DATA" "$EVAL_MATH_DATA"; do
  if [[ ! -f "$data_file" ]]; then
    echo "[ERROR] missing data file: $data_file"
    echo "[HINT] set AUTO_PREPARE_DATA=1 or provide TRAIN_DATA/EVAL_MATH_DATA in Retool base format."
    exit 1
  fi
done

if [[ ! "$N_SAMPLES_PER_EVAL_PROMPT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] N_SAMPLES_PER_EVAL_PROMPT must be a positive integer: $N_SAMPLES_PER_EVAL_PROMPT"
  exit 1
fi
if [[ ! "$FINAL_EVAL_N_SAMPLES_PER_PROMPT" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] FINAL_EVAL_N_SAMPLES_PER_PROMPT must be a non-negative integer: $FINAL_EVAL_N_SAMPLES_PER_PROMPT"
  exit 1
fi
if [[ ! "$EPS_CLIP" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]]; then
  echo "[ERROR] EPS_CLIP must be a non-negative number: $EPS_CLIP"
  exit 1
fi
if [[ ! "$EPS_CLIP_HIGH" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]]; then
  echo "[ERROR] EPS_CLIP_HIGH must be a non-negative number: $EPS_CLIP_HIGH"
  exit 1
fi

if [[ "$REQUIRE_MODEL_PATHS" == "1" ]]; then
  for model_path in "$HF_CHECKPOINT" "$REF_LOAD"; do
    if [[ ! -e "$model_path" ]]; then
      echo "[ERROR] missing model path: $model_path"
      echo "[HINT] override HF_CHECKPOINT/REF_LOAD, or set REQUIRE_MODEL_PATHS=0 for dry script checks."
      echo "[HINT] for upstream Qwen3-4B Retool, set MODEL_CONFIG, HF_CHECKPOINT, REF_LOAD and ROTARY_BASE=5000000."
      exit 1
    fi
  done
fi

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_KEY:-}" ]]; then
  echo "[ERROR] WANDB_KEY is required when WANDB_MODE=online."
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "harnessr1" ]]; then
  echo "[WARN] CONDA_DEFAULT_ENV is not harnessr1; continuing with current shell."
fi

cd "$root_path"
export MODEL_ARGS_ROTARY_BASE="${MODEL_ARGS_ROTARY_BASE:-${ROTARY_BASE:-1000000}}"
source "$MODEL_CONFIG"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE
export WANDB_DIR
unset LD_PRELOAD || true
mkdir -p "$WANDB_DIR"
mkdir -p "$SAVE_PATH"

apply_chat_template_yaml="false"
if [[ "$APPLY_CHAT_TEMPLATE" == "1" ]]; then
  apply_chat_template_yaml="true"
fi

# 数据：目标路径和每题采样数；算法：生成同构 eval YAML，让训练期和最终评测只在采样数上不同。
write_eval_config() {
  local config_path="$1"
  local samples_per_prompt="$2"

  mkdir -p "$(dirname "$config_path")"
  {
    printf 'eval:\n'
    printf '  defaults:\n'
    printf '    input_key: prompt\n'
    printf '    label_key: label\n'
    printf '    metadata_key: %s\n' "$METADATA_KEY"
    printf '    apply_chat_template: %s\n' "$apply_chat_template_yaml"
    printf '    n_samples_per_eval_prompt: %s\n' "$samples_per_prompt"
    printf '    max_prompt_len: %s\n' "$EVAL_MAX_PROMPT_LEN"
    printf '    max_response_len: %s\n' "$EVAL_MAX_RESPONSE_LEN"
    printf '    top_p: %s\n' "$EVAL_TOP_P"
    printf '  datasets:\n'
    printf '    - name: math\n'
    printf '      path: %s\n' "$EVAL_MATH_DATA"
  } > "$config_path"
}

write_eval_config "$EVAL_CONFIG_PATH" "$N_SAMPLES_PER_EVAL_PROMPT"
if [[ "$FINAL_EVAL_N_SAMPLES_PER_PROMPT" -gt 0 ]]; then
  write_eval_config "$FINAL_EVAL_CONFIG_PATH" "$FINAL_EVAL_N_SAMPLES_PER_PROMPT"
fi

if [[ "$CLEANUP_BEFORE_RUN" == "1" ]]; then
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  sleep 3
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [[ "$NVLINK_COUNT" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "[INFO] HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

set -x

CKPT_ARGS=(
   --hf-checkpoint "$HF_CHECKPOINT"
   --ref-load "$REF_LOAD"
   --save "$SAVE_PATH"
   --save-interval "$SAVE_INTERVAL"
)
if [[ "$SAVE_ONLY_AT_END" == "1" ]]; then
   CKPT_ARGS+=(--save-only-at-end)
fi
if [[ "$SAVE_MODEL_ONLY" == "1" ]]; then
   CKPT_ARGS+=(--no-save-optim --no-save-rng)
fi
if [[ -n "$ROTARY_BASE" ]]; then
   CKPT_ARGS+=(--rotary-base "$ROTARY_BASE")
fi

ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_DATA"
   --input-key prompt
   --label-key label
   --metadata-key "$METADATA_KEY"
   --rollout-shuffle
   --reward-key score
   --num-rollout "$NUM_ROLLOUT"
   --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
   --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
   --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN"
   --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
   --rollout-temperature "$ROLLOUT_TEMPERATURE"
   --eval-interval "$EVAL_INTERVAL"
   --eval-config "$EVAL_CONFIG_PATH"
   --eval-input-key prompt
   --eval-label-key label
   --n-samples-per-eval-prompt "$N_SAMPLES_PER_EVAL_PROMPT"
   --eval-max-prompt-len "$EVAL_MAX_PROMPT_LEN"
   --eval-max-response-len "$EVAL_MAX_RESPONSE_LEN"
   --eval-top-p "$EVAL_TOP_P"
   --global-batch-size "$GLOBAL_BATCH_SIZE"
   --balance-data
)
if [[ "$APPLY_CHAT_TEMPLATE" == "1" ]]; then
   ROLLOUT_ARGS+=(--apply-chat-template)
fi

FINAL_EVAL_TEMPLATE_ARGS=()
if [[ "$APPLY_CHAT_TEMPLATE" == "1" ]]; then
   FINAL_EVAL_TEMPLATE_ARGS+=(--apply-chat-template)
fi

PERF_ARGS=(
   --tensor-model-parallel-size "$TP"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "$CP"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef "$KL_LOSS_COEF"
   --kl-loss-type low_var_kl
   --entropy-coef "$ENTROPY_COEF"
   --eps-clip "$EPS_CLIP"
   --eps-clip-high "$EPS_CLIP_HIGH"
)
if [[ "$USE_TIS" == "1" ]]; then
   GRPO_ARGS+=(--use-tis --tis-clip "$TIS_CLIP")
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-mode "$WANDB_MODE"
   --wandb-dir "$WANDB_DIR"
   --wandb-team "$WANDB_TEAM"
   --wandb-project "$WANDB_PROJECT"
   --wandb-group "$WANDB_GROUP"
)
FINAL_EVAL_WANDB_ARGS=(
   --use-wandb
   --wandb-mode "$WANDB_MODE"
   --wandb-dir "$WANDB_DIR"
   --wandb-team "$WANDB_TEAM"
   --wandb-project "$WANDB_PROJECT"
   --wandb-group "$FINAL_EVAL_WANDB_GROUP"
)
if [[ -n "${WANDB_KEY:-}" ]]; then
   WANDB_ARGS+=(--wandb-key "$WANDB_KEY")
   FINAL_EVAL_WANDB_ARGS+=(--wandb-key "$WANDB_KEY")
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "$ROLLOUT_NUM_GPUS_PER_ENGINE"
   --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path "$CUSTOM_GENERATE_FUNCTION_PATH"
   --custom-rm-path "$CUSTOM_RM_PATH"
)
if [[ -n "$CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH" ]]; then
   CUSTOM_ARGS+=(--custom-eval-rollout-log-function-path "$CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH")
fi

if [[ "$DRY_RUN" == "1" ]]; then
   set +x
   echo "[INFO] DRY_RUN=1, skip Ray start and training submission."
   echo "[INFO] train data: $TRAIN_DATA"
   echo "[INFO] eval data: $EVAL_MATH_DATA"
   echo "[INFO] model config: $MODEL_CONFIG"
   echo "[INFO] hf checkpoint: $HF_CHECKPOINT"
   echo "[INFO] ref load: $REF_LOAD"
   echo "[INFO] save path: $SAVE_PATH"
   echo "[INFO] save only at end: $SAVE_ONLY_AT_END"
   echo "[INFO] save model only: $SAVE_MODEL_ONLY"
   echo "[INFO] save interval: $SAVE_INTERVAL"
   echo "[INFO] GRPO clip: left=$EPS_CLIP, right=$EPS_CLIP_HIGH"
   echo "[INFO] eval config: $EVAL_CONFIG_PATH"
   echo "[INFO] periodic eval samples per prompt: $N_SAMPLES_PER_EVAL_PROMPT"
   echo "[INFO] final eval samples per prompt: $FINAL_EVAL_N_SAMPLES_PER_PROMPT"
   if [[ "$FINAL_EVAL_N_SAMPLES_PER_PROMPT" -gt 0 ]]; then
      echo "[INFO] final eval config: $FINAL_EVAL_CONFIG_PATH"
      echo "[INFO] final eval W&B group: $FINAL_EVAL_WANDB_GROUP"
   fi
   if [[ -n "$CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH" ]]; then
      echo "[INFO] custom eval logger: $CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH"
   fi
   if [[ -n "${RETOOL_SKILL_TRACE_OUTPUT_PATH:-}" ]]; then
      echo "[INFO] Retool skill trace: $RETOOL_SKILL_TRACE_OUTPUT_PATH"
   fi
   exit 0
fi

if ray job list --address="$RAY_DASHBOARD_ADDRESS" >/dev/null 2>&1; then
   echo "[INFO] Reusing existing Ray cluster at $RAY_DASHBOARD_ADDRESS."
else
   ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus "$NUM_GPUS" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"$root_path/Megatron-LM/:$RETOOL_SCRIPT_DIR:$RELEASE_RETOOL_CODE_DIR:$root_path\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"$HAS_NVLINK\",\"WANDB_MODE\":\"$WANDB_MODE\",\"WANDB_DIR\":\"$WANDB_DIR\",\"RETOOL_SKILL_TRACE_TRAIN_TOKENS\":\"${RETOOL_SKILL_TRACE_TRAIN_TOKENS:-}\",\"RETOOL_SKILL_TRACE_MAX_SAMPLES\":\"${RETOOL_SKILL_TRACE_MAX_SAMPLES:-}\",\"RETOOL_SKILL_TRACE_MAX_CHARS\":\"${RETOOL_SKILL_TRACE_MAX_CHARS:-}\",\"RETOOL_SKILL_TRACE_TOKEN_IDS\":\"${RETOOL_SKILL_TRACE_TOKEN_IDS:-}\",\"RETOOL_SKILL_TRACE_OUTPUT_PATH\":\"${RETOOL_SKILL_TRACE_OUTPUT_PATH:-}\"}}"

ray job submit --address="$RAY_DASHBOARD_ADDRESS" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "$NUM_GPUS" \
   --rollout-num-gpus "$NUM_GPUS" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"

if [[ "$FINAL_EVAL_N_SAMPLES_PER_PROMPT" -gt 0 ]]; then
   if [[ ! -f "$SAVE_PATH/latest_checkpointed_iteration.txt" ]]; then
      echo "[ERROR] final checkpoint marker missing: $SAVE_PATH/latest_checkpointed_iteration.txt"
      exit 1
   fi

   echo "[INFO] Training completed; starting final pass@1/2/4/8/16 eval from $SAVE_PATH"
   ray job submit --address="$RAY_DASHBOARD_ADDRESS" \
      --runtime-env-json="$RUNTIME_ENV_JSON" \
      -- python3 train.py \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node "$NUM_GPUS" \
      --rollout-num-gpus "$NUM_GPUS" \
      --colocate \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "$HF_CHECKPOINT" \
      --ref-load "$REF_LOAD" \
      --load "$SAVE_PATH" \
      --no-load-optim \
      --no-load-rng \
      --prompt-data "$TRAIN_DATA" \
      --input-key prompt \
      --label-key label \
      --metadata-key "$METADATA_KEY" \
      --rollout-shuffle \
      --reward-key score \
      --num-rollout 0 \
      --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
      --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
      --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" \
      --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" \
      --rollout-temperature "$ROLLOUT_TEMPERATURE" \
      --eval-interval 1 \
      --eval-config "$FINAL_EVAL_CONFIG_PATH" \
      --eval-input-key prompt \
      --eval-label-key label \
      --n-samples-per-eval-prompt "$FINAL_EVAL_N_SAMPLES_PER_PROMPT" \
      --eval-max-prompt-len "$EVAL_MAX_PROMPT_LEN" \
      --eval-max-response-len "$EVAL_MAX_RESPONSE_LEN" \
      --eval-top-p "$EVAL_TOP_P" \
      --global-batch-size "$GLOBAL_BATCH_SIZE" \
      --balance-data \
      "${FINAL_EVAL_TEMPLATE_ARGS[@]}" \
      --optimizer adam \
      --lr 1e-6 \
      --lr-decay-style constant \
      --lr-decay-iters 1 \
      --weight-decay 0.1 \
      --adam-beta1 0.9 \
      --adam-beta2 0.98 \
      "${GRPO_ARGS[@]}" \
      "${FINAL_EVAL_WANDB_ARGS[@]}" \
      "${PERF_ARGS[@]}" \
      "${SGLANG_ARGS[@]}" \
      "${MISC_ARGS[@]}" \
      "${CUSTOM_ARGS[@]}"
fi
