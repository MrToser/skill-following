#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail
source "$(dirname "$0")/../core/common.sh"

model_key="${1:?usage: prepare_model_reference.sh qwen35_4b}"
[[ "$model_key" == "qwen35_4b" ]] || { echo "[ERROR] unsupported model key: $model_key" >&2; exit 1; }
probe_experiment=main_qwen35_4b_o_math
load_experiment_environment "$probe_experiment" train 20260814

if [[ ! -d "$HF_CHECKPOINT" ]]; then
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[PENDING] Hugging Face checkpoint will be downloaded before conversion: $HF_CHECKPOINT"
  else
    echo "[ERROR] missing Hugging Face checkpoint: $HF_CHECKPOINT" >&2
    exit 1
  fi
fi
[[ -f "$MODEL_CONFIG" ]] || { echo "[ERROR] missing model config: $MODEL_CONFIG" >&2; exit 1; }
if [[ -f "$REF_LOAD/latest_checkpointed_iteration.txt" ]]; then
  echo "[READY] torch_dist checkpoint already exists: $REF_LOAD"
  exit 0
fi
source "$MODEL_CONFIG"

# 数据：训练模型参数。
# 算法：转换器不接收仅由新版本 Qwen3.5 runtime 使用的 --use-gated-attention，转换时过滤该无值开关，训练仍继续读取原始 MODEL_ARGS。
conversion_model_args=()
for model_arg in "${MODEL_ARGS[@]}"; do
  if [[ "$model_arg" == "--use-gated-attention" ]]; then
    echo "[CONVERT] omit unsupported converter argument: $model_arg"
    continue
  fi
  conversion_model_args+=("$model_arg")
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[READY] conversion model=$model_key"
  echo "[READY] source=$HF_CHECKPOINT"
  echo "[READY] destination=$REF_LOAD"
  printf '[READY] converter_args='
  printf '%q ' "${conversion_model_args[@]}"
  printf '\n'
  exit 0
fi

activate_env harnessr1
configure_cuda_runtime
mkdir -p "$REF_LOAD"
PYTHONPATH="$root_path/Megatron-LM:$root_path" "$python_bin" \
  "$root_path/tools/convert_hf_to_torch_dist.py" \
  "${conversion_model_args[@]}" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --save "$REF_LOAD"
[[ -f "$REF_LOAD/latest_checkpointed_iteration.txt" ]] || {
  echo "[ERROR] conversion finished without latest_checkpointed_iteration.txt" >&2
  exit 1
}
