#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
baseline_root="$(cd -- "$script_dir/../.." && pwd)"
suite_root="$baseline_root"
root_path="${ROOT_PATH:-$PWD}"
seed="${SEED:-20260818}"
retool_script="$script_dir/train_source_retool.sh"
search_script="$script_dir/train_source_search_r1.sh"
qwen35_reference="$root_path/model/Qwen3.5-4B_torch_dist/latest_checkpointed_iteration.txt"
checkpoint_root="${QWEN35_SOURCE_CHECKPOINT_ROOT:-$baseline_root/data/checkpoints/skill_following_source_baselines_qwen35_4b}"
retool_checkpoint="$checkpoint_root/source_retool_qwen35_4b"
search_checkpoint="$checkpoint_root/source_search_r1_qwen35_4b"

# 数据：Qwen3.5-4B reference、两套 source 入口和准备后的训练数据。算法：申请 GPU 前验证两套源方法均可解析到隔离输出目录。
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  BASELINE_MODEL_KEY=qwen35_4b CHECKPOINT="$retool_checkpoint" NUM_GPUS=4 TP=2 SEED="$seed" SKIP_DATA_PREP=1 DRY_RUN=1 bash "$retool_script"
  BASELINE_MODEL_KEY=qwen35_4b CHECKPOINT="$search_checkpoint" NUM_GPUS=4 TP=2 SEED="$seed" SKIP_DATA_PREP=1 DRY_RUN=1 bash "$search_script"
  echo "[READY] serial source baselines model=Qwen3.5-4B GPUs=4 TP=2 order=ReTool,Search-R1"
  echo "[READY] checkpoint_root=$checkpoint_root"
  exit 0
fi

[[ -f "$qwen35_reference" ]] || { echo "[ERROR] missing Qwen3.5-4B torch_dist reference: $qwen35_reference" >&2; exit 1; }
mkdir -p "$checkpoint_root"

failed_methods=()
run_method() {
  local method_name="$1"
  local method_script="$2"
  local method_checkpoint="$3"
  echo "[BATCH] start method=$method_name model=Qwen3.5-4B"
  if BASELINE_MODEL_KEY=qwen35_4b CHECKPOINT="$method_checkpoint" NUM_GPUS=4 TP=2 SEED="$seed" SKIP_DATA_PREP=1 bash "$method_script"; then
    echo "[BATCH] complete method=$method_name model=Qwen3.5-4B"
  else
    failed_methods+=("$method_name")
    echo "[BATCH] failed method=$method_name model=Qwen3.5-4B" >&2
  fi
}

# 数据：相同 Qwen3.5-4B checkpoint 下的两类 source baseline。算法：先完成 Math/ReTool，再完成 Search/Search-R1，失败项在末尾统一报告。
run_method source-retool "$retool_script" "$retool_checkpoint"
run_method source-search-r1 "$search_script" "$search_checkpoint"

if (( ${#failed_methods[@]} > 0 )); then
  printf '[ERROR] failed source methods=' >&2
  printf '%s ' "${failed_methods[@]}" >&2
  printf '\n' >&2
  exit 1
fi
echo "[PASS] Qwen3.5-4B Source ReTool and Source Search-R1 completed"
