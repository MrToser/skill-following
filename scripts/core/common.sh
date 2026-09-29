#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail

suite_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root_path="${ROOT_PATH:-$PWD}"
python_bin="${PYTHON_BIN:-python3}"
conda_setup="${CONDA_SETUP:-}"
package_name="skill_following"

export ROOT_PATH="$root_path"
export PYTHONPATH="$suite_root/code:$suite_root/code/data_prep:$root_path/Megatron-LM:$root_path/examples/search-r1:$root_path/examples/retool:$root_path${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/tmp/skill_following_tilelang_cache}"
export TILELANG_TMP_DIR="${TILELANG_TMP_DIR:-$TILELANG_CACHE_DIR/tmp}"

# 数据：实验 ID、phase 和 seed。算法：从注册表 TSV 安全导出环境，不使用 eval。
load_experiment_environment() {
  local experiment_id="$1"
  local phase="$2"
  local seed="$3"
  local key
  local value
  while IFS=$'\t' read -r key value; do
    [[ "$key" =~ ^[A-Z][A-Z0-9_]*$ ]] || { echo "[ERROR] invalid registry key: $key" >&2; return 1; }
    printf -v "$key" '%s' "$value"
    export "$key"
  done < <("$python_bin" -m "$package_name.registry" env "$experiment_id" --phase "$phase" --seed "$seed")
}

# Data: prepared eval YAML, artifact directory, and release package. Algorithm: copy
# the user-data config for this run and point its per-dataset plugin hooks at the
# package shipped with this release.
prepare_eval_config() {
  local source_config="$1"
  local phase="$2"
  local output_config="$ARTIFACT_ROOT/config/${phase}_eval_config.yaml"
  mkdir -p "$(dirname "$output_config")"
  "$python_bin" - "$source_config" "$output_config" "$package_name" <<'PY'
import sys
from pathlib import Path

import yaml

source, output, package = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
if not isinstance(config, dict) or not isinstance(config.get("eval"), dict):
    raise ValueError(f"invalid eval config: {source}")
evaluation = config["eval"]
defaults = evaluation.setdefault("defaults", {})
if not isinstance(defaults, dict):
    raise ValueError(f"eval.defaults must be a mapping: {source}")
plugin_functions = {
    "custom_rm_path": "reward_func",
    "custom_generate_function_path": "generate",
    "custom_eval_rollout_log_function_path": "log_eval_rollout_data",
}
for key, function in plugin_functions.items():
    if key in defaults:
        defaults[key] = f"{package}.entrypoint.{function}"
datasets = evaluation.setdefault("datasets", [])
if not isinstance(datasets, list):
    raise ValueError(f"eval.datasets must be a list: {source}")
for dataset in datasets:
    if not isinstance(dataset, dict):
        raise ValueError(f"eval dataset entries must be mappings: {source}")
    for key, function in plugin_functions.items():
        if key in dataset:
            dataset[key] = f"{package}.entrypoint.{function}"
Path(output).write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
print(output)
PY
}

# 数据：conda 环境名。算法：在 nounset 下容忍 activation hook 的可选变量。
activate_env() {
  local environment_name="$1"
  set +u
  if [[ -n "$conda_setup" ]]; then source "$conda_setup"; conda activate "$environment_name"; fi
  set -u
}

# 数据：harnessr1 环境内 CUDA wheel。算法：为 Transformer Engine 构造稳定动态库路径。
configure_cuda_runtime() {
  [[ -n "${CONDA_PREFIX:-}" ]] || return 0
  local nvidia_root="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
  export CUDA_HOME="$CONDA_PREFIX"
  export CUDA_PATH="$CUDA_HOME"
  local cuda_library_path="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib"
  local package_name_part
  for package_name_part in cuda_runtime cuda_nvrtc cublas cudnn cufft curand cusolver cusparse cusparselt nccl nvjitlink nvtx; do
    local package_lib="$nvidia_root/$package_name_part/lib"
    if [[ -d "$package_lib" ]]; then
      cuda_library_path="${cuda_library_path:+$cuda_library_path:}$package_lib"
    fi
  done
  export LD_LIBRARY_PATH="$cuda_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

# 数据：当前实验 task mode。算法：只为 Search 或 Joint 启动共享本地 retriever。
start_retriever_if_needed() {
  retriever_pid=""
  if [[ "$SF_EXP_TASK_MODE" == "math" || "${SKIP_RETRIEVER:-0}" == "1" ]]; then
    return 0
  fi
  local index_path="${INDEX_PATH:-$root_path/Index/e5_Flat.index}"
  local corpus_path="${CORPUS_PATH:-$root_path/Index/wiki-18.jsonl}"
  local retriever_model="${RETRIEVER_MODEL:-$root_path/model/e5-base-v2}"
  local retriever_script="$root_path/examples/search-r1/local_dense_retriever/retrieval_server.py"
  local search_url="${UNIFIED_SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
  local retriever_devices="${RETRIEVER_CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
  for required_path in "$index_path" "$corpus_path" "$retriever_model" "$retriever_script"; do
    [[ -e "$required_path" ]] || { echo "[ERROR] missing retriever input: $required_path" >&2; return 1; }
  done
  activate_env retriever
  mkdir -p "$TRACE_DIR"
  CUDA_VISIBLE_DEVICES="$retriever_devices" python "$retriever_script" \
    --index_path "$index_path" \
    --corpus_path "$corpus_path" \
    --topk "${UNIFIED_SEARCH_TOPK:-3}" \
    --retriever_name e5 \
    --retriever_model "$retriever_model" \
    --faiss_gpu > "$TRACE_DIR/retriever.log" 2>&1 &
  retriever_pid=$!
  local attempt
  for attempt in $(seq 1 180); do
    if curl -fsS -X POST "$search_url" -H 'Content-Type: application/json' \
      -d '{"queries":["retriever health check"],"topk":1,"return_scores":false}' >/dev/null 2>&1; then
      export UNIFIED_SEARCH_URL="$search_url"
      export UNIFIED_SEARCH_TOPK="${UNIFIED_SEARCH_TOPK:-3}"
      export UNIFIED_SEARCH_CONCURRENCY="${UNIFIED_SEARCH_CONCURRENCY:-32}"
      return 0
    fi
    if ! kill -0 "$retriever_pid" 2>/dev/null; then
      tail -n 100 "$TRACE_DIR/retriever.log" >&2 || true
      return 1
    fi
    sleep 10
  done
  echo "[ERROR] retriever did not become ready" >&2
  return 1
}

# 数据：retriever PID。算法：只清理由当前脚本创建的进程。
stop_retriever() {
  if [[ -n "${retriever_pid:-}" ]]; then
    kill "$retriever_pid" 2>/dev/null || true
    wait "$retriever_pid" 2>/dev/null || true
  fi
}

# 数据：实验输入路径。算法：dry-run 报告待补资产，正式运行则严格失败。
run_preflight() {
  local experiment_id="$1"
  local phase="$2"
  local seed="$3"
  if "$python_bin" -m "$package_name.registry" preflight "$experiment_id" --phase "$phase" --seed "$seed"; then
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[BLOCKED] registry is valid, but this future run still has missing local artifacts."
    return 2
  fi
  return 1
}
