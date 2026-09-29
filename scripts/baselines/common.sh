#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail

baseline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
suite_root="$baseline_root"
root_path="${ROOT_PATH:-$PWD}"
python_bin="${PYTHON_BIN:-python3}"
conda_setup="${CONDA_SETUP:-}"
export PYTHONPATH="$baseline_root/code/baselines:$suite_root/code:$suite_root/code/data_prep:$root_path/Megatron-LM:$root_path/examples/search-r1:$root_path/examples/retool:$root_path${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/tmp/skill_following_baseline_tilelang_cache}"
export TILELANG_TMP_DIR="${TILELANG_TMP_DIR:-$TILELANG_CACHE_DIR/tmp}"

# 数据：公开基线使用的模型键。算法：把模型键解析为统一的显示名、模型配置、HF 权重和 torch_dist reference。
resolve_baseline_model() {
  local model_key="${1:?missing baseline model key}"
  case "$model_key" in
    qwen35_4b)
      BASELINE_MODEL_DISPLAY_NAME="Qwen3.5-4B"
      BASELINE_MODEL_CONFIG="$root_path/scripts/models/qwen3.5-4B.sh"
      BASELINE_HF_CHECKPOINT="$root_path/model/Qwen3.5-4B"
      BASELINE_REF_LOAD="$root_path/model/Qwen3.5-4B_torch_dist"
      ;;
    *)
      echo "[ERROR] unsupported BASELINE_MODEL_KEY=$model_key" >&2
      return 1
      ;;
  esac
  export BASELINE_MODEL_KEY="$model_key"
  export BASELINE_MODEL_DISPLAY_NAME BASELINE_MODEL_CONFIG BASELINE_HF_CHECKPOINT BASELINE_REF_LOAD
}

activate_env() {
  local environment_name="$1"
  set +u
  if [[ -n "$conda_setup" ]]; then source "$conda_setup"; conda activate "$environment_name"; fi
  set -u
}

# 数据：harnessr1 CUDA wheel。算法：构造 Transformer Engine 可见的动态库路径。
configure_cuda_runtime() {
  [[ -n "${CONDA_PREFIX:-}" ]] || return 0
  local nvidia_root="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export CUDA_HOME="$CONDA_PREFIX"
  export CUDA_PATH="$CUDA_HOME"
  local cuda_library_path="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib"
  local package_name
  for package_name in cuda_runtime cuda_nvrtc cublas cudnn cufft curand cusolver cusparse cusparselt nccl nvjitlink nvtx; do
    local package_lib="$nvidia_root/$package_name/lib"
    if [[ -d "$package_lib" ]]; then
      cuda_library_path="${cuda_library_path:+$cuda_library_path:}$package_lib"
    fi
  done
  export LD_LIBRARY_PATH="$cuda_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

require_paths() {
  local path
  for path in "$@"; do
    [[ -e "$path" ]] || { echo "[ERROR] missing required path: $path" >&2; return 1; }
  done
}

start_ray() {
  local gpu_count="$1"
  ray stop --force >/dev/null 2>&1 || true
  ray start --head --node-ip-address "${MASTER_ADDR:-127.0.0.1}" --num-gpus "$gpu_count" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
}

start_retriever() {
  local trace_dir="$1"
  local index_path="${INDEX_PATH:-$root_path/Index/e5_Flat.index}"
  local corpus_path="${CORPUS_PATH:-$root_path/Index/wiki-18.jsonl}"
  local retriever_model="${RETRIEVER_MODEL:-$root_path/model/e5-base-v2}"
  local retriever_script="$root_path/examples/search-r1/local_dense_retriever/retrieval_server.py"
  local search_url="${UNIFIED_SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
  require_paths "$index_path" "$corpus_path" "$retriever_model" "$retriever_script"
  mkdir -p "$trace_dir"
  activate_env retriever
  CUDA_VISIBLE_DEVICES="${RETRIEVER_CUDA_VISIBLE_DEVICES:-0,1}" python "$retriever_script" \
    --index_path "$index_path" \
    --corpus_path "$corpus_path" \
    --topk "${UNIFIED_SEARCH_TOPK:-3}" \
    --retriever_name e5 \
    --retriever_model "$retriever_model" \
    --faiss_gpu > "$trace_dir/retriever.log" 2>&1 &
  retriever_pid=$!
  local attempt
  for attempt in $(seq 1 180); do
    if curl -fsS -X POST "$search_url" -H 'Content-Type: application/json' -d '{"queries":["health check"],"topk":1,"return_scores":false}' >/dev/null 2>&1; then
      export UNIFIED_SEARCH_URL="$search_url"
      return 0
    fi
    if ! kill -0 "$retriever_pid" 2>/dev/null; then
      tail -n 100 "$trace_dir/retriever.log" >&2 || true
      return 1
    fi
    sleep 10
  done
  echo "[ERROR] retriever did not become ready" >&2
  return 1
}

stop_retriever() {
  if [[ -n "${retriever_pid:-}" ]]; then
    kill "$retriever_pid" 2>/dev/null || true
    wait "$retriever_pid" 2>/dev/null || true
  fi
}

prepare_baseline_data() {
  if [[ "${SKIP_DATA_PREP:-0}" == "1" ]]; then
    require_paths "$baseline_root/data/manifest.json" "$baseline_root/data/prepared"
    return 0
  fi
  "$python_bin" -m skill_following_baselines.data --root "$baseline_root" --suite-root "$suite_root"
}
