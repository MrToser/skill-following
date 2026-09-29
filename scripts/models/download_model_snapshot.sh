#!/usr/bin/env bash
# LOCKED: false

set -euo pipefail
source "$(dirname "$0")/../core/common.sh"

model_key="${1:?usage: download_model_snapshot.sh MODEL_KEY [--dry-run] [--metadata-only]}"
shift
download_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--metadata-only)
      download_args+=("$1")
      shift
      ;;
    *)
      echo "[ERROR] usage: download_model_snapshot.sh MODEL_KEY [--dry-run] [--metadata-only]" >&2
      exit 1
      ;;
  esac
done

# 数据：公开 Hugging Face 模型。算法：使用当前环境代理配置执行可恢复下载。
# 数据：代理网络与 Hugging Face Hub 传输后端。
# 算法：禁用在当前代理下可能长时间无进度的 Xet 通道，保留标准 HTTP 断点续传并放宽大分片超时。
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
"$python_bin" -m "$package_name.models.download" "$model_key" "${download_args[@]}"
