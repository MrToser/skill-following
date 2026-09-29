# LOCKED: false
"""按实验注册表下载公开 Hugging Face 模型快照。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from ..logger import my_logger
from ..registry import DEFAULT_ROOT, expand_value, load_registry


LOGGER = my_logger("skill_following_experiment_suite.download_model")
METADATA_ALLOW_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.model",
    "*.py",
    "*.txt",
    ".gitattributes",
    "LICENSE*",
    "README*",
)
SNAPSHOT_ALLOW_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.model",
    "*.py",
    "*.safetensors",
    "*.txt",
)


# 数据：本地 Hugging Face 目录。算法：验证运行时解析模型和 tokenizer 所需的小体积元数据，不要求权重分片存在。
def metadata_ready(local_dir: Path) -> bool:
    config_path = local_dir / "config.json"
    tokenizer_config = local_dir / "tokenizer_config.json"
    if not config_path.is_file() or not tokenizer_config.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not str(config.get("model_type") or "").strip():
        return False
    tokenizer_assets = (
        local_dir / "tokenizer.json",
        local_dir / "tokenizer.model",
        local_dir / "vocab.json",
    )
    return any(path.is_file() and path.stat().st_size > 0 for path in tokenizer_assets)


# 数据：本地模型目录。算法：同时验证配置、tokenizer 与索引声明的全部权重分片，拒绝仅下载了 config 的半成品。
def snapshot_ready(local_dir: Path) -> bool:
    if not metadata_ready(local_dir):
        return False
    index_path = local_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map") or {}
        shards = {str(name) for name in weight_map.values()}
        return bool(shards) and all(
            (local_dir / shard).is_file() and (local_dir / shard).stat().st_size > 0
            for shard in shards
        )
    single_weight = local_dir / "model.safetensors"
    return single_weight.is_file() and single_weight.stat().st_size > 0


# 数据：模型键与注册表。算法：解析公开仓库和本地目录，拒绝未注册模型。
def resolve_model(model_key: str) -> tuple[str, Path]:
    registry = load_registry()
    assert model_key in registry["models"], f"unknown model key: {model_key}"
    root = Path(os.environ.get("ROOT_PATH", str(DEFAULT_ROOT))).resolve()
    model = expand_value(registry["models"][model_key], root=root, seed=0)
    repo_id = str(model["hf_repo"]).strip()
    local_dir = Path(str(model["hf_checkpoint"])).resolve()
    assert repo_id and str(local_dir)
    return repo_id, local_dir


# 数据：公开仓库与目标目录。算法：支持断点续传，并以 config.json 验证完整模型快照。
def download_model(model_key: str, *, dry_run: bool, metadata_only: bool = False) -> Path:
    repo_id, local_dir = resolve_model(model_key)
    ready_fn = metadata_ready if metadata_only else snapshot_ready
    asset_kind = "metadata" if metadata_only else "snapshot"
    if ready_fn(local_dir):
        LOGGER.info("model %s ready: model=%s path=%s", asset_kind, model_key, local_dir)
        return local_dir
    if dry_run:
        LOGGER.info(
            "model %s pending: model=%s repo=%s path=%s",
            asset_kind,
            model_key,
            repo_id,
            local_dir,
        )
        return local_dir
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "download model %s: model=%s repo=%s path=%s",
        asset_kind,
        model_key,
        repo_id,
        local_dir,
    )
    max_workers = int(os.environ.get("MODEL_DOWNLOAD_WORKERS", "1"))
    assert max_workers > 0, "MODEL_DOWNLOAD_WORKERS must be positive"
    download_kwargs = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "max_workers": max_workers,
    }
    if metadata_only:
        download_kwargs["allow_patterns"] = list(METADATA_ALLOW_PATTERNS)
    else:
        download_kwargs["allow_patterns"] = list(SNAPSHOT_ALLOW_PATTERNS)
    snapshot_download(**download_kwargs)
    assert ready_fn(local_dir), f"download finished without complete model {asset_kind}: {local_dir}"
    LOGGER.info("model %s complete: model=%s path=%s", asset_kind, model_key, local_dir)
    return local_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_key")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    download_model(
        args.model_key,
        dry_run=args.dry_run,
        metadata_only=args.metadata_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
