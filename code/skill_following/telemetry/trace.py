# LOCKED: false
"""统一训练轨迹的 loss-mask 可视化记录。"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from slime.utils.types import Sample

from ..logger import my_logger
from ..state import UnifiedSkillState


LOGGER = my_logger("skill_following_experiment_suite.trace")
TRACE_ENABLED_ENV = "SF_TRACE_TRAIN_TOKENS"
TRACE_PATH_ENV = "SF_TRACE_OUTPUT_PATH"
TRACE_MAX_SAMPLES_ENV = "SF_TRACE_MAX_SAMPLES"
TRACE_MAX_CHARS_ENV = "SF_TRACE_MAX_CHARS"
AUDIT_TRACE_PATH_ENV = "SF_AUDIT_TRACE_PATH"
DEFAULT_MAX_SAMPLES = 5
DEFAULT_MAX_CHARS = 12000
_FALLBACK_TRACE_COUNT = 0


# 数据：环境变量原始值。算法：解析常见真值字符串。
def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# 数据：环境变量名和正整数默认值。算法：非法值回退并记录 warning。
def positive_int_env(name: str, default: int) -> int:
    assert default > 0
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        LOGGER.warning("invalid %s=%r, fallback=%s", name, raw_value, default)
        return default
    return value if value > 0 else default


# 数据：长文本与字符上限。算法：保留前缀并显式报告省略字符数。
def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n...[truncated {len(text) - max_chars} chars]"


# 数据：response token 与 loss mask。算法：合并连续同 mask token 为可读分段。
def build_segments(tokenizer: Any, token_ids: list[int], loss_mask: list[int]) -> list[dict[str, Any]]:
    assert len(token_ids) == len(loss_mask)
    if not token_ids:
        return []
    segments: list[dict[str, Any]] = []
    start = 0
    current_mask = int(loss_mask[0])
    for index in range(1, len(loss_mask) + 1):
        if index < len(loss_mask) and int(loss_mask[index]) == current_mask:
            continue
        segment_ids = token_ids[start:index]
        segments.append(
            {
                "loss_mask": current_mask,
                "label": "TRAIN" if current_mask == 1 else "ENV",
                "token_count": len(segment_ids),
                "text": tokenizer.decode(segment_ids, skip_special_tokens=False),
            }
        )
        if index < len(loss_mask):
            start = index
            current_mask = int(loss_mask[index])
    return segments


# 数据：sample 全局 index 与每步采样规模。算法：每个 rollout step 只记录前 N 条。
def should_trace(args: Any, sample: Sample, max_samples: int) -> bool:
    global _FALLBACK_TRACE_COUNT
    sample_index = getattr(sample, "index", None)
    batch_size = int(getattr(args, "rollout_batch_size", 0) or 0)
    samples_per_prompt = int(getattr(args, "n_samples_per_prompt", 0) or 0)
    if sample_index is not None and batch_size > 0 and samples_per_prompt > 0:
        return int(sample_index) % (batch_size * samples_per_prompt) < max_samples
    if _FALLBACK_TRACE_COUNT >= max_samples:
        return False
    _FALLBACK_TRACE_COUNT += 1
    return True


# 数据：trace 路径与一条 JSON 记录。算法：使用 append 文件描述符避免并发覆盖。
def append_record(path: str, record: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, line.encode("utf-8"))
    finally:
        os.close(descriptor)


# 数据：完整 rollout、状态与 reward 函数。算法：记录 prompt、TRAIN/ENV 分段和同轨迹 reward 预览。
def maybe_log_trace(
    *,
    args: Any,
    tokenizer: Any,
    sample: Sample,
    prompt_text: str,
    response_token_ids: list[int],
    state: UnifiedSkillState,
    reward_fn: Callable[[Sample], dict[str, Any]],
) -> None:
    audit_path = os.environ.get(AUDIT_TRACE_PATH_ENV, "").strip()
    if audit_path:
        try:
            append_record(
                audit_path,
                {
                    "time_unix": time.time(),
                    "experiment_id": state.experiment_id,
                    "sample": {
                        "index": getattr(sample, "index", None),
                        "group_index": getattr(sample, "group_index", None),
                        "rollout_id": getattr(sample, "rollout_id", None),
                        "status": getattr(
                            getattr(sample, "status", None),
                            "value",
                            str(getattr(sample, "status", "")),
                        ),
                    },
                    "skill_state": state.to_dict(),
                    "reward": reward_fn(sample),
                },
            )
        except Exception as exc:
            LOGGER.warning("skip audit trace because of %s: %s", type(exc).__name__, exc)
    if not env_flag(TRACE_ENABLED_ENV):
        return
    trace_path = os.environ.get(TRACE_PATH_ENV, "").strip()
    if not trace_path:
        LOGGER.warning("%s is enabled but %s is empty", TRACE_ENABLED_ENV, TRACE_PATH_ENV)
        return
    max_samples = positive_int_env(TRACE_MAX_SAMPLES_ENV, DEFAULT_MAX_SAMPLES)
    if not should_trace(args, sample, max_samples):
        return
    try:
        max_chars = positive_int_env(TRACE_MAX_CHARS_ENV, DEFAULT_MAX_CHARS)
        loss_mask = list(sample.loss_mask or [])
        segments = build_segments(tokenizer, response_token_ids, loss_mask)
        for segment in segments:
            segment["text"] = truncate_text(segment["text"], max_chars)
        record = {
            "time_unix": time.time(),
            "experiment_id": state.experiment_id,
            "sample": {
                "index": getattr(sample, "index", None),
                "group_index": getattr(sample, "group_index", None),
                "rollout_id": getattr(sample, "rollout_id", None),
                "status": getattr(getattr(sample, "status", None), "value", str(getattr(sample, "status", ""))),
                "label": getattr(sample, "label", None),
            },
            "state_prompt": truncate_text(prompt_text, max_chars),
            "segments": segments,
            "skill_state": state.to_dict(),
            "reward": reward_fn(sample),
        }
        append_record(trace_path, record)
    except Exception as exc:  # 诊断失败不能中断 RL。
        LOGGER.warning("skip trace because of %s: %s", type(exc).__name__, exc)
