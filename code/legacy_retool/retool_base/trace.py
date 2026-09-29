# LOCKED: false
"""Optional token-level Retool trace diagnostics."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from slime.utils.types import Sample

from .constants import (
    DEFAULT_TRACE_MAX_CHARS,
    DEFAULT_TRACE_MAX_SAMPLES,
    TRACE_MAX_CHARS_ENV,
    TRACE_MAX_SAMPLES_ENV,
    TRACE_OUTPUT_PATH_ENV,
    TRACE_TOKEN_IDS_ENV,
    TRACE_TRAIN_TOKENS_ENV,
)
from .logger import LOGGER
from .models import RetoolRuntimeConfig, RetoolSkillState
from .reward import get_sample_status


_TRACE_PRINT_COUNT = 0
_TRACE_OUTPUT_MISSING_WARNED = False


# Data: environment variable raw string. Algorithm: parse common truthy strings into a boolean trace switch.
def env_flag(name: str) -> bool:
    assert name, "environment variable name must not be empty"
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Data: environment variable name and positive default. Algorithm: read a positive integer with warning fallback.
def read_positive_int_env(name: str, default: int) -> int:
    assert name, "environment variable name must not be empty"
    assert default > 0, "default must be positive"
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError:
        LOGGER.warning("Invalid %s=%r; fallback to %s.", name, raw_value, default)
        return default
    if value <= 0:
        LOGGER.warning("Invalid %s=%r; fallback to %s.", name, raw_value, default)
        return default
    return value


# Data: trace text and maximum length. Algorithm: keep trace files bounded while preserving readable prefixes.
def truncate_for_trace(text: str, max_chars: int) -> str:
    assert isinstance(text, str), "text must be a string"
    assert max_chars > 0, "max_chars must be positive"
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n...[truncated {omitted} chars]"


# Data: tokenizer and token id sequence. Algorithm: decode with special tokens preserved for audit alignment.
def decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    assert token_ids is not None, "token_ids must not be None"
    if not token_ids:
        return ""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(token_ids)


# Data: response token ids and same-length loss mask. Algorithm: split continuous train/env segments.
def build_loss_mask_segments(tokenizer: Any, response_token_ids: list[int], loss_mask: list[int]) -> list[dict[str, Any]]:
    assert len(response_token_ids) == len(loss_mask), "response_token_ids and loss_mask length mismatch"
    if not response_token_ids:
        return []

    segments: list[dict[str, Any]] = []
    start = 0
    current_mask = int(loss_mask[0])
    for index in range(1, len(loss_mask) + 1):
        is_boundary = index == len(loss_mask) or int(loss_mask[index]) != current_mask
        if not is_boundary:
            continue

        segment_token_ids = response_token_ids[start:index]
        segments.append(
            {
                "mask": current_mask,
                "token_count": len(segment_token_ids),
                "token_ids": segment_token_ids,
                "text": decode_token_ids(tokenizer, segment_token_ids),
            }
        )
        if index < len(loss_mask):
            start = index
            current_mask = int(loss_mask[index])

    return segments


# Data: sample fields that may or may not exist across train/eval. Algorithm: build a compact stable id.
def format_sample_id(sample: Sample) -> str:
    parts = []
    for field_name in ("group_index", "index", "rollout_id"):
        value = getattr(sample, field_name, None)
        if value is not None:
            parts.append(f"{field_name}={value}")
    return ", ".join(parts) if parts else "sample_id=unknown"


# Data: rollout args, sample index, and per-step cap. Algorithm: trace first N samples per rollout step.
def should_log_training_token_trace(args: Any, sample: Sample, max_samples: int) -> tuple[bool, str]:
    global _TRACE_PRINT_COUNT

    assert max_samples > 0, "max_samples must be positive"
    rollout_batch_size = int(getattr(args, "rollout_batch_size", 0) or 0)
    n_samples_per_prompt = int(getattr(args, "n_samples_per_prompt", 0) or 0)
    sample_index = getattr(sample, "index", None)
    if sample_index is not None and rollout_batch_size > 0 and n_samples_per_prompt > 0:
        samples_per_step = rollout_batch_size * n_samples_per_prompt
        step_index = int(sample_index) // samples_per_step
        step_position = int(sample_index) % samples_per_step
        trace_context = f"rollout_step={step_index}, step_sample_position={step_position}"
        return step_position < max_samples, trace_context

    if _TRACE_PRINT_COUNT >= max_samples:
        return False, "rollout_step=unknown, fallback_scope=process"
    _TRACE_PRINT_COUNT += 1
    return True, f"rollout_step=unknown, fallback_scope=process, process_trace_index={_TRACE_PRINT_COUNT - 1}"


# Data: trace output path and one JSON-safe record. Algorithm: append one compact JSON line atomically.
def append_trace_record(trace_path: str, record: dict[str, Any]) -> None:
    assert trace_path.strip(), "trace_path must not be empty"
    parent_dir = os.path.dirname(trace_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    file_descriptor = os.open(trace_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(file_descriptor, line.encode("utf-8"))
    finally:
        os.close(file_descriptor)


# Data: generated sample, tokenizer, loss mask, protocol state. Algorithm: write Search-style token trace JSONL.
def maybe_log_training_token_trace(
    *,
    args: Any,
    tokenizer: Any,
    sample: Sample,
    prompt_text: str,
    response_token_ids: list[int],
    loss_mask: list[int],
    skill_state: RetoolSkillState,
    runtime: RetoolRuntimeConfig,
) -> None:
    global _TRACE_OUTPUT_MISSING_WARNED

    if not env_flag(TRACE_TRAIN_TOKENS_ENV):
        return

    trace_path = os.environ.get(TRACE_OUTPUT_PATH_ENV, "").strip()
    if not trace_path:
        if not _TRACE_OUTPUT_MISSING_WARNED:
            LOGGER.warning("%s is enabled but %s is not set; skip token trace.", TRACE_TRAIN_TOKENS_ENV, TRACE_OUTPUT_PATH_ENV)
            _TRACE_OUTPUT_MISSING_WARNED = True
        return

    max_samples = read_positive_int_env(TRACE_MAX_SAMPLES_ENV, DEFAULT_TRACE_MAX_SAMPLES)
    should_log, trace_context = should_log_training_token_trace(args, sample, max_samples)
    if not should_log:
        return

    try:
        max_chars = read_positive_int_env(TRACE_MAX_CHARS_ENV, DEFAULT_TRACE_MAX_CHARS)
        include_token_ids = env_flag(TRACE_TOKEN_IDS_ENV)
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        segments = build_loss_mask_segments(tokenizer, response_token_ids, loss_mask)
        train_token_count = sum(segment["token_count"] for segment in segments if segment["mask"] == 1)
        state_token_count = len(prompt_token_ids) + sum(segment["token_count"] for segment in segments if segment["mask"] == 0)
        trace_segments = []
        for index, segment in enumerate(segments, start=1):
            trace_segment = {
                "segment_index": index,
                "label": "TRAIN" if segment["mask"] == 1 else "ENV",
                "loss_mask": segment["mask"],
                "token_count": segment["token_count"],
                "text": truncate_for_trace(segment["text"], max_chars),
            }
            if include_token_ids:
                trace_segment["token_ids"] = segment["token_ids"]
            trace_segments.append(trace_segment)

        reward_preview: dict[str, Any]
        try:
            reward_preview = runtime.compute_reward(sample)
        except Exception as exc:  # pragma: no cover - trace must never fail rollout
            reward_preview = {"trace_reward_error": f"{type(exc).__name__}: {exc}"}

        full_skill_state = skill_state.to_dict()
        if runtime.trace_skill_state_fields is None:
            trace_skill_state = full_skill_state
        else:
            missing_state_fields = set(runtime.trace_skill_state_fields) - full_skill_state.keys()
            assert not missing_state_fields, f"trace state missing fields: {sorted(missing_state_fields)}"
            trace_skill_state = {
                field_name: full_skill_state[field_name]
                for field_name in runtime.trace_skill_state_fields
            }

        if runtime.trace_reward_fields is None or "trace_reward_error" in reward_preview:
            trace_reward = reward_preview
        else:
            missing_reward_fields = set(runtime.trace_reward_fields) - reward_preview.keys()
            assert not missing_reward_fields, f"trace reward missing fields: {sorted(missing_reward_fields)}"
            trace_reward = {
                field_name: reward_preview[field_name]
                for field_name in runtime.trace_reward_fields
            }

        trace_record = {
            "time_unix": time.time(),
            "sample": {
                "group_index": getattr(sample, "group_index", None),
                "index": getattr(sample, "index", None),
                "rollout_id": getattr(sample, "rollout_id", None),
                "status": get_sample_status(sample),
                "label": getattr(sample, "label", None),
            },
            "trace_context": trace_context,
            "tokens": {
                "prompt_state": len(prompt_token_ids),
                "response_total": len(response_token_ids),
                "train_mask_1": train_token_count,
                "state_mask_0": state_token_count,
            },
            "skill_state": trace_skill_state,
            "reward": trace_reward,
            "state_prompt": truncate_for_trace(prompt_text, max_chars),
            "segments": trace_segments,
        }
        if runtime.include_redundant_trace_views:
            response_text = decode_token_ids(tokenizer, response_token_ids)
            train_only_text = "".join(segment["text"] for segment in segments if segment["mask"] == 1)
            trace_record |= {
                "pid": os.getpid(),
                "sample_id": format_sample_id(sample),
                "response_trajectory": truncate_for_trace(response_text, max_chars),
                "full_trajectory": truncate_for_trace(prompt_text + response_text, max_chars),
                "train_only_trajectory": truncate_for_trace(train_only_text, max_chars),
            }
        append_trace_record(trace_path, trace_record)
    except Exception as exc:  # pragma: no cover - diagnostics should not interrupt RL
        LOGGER.warning("Skip Retool skill token trace because of %s: %s", type(exc).__name__, exc)
